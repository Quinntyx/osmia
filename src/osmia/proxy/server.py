"""Local OpenAI-compatible proxy with model slugs and agent control plane."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid

import httpx
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse

from .. import __version__
from ..events import EventLog
from ..providers import Provider, ProviderStreamError, provider_for
from .config import Config, ModelRoute

MAX_BODY = 32 * 1024 * 1024
AGENT_RE = re.compile(r"^/v1/agents/([A-Za-z0-9_.:-]+)(?:/(steer|stop|prompt))?$")


class ProxyError(RuntimeError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def aggregate_stream(chunks: Iterator[dict[str, Any]], model: str) -> dict[str, Any]:
    """Collect OpenAI streaming chunks into one Chat Completions response."""
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    completion_id: str | None = None
    created: int | None = None

    for chunk in chunks:
        completion_id = completion_id or chunk.get("id")
        created = created or chunk.get("created")
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(str(delta["content"]))
            if delta.get("reasoning_content"):
                reasoning.append(str(delta["reasoning_content"]))
            for call in delta.get("tool_calls") or []:
                index = int(call.get("index", 0))
                slot = tool_calls.setdefault(
                    index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if call.get("id"):
                    slot["id"] = call["id"]
                function = call.get("function") or {}
                if function.get("name"):
                    slot["function"]["name"] += function["name"]
                if function.get("arguments"):
                    slot["function"]["arguments"] += function["arguments"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content) if content else None,
    }
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    return {
        "id": completion_id or "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
                or ("tool_calls" if tool_calls else "stop"),
            }
        ],
        "usage": usage
        or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


class Proxy:
    """Startable local proxy.  Construct, ``start()`` (or ``serve()``), then use."""

    def __init__(
        self,
        config: Config | None = None,
        config_path: str | Path | None = None,
        host: str | None = None,
        port: int | None = None,
        providers: dict[str, Provider] | None = None,
    ) -> None:
        self.config = config or Config.load(
            Path(config_path) if config_path is not None else None
        )
        self.host = host or self.config.host
        self.port = self.config.port if port is None else int(port)
        self._providers: dict[str, Provider] = dict(providers or {})
        self._provider_lock = threading.Lock()
        self.agents: dict[str, dict[str, Any]] = {}
        self.agent_events: dict[str, EventLog] = {}
        self.steering: dict[str, list[dict[str, Any]]] = {}
        self._registry_lock = threading.Lock()
        self.events = EventLog()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.started_at = time.time()

    @property
    def bound_port(self) -> int:
        return int(self._server.server_address[1]) if self._server else self.port

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.bound_port}"

    def provider(self, name: str) -> Provider:
        with self._provider_lock:
            provider = self._providers.get(name)
            if provider is None:
                provider = provider_for(name, self.config)
                self._providers[name] = provider
            return provider

    def start(self) -> Proxy:
        if self._server is not None:
            return self
        try:
            server = ThreadingHTTPServer((self.host, self.port), _Handler)
        except OSError as exc:
            raise ProxyError(
                f"Cannot bind {self.host}:{self.port}: {exc}. Is another Osmia proxy running?"
            ) from exc
        server.daemon_threads = True
        server.proxy = self  # type: ignore[attr-defined]
        self._server = server
        if self.port == 0:
            self.port = server.server_address[1]
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        self.events.append("proxy.start", {"host": self.host, "port": self.bound_port})
        return self

    def serve(self) -> None:
        self.start()
        assert self._server is not None
        try:
            self._server.serve_forever()
        finally:
            self.stop()

    def stop(self) -> None:
        server, self._server = self._server, None
        thread, self._thread = self._thread, None
        if server is not None:
            server.shutdown()
            server.server_close()
            self.events.append("proxy.stop", {})
        if thread is not None:
            thread.join(timeout=5)

    def __enter__(self) -> Proxy:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def resolve(self, model: str | None) -> ModelRoute:
        try:
            return self.config.route(model)
        except KeyError as exc:
            raise ProxyError(str(exc).strip("'\""), status=404) from exc

    def model_list(self) -> dict[str, Any]:
        data = []
        for slug, route in sorted(self.config.routes().items()):
            data.append(
                {
                    "id": slug,
                    "object": "model",
                    "created": int(self.started_at),
                    "owned_by": route.provider,
                }
            )
        return {"object": "list", "data": data}

    def register_agent(self, agent_id: str, state: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        with self._registry_lock:
            state = dict(state)
            state["id"] = agent_id
            state["last_seen"] = now
            state["stale"] = False
            self.agents[agent_id] = state
            self.agent_events.setdefault(agent_id, EventLog())
        return state

    def public_agent(self, agent_id: str) -> dict[str, Any] | None:
        with self._registry_lock:
            state = self.agents.get(agent_id)
            if state is None:
                return None
            state = dict(state)
            age = time.time() - float(state.get("last_seen") or 0)
            state["stale"] = age > self.config.agent_stale_seconds
            state.pop("last_seen", None)
            return state

    def list_agents(self) -> list[dict[str, Any]]:
        with self._registry_lock:
            ids = sorted(self.agents)
        return [state for agent_id in ids if (state := self.public_agent(agent_id))]

    def unregister_agent(self, agent_id: str) -> None:
        with self._registry_lock:
            self.agents.pop(agent_id, None)
            self.agent_events.pop(agent_id, None)
            self.steering.pop(agent_id, None)

    def event_snapshot(
        self, agent_id: str | None, after: int, limit: int
    ) -> list[dict[str, Any]]:
        if agent_id:
            log = self.agent_events.get(agent_id)
            if log is None:
                return []
            return log.since(after, limit)
        return self.events.since(after, limit)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Osmia/" + __version__

    @property
    def proxy(self) -> Proxy:
        return self.server.proxy  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ANN001
        return

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header(
            "Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS"
        )
        self.send_header("Access-Control-Max-Age", "600")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send_json(
            status,
            {"error": {"message": message, "type": "osmia_error", "code": status}},
        )

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ProxyError("Invalid Content-Length") from exc
        if length > MAX_BODY:
            raise ProxyError("Request body is too large", status=413)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProxyError(f"Request body is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ProxyError("Request body must be a JSON object")
        return data

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "service": "osmia-proxy",
                        "version": __version__,
                        "default_model": self.proxy.config.default_model,
                        "models": sorted(self.proxy.config.routes()),
                        "agents": len(self.proxy.agents),
                        "uptime": time.time() - self.proxy.started_at,
                    },
                )
            elif parsed.path in ("/v1/models", "/models"):
                self._send_json(HTTPStatus.OK, self.proxy.model_list())
            elif parsed.path in ("/v1/agents", "/agents"):
                self._send_json(
                    HTTPStatus.OK,
                    {"object": "list", "data": self.proxy.list_agents()},
                )
            elif parsed.path in ("/v1/events", "/events", "/v1/events.json"):
                query = parse_qs(parsed.query)
                after = int((query.get("after") or query.get("since") or ["0"])[0])
                limit = min(1000, int((query.get("limit") or ["200"])[0]))
                agent_id = (query.get("agent") or [None])[0]
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "data": self.proxy.event_snapshot(agent_id, after, limit),
                    },
                )
            else:
                match = AGENT_RE.match(parsed.path)
                if match:
                    state = self.proxy.public_agent(match.group(1))
                    if state is None:
                        self._error(404, "Unknown agent")
                    else:
                        self._send_json(HTTPStatus.OK, state)
                else:
                    self._error(404, f"Unknown route {parsed.path}")
        except (ProxyError, ValueError) as exc:
            self._error(getattr(exc, "status", 400), str(exc))

    def do_PUT(self) -> None:  # noqa: N802
        match = AGENT_RE.match(urlparse(self.path).path)
        if not match or match.group(2):
            self._error(404, "Unknown route")
            return
        try:
            payload = self._read_json()
            state = self.proxy.register_agent(match.group(1), payload)
            self._send_json(
                HTTPStatus.OK,
                {"ok": True, "id": state["id"], "stale": False},
            )
        except ProxyError as exc:
            self._error(exc.status, str(exc))

    def do_DELETE(self) -> None:  # noqa: N802
        match = AGENT_RE.match(urlparse(self.path).path)
        if not match or match.group(2):
            self._error(404, "Unknown route")
            return
        self.proxy.unregister_agent(match.group(1))
        self._send_json(HTTPStatus.OK, {"ok": True})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/v1/chat/completions", "/chat/completions"):
            self._chat_completions()
            return
        if path in ("/v1/register", "/v1/heartbeat"):
            self._register_alias()
            return
        match = AGENT_RE.match(path)
        if match and match.group(2):
            self._agent_control(match.group(1), match.group(2))
            return
        self._error(404, f"Unknown route {path}")

    def _register_alias(self) -> None:
        try:
            payload = self._read_json()
        except ProxyError as exc:
            self._error(exc.status, str(exc))
            return
        agent_id = payload.pop("id", None) or payload.pop("agent_id", None)
        if not agent_id:
            self._error(400, "`id` is required")
            return
        state = self.proxy.register_agent(str(agent_id), payload)
        self._send_json(
            HTTPStatus.OK, {"ok": True, "id": state["id"], "port": state.get("port")}
        )

    def _forward(self, attach_url: str, action: str, body: dict[str, Any]) -> bool:
        try:
            response = httpx.post(
                attach_url.rstrip("/") + "/" + action,
                json=body,
                timeout=2.0,
            )
            return response.status_code < 400
        except Exception:
            return False

    def _agent_control(self, agent_id: str, action: str) -> None:
        state = self.proxy.public_agent(agent_id)
        if state is None:
            self._error(404, "Unknown agent")
            return
        attach_url = state.get("attach_url") or state.get("url")
        if action == "stop":
            if attach_url:
                self._forward(str(attach_url), "stop", {})
            with self.proxy._registry_lock:
                self.proxy.agents[agent_id]["stop_requested"] = True
            self._send_json(HTTPStatus.OK, {"ok": True, "action": "stop"})
            return
        try:
            payload = self._read_json()
        except ProxyError as exc:
            self._error(exc.status, str(exc))
            return
        text = str(payload.get("text") or "").strip()
        if not text:
            self._error(
                400,
                "Prompt text is required"
                if action == "prompt"
                else "Steering message text is required",
            )
            return
        if action == "prompt":
            # A prompt is a new user turn, so it must reach the agent itself; there is
            # no proxy-side queue that could replay it later.
            if not attach_url:
                self._error(409, "Agent has no control endpoint to receive a prompt")
                return
            if not self._forward(str(attach_url), "prompt", {"text": text}):
                self._error(502, "Agent control endpoint is unreachable")
                return
            self._send_json(
                HTTPStatus.ACCEPTED,
                {"ok": True, "action": "prompt", "forwarded": True},
            )
            return
        if attach_url and self._forward(str(attach_url), "steer", {"text": text}):
            self._send_json(
                HTTPStatus.OK, {"ok": True, "action": "steer", "forwarded": True}
            )
            return
        seq = self.proxy.agent_events.setdefault(agent_id, EventLog()).append(
            "steer.queued", {"text": text}
        )
        with self.proxy._registry_lock:
            self.proxy.steering.setdefault(agent_id, []).append(
                {"text": text, "seq": seq}
            )
        self._send_json(HTTPStatus.OK, {"ok": True, "action": "steer", "seq": seq})

    def _chat_completions(self) -> None:
        try:
            payload = self._read_json()
        except ProxyError as exc:
            self._error(exc.status, str(exc))
            return
        try:
            route = self.proxy.resolve(payload.get("model"))
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ProxyError("`messages` must be a non-empty list")
            stream = bool(payload.get("stream"))
            provider = self.proxy.provider(route.provider)
            started = time.time()
            chunks = provider.stream(route, payload)
            if stream:
                self._stream_response(route, chunks, provider, started)
            else:
                try:
                    result = aggregate_stream(chunks, route.slug)
                finally:
                    provider.close()
                self.proxy.events.append(
                    "chat.completion",
                    {
                        "model": route.slug,
                        "provider": route.provider,
                        "stream": False,
                        "duration": time.time() - started,
                    },
                )
                self._send_json(HTTPStatus.OK, result)
        except ProxyError as exc:
            self._error(exc.status, str(exc))
        except ProviderStreamError as exc:
            self._error(exc.status, str(exc))
        except Exception as exc:
            self._error(500, f"Internal Osmia proxy error: {type(exc).__name__}: {exc}")

    def _stream_response(
        self,
        route: ModelRoute,
        chunks: Iterator[dict[str, Any]],
        provider: Provider,
        started: float,
    ) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()
        self.close_connection = True
        try:
            for chunk in chunks:
                chunk["model"] = route.slug
                data = json.dumps(chunk, ensure_ascii=False)
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.proxy.events.append(
                "chat.completion",
                {
                    "model": route.slug,
                    "provider": route.provider,
                    "stream": True,
                    "duration": time.time() - started,
                },
            )
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                error = json.dumps(
                    {
                        "error": {
                            "message": f"{type(exc).__name__}: {exc}",
                            "type": "osmia_stream_error",
                        }
                    }
                )
                self.wfile.write(f"data: {error}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass
        finally:
            provider.close()
