"""Per-agent discovery/control server and proxy registration."""

from __future__ import annotations

import json
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx


def _find_port(start: int, count: int) -> int:
    for offset in range(count):
        port = start + offset
        if port <= 0 or port > 65535:
            continue
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
        finally:
            probe.close()
    raise RuntimeError(
        f"No free port in range {start}-{start + count - 1}; adjust `agent_port_count`"
    )


class AgentDiscovery:
    def __init__(
        self,
        agent: Any,
        port_start: int,
        port_count: int,
        heartbeat: float = 2.0,
    ) -> None:
        self.agent = agent
        self.port_start = int(port_start)
        self.port_count = max(1, int(port_count))
        self.heartbeat = max(0.5, float(heartbeat))
        self.port: int | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._registered = False
        self._last_error: str | None = None
        self._register_thread: threading.Thread | None = None

    @property
    def url(self) -> str | None:
        return f"http://127.0.0.1:{self.port}" if self.port else None

    def start(self) -> None:
        port = _find_port(self.port_start, self.port_count)
        discovery = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "OsmiaAgent/1.0"

            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ANN001
                return

            def _json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    data = json.loads(raw.decode("utf-8"))
                except Exception:
                    return {}
                return data if isinstance(data, dict) else {}

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/health":
                    self._json(HTTPStatus.OK, {"ok": True, "id": discovery.agent.id})
                elif parsed.path in ("/state", "/"):
                    query = parse_qs(parsed.query)
                    include = (query.get("messages") or ["1"])[0] != "0"
                    self._json(
                        HTTPStatus.OK,
                        discovery.agent.state(include_messages=include),
                    )
                elif parsed.path in ("/events", "/events.json"):
                    query = parse_qs(parsed.query)
                    after = int((query.get("after") or query.get("since") or ["0"])[0])
                    limit = min(1000, int((query.get("limit") or ["200"])[0]))
                    self._json(
                        HTTPStatus.OK,
                        {"object": "list", "data": discovery.agent.events.since(after, limit)},
                    )
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "unknown route"})

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/steer":
                    payload = self._body()
                    text = str(payload.get("text") or "").strip()
                    if not text:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": "text is required"})
                        return
                    seq = discovery.agent.steer(text)
                    self._json(HTTPStatus.OK, {"ok": True, "seq": seq})
                elif path == "/stop":
                    discovery.agent.stop()
                    self._json(HTTPStatus.OK, {"ok": True})
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "unknown route"})

        try:
            self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        except OSError as exc:
            raise RuntimeError(f"Cannot bind agent discovery port {port}: {exc}") from exc
        self._server.daemon_threads = True
        self.port = port
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.agent.events.append("discovery.start", {"url": self.url})
        self._register_thread = threading.Thread(target=self._register_loop, daemon=True)
        self._register_thread.start()

    def _register_loop(self) -> None:
        self._put_state()
        while not self._stop.wait(self.heartbeat):
            self._put_state()

    def _put_state(self) -> None:
        if not self.url:
            return
        snapshot = self.agent.state(include_messages=False)
        snapshot["port"] = self.port
        snapshot["attach_url"] = self.url
        snapshot["events"] = self.agent.events.snapshot(limit=100)
        snapshot["last_event"] = self.agent.events.last_seq
        try:
            response = httpx.put(
                f"{self.agent.registry_url}/v1/agents/{self.agent.id}",
                json=snapshot,
                timeout=1.5,
            )
            ok = response.status_code < 400
            if ok:
                self._registered = True
                self._last_error = None
            else:
                self._registered = False
                self._last_error = f"proxy returned {response.status_code}"
        except Exception as exc:
            if self._registered or self._last_error is None:
                self.agent.events.append(
                    "registry.error", {"message": f"{type(exc).__name__}: {exc}"}
                )
            self._registered = False
            self._last_error = str(exc)

    def stop(self) -> None:
        self._stop.set()
        if self._registered:
            try:
                httpx.delete(
                    f"{self.agent.registry_url}/v1/agents/{self.agent.id}",
                    timeout=1.5,
                )
            except Exception:
                pass
        if self._register_thread is not None:
            self._register_thread.join(timeout=2)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.port = None
        self.agent.events.append("discovery.stop", {})
