"""Upstream providers behind the OpenAI Chat Completions surface."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterator

import httpx

from .auth import ensure_codex_auth, refresh_codex, LoginError
from .proxy.config import Config, ModelRoute


class ProviderStreamError(RuntimeError):
    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


def _new_completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _chunk(
    model: str,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
    completion_id: str | None = None,
    created: int | None = None,
) -> dict[str, Any]:
    choices: list[dict[str, Any]] = []
    if delta is not None or finish_reason is not None:
        choices.append(
            {
                "index": 0,
                "delta": delta or {},
                "finish_reason": finish_reason,
            }
        )
    chunk: dict[str, Any] = {
        "id": completion_id or _new_completion_id(),
        "object": "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


CHAT_FIELDS = {
    "messages",
    "tools",
    "tool_choice",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "response_format",
    "logprobs",
    "top_logprobs",
    "parallel_tool_calls",
    "user",
    "metadata",
    "modalities",
    "prediction",
    "reasoning",
    "reasoning_effort",
    "verbosity",
    "stream_options",
}


class Provider:
    name = "base"

    def stream(
        self, route: ModelRoute, payload: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class OpenRouterProvider(Provider):
    name = "openrouter"

    def __init__(self, config: Config) -> None:
        self.config = config

    def _api_key(self) -> str:
        key = self.config.provider("openrouter").get("api_key")
        if not key:
            raise ProviderStreamError(
                "OpenRouter is not logged in. Run `osmia login openrouter`.", status=401
            )
        return str(key)

    def stream(
        self, route: ModelRoute, payload: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        key = self._api_key()
        body: dict[str, Any] = {
            key_name: payload[key_name]
            for key_name in CHAT_FIELDS
            if key_name in payload and payload[key_name] is not None
        }
        body["model"] = route.model
        body["stream"] = True
        stream_options = dict(body.get("stream_options") or {})
        stream_options["include_usage"] = True
        body["stream_options"] = stream_options

        base_url = str(route.options.get("base_url") or "https://openrouter.ai/api/v1")
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://osmia.local",
            "X-Title": "Osmia",
        }
        headers.update(route.options.get("headers") or {})

        try:
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
                with client.stream("POST", url, json=body, headers=headers) as response:
                    if response.status_code >= 400:
                        text = response.read().decode("utf-8", "replace")
                        raise ProviderStreamError(
                            f"OpenRouter returned {response.status_code}: {text}",
                            status=response.status_code,
                        )
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", "replace")
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(event, dict) and event.get("error"):
                            raise ProviderStreamError(f"OpenRouter error: {event['error']}")
                        if isinstance(event, dict):
                            yield event
        except httpx.HTTPError as exc:
            raise ProviderStreamError(f"OpenRouter request failed: {exc}") from exc


def _content_parts(content: Any, role: str) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        if not content:
            return []
        kind = "input_text" if role == "user" else "output_text"
        return [{"type": kind, "text": content}]
    if not isinstance(content, list):
        return [{"type": "input_text", "text": str(content)}]
    result: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            result.append(
                {"type": "input_text" if role == "user" else "output_text", "text": part}
            )
        elif isinstance(part, dict):
            part_type = part.get("type")
            if part_type in ("text", "input_text", "output_text"):
                result.append(
                    {
                        "type": "input_text" if role == "user" else "output_text",
                        "text": part.get("text", ""),
                    }
                )
            elif part_type == "image_url":
                image = part.get("image_url")
                if isinstance(image, dict):
                    image = image.get("url")
                if image:
                    result.append({"type": "input_image", "image_url": image})
    return result


def chat_to_responses(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate OpenAI Chat Completions messages/tools into Responses input."""
    messages = payload.get("messages") or []
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in ("system", "developer"):
            text = content if isinstance(content, str) else json.dumps(content)
            if text:
                instructions.append(text)
            continue
        if role == "user":
            parts = _content_parts(content, "user")
            input_items.append({"role": "user", "content": parts})
            continue
        if role == "assistant":
            parts = _content_parts(content, "assistant")
            if parts:
                input_items.append({"role": "assistant", "content": parts})
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id") or "",
                        "name": function.get("name") or "",
                        "arguments": function.get("arguments") or "{}",
                    }
                )
            continue
        if role in ("tool", "function"):
            call_id = message.get("tool_call_id") or message.get("name") or ""
            text = content if isinstance(content, str) else json.dumps(content)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": text or "",
                }
            )

    tools: list[dict[str, Any]] = []
    for definition in payload.get("tools") or []:
        if not isinstance(definition, dict):
            continue
        if definition.get("type") != "function":
            continue
        function = definition.get("function") or {}
        if not function.get("name"):
            continue
        tools.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function.get("description") or "",
                "parameters": function.get("parameters") or {"type": "object"},
                "strict": False,
            }
        )

    body: dict[str, Any] = {
        "instructions": "\n\n".join(instructions),
        "input": input_items,
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    if tools:
        body["tools"] = tools
    if payload.get("tool_choice") is not None:
        choice = payload["tool_choice"]
        if isinstance(choice, dict) and isinstance(choice.get("function"), dict):
            choice = {"type": "function", "name": choice["function"].get("name")}
        body["tool_choice"] = choice
    if payload.get("parallel_tool_calls") is not None:
        body["parallel_tool_calls"] = bool(payload["parallel_tool_calls"])
    else:
        body["parallel_tool_calls"] = True
    if payload.get("reasoning") is not None:
        body["reasoning"] = payload["reasoning"]
    elif payload.get("reasoning_effort"):
        body["reasoning"] = {"effort": payload["reasoning_effort"]}
    if payload.get("verbosity"):
        body["text"] = {"verbosity": payload["verbosity"]}
    return body


def _usage_from_response(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    return {
        "prompt_tokens": int(usage.get("input_tokens") or 0),
        "completion_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "prompt_tokens_details": {
            "cached_tokens": int(details.get("cached_tokens") or 0)
        },
        "completion_tokens_details": {
            "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0)
        },
    }


def _ws_connect(ws_client: Any, ws_url: str, headers: dict[str, str]) -> Any:
    """Connect while staying compatible with several websockets releases."""
    common = {"open_timeout": 30, "close_timeout": 5, "max_size": None}
    variants = [
        {"additional_headers": headers, "legacy": True},
        {"additional_headers": headers},
        {"extra_headers": headers},
    ]
    last_type_error: TypeError | None = None
    for variant in variants:
        try:
            return ws_client.connect(ws_url, **common, **variant)
        except TypeError as exc:
            last_type_error = exc
            continue
    raise last_type_error or TypeError("Unsupported websockets connect signature")


class CodexProvider(Provider):
    name = "codex"

    def __init__(self, config: Config) -> None:
        self.config = config

    def _headers(self, values: dict[str, Any], session_id: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {values['access_token']}",
            "originator": "osmia",
            "OpenAI-Beta": "responses_websockets=2026-02-06",
            "session_id": session_id,
            "conversation_id": session_id,
            "x-client-request-id": str(uuid.uuid4()),
            "User-Agent": "osmia/0.1",
        }
        if values.get("account_id"):
            headers["chatgpt-account-id"] = str(values["account_id"])
        return headers

    def stream(
        self, route: ModelRoute, payload: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        session_id = str(payload.get("prompt_cache_key") or uuid.uuid4().hex)
        body = chat_to_responses(payload)
        body["model"] = route.model
        body["prompt_cache_key"] = session_id
        for key in ("reasoning", "text", "service_tier", "instructions"):
            if key in route.options:
                body[key] = route.options[key]
        for key in ("tools", "tool_choice", "parallel_tool_calls", "include", "store"):
            if key in route.options:
                body[key] = route.options[key]

        base_url = str(
            route.options.get("base_url") or "https://chatgpt.com/backend-api/codex"
        ).rstrip("/")
        http_url = base_url + "/responses"
        ws_url = http_url.replace("https://", "wss://", 1).replace(
            "http://", "ws://", 1
        )
        send_body = dict(body)
        send_body["type"] = "response.create"

        try:
            values = ensure_codex_auth(self.config)
        except LoginError as exc:
            raise ProviderStreamError(str(exc), status=401) from exc

        events = self._run_websocket(
            ws_url,
            send_body,
            self._headers(values, session_id),
            route.model,
            session_id,
        )
        try:
            for chunk in events:
                yield chunk
        except ProviderStreamError:
            raise
        except Exception as exc:
            raise ProviderStreamError(f"Codex stream failed: {exc}") from exc

    def _run_websocket(
        self,
        ws_url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        model: str,
        session_id: str,
    ) -> Iterator[dict[str, Any]]:
        import websockets.sync.client as ws_client
        from websockets.exceptions import InvalidStatus

        last_error: Exception | None = None
        connection = None
        for attempt in range(2):
            try:
                connection = _ws_connect(ws_client, ws_url, headers)
                break
            except InvalidStatus as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 401 and attempt == 0:
                    values = refresh_codex(self.config)
                    headers = self._headers(values, session_id)
                    continue
                raise ProviderStreamError(f"Codex WebSocket handshake failed: {exc}") from exc
            except Exception as exc:
                raise ProviderStreamError(
                    f"Codex WebSocket connection failed: {exc}"
                ) from exc
        if connection is None:
            raise ProviderStreamError(f"Codex WebSocket connection failed: {last_error}")

        completion_id = _new_completion_id()
        created = int(time.time())
        tool_buffers: dict[str, dict[str, Any]] = {}
        emitted_text = False

        try:
            connection.send(json.dumps(body))
            while True:
                try:
                    raw = connection.recv(timeout=600)
                except TimeoutError as exc:
                    raise ProviderStreamError("Codex stream timed out waiting for events") from exc
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "replace")
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                kind = event.get("type")

                if kind == "response.output_text.delta" or kind == "response.refusal.delta":
                    text = event.get("delta") or ""
                    if text:
                        emitted_text = True
                        yield _chunk(
                            model,
                            {"content": text},
                            completion_id=completion_id,
                            created=created,
                        )
                elif kind in (
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                ):
                    text = event.get("delta") or ""
                    if text:
                        yield _chunk(
                            model,
                            {"reasoning_content": text},
                            completion_id=completion_id,
                            created=created,
                        )
                elif kind == "response.output_item.added":
                    item = event.get("item") or {}
                    if item.get("type") != "function_call":
                        continue
                    index = int(event.get("output_index") or len(tool_buffers))
                    item_id = str(item.get("id") or event.get("item_id") or index)
                    buffer = {
                        "index": index,
                        "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}",
                        "name": item.get("name") or "",
                        "arguments": item.get("arguments") or "",
                        "emitted": 0,
                        "header": False,
                        "item_id": item_id,
                    }
                    tool_buffers[item_id] = buffer
                    buffer["header"] = True
                    yield _chunk(
                        model,
                        {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": buffer["id"],
                                    "type": "function",
                                    "function": {
                                        "name": buffer["name"],
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                        completion_id=completion_id,
                        created=created,
                    )
                elif kind == "response.function_call_arguments.delta":
                    item_id = str(event.get("item_id") or "")
                    buffer = tool_buffers.get(item_id)
                    if buffer is None:
                        continue
                    delta = event.get("delta") or ""
                    if delta:
                        buffer["arguments"] += delta
                        buffer["emitted"] += len(delta)
                        yield _chunk(
                            model,
                            {
                                "tool_calls": [
                                    {
                                        "index": buffer["index"],
                                        "function": {"arguments": delta},
                                    }
                                ]
                            },
                            completion_id=completion_id,
                            created=created,
                        )
                elif kind == "response.output_item.done":
                    item = event.get("item") or {}
                    if item.get("type") != "function_call":
                        continue
                    item_id = str(item.get("id") or event.get("item_id") or "")
                    buffer = tool_buffers.get(item_id)
                    arguments = item.get("arguments") or ""
                    if buffer is None:
                        index = int(event.get("output_index") or len(tool_buffers))
                        buffer = {
                            "index": index,
                            "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}",
                            "name": item.get("name") or "",
                            "arguments": "",
                            "emitted": 0,
                            "header": True,
                            "item_id": item_id or str(index),
                        }
                        tool_buffers[buffer["item_id"]] = buffer
                        yield _chunk(
                            model,
                            {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": buffer["id"],
                                        "type": "function",
                                        "function": {"name": buffer["name"], "arguments": ""},
                                    }
                                ]
                            },
                            completion_id=completion_id,
                            created=created,
                        )
                    elif not buffer["name"] and item.get("name"):
                        buffer["name"] = str(item["name"])
                    if len(arguments) > buffer["emitted"]:
                        remaining = arguments[buffer["emitted"] :]
                        buffer["arguments"] = arguments
                        buffer["emitted"] = len(arguments)
                        yield _chunk(
                            model,
                            {
                                "tool_calls": [
                                    {
                                        "index": buffer["index"],
                                        "function": {"arguments": remaining},
                                    }
                                ]
                            },
                            completion_id=completion_id,
                            created=created,
                        )
                elif kind in ("response.completed", "response.done", "response.incomplete"):
                    response = event.get("response") or {}
                    usage = _usage_from_response(response.get("usage"))
                    if not tool_buffers:
                        for index, item in enumerate(response.get("output") or []):
                            if item.get("type") != "function_call":
                                continue
                            call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
                            name = item.get("name") or ""
                            arguments = item.get("arguments") or "{}"
                            tool_buffers[str(item.get("id") or index)] = {
                                "index": index,
                                "id": call_id,
                                "name": name,
                                "arguments": arguments,
                                "emitted": len(arguments),
                                "header": True,
                            }
                            yield _chunk(
                                model,
                                {
                                    "tool_calls": [
                                        {
                                            "index": index,
                                            "id": call_id,
                                            "type": "function",
                                            "function": {
                                                "name": name,
                                                "arguments": arguments,
                                            },
                                        }
                                    ]
                                },
                                completion_id=completion_id,
                                created=created,
                            )
                    if not emitted_text and not tool_buffers:
                        for item in response.get("output") or []:
                            if item.get("type") == "message":
                                for part in item.get("content") or []:
                                    text = part.get("text")
                                    if text:
                                        emitted_text = True
                                        yield _chunk(
                                            model,
                                            {"content": text},
                                            completion_id=completion_id,
                                            created=created,
                                        )
                    finish = "tool_calls" if tool_buffers else "stop"
                    if kind == "response.incomplete":
                        details = response.get("incomplete_details") or {}
                        if details.get("reason") in ("max_output_tokens", "length"):
                            finish = "length"
                    yield _chunk(
                        model,
                        {},
                        finish_reason=finish,
                        usage=usage,
                        completion_id=completion_id,
                        created=created,
                    )
                    break
                elif kind in ("response.failed", "error"):
                    response = event.get("response") or {}
                    error = response.get("error") or event.get("error") or event
                    if isinstance(error, dict):
                        message = error.get("message") or json.dumps(error)
                    else:
                        message = str(error)
                    raise ProviderStreamError(f"Codex error: {message}")
        finally:
            try:
                connection.close()
            except Exception:
                pass


def provider_for(name: str, config: Config) -> Provider:
    if name == "openrouter":
        return OpenRouterProvider(config)
    if name == "codex":
        return CodexProvider(config)
    raise ProviderStreamError(f"Unsupported provider {name!r}", status=500)
