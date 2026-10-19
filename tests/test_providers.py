from __future__ import annotations

import json
import threading
import time
from typing import Any

import httpx
import pytest
from websockets.sync.server import serve

from osmia.providers import (
    CodexProvider,
    OpenRouterProvider,
    chat_to_responses,
)
from osmia.proxy.config import Config, ModelRoute


def test_chat_to_responses_translation() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "developer", "content": "Use tools."},
            {"role": "user", "content": "add 1 and 2"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "3"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "integer"}},
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }
    body = chat_to_responses(payload)
    assert body["instructions"] == "Be brief.\n\nUse tools."
    assert body["stream"] is True
    assert body["store"] is False
    assert body["input"][0] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "add 1 and 2"}],
    }
    assert body["input"][1] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "add",
        "arguments": '{"a":1,"b":2}',
    }
    assert body["input"][2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "3",
    }
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["name"] == "add"
    assert body["tools"][0]["strict"] is False


def test_codex_websocket_event_translation(tmp_path) -> None:
    received: list[dict[str, Any]] = []

    def handler(ws) -> None:  # noqa: ANN001
        message = json.loads(ws.recv())
        received.append(message)
        events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "add",
                    "arguments": "",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": '{"a":',
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": "1}",
            },
            {"type": "response.output_text.delta", "delta": "hello"},
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 4,
                        "total_tokens": 7,
                        "input_tokens_details": {"cached_tokens": 1},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            },
        ]
        for event in events:
            ws.send(json.dumps(event))

    server = serve(handler, "127.0.0.1", 0)
    port = server.socket.getsockname()[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    try:
        data = {
            "host": "127.0.0.1",
            "port": 0,
            "default_model": "codex-test",
            "models": {"codex-test": {"provider": "codex", "model": "gpt-test-codex"}},
            "providers": {
                "codex": {
                    "access_token": "token",
                    "expires_at": time.time() + 3600,
                    "account_id": "acct_1",
                }
            },
        }
        config = Config(tmp_path / "config.json", data)
        route = ModelRoute(
            slug="codex-test",
            provider="codex",
            model="gpt-test-codex",
            options={"base_url": f"http://127.0.0.1:{port}"},
        )
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }
        chunks = list(CodexProvider(config).stream(route, payload))
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert received and received[0]["type"] == "response.create"
    assert received[0]["model"] == "gpt-test-codex"
    assert received[0]["store"] is False
    content = "".join(
        choice.get("delta", {}).get("content", "")
        for chunk in chunks
        for choice in chunk.get("choices", [])
        if choice.get("delta", {}).get("content")
    )
    arguments = "".join(
        call["function"].get("arguments", "")
        for chunk in chunks
        for choice in chunk.get("choices", [])
        for call in choice.get("delta", {}).get("tool_calls", [])
    )
    final = chunks[-1]
    assert content == "hello"
    assert arguments == '{"a":1}'
    assert final["choices"][0]["finish_reason"] == "tool_calls"
    assert final["usage"]["prompt_tokens"] == 3
    assert final["usage"]["completion_tokens"] == 4
    assert final["usage"]["prompt_tokens_details"]["cached_tokens"] == 1


def test_openrouter_stream_request_and_relay(tmp_path, monkeypatch) -> None:
    recorded: dict[str, Any] = {}

    class FakeResponse:
        status_code = 200

        def iter_lines(self):  # noqa: ANN201
            yield 'data: {"id":"x","choices":[{"index":0,"delta":{"content":"ok"}}]}'
            yield "data: [DONE]"

        def read(self) -> bytes:
            return b""

    class FakeStream:
        def __enter__(self) -> FakeResponse:
            return FakeResponse()

        def __exit__(self, *args: Any) -> None:
            return None

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def stream(self, method: str, url: str, json: Any, headers: Any):  # noqa: ANN201
            recorded.update({"method": method, "url": url, "json": json, "headers": headers})
            return FakeStream()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    data = {
        "host": "127.0.0.1",
        "port": 0,
        "default_model": "or-test",
        "models": {"or-test": {"provider": "openrouter", "model": "vendor/model"}},
        "providers": {"openrouter": {"api_key": "sk-or-test"}},
    }
    config = Config(tmp_path / "config.json", data)
    route = ModelRoute(slug="or-test", provider="openrouter", model="vendor/model")
    chunks = list(
        OpenRouterProvider(config).stream(
            route,
            {
                "messages": [{"role": "user", "content": "hi"}],
                "model": "ignored",
                "provider": {"only": ["somewhere"]},
            },
        )
    )
    assert recorded["url"].endswith("/chat/completions")
    assert recorded["headers"]["Authorization"] == "Bearer sk-or-test"
    assert recorded["json"]["model"] == "vendor/model"
    assert recorded["json"]["stream"] is True
    assert recorded["json"]["stream_options"]["include_usage"] is True
    assert "provider" not in recorded["json"]
    assert chunks[0]["choices"][0]["delta"]["content"] == "ok"


def test_openrouter_missing_key(tmp_path) -> None:
    config = Config(tmp_path / "config.json", {"providers": {"openrouter": {"api_key": None}}})
    route = ModelRoute(slug="x", provider="openrouter", model="x")
    with pytest.raises(Exception) as exc:
        list(OpenRouterProvider(config).stream(route, {"messages": []}))
    assert "not logged in" in str(exc.value)
