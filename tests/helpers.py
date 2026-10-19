"""Shared fakes for Osmia tests."""

from __future__ import annotations

import threading
from typing import Any, Iterator

from osmia.providers import Provider
from osmia.proxy.config import ModelRoute


def text_chunks(text: str, model: str = "fake") -> list[dict[str, Any]]:
    return [
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]


def tool_chunks(
    call_id: str,
    name: str,
    arguments: str,
    model: str = "fake",
) -> list[dict[str, Any]]:
    return [
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": arguments},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    ]


class ScriptedProvider(Provider):
    name = "codex"

    def __init__(self, scripts: list[list[dict[str, Any]]]) -> None:
        self.scripts = scripts
        self.requests: list[dict[str, Any]] = []
        self.on_request = None
        self._lock = threading.Lock()

    def stream(self, route: ModelRoute, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        with self._lock:
            index = len(self.requests)
            self.requests.append(payload)
        callback = self.on_request
        if callback is not None:
            callback(index, payload)
        script = self.scripts[min(index, len(self.scripts) - 1)]
        yield from script

    def close(self) -> None:
        return None
