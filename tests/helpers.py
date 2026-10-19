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
            "choices": [
                {"index": 0, "delta": {"content": text}, "finish_reason": None}
            ],
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

    def stream(
        self, route: ModelRoute, payload: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
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


def wait_until(predicate: Any, timeout: float = 10.0, what: str = "condition") -> None:
    """Spin until ``predicate()`` is truthy; fail loudly instead of hanging a test."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"{what} not met within {timeout}s")


def drive_prompts(
    agent: Any, prompts: list[str], timeout: float = 10.0
) -> threading.Thread:
    """Submit ``prompts`` whenever the agent waits for input, then stop the run.

    Thread failures are recorded on ``thread.errors`` so a stuck driver cannot be
    mistaken for a stalled agent.
    """
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for text in prompts:
                wait_until(
                    lambda: agent.awaiting_input, timeout, "agent awaiting input"
                )
                turns = agent.turns
                agent.submit(text)
                # Wait on observable progress, not on the awaiting flag, which may
                # still read True in the instant before the prompt is picked up.
                wait_until(
                    lambda: agent.turns > turns or agent.stop_requested,
                    timeout,
                    "follow-up turn to run",
                )
            wait_until(lambda: agent.awaiting_input, timeout, "agent awaiting input")
            agent.stop()
        except Exception as exc:  # pragma: no cover - surfaced by the test
            errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.errors = errors  # type: ignore[attr-defined]
    thread.start()
    return thread
