from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from osmia.proxy.config import Config
from osmia.proxy.server import Proxy, aggregate_stream
from tests.helpers import ScriptedProvider, text_chunks


def make_config(tmp_path: Path, port: int = 0) -> Config:
    data = {
        "host": "127.0.0.1",
        "port": port,
        "default_model": "fake",
        "models": {
            "fake": {"provider": "codex", "model": "upstream-fake"},
            "other": {"provider": "codex", "model": "other-fake"},
        },
        "providers": {
            "codex": {"access_token": "x", "expires_at": time.time() + 3600},
            "openrouter": {"api_key": None},
        },
    }
    return Config(tmp_path / "config.json", data)


def test_health_models_and_unknown_slug(tmp_path) -> None:
    config = make_config(tmp_path)
    proxy = Proxy(config=config, port=0, providers={"codex": ScriptedProvider([])}).start()
    try:
        health = httpx.get(proxy.url + "/health").json()
        assert health["ok"] is True
        assert health["default_model"] == "fake"
        models = httpx.get(proxy.url + "/v1/models").json()
        assert {item["id"] for item in models["data"]} == {"fake", "other"}
        response = httpx.post(
            proxy.url + "/v1/chat/completions",
            json={"model": "nope", "messages": [{"role": "user", "content": "x"}]},
        )
        assert response.status_code == 404
        assert "Unknown model slug" in response.json()["error"]["message"]
        assert "fake" in response.json()["error"]["message"]
    finally:
        proxy.stop()


def test_stream_relay_and_nonstream_aggregation(tmp_path) -> None:
    provider = ScriptedProvider([text_chunks("streamed answer")])
    config = make_config(tmp_path)
    proxy = Proxy(config=config, port=0, providers={"codex": provider}).start()
    try:
        with httpx.stream(
            "POST",
            proxy.url + "/v1/chat/completions",
            json={
                "model": "fake",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "provider": {"only": ["ignored"]},
            },
        ) as response:
            assert response.status_code == 200
            data = [
                line[6:]
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]
        assert data[-1] == "[DONE]"
        chunks = [json.loads(item) for item in data[:-1]]
        assert "".join(
            choice["delta"].get("content", "")
            for chunk in chunks
            for choice in chunk["choices"]
        ) == "streamed answer"
        assert chunks[0]["model"] == "fake"

        response = httpx.post(
            proxy.url + "/v1/chat/completions",
            json={"model": "fake", "messages": [{"role": "user", "content": "hi"}]},
        )
        result = response.json()
        assert result["choices"][0]["message"]["content"] == "streamed answer"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["total_tokens"] == 2
        assert len(provider.requests) == 2
    finally:
        proxy.stop()


def test_aggregate_tool_calls() -> None:
    from tests.helpers import tool_chunks

    result = aggregate_stream(iter(tool_chunks("call_1", "add", '{"a":1}')), "fake")
    message = result["choices"][0]["message"]
    assert message["tool_calls"][0]["id"] == "call_1"
    assert message["tool_calls"][0]["function"]["name"] == "add"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"a":1}'
    assert result["choices"][0]["finish_reason"] == "tool_calls"


def test_agent_registry_and_aliases(tmp_path) -> None:
    config = make_config(tmp_path)
    proxy = Proxy(config=config, port=0, providers={"codex": ScriptedProvider([])}).start()
    try:
        state = {
            "status": "running",
            "turns": 3,
            "model": "fake",
            "attach_url": "http://127.0.0.1:1",
        }
        response = httpx.put(proxy.url + "/v1/agents/agent-1", json=state)
        assert response.status_code == 200
        listed = httpx.get(proxy.url + "/v1/agents").json()["data"]
        assert listed[0]["id"] == "agent-1"
        assert listed[0]["turns"] == 3
        assert listed[0]["stale"] is False

        with proxy._registry_lock:
            proxy.agents["agent-1"]["last_seen"] = time.time() - 100
        stale = httpx.get(proxy.url + "/v1/agents/agent-1").json()
        assert stale["stale"] is True

        response = httpx.post(
            proxy.url + "/v1/register",
            json={"id": "agent-2", "status": "idle"},
        )
        assert response.status_code == 200
        assert response.json()["id"] == "agent-2"

        response = httpx.post(
            proxy.url + "/v1/heartbeat",
            json={"id": "agent-3", "status": "idle"},
        )
        assert response.status_code == 200

        response = httpx.post(
            proxy.url + "/v1/agents/agent-1/steer",
            json={"text": "keep going"},
        )
        assert response.status_code == 200
        assert response.json()["action"] == "steer"

        response = httpx.delete(proxy.url + "/v1/agents/agent-2")
        assert response.status_code == 200
        assert httpx.get(proxy.url + "/v1/agents/agent-2").status_code == 404
    finally:
        proxy.stop()


def test_proxy_requires_valid_messages(tmp_path) -> None:
    config = make_config(tmp_path)
    proxy = Proxy(config=config, port=0, providers={"codex": ScriptedProvider([])}).start()
    try:
        response = httpx.post(
            proxy.url + "/v1/chat/completions",
            json={"model": "fake", "messages": []},
        )
        assert response.status_code == 400
        assert "messages" in response.json()["error"]["message"]
    finally:
        proxy.stop()
