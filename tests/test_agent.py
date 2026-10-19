from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from osmia.agent import Agent
from osmia.proxy.config import Config
from osmia.proxy.server import Proxy
from osmia.tui import TUI
from tests.helpers import (
    ScriptedProvider,
    drive_prompts,
    text_chunks,
    tool_chunks,
    wait_until,
)


def make_proxy(tmp_path: Path, provider: ScriptedProvider) -> Proxy:
    data = {
        "host": "127.0.0.1",
        "port": 0,
        "default_model": "fake",
        "agent_port_count": 20,
        "models": {"fake": {"provider": "codex", "model": "upstream-fake"}},
        "providers": {
            "codex": {"access_token": "x", "expires_at": time.time() + 3600},
            "openrouter": {"api_key": None},
        },
    }
    return Proxy(
        config=Config(tmp_path / "config.json", data),
        port=0,
        providers={"codex": provider},
    ).start()


def test_agent_tool_loop_end_to_end(tmp_path) -> None:
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    provider = ScriptedProvider(
        [
            tool_chunks("call_1", "add", '{"a": 2, "b": 3}'),
            text_chunks("The answer is 5."),
        ]
    )
    proxy = make_proxy(tmp_path, provider)
    agent = Agent(
        tools=[add],
        model="fake",
        system_prompt="You are a calculator.",
        proxy_url=proxy.url,
        tui=False,
        agent_port_count=20,
    )
    try:
        result = agent.run("What is 2 + 3?")
    finally:
        proxy.stop()

    assert result == "The answer is 5."
    assert agent.status == "done"
    assert agent.attach_url is None
    roles = [message["role"] for message in agent.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert agent.messages[3]["content"] == "5"
    assert agent.messages[3]["tool_call_id"] == "call_1"
    assert provider.requests[1]["messages"][3]["role"] == "tool"
    kinds = [event["kind"] for event in agent.events.snapshot(limit=100)]
    assert "tool.call" in kinds
    assert "tool.result" in kinds
    assert "agent.start" in kinds
    assert "agent.end" in kinds


def test_agent_constructor_prompt_is_used(tmp_path) -> None:
    provider = ScriptedProvider([text_chunks("constructed")])
    proxy = make_proxy(tmp_path, provider)
    agent = Agent(
        prompt="Say constructed.",
        model="fake",
        proxy_url=proxy.url,
        tui=False,
        agent_port_count=20,
    )
    try:
        assert agent.run() == "constructed"
    finally:
        proxy.stop()
    assert agent.messages[0] == {"role": "user", "content": "Say constructed."}


def test_agent_steering_extends_conversation(tmp_path) -> None:
    provider = ScriptedProvider([text_chunks("first"), text_chunks("second")])
    proxy = make_proxy(tmp_path, provider)
    agent = Agent(
        model="fake",
        proxy_url=proxy.url,
        tui=False,
        agent_port_count=20,
    )

    def steer_on_first(index: int, payload: dict[str, Any]) -> None:
        if index == 0:
            agent.steer("Actually answer second.")

    provider.on_request = steer_on_first
    try:
        result = agent.run("Say something.")
    finally:
        proxy.stop()

    assert result == "second"
    assert [message["role"] for message in agent.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert agent.messages[2]["content"] == "Actually answer second."


def test_agent_stop_breaks_loop(tmp_path) -> None:
    provider = ScriptedProvider([text_chunks("partial answer")])
    proxy = make_proxy(tmp_path, provider)
    agent = Agent(model="fake", proxy_url=proxy.url, tui=False, agent_port_count=20)

    def stop_on_request(index: int, payload: dict[str, Any]) -> None:
        agent.stop()

    provider.on_request = stop_on_request
    try:
        result = agent.run("Do work.")
    finally:
        proxy.stop()

    assert result == "partial answer"
    assert agent.status == "done"
    assert "agent.stopped" in [event["kind"] for event in agent.events.snapshot()]


def test_discovery_state_events_steer_stop_and_registry(tmp_path) -> None:
    provider = ScriptedProvider([text_chunks("unused")])
    proxy = make_proxy(tmp_path, provider)
    agent = Agent(model="fake", proxy_url=proxy.url, tui=False, agent_port_count=20)
    agent._start_discovery()
    try:
        assert agent.attach_url is not None
        discovery_url = agent.attach_url

        deadline = time.time() + 3
        while time.time() < deadline:
            listed = httpx.get(proxy.url + "/v1/agents").json()["data"]
            if listed:
                break
            time.sleep(0.05)
        assert listed[0]["id"] == agent.id
        assert listed[0]["attach_url"] == discovery_url

        state = httpx.get(discovery_url + "/state").json()
        assert state["id"] == agent.id
        assert state["messages"] == []
        hidden = httpx.get(discovery_url + "/state?messages=0").json()
        assert "messages" not in hidden

        httpx.post(
            discovery_url + "/steer",
            json={"text": "direct steering"},
        )
        assert agent._steering == ["direct steering"]

        response = httpx.post(
            proxy.url + f"/v1/agents/{agent.id}/steer",
            json={"text": "forwarded steering"},
        )
        assert response.status_code == 200
        assert response.json().get("forwarded") is True
        assert "forwarded steering" in agent._steering

        events = httpx.get(discovery_url + "/events.json").json()["data"]
        assert events
        assert any(event["kind"] == "agent.steer" for event in events)

        response = httpx.post(
            discovery_url + "/prompt",
            json={"text": "direct prompt"},
        )
        assert response.status_code == 202
        assert response.json()["queued"] == 1
        assert agent.pending_prompts == 1

        assert (
            httpx.post(discovery_url + "/prompt", json={"text": ""}).status_code == 400
        )

        response = httpx.post(
            proxy.url + f"/v1/agents/{agent.id}/prompt",
            json={"text": "forwarded prompt"},
        )
        assert response.status_code == 202
        assert response.json()["action"] == "prompt"
        assert agent.pending_prompts == 2
        assert (
            httpx.post(
                proxy.url + "/v1/agents/ghost/prompt", json={"text": "x"}
            ).status_code
            == 404
        )

        kinds = [
            event["kind"]
            for event in httpx.get(discovery_url + "/events.json").json()["data"]
        ]
        assert kinds.count("agent.prompt") == 2

        response = httpx.post(proxy.url + f"/v1/agents/{agent.id}/stop", json={})
        assert response.status_code == 200
        assert agent._stop.is_set()
    finally:
        agent._discovery.stop()
        agent._discovery = None
        proxy.stop()


class QuietTUI(TUI):
    """Interactive UI stub that records transcript blocks instead of drawing them."""

    interactive = True

    def __init__(self) -> None:
        super().__init__(enabled=False)
        self.seen: list[str] = []
        self.blocks: list[tuple[str, str]] = []

    def start(self, agent: Any, model: str | None) -> None:
        self.agent = agent

    def stop(self) -> None:
        return None

    def event(self, event: dict[str, Any]) -> None:
        payload = event.get("payload") or {}
        if event.get("kind") == "agent.prompt":
            self.blocks.append(("user", str(payload.get("text") or "")))

    def user_message(self, text: str) -> None:
        self.seen.append(text)


def test_tui_submit_routes_prompts_and_steering() -> None:
    class StubAgent:
        def __init__(self, awaiting: bool) -> None:
            self.awaiting_input = awaiting
            self.submitted: list[str] = []
            self.steered: list[str] = []

        def submit(self, text: str) -> int:
            self.submitted.append(text)
            return 1

        def steer(self, text: str) -> int:
            self.steered.append(text)
            return 1

    waiting = StubAgent(True)
    tui = TUI(enabled=False)
    tui.agent = waiting
    tui._line = list("follow-up prompt")
    tui._submit()
    assert waiting.submitted == ["follow-up prompt"]
    assert waiting.steered == []

    busy = StubAgent(False)
    tui.agent = busy
    tui._line = list("steer this way")
    tui._submit()
    assert busy.steered == ["steer this way"]
    assert busy.submitted == []


def test_interactive_tui_accepts_followup_prompts(tmp_path) -> None:
    provider = ScriptedProvider([text_chunks("first"), text_chunks("second")])
    proxy = make_proxy(tmp_path, provider)
    tui = QuietTUI()
    agent = Agent(model="fake", proxy_url=proxy.url, tui=tui, agent_port_count=20)
    driver = drive_prompts(agent, ["Actually answer second."])
    try:
        assert agent.run("Say something.") == "second"
    finally:
        driver.join(5)
        proxy.stop()

    assert tui.seen == ["Say something."]
    assert tui.blocks == [("user", "Actually answer second.")]
    assert [message["role"] for message in agent.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert agent.messages[2]["content"] == "Actually answer second."
    assert provider.requests[1]["messages"][-1]["content"] == "Actually answer second."
    assert agent.turns == 2


def test_promptless_headless_run_is_driven_over_rest(tmp_path) -> None:
    provider = ScriptedProvider([text_chunks("first"), text_chunks("second")])
    proxy = make_proxy(tmp_path, provider)
    agent = Agent(model="fake", proxy_url=proxy.url, tui=False, agent_port_count=20)
    result: dict[str, str] = {}

    def worker() -> None:
        result["text"] = agent.run()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        wait_until(lambda: agent.attach_url is not None)
        wait_until(lambda: agent.awaiting_input)
        control = str(agent.attach_url)

        state = httpx.get(control + "/state").json()
        assert state["accepts_prompts"] is True
        assert state["awaiting_input"] is True
        assert state["prompts_queued"] == 0

        first = httpx.post(control + "/prompt", json={"text": "Say something."})
        assert first.status_code == 202
        assert first.json()["ok"] is True
        wait_until(lambda: len(provider.requests) == 1)
        wait_until(lambda: agent.awaiting_input)

        second = httpx.post(control + "/prompt", json={"text": "Say it again."})
        assert second.status_code == 202
        wait_until(lambda: len(provider.requests) == 2)
        wait_until(lambda: agent.awaiting_input)

        assert httpx.post(control + "/prompt", json={"text": "  "}).status_code == 400
        httpx.post(control + "/stop")
        thread.join(5)
    finally:
        proxy.stop()

    assert not thread.is_alive()
    assert result["text"] == "second"
    assert agent.awaiting_input is False
    assert [message["role"] for message in agent.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert agent.messages[2]["content"] == "Say it again."
    kinds = [event["kind"] for event in agent.events.snapshot()]
    assert "agent.prompt" in kinds
    assert "agent.awaiting" in kinds


def test_events_sse_streams_events(tmp_path) -> None:
    provider = ScriptedProvider([text_chunks("streamed over sse")])
    proxy = make_proxy(tmp_path, provider)
    agent = Agent(model="fake", proxy_url=proxy.url, tui=False, agent_port_count=20)

    def worker() -> None:
        agent.run()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    events: list[dict[str, Any]] = []
    ids: list[str] = []
    try:
        wait_until(lambda: agent.awaiting_input)
        with httpx.stream(
            "GET", str(agent.attach_url) + "/events.sse", timeout=20
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            agent.submit("Say something.")
            for line in response.iter_lines():
                if line.startswith("id: "):
                    ids.append(line[4:])
                elif line.startswith("data: "):
                    events.append(json.loads(line[6:]))
                    if (
                        len([e for e in events if e["kind"] == "assistant.message"])
                        >= 1
                    ):
                        break
        agent.stop()
        thread.join(5)
    finally:
        proxy.stop()

    assert not thread.is_alive()
    kinds = [event["kind"] for event in events]
    assert "agent.prompt" in kinds
    assert "turn.start" in kinds
    assert "assistant.delta" in kinds
    assert ids and ids == sorted(ids, key=int)
    assert [event["seq"] for event in events] == sorted(
        event["seq"] for event in events
    )


def test_agent_without_prompt_requires_interactive_tui(tmp_path) -> None:
    provider = ScriptedProvider([text_chunks("unused")])
    proxy = make_proxy(tmp_path, provider)
    agent = Agent(model="fake", proxy_url=proxy.url, tui=False, control_api=False)
    try:
        try:
            agent.run()
        except ValueError as exc:
            assert "needs a prompt" in str(exc)
        else:
            raise AssertionError("run() without a prompt should fail headless")
    finally:
        proxy.stop()
    assert provider.requests == []
    assert agent.attach_url is None
