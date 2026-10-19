"""The Osmia Agent: a small tool-calling loop over the local proxy."""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Sequence
from urllib.parse import urlparse

from .client import ProxyClient, ProxyError, default_proxy_url
from .discovery import AgentDiscovery
from .events import EventLog
from .proxy.config import Config
from .tools import Tool, as_tool
from .tui import TUI


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


class Agent:
    """An agent with Python callables as tools and a live control endpoint.

    ``Agent(prompt="do the task", tools=[...]).run()`` is the whole happy path. While it
    runs, the agent registers itself with the proxy and serves ``/state``, ``/events``,
    ``/events.sse``, ``/prompt``, ``/steer`` and ``/stop`` on a local port, so a session
    is fully drivable over HTTP: submit prompts, watch the event stream, steer, or stop.
    Pass ``control_api=False`` to turn that surface off.
    """

    def __init__(
        self,
        tools: Sequence[Callable[..., Any] | Tool] | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        prompt: str | None = None,
        proxy_url: str | None = None,
        tui: bool | TUI | None = None,
        max_turns: int = 50,
        temperature: float | None = None,
        max_tokens: int | None = None,
        name: str | None = None,
        agent_port_count: int | None = None,
        verbose: bool = False,
        client: ProxyClient | None = None,
        heartbeat: float = 2.0,
        max_tool_output: int = 200_000,
        control_api: bool = True,
        session: bool | None = None,
    ) -> None:
        self.id = "agent-" + uuid.uuid4().hex[:10]
        self.name = name or self.id
        self.model = model
        self.system_prompt = system_prompt
        self.proxy_url = (proxy_url or default_proxy_url()).rstrip("/")
        self.registry_url = self.proxy_url
        self.max_turns = max(1, int(max_turns))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tool_output = max(1000, int(max_tool_output))
        self.verbose = bool(verbose)
        self.tools: list[Tool] = [as_tool(item) for item in (tools or [])]
        self._tools_by_name = {tool.name: tool for tool in self.tools}
        self.messages: list[dict[str, Any]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        self.prompt: str | None = None
        if prompt is not None:
            text = str(prompt)
            if not text.strip():
                raise ValueError("Prompt must not be empty")
            self.prompt = text
            self.messages.append({"role": "user", "content": text})
        self.events = EventLog()
        self._client = client or ProxyClient(self.proxy_url)
        self._tui_arg = tui
        self._tui: TUI | None = None
        self._stop = threading.Event()
        self._steering: list[str] = []
        self._steering_lock = threading.Lock()
        self._inbox: queue.Queue[str] = queue.Queue()
        self._awaiting = False
        self._awaiting_lock = threading.Lock()
        self._control_api = bool(control_api)
        self._session_arg = session
        self._session_active: bool | None = None
        self._running = False
        self.status = "idle"
        self.turns = 0
        self.started_at: float | None = None
        self.last_error: str | None = None
        self._discovery: AgentDiscovery | None = None
        self._heartbeat = heartbeat
        self._port_count = agent_port_count
        self._stop_notified = False

    # ------------------------------------------------------------------ events
    def _emit(self, kind: str, payload: dict[str, Any] | None = None) -> int:
        seq = self.events.append(kind, payload)
        if self._tui is not None:
            self._tui.event(
                {
                    "seq": seq,
                    "time": time.time(),
                    "kind": kind,
                    "payload": payload or {},
                }
            )
        return seq

    # ------------------------------------------------------------------ run
    def run(self, prompt: str | None = None) -> str:
        if self._running:
            raise RuntimeError("Agent is already running")
        if prompt is not None:
            text = str(prompt)
            if not text.strip():
                raise ValueError("Prompt must not be empty")
            self.prompt = text
            self.messages.append({"role": "user", "content": text})
        self._stop.clear()
        self._stop_notified = False
        self.status = "starting"
        self.started_at = time.time()
        self.turns = 0
        self.last_error = None
        self._running = True
        self._tui = self._make_tui()
        try:
            self._start_discovery()
            self._session_active = self._session_mode()
            if not self._has_user_message() and not self._input_available():
                raise ValueError(
                    "Agent.run() needs a prompt unless the TUI or control API can "
                    "supply one"
                )
            if self._tui is not None:
                self._tui.start(self, self.model)
                if self.prompt:
                    self._tui.user_message(self.prompt)
            self.status = "running"
            self._emit(
                "agent.start",
                {
                    "model": self.model or "(proxy default)",
                    "tools": [tool.name for tool in self.tools],
                    "proxy": self.proxy_url,
                },
            )
            try:
                final = self._loop()
            except KeyboardInterrupt:
                self.stop()
                final = ""
            self.status = "done"
            self._emit("agent.end", {"status": "done"})
            return final
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.status = "error"
            self._emit("agent.error", {"message": self.last_error})
            raise
        finally:
            self._running = False
            if self._tui is not None:
                self._tui.stop()
            if self._discovery is not None:
                self._discovery.stop()
                self._discovery = None
            if self.status == "running":
                self.status = "done"

    def _make_tui(self) -> TUI | None:
        if isinstance(self._tui_arg, TUI):
            return self._tui_arg
        if self._tui_arg is False:
            return None
        enabled = self._tui_arg is True or (sys.stdin.isatty() and sys.stdout.isatty())
        return TUI(enabled=enabled, verbose=self.verbose) if enabled else None

    @staticmethod
    def _interactive_tui(tui: Any) -> bool:
        """True when the configured UI can accept prompts typed by a human."""
        if isinstance(tui, TUI):
            return bool(getattr(tui, "interactive", False))
        if tui is False:
            return False
        return sys.stdin.isatty() and sys.stdout.isatty()

    def _has_user_message(self) -> bool:
        return any(message.get("role") == "user" for message in self.messages)

    def _input_available(self) -> bool:
        """True when a prompt can still arrive, from the TUI or the control API."""
        if not self._inbox.empty():
            return True
        if isinstance(self._tui, TUI) and getattr(self._tui, "interactive", False):
            return True
        return self._control_api and self.attach_url is not None

    def _session_mode(self) -> bool:
        """True when the run keeps listening for prompts instead of returning.

        Defaults to a session for a prompt-less start (the caller expects input from
        the TUI or the control API) and for an interactive TUI, and to one shot when a
        headless caller passed a prompt. ``Agent(session=True)`` forces a session so a
        control plane can seed the first prompt and keep driving the same run.
        """
        if self._session_arg is not None:
            return bool(self._session_arg)
        if not self._has_user_message():
            return True
        return isinstance(self._tui, TUI) and getattr(self._tui, "interactive", False)

    @property
    def session(self) -> bool:
        """True when this run waits for prompts instead of returning after a turn.

        Resolved once per run so ``/state`` keeps reporting the mode the run is in,
        even after the first user message makes the guess ambiguous.
        """
        if self._session_active is not None:
            return self._session_active
        return self._session_mode()

    def _start_discovery(self) -> None:
        if not self._control_api:
            return
        parsed = urlparse(self.proxy_url)
        proxy_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        start = proxy_port + 1
        if start > 65000:
            start = 40000
        count = self._port_count
        if count is None:
            try:
                count = Config.load(create=False).agent_port_count
            except Exception:
                count = 100
        self._discovery = AgentDiscovery(
            agent=self,
            port_start=start,
            port_count=count,
            heartbeat=self._heartbeat,
        )
        self._discovery.start()

    @property
    def attach_url(self) -> str | None:
        return self._discovery.url if self._discovery else None

    def _loop(self) -> str:
        last_text = ""
        session = self.session
        if not self._has_user_message():
            opening = self._await_prompt()
            if opening is None:
                return last_text
            self.messages.append({"role": "user", "content": opening})
        while True:
            completed = False
            for _ in range(self.max_turns):
                if self._stop.is_set():
                    break
                self._drain_steering()
                self.turns += 1
                self._emit("turn.start", {"turn": self.turns})
                text, calls = self._turn()
                last_text = text
                if self._stop.is_set():
                    break
                if calls:
                    self._execute_tools(calls)
                    continue
                if self._steering_pending():
                    continue
                completed = True
                break
            if self._stop.is_set():
                return last_text
            if not completed:
                raise RuntimeError(f"Agent exceeded max_turns={self.max_turns}")
            if not session:
                return last_text
            if self._steering_pending():
                continue
            followup = self._await_prompt()
            if followup is None:
                return last_text
            self.messages.append({"role": "user", "content": followup})

    # --------------------------------------------------------------- one turn
    def _turn(self) -> tuple[str, list[ToolCall]]:
        kwargs: dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        schemas = [tool.openai_schema() for tool in self.tools]
        if schemas:
            kwargs["tools"] = schemas

        content: list[str] = []
        slots: dict[int, dict[str, Any]] = {}
        try:
            stream = self._client.stream_chat(self.messages, model=self.model, **kwargs)
            for chunk in stream:
                for choice in chunk.get("choices") or []:
                    if choice.get("index", 0) != 0:
                        continue
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        content.append(str(text))
                        self._emit("assistant.delta", {"text": str(text)})
                    reasoning = delta.get("reasoning_content")
                    if reasoning and (self.verbose or self._tui is not None):
                        self._emit(
                            "assistant.reasoning.delta", {"text": str(reasoning)}
                        )
                    for raw_call in delta.get("tool_calls") or []:
                        index = int(raw_call.get("index", 0))
                        slot = slots.setdefault(
                            index,
                            {"id": "", "name": "", "arguments": ""},
                        )
                        if raw_call.get("id"):
                            slot["id"] = str(raw_call["id"])
                        function = raw_call.get("function") or {}
                        if function.get("name"):
                            slot["name"] += str(function["name"])
                        if function.get("arguments"):
                            slot["arguments"] += str(function["arguments"])
        except ProxyError as exc:
            raise RuntimeError(str(exc)) from exc

        text = "".join(content)
        calls = [
            ToolCall(
                id=slot["id"] or "call_" + uuid.uuid4().hex,
                name=slot["name"],
                arguments=slot["arguments"] or "{}",
            )
            for _, slot in sorted(slots.items())
        ]
        if text or calls:
            message: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in calls
                ]
            self.messages.append(message)
        self._emit(
            "assistant.message",
            {
                "text": text,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in calls
                ],
            },
        )
        return text, calls

    def _execute_tools(self, calls: list[ToolCall]) -> None:
        for call in calls:
            if self._stop.is_set():
                break
            self._emit(
                "tool.call",
                {"name": call.name, "arguments": call.arguments, "call_id": call.id},
            )
            tool = self._tools_by_name.get(call.name)
            if tool is None:
                result = f"Error: unknown tool {call.name!r}"
            else:
                try:
                    result = tool.invoke(call.arguments)
                except Exception as exc:
                    result = f"Error ({type(exc).__name__}): {exc}"
            if len(result) > self.max_tool_output:
                result = result[: self.max_tool_output] + "\n...[truncated]"
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": result,
                }
            )
            self._emit(
                "tool.result",
                {
                    "name": call.name,
                    "call_id": call.id,
                    "result": result,
                },
            )

    # --------------------------------------------------------------- steering
    def steer(self, text: str) -> int:
        message = str(text).strip()
        if not message:
            raise ValueError("Steering message must not be empty")
        with self._steering_lock:
            self._steering.append(message)
        return self._emit("agent.steer", {"text": message})

    def _drain_steering(self) -> None:
        with self._steering_lock:
            pending, self._steering = self._steering, []
        for text in pending:
            self.messages.append({"role": "user", "content": text})

    def _steering_pending(self) -> bool:
        with self._steering_lock:
            return bool(self._steering)

    # ----------------------------------------------------------------- prompts
    def submit(self, text: str) -> int:
        """Queue a prompt from any surface (human or control API).

        The prompt becomes the next user message at the next turn boundary, so it is
        safe to call while a turn is running. Use ``steer(text)`` to influence the
        turn that is already in flight instead of starting a follow-up turn.
        """
        message = str(text).strip()
        if not message:
            raise ValueError("Prompt text must not be empty")
        self._inbox.put(message)
        return self._emit(
            "agent.prompt", {"text": message, "queued": self.pending_prompts}
        )

    @property
    def awaiting_input(self) -> bool:
        """True while the run is blocked waiting for a prompt."""
        with self._awaiting_lock:
            return self._awaiting

    @property
    def pending_prompts(self) -> int:
        """Prompts submitted but not yet delivered to the conversation."""
        return self._inbox.qsize()

    @property
    def stop_requested(self) -> bool:
        """True once ``stop()`` (or Ctrl+C) asked the run to end."""
        return self._stop.is_set()

    def _set_awaiting(self, value: bool) -> None:
        with self._awaiting_lock:
            if self._awaiting == value:
                return
            self._awaiting = value
        self._emit("agent.awaiting", {"awaiting": value})

    def _await_prompt(self) -> str | None:
        """Block until a prompt arrives from any input surface, or the run stops."""
        self._set_awaiting(True)
        try:
            while not self._stop.is_set():
                try:
                    prompt = self._inbox.get(timeout=0.2)
                except queue.Empty:
                    continue
                # Flip the flag before the prompt reaches the conversation so an
                # observer that just submitted it never reads a stale "still waiting".
                self._set_awaiting(False)
                return prompt
            return None
        finally:
            self._set_awaiting(False)

    def stop(self) -> None:
        self._stop.set()
        if not self._stop_notified:
            self._stop_notified = True
            if self.status == "running":
                self.status = "stopping"
            self._emit("agent.stopped", {})

    # ----------------------------------------------------------------- state
    def state(self, include_messages: bool = True) -> dict[str, Any]:
        state: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "pid": os.getpid(),
            "status": self.status,
            "prompt": self.prompt,
            "model": self.model or "(proxy default)",
            "proxy_url": self.proxy_url,
            "started_at": self.started_at,
            "turns": self.turns,
            "tool_names": [tool.name for tool in self.tools],
            "steering_queued": len(self._steering),
            "prompts_queued": self.pending_prompts,
            "awaiting_input": self.awaiting_input,
            "accepts_prompts": self._input_available(),
            "session": self.session,
            "control_api": self._control_api,
            "last_event": self.events.last_seq,
            "last_error": self.last_error,
            "attach_url": self.attach_url,
        }
        if include_messages:
            state["messages"] = json.loads(json.dumps(self.messages, default=str))
        return state
