"""Minimal terminal UI: live agent status, steering input, markdown-ish output."""

from __future__ import annotations

import os
import sys
import termios
import threading
import tty
from typing import Any

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"
YELLOW = "\x1b[33m"


class TUI:
    def __init__(self, enabled: bool = True, verbose: bool = False) -> None:
        self.enabled = bool(enabled and sys.stdin.isatty() and sys.stdout.isatty())
        self.verbose = verbose
        self.agent: Any = None
        self.model: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._saved: list[Any] | None = None
        self._line: list[str] = []
        self._last_was_delta = False
        self._color = self.enabled and os.environ.get("NO_COLOR") is None

    def _c(self, code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if self._color else text

    def start(self, agent: Any, model: str | None) -> None:
        self.agent = agent
        self.model = model
        if not self.enabled:
            return
        print(
            self._c(BOLD, f"Osmia agent {agent.id}")
            + self._c(DIM, f"  model={model or '(proxy default)'}  proxy={agent.proxy_url}")
        )
        print(
            self._c(
                DIM,
                "Type a message and press Enter to steer. Ctrl+Enter inserts a newline; "
                "Ctrl+D stops the agent.",
            )
        )
        self._fd = sys.stdin.fileno()
        try:
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except (termios.error, OSError):
            self.enabled = False
            return
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        self._prompt()

    def stop(self) -> None:
        self._stop.set()
        if self._fd is not None and self._saved is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:
                pass
            self._saved = None
            if self.enabled:
                print()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)
        self._thread = None

    def _prompt(self) -> None:
        if not self.enabled or self._stop.is_set():
            return
        sys.stdout.write(self._c(CYAN, "\nsteer> ") + "".join(self._line))
        sys.stdout.flush()

    def _read_loop(self) -> None:
        assert self._fd is not None
        try:
            while not self._stop.is_set():
                try:
                    char = os.read(self._fd, 1)
                except OSError:
                    break
                if not char:
                    break
                if char in (b"\r", b"\n"):
                    # Common terminal encodings for Ctrl+Enter insert a newline instead.
                    text = "".join(self._line).strip()
                    self._line = []
                    if text:
                        self.agent.steer(text)
                    self._prompt()
                elif char == b"\x04":  # Ctrl+D
                    if not self._line:
                        self.agent.stop()
                        break
                elif char in (b"\x7f", b"\x08"):
                    if self._line:
                        self._line.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                elif char == b"\x1b":
                    sequence = self._read_escape()
                    if sequence in ("\r", "[13;5u", "[27;5;13~", "[13;5~"):
                        self._line.append("\n")
                        sys.stdout.write("\nsteer> " + "".join(self._line.splitlines()[-1:]))
                        sys.stdout.flush()
                elif char == b"\x03":
                    self.agent.stop()
                    break
                else:
                    try:
                        decoded = char.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    self._line.append(decoded)
                    sys.stdout.write(decoded)
                    sys.stdout.flush()
        finally:
            if self._saved is not None and self._fd is not None:
                try:
                    termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
                except Exception:
                    pass

    def _read_escape(self) -> str:
        assert self._fd is not None
        import select

        collected = ""
        while True:
            ready, _, _ = select.select([self._fd], [], [], 0.05)
            if not ready:
                break
            try:
                nxt = os.read(self._fd, 1)
            except OSError:
                break
            if not nxt:
                break
            collected += nxt.decode("latin-1")
            if collected.endswith("~") or collected.endswith("u") or len(collected) > 12:
                break
        return collected

    def event(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        kind = event.get("kind")
        payload = event.get("payload") or {}
        if kind == "turn.start":
            self._end_delta()
            print(self._c(DIM, f"\n── turn {payload.get('turn', '?')} ──"))
        elif kind == "assistant.reasoning.delta":
            if self.verbose:
                self._end_delta()
                sys.stdout.write(self._c(DIM, str(payload.get("text", ""))))
                sys.stdout.flush()
        elif kind == "assistant.delta":
            text = str(payload.get("text", ""))
            if text:
                sys.stdout.write(text)
                sys.stdout.flush()
                self._last_was_delta = True
        elif kind == "assistant.message":
            self._end_delta()
        elif kind == "tool.call":
            self._end_delta()
            arguments = payload.get("arguments") or ""
            print(self._c(YELLOW, f"→ {payload.get('name')}({arguments})"))
        elif kind == "tool.result":
            self._end_delta()
            result = str(payload.get("result") or "")
            self._print_result(result)
        elif kind == "agent.steer":
            self._end_delta()
            print(self._c(CYAN, f"✎ steering: {payload.get('text')}"))
        elif kind == "agent.error":
            self._end_delta()
            print(self._c(RED, f"! {payload.get('message')}"))
        elif kind == "agent.stopped":
            self._end_delta()
            print(self._c(YELLOW, "■ stop requested"))
        elif self.verbose and kind.startswith(("registry.", "discovery.")):
            self._end_delta()
            print(self._c(DIM, f"[{kind}] {payload}"))

    def _end_delta(self) -> None:
        if self._last_was_delta:
            print()
            self._last_was_delta = False

    def _print_result(self, result: str) -> None:
        lines = result.splitlines() or [""]
        diff = result.startswith("diff --git") or any(
            line.startswith(("+++ ", "--- ", "@@ ")) for line in lines[:20]
        )
        limit = 60 if not self.verbose else 400
        for line in lines[:limit]:
            if diff and line.startswith("+") and not line.startswith("+++"):
                print(self._c(GREEN, line))
            elif diff and line.startswith("-") and not line.startswith("---"):
                print(self._c(RED, line))
            elif diff and line.startswith("@@"):
                print(self._c(CYAN, line))
            else:
                print(self._c(DIM, line))
        if len(lines) > limit:
            print(self._c(DIM, f"... {len(lines) - limit} more lines"))
