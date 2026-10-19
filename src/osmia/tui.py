"""Full-screen terminal UI with streaming output and follow-up prompts."""

from __future__ import annotations

import codecs
import os
import queue
import select
import shutil
import sys
import termios
import threading
import time
import tty
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
REVERSE = "\x1b[7m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"
YELLOW = "\x1b[33m"

ALT_ENTER = "\x1b[?1049h"
ALT_EXIT = "\x1b[?1049l"
CURSOR_ON = "\x1b[?25h"
CURSOR_OFF = "\x1b[?25l"
CLEAR = "\x1b[2J"
CLEAR_LINE = "\x1b[K"
HOME = "\x1b[H"


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in "WF" else 1
    return width


def _sanitize(text: str) -> str:
    result = []
    for char in str(text):
        code = ord(char)
        if char in ("\n", "\t"):
            result.append(char)
        elif code < 32 or code == 127:
            continue
        else:
            result.append(char)
    return "".join(result)


def _looks_failed(text: str) -> bool:
    """Best-effort failure detection for tool output the UI did not run itself."""
    stripped = str(text).strip()
    if not stripped:
        return False
    if stripped.startswith(("Error", "error:", "Traceback")):
        return True
    first = stripped.splitlines()[0]
    if first.startswith("exit code:"):
        try:
            return int(first.split(":", 1)[1].strip()) != 0
        except ValueError:
            return first.startswith("exit code: timeout")
    return False


def _wrap(text: str, width: int) -> list[str]:
    width = max(1, width)
    result: list[str] = []
    for raw in _sanitize(text).split("\n"):
        if not raw:
            result.append("")
            continue
        line = raw
        while _display_width(line) > width:
            take = 0
            used = 0
            for char in line:
                char_width = (
                    0
                    if unicodedata.combining(char)
                    else (2 if unicodedata.east_asian_width(char) in "WF" else 1)
                )
                if used + char_width > width:
                    break
                used += char_width
                take += 1
            if take == 0:
                take = 1
            result.append(line[:take])
            line = line[take:]
        result.append(line)
    return result or [""]


@dataclass
class _Block:
    kind: str
    text: str


class TUI:
    """Alternate-screen agent UI.

    The UI deliberately owns no model or tool state. It listens to agent events,
    maintains a transcript, and offers a follow-up queue once a turn settles.
    """

    interactive = True

    def __init__(self, enabled: bool = True, verbose: bool = False) -> None:
        self.enabled = bool(enabled and sys.stdin.isatty() and sys.stdout.isatty())
        self.verbose = verbose
        self.agent: Any = None
        self.model: str | None = None
        self._stop = threading.Event()
        self._dirty = threading.Event()
        self._input_thread: threading.Thread | None = None
        self._render_thread: threading.Thread | None = None
        self._fd: int | None = None
        self._saved: list[Any] | None = None
        self._entered = False
        self._color = self.enabled and os.environ.get("NO_COLOR") is None
        self._lock = threading.RLock()
        self._blocks: list[_Block] = []
        self._stream_kind: str | None = None
        self._stream_text: list[str] = []
        self._line: list[str] = []
        self._cursor = 0
        self._history: list[str] = []
        self._history_pos: int | None = None
        self._history_draft = ""
        self._followups: queue.Queue[str | None] = queue.Queue()
        self._last_size = (0, 0)

    # --------------------------------------------------------------- lifecycle
    def start(self, agent: Any, model: str | None) -> None:
        self.agent = agent
        self.model = model
        if not self.enabled:
            return
        self._fd = sys.stdin.fileno()
        try:
            self._saved = termios.tcgetattr(self._fd)
            tty.setraw(self._fd)
            sys.stdout.write(ALT_ENTER + CLEAR + HOME + CURSOR_ON)
            sys.stdout.flush()
            self._entered = True
        except (termios.error, OSError):
            self._restore_terminal()
            self.enabled = False
            return
        self._add_block(
            "info",
            f"Osmia agent {getattr(agent, 'id', '?')} · "
            f"{model or '(proxy default)'} · {getattr(agent, 'proxy_url', '')}",
        )
        self._render_thread = threading.Thread(
            target=self._render_loop, name="osmia-tui-render", daemon=True
        )
        self._render_thread.start()
        self._input_thread = threading.Thread(
            target=self._read_loop, name="osmia-tui-input", daemon=True
        )
        self._input_thread.start()
        self._wake()

    def stop(self) -> None:
        self._stop.set()
        self._dirty.set()
        self._restore_terminal()
        for thread in (self._input_thread, self._render_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=1)
        self._input_thread = None
        self._render_thread = None
        with self._lock:
            if self._entered:
                sys.stdout.write(RESET + CURSOR_ON + ALT_EXIT)
                sys.stdout.flush()
                self._entered = False

    def _restore_terminal(self) -> None:
        if self._fd is not None and self._saved is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:
                pass
            self._saved = None

    def _wake(self) -> None:
        self._dirty.set()

    # --------------------------------------------------------------- transcript
    def _c(self, code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if self._color else text

    def _add_block(self, kind: str, text: str) -> None:
        with self._lock:
            self._flush_stream_locked()
            self._blocks.append(_Block(kind, text))
        self._wake()

    def _append_stream(self, kind: str, text: str) -> None:
        with self._lock:
            if self._stream_kind != kind:
                self._flush_stream_locked()
                self._stream_kind = kind
            self._stream_text.append(text)
        self._wake()

    def _flush_stream_locked(self) -> None:
        if self._stream_kind is not None:
            text = "".join(self._stream_text)
            if text:
                self._blocks.append(_Block(self._stream_kind, text))
            self._stream_kind = None
            self._stream_text = []

    def _flush_stream(self) -> None:
        with self._lock:
            self._flush_stream_locked()
        self._wake()

    def user_message(self, text: str) -> None:
        if not self.enabled or not text:
            return
        self._add_block("user", text)

    # --------------------------------------------------------------- agent events
    def event(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        kind = event.get("kind")
        payload = event.get("payload") or {}
        if kind == "turn.start":
            self._add_block("turn", f"turn {payload.get('turn', '?')}")
        elif kind == "assistant.reasoning.delta":
            text = str(payload.get("text", ""))
            if text:
                self._append_stream("reasoning", text)
        elif kind == "assistant.delta":
            text = str(payload.get("text", ""))
            if text:
                self._append_stream("assistant", text)
        elif kind == "assistant.message":
            with self._lock:
                had_stream = self._stream_kind == "assistant" and bool(
                    "".join(self._stream_text)
                )
                self._flush_stream_locked()
            text = str(payload.get("text") or "")
            if text and not had_stream:
                self._add_block("assistant", text)
            self._wake()
        elif kind == "tool.call":
            self._add_block(
                "tool_call",
                f"{payload.get('name', '?')}({payload.get('arguments') or ''})",
            )
        elif kind == "tool.result":
            self._add_block("tool_result", str(payload.get("result") or ""))
        elif kind == "agent.prompt":
            text = str(payload.get("text") or "")
            if text:
                self._add_block("user", text)
        elif kind == "agent.steer":
            self._add_block("steer", str(payload.get("text") or ""))
        elif kind == "agent.error":
            self._add_block("error", str(payload.get("message") or "error"))
        elif kind == "agent.stopped":
            self._add_block("stopped", "stop requested")
        elif kind == "agent.start":
            tools = payload.get("tools") or []
            if tools:
                self._add_block("info", "tools: " + ", ".join(map(str, tools)))
        elif kind == "agent.end":
            pass
        elif self.verbose and kind.startswith(("registry.", "discovery.")):
            self._add_block("info", f"[{kind}] {payload}")

    # --------------------------------------------------------------- input
    def _waiting_for_input(self) -> bool:
        return bool(getattr(self.agent, "awaiting_input", False))

    def _read_loop(self) -> None:
        assert self._fd is not None
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while not self._stop.is_set():
                size = shutil.get_terminal_size((80, 24))
                if (size.columns, size.lines) != self._last_size:
                    self._last_size = (size.columns, size.lines)
                    self._wake()
                ready, _, _ = select.select([self._fd], [], [], 0.2)
                if not ready:
                    continue
                data = os.read(self._fd, 1)
                if not data:
                    break
                if data == b"\x1b":
                    self._handle_escape(self._read_escape())
                    continue
                if data in (b"\r", b"\n"):
                    self._submit()
                elif data == b"\x03":  # Ctrl+C
                    self._interrupt()
                elif data == b"\x04":  # Ctrl+D
                    self._eof()
                elif data in (b"\x7f", b"\x08"):
                    self._backspace()
                elif data == b"\x15":  # Ctrl+U
                    self._line.clear()
                    self._cursor = 0
                    self._wake()
                elif data == b"\x17":  # Ctrl+W
                    self._delete_word()
                elif data < b" ":
                    continue
                else:
                    text = decoder.decode(data)
                    if text:
                        self._insert(text)
        finally:
            self._restore_terminal()

    def _read_escape(self) -> str:
        assert self._fd is not None
        collected = ""
        while True:
            ready, _, _ = select.select([self._fd], [], [], 0.03)
            if not ready:
                break
            try:
                nxt = os.read(self._fd, 1)
            except OSError:
                break
            if not nxt:
                break
            collected += nxt.decode("latin-1")
            if collected and (collected[-1].isalpha() or collected.endswith("~")):
                break
            if len(collected) > 16:
                break
        return collected

    def _handle_escape(self, sequence: str) -> None:
        if not sequence:
            return
        if sequence in ("[A", "OA"):
            self._history_move(-1)
        elif sequence in ("[B", "OB"):
            self._history_move(1)
        elif sequence in ("[C", "OC"):
            self._cursor = min(len(self._line), self._cursor + 1)
            self._wake()
        elif sequence in ("[D", "OD"):
            self._cursor = max(0, self._cursor - 1)
            self._wake()
        elif sequence in ("[H", "OH", "[1~", "[7~"):
            self._cursor = 0
            self._wake()
        elif sequence in ("[F", "OF", "[4~", "[8~"):
            self._cursor = len(self._line)
            self._wake()
        elif sequence == "[3~":
            if self._cursor < len(self._line):
                self._line.pop(self._cursor)
                self._wake()

    def _insert(self, text: str) -> None:
        for char in text:
            if char in ("\r", "\n"):
                continue
            self._line.insert(self._cursor, char)
            self._cursor += 1
        self._wake()

    def _backspace(self) -> None:
        if self._cursor > 0:
            self._line.pop(self._cursor - 1)
            self._cursor -= 1
            self._wake()

    def _delete_word(self) -> None:
        while self._cursor > 0 and self._line[self._cursor - 1].isspace():
            self._line.pop(self._cursor - 1)
            self._cursor -= 1
        while self._cursor > 0 and not self._line[self._cursor - 1].isspace():
            self._line.pop(self._cursor - 1)
            self._cursor -= 1
        self._wake()

    def _history_move(self, direction: int) -> None:
        if not self._history:
            return
        if self._history_pos is None:
            if direction > 0:
                return
            self._history_draft = "".join(self._line)
            self._history_pos = len(self._history) - 1
        else:
            self._history_pos += direction
            if self._history_pos < 0:
                self._history_pos = 0
            elif self._history_pos >= len(self._history):
                self._history_pos = None
        if self._history_pos is None:
            self._line = list(self._history_draft)
        else:
            self._line = list(self._history[self._history_pos])
        self._cursor = len(self._line)
        self._wake()

    def _submit(self) -> None:
        text = "".join(self._line).strip()
        if not text:
            return
        self._line.clear()
        self._cursor = 0
        self._history.append(text)
        self._history_pos = None
        try:
            if self._waiting_for_input():
                # The agent renders the user block from the agent.prompt event.
                self.agent.submit(text)
            else:
                self.agent.steer(text)
        except Exception:
            pass
        self._wake()

    def _interrupt(self) -> None:
        try:
            self.agent.stop()
        except Exception:
            pass
        self._wake()

    def _eof(self) -> None:
        if self._line:
            return
        try:
            self.agent.stop()
        except Exception:
            pass
        self._wake()

    # --------------------------------------------------------------- rendering
    def _render_loop(self) -> None:
        while not self._stop.is_set():
            # Block until something changed; idle turns must not burn CPU.
            if not self._dirty.wait(0.2):
                continue
            self._dirty.clear()
            if self._stop.is_set():
                break
            time.sleep(0.02)  # coalesce bursts of deltas into one frame
            self._render()

    def _render(self) -> None:
        with self._lock:
            if not self.enabled or not self._entered:
                return
            size = shutil.get_terminal_size((80, 24))
            cols = max(20, size.columns)
            rows = max(6, size.lines)
            body = self._body_lines(cols, rows - 2)
            status = self._status_line(cols)
            prompt, cursor_col = self._input_line(cols)
            out = [CURSOR_OFF, HOME, CLEAR + "\x1b[J"]
            for line in body:
                out.append(line)
                out.append(CLEAR_LINE)
                out.append("\r\n")
            out.append(f"\x1b[{rows - 1};1H" + status + CLEAR_LINE)
            out.append(f"\x1b[{rows};1H" + prompt + CLEAR_LINE)
            out.append(f"\x1b[{rows};{cursor_col}H" + CURSOR_ON)
            try:
                sys.stdout.write("".join(out))
                sys.stdout.flush()
            except OSError:
                self._stop.set()

    def _body_lines(self, width: int, height: int) -> list[str]:
        blocks = list(self._blocks)
        if self._stream_kind is not None:
            text = "".join(self._stream_text)
            if text:
                blocks.append(_Block(self._stream_kind, text))
        lines: list[str] = []
        index = 0
        while index < len(blocks):
            if blocks[index].kind in ("tool_call", "tool_result"):
                group: list[_Block] = []
                while index < len(blocks) and blocks[index].kind in (
                    "tool_call",
                    "tool_result",
                ):
                    group.append(blocks[index])
                    index += 1
                lines.extend(self._tool_group_lines(group, width))
                continue
            lines.extend(self._block_lines(blocks[index], width))
            index += 1
        if len(lines) > height:
            hidden = len(lines) - (height - 1)
            return [self._c(DIM, f"… {hidden} earlier lines")] + lines[-(height - 1) :]
        return lines

    def _tool_group_lines(self, group: list[_Block], width: int) -> list[str]:
        """Render consecutive tool blocks as one rail-grouped activity panel.

        Consecutive ``tool_call``/``tool_result`` blocks become a single unit with a
        header, a left rail per member, and a closing footer, so a burst of tool work
        reads as one step instead of a bare list of lines.
        """
        calls = [block for block in group if block.kind == "tool_call"]
        results = [block for block in group if block.kind == "tool_result"]
        done = len(results)
        failed = sum(1 for block in results if _looks_failed(block.text))
        total = len(calls) or done
        if done >= total:
            glyph, label = "✓", "tools done"
            head_color = GREEN if not failed else RED
        else:
            glyph, label = "✻", "tools running"
            head_color = YELLOW
        count = f" · {done}/{total}" if total else ""
        failure = f" · {failed} failed" if failed else ""
        lines = [self._c(head_color, f"╭─ {glyph} {label}{count}{failure}")]

        rail = self._c(DIM, "│ ")
        pending = iter(results)
        result = None
        for call in calls:
            for line in _wrap(call.text, width - 4):
                lines.append(rail + self._c(YELLOW, "● " + line))
            try:
                result = next(pending)
            except StopIteration:
                result = None
            if result is None:
                continue
            lines.extend(self._tool_result_lines(result.text, width, rail))
        for leftover in pending:  # a result with no recorded call
            lines.extend(self._tool_result_lines(leftover.text, width, rail))
        lines.append(self._c(DIM, "╰─"))
        return lines

    def _tool_result_lines(self, text: str, width: int, rail: str) -> list[str]:
        lines = _wrap(text, width - 6)
        limit = 60 if self.verbose else 24
        diff = text.startswith("diff --git") or any(
            line.startswith(("+++ ", "--- ", "@@ ")) for line in lines[:20]
        )
        failed = _looks_failed(text)
        out: list[str] = []
        for position, line in enumerate(lines[:limit]):
            prefix = "" if position else ("✗ " if failed else "↳ ")
            if diff and line.startswith("+") and not line.startswith("+++"):
                out.append(rail + self._c(GREEN, "  " + prefix + line))
            elif diff and line.startswith("-") and not line.startswith("---"):
                out.append(rail + self._c(RED, "  " + prefix + line))
            elif diff and line.startswith("@@"):
                out.append(rail + self._c(CYAN, "  " + prefix + line))
            elif failed:
                out.append(rail + self._c(RED, "  " + prefix + line))
            else:
                out.append(rail + self._c(DIM, "  " + prefix + line))
        if len(lines) > limit:
            out.append(rail + self._c(DIM, f"    … {len(lines) - limit} more lines"))
        return out

    def _block_lines(self, block: _Block, width: int) -> Iterable[str]:
        kind = block.kind
        if kind == "user":
            wrapped = _wrap(block.text, width - 2)
            yield self._c(CYAN + BOLD, "❯ " + wrapped[0])
            for line in wrapped[1:]:
                yield self._c(CYAN, "  " + line)
        elif kind == "assistant":
            for line in _wrap(block.text, width):
                yield line
        elif kind == "reasoning":
            yield self._c(DIM + BOLD, "✻ thinking")
            for line in _wrap(block.text, width - 2):
                yield self._c(DIM, "  " + line)
        elif kind == "tool_call":
            for line in _wrap(block.text, width - 2):
                yield self._c(YELLOW, "● " + line)
        elif kind == "tool_result":
            for line in self._tool_result_lines(block.text, width, ""):
                yield line
        elif kind == "turn":
            yield self._c(DIM, f"── {block.text} ──")
        elif kind == "steer":
            yield self._c(CYAN, "✎ steering: " + block.text)
        elif kind == "error":
            yield self._c(RED, "! " + block.text)
        elif kind == "stopped":
            yield self._c(YELLOW, "■ " + block.text)
        else:
            for line in _wrap(block.text, width):
                yield self._c(DIM, line)

    def _status_line(self, width: int) -> str:
        agent = self.agent
        model = self.model or "(proxy default)"
        turns = getattr(agent, "turns", 0)
        status = getattr(agent, "status", "?")
        waiting = self._waiting_for_input()
        if waiting:
            state = "waiting for input"
        elif status == "running":
            state = "working"
        else:
            state = str(status)
        left = f" Osmia · {model} · turn {turns}"
        if waiting:
            hint = "Enter send · Ctrl+D exit"
        elif status == "running":
            hint = "Enter steer · Ctrl+C stop"
        else:
            hint = ""
        right = f"{state} · {hint} " if hint else f"{state} "
        if _display_width(left) + _display_width(right) >= width:
            right = f"{state} "
        text = (
            left
            + " " * max(1, width - _display_width(left) - _display_width(right))
            + right
        )
        return self._c(REVERSE, text[:width])

    def _input_line(self, width: int) -> tuple[str, int]:
        prompt = "❯ "
        prompt_width = _display_width(prompt)
        line = "".join(self._line)
        available = max(1, width - prompt_width)
        start = 0
        while _display_width(line[start : self._cursor]) >= available:
            start += 1
        visible = line[start:]
        if _display_width(visible) > available:
            visible = visible[: max(1, available)]
        cursor_col = 1 + prompt_width + _display_width(line[start : self._cursor])
        cursor_col = min(max(1, cursor_col), width)
        return self._c(CYAN + BOLD, prompt) + visible, cursor_col
