"""Bounded, inspectable event log shared by TUI and control plane."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any


class EventLog:
    def __init__(self, max_events: int = 2000) -> None:
        self._lock = threading.Lock()
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._seq = 0

    def append(self, kind: str, payload: dict[str, Any] | None = None) -> int:
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._events.append(
                {
                    "seq": seq,
                    "time": time.time(),
                    "kind": kind,
                    "payload": payload or {},
                }
            )
            return seq

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    def since(self, seq: int, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            events = [event for event in self._events if event["seq"] > seq]
        return events[-limit:]

    def snapshot(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)[-limit:]
