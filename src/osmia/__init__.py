"""Osmia: a minimal Python harness with a local model proxy.

The proxy is imported lazily so an agent process can import ``osmia.Agent`` without
loading provider authentication code.
"""

__version__ = "0.1.0"

from typing import Any

from .agent import Agent  # noqa: E402
from .client import ProxyClient, ProxyError  # noqa: E402
from .tools import Tool, ToolMetadata, tool  # noqa: E402

__all__ = [
    "Agent",
    "Proxy",
    "ProxyClient",
    "ProxyError",
    "Tool",
    "ToolMetadata",
    "tool",
    "__version__",
]


def __getattr__(name: str) -> Any:
    if name == "Proxy":
        from .proxy.server import Proxy

        return Proxy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
