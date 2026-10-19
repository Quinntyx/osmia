"""Local OpenAI-compatible proxy package."""

from typing import Any

from .config import Config, ModelRoute, config_path

__all__ = ["Config", "ModelRoute", "Proxy", "ProxyError", "config_path"]


def __getattr__(name: str) -> Any:
    if name in ("Proxy", "ProxyError"):
        from .server import Proxy, ProxyError

        return {"Proxy": Proxy, "ProxyError": ProxyError}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
