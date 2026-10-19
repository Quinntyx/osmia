"""Osmia proxy configuration and model-slug routing."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_PROVIDERS = ("openrouter", "codex")

DEFAULT_MODELS: dict[str, dict[str, Any]] = {
    "gpt-5.1-codex": {"provider": "codex", "model": "gpt-5.1-codex"},
    "gpt-5.1-codex-mini": {"provider": "codex", "model": "gpt-5.1-codex-mini"},
    "gpt-5.1": {"provider": "codex", "model": "gpt-5.1"},
    "claude-sonnet-4.5": {
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4.5",
    },
    "gpt-4.1": {"provider": "openrouter", "model": "openai/gpt-4.1"},
}


def config_path() -> Path:
    explicit = os.environ.get("OSMIA_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("OSMIA_HOME")
    if home:
        return Path(home).expanduser() / "config.json"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "osmia" / "config.json"


def default_config_data() -> dict[str, Any]:
    return {
        "host": "127.0.0.1",
        "port": 8765,
        "default_model": "gpt-5.1-codex",
        "agent_port_count": 100,
        "agent_stale_seconds": 10,
        "models": json.loads(json.dumps(DEFAULT_MODELS)),
        "providers": {
            "openrouter": {"api_key": None},
            "codex": {
                "access_token": None,
                "refresh_token": None,
                "expires_at": None,
                "account_id": None,
            },
        },
    }


@dataclass
class ModelRoute:
    slug: str
    provider: str
    model: str
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"provider": self.provider, "model": self.model}
        if self.options:
            payload["options"] = dict(self.options)
        return payload


class Config:
    """Mutable config with atomic, permission-restricted persistence."""

    def __init__(self, path: Path | None = None, data: dict[str, Any] | None = None) -> None:
        self.path = Path(path) if path is not None else config_path()
        self._lock = threading.RLock()
        self._data = data if data is not None else default_config_data()
        self._normalize()

    def _normalize(self) -> None:
        defaults = default_config_data()
        for key, value in defaults.items():
            self._data.setdefault(key, value)
        models = self._data.get("models")
        if not isinstance(models, dict) or not models:
            self._data["models"] = json.loads(json.dumps(DEFAULT_MODELS))
        providers = self._data.get("providers")
        if not isinstance(providers, dict):
            self._data["providers"] = {}
        for name in SUPPORTED_PROVIDERS:
            self._data["providers"].setdefault(name, defaults["providers"][name])
        if self._data.get("default_model") not in self._data.get("models", {}):
            self._data["default_model"] = next(iter(self._data["models"]))

    @classmethod
    def load(cls, path: Path | None = None, create: bool = True) -> Config:
        target = Path(path) if path is not None else config_path()
        if target.exists():
            with target.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError(f"Osmia config must be a JSON object: {target}")
            return cls(target, data)
        config = cls(target)
        if create:
            config.save()
        return config

    @property
    def host(self) -> str:
        return str(self._data.get("host") or "127.0.0.1")

    @property
    def port(self) -> int:
        return int(self._data.get("port") or 8765)

    @property
    def default_model(self) -> str:
        return str(self._data["default_model"])

    @property
    def agent_port_count(self) -> int:
        try:
            return max(1, int(self._data.get("agent_port_count") or 100))
        except (TypeError, ValueError):
            return 100

    @property
    def agent_stale_seconds(self) -> float:
        try:
            return max(2.0, float(self._data.get("agent_stale_seconds") or 10))
        except (TypeError, ValueError):
            return 10.0

    def routes(self) -> dict[str, ModelRoute]:
        result: dict[str, ModelRoute] = {}
        for slug, raw in self._data.get("models", {}).items():
            if not isinstance(raw, dict):
                continue
            provider = str(raw.get("provider") or "").strip()
            model = str(raw.get("model") or slug).strip()
            options = raw.get("options")
            result[slug] = ModelRoute(
                slug=slug,
                provider=provider,
                model=model,
                options=dict(options) if isinstance(options, dict) else {},
            )
        return result

    def route(self, slug: str | None) -> ModelRoute:
        chosen = (slug or "").strip() or self.default_model
        routes = self.routes()
        if chosen not in routes:
            available = ", ".join(sorted(routes))
            raise KeyError(
                f"Unknown model slug {chosen!r}. Configured models: {available}"
            )
        return routes[chosen]

    def provider(self, name: str) -> dict[str, Any]:
        with self._lock:
            raw = self._data.get("providers", {}).get(name)
            return json.loads(json.dumps(raw)) if isinstance(raw, dict) else {}

    def update_provider(self, name: str, values: dict[str, Any]) -> None:
        with self._lock:
            providers = self._data.setdefault("providers", {})
            current = providers.setdefault(name, {})
            current.update(values)
            self.save()

    def clear_provider(self, name: str) -> None:
        with self._lock:
            self._data.setdefault("providers", {})[name] = json.loads(
                json.dumps(default_config_data()["providers"][name])
            )
            self.save()

    def to_dict(self, redact: bool = True) -> dict[str, Any]:
        with self._lock:
            data = json.loads(json.dumps(self._data))
        if redact:
            for values in data.get("providers", {}).values():
                if not isinstance(values, dict):
                    continue
                for key in list(values):
                    if values[key]:
                        values[key] = "<set>"
                    else:
                        values[key] = "<unset>"
        return data

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self._data, indent=2, sort_keys=False) + "\n"
            fd, tmp_name = tempfile.mkstemp(
                prefix=".osmia-config-", suffix=".json", dir=str(self.path.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(tmp_name, 0o600)
                os.replace(tmp_name, self.path)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]
