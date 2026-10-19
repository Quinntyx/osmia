"""HTTP client for the Osmia proxy, usable independently of ``Agent``."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import httpx


class ProxyError(RuntimeError):
    pass


def default_proxy_url() -> str:
    explicit = os.environ.get("OSMIA_PROXY_URL")
    if explicit:
        return explicit.rstrip("/")
    from .proxy.config import Config

    try:
        config = Config.load(create=False)
        return f"http://{config.host}:{config.port}"
    except Exception:
        return "http://127.0.0.1:8765"


class ProxyClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = (base_url or default_proxy_url()).rstrip("/")
        self.api_key = api_key or os.environ.get("OSMIA_PROXY_TOKEN")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _url(self, path: str) -> str:
        return self.base_url + "/" + path.lstrip("/")

    def _raise(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            payload = response.json()
            message = payload.get("error", {}).get("message") or json.dumps(payload)
        except Exception:
            message = response.text
        raise ProxyError(f"Osmia proxy {response.status_code}: {message}")

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float = 10.0,
        **kwargs: Any,
    ) -> httpx.Response:
        try:
            response = httpx.request(
                method, self._url(path), timeout=timeout, **kwargs
            )
        except httpx.HTTPError as exc:
            raise ProxyError(
                f"Cannot reach the Osmia proxy at {self.base_url}: {exc}"
            ) from exc
        self._raise(response)
        return response

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health").json()

    def models(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/v1/models").json().get("data") or [])

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"messages": messages, "stream": False}
        if model:
            payload["model"] = model
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)
        response = self._request(
            "POST",
            "/v1/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        return response.json()

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        payload: dict[str, Any] = {"messages": messages, "stream": True}
        if model:
            payload["model"] = model
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)
        started = time.time()
        try:
            with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=30.0)) as client:
                with client.stream(
                    "POST",
                    self._url("/v1/chat/completions"),
                    json=payload,
                    headers=self._headers(),
                ) as response:
                    if response.status_code >= 400:
                        text = response.read().decode("utf-8", "replace")
                        raise ProxyError(
                            f"Osmia proxy {response.status_code}: {text}"
                        )
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", "replace")
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(event, dict) and event.get("error"):
                            raise ProxyError(str(event["error"]))
                        if isinstance(event, dict):
                            yield event
        except httpx.HTTPError as exc:
            raise ProxyError(f"Osmia proxy request failed after {time.time() - started:.1f}s: {exc}") from exc

    def agents(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/v1/agents").json().get("data") or [])

    def agent(self, agent_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/agents/{agent_id}").json()

    def steer_agent(self, agent_id: str, text: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/agents/{agent_id}/steer",
            json={"text": text},
            headers=self._headers(),
        ).json()

    def stop_agent(self, agent_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/agents/{agent_id}/stop",
            json={},
            headers=self._headers(),
        ).json()

    def events(self, after: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        return list(
            self._request(
                "GET",
                "/v1/events",
                params={"after": after, "limit": limit},
            ).json().get("data")
            or []
        )
