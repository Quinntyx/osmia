from __future__ import annotations

import base64
import json
import os
import stat
import time
from pathlib import Path

import pytest

from osmia import auth
from osmia.auth import (
    LoginError,
    Pkce,
    _account_id,
    _expiry,
    _jwt_claims,
    ensure_codex_auth,
    login_openrouter_key,
    refresh_codex,
    save_openrouter_key,
)
from osmia.proxy.config import Config


def jwt(payload: dict, token_type: str = "access") -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{header}.{body}.sig-{token_type}"


def test_pkce_challenge_is_derived() -> None:
    pkce = Pkce.generate()
    assert len(pkce.verifier) >= 43
    import hashlib

    expected = base64.urlsafe_b64encode(
        hashlib.sha256(pkce.verifier.encode()).digest()
    ).decode().rstrip("=")
    assert pkce.challenge == expected


def test_jwt_helpers() -> None:
    token = jwt(
        {
            "exp": 12345,
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_42"},
        }
    )
    assert _jwt_claims(token)["exp"] == 12345
    assert _account_id({"access_token": token}) == "acct_42"
    assert _expiry({"access_token": token, "expires_in": 10}) > time.time()


def test_config_round_trip_and_permissions(tmp_path) -> None:
    path = tmp_path / "nested" / "config.json"
    config = Config(path)
    config.update_provider("openrouter", {"api_key": "sk-secret"})
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    loaded = Config.load(path)
    assert loaded.provider("openrouter")["api_key"] == "sk-secret"
    redacted = loaded.to_dict()
    assert redacted["providers"]["openrouter"]["api_key"] == "<set>"
    assert loaded.to_dict(redact=False)["providers"]["openrouter"]["api_key"] == "sk-secret"


def test_openrouter_key_login_uses_env(tmp_path, monkeypatch) -> None:
    config = Config(tmp_path / "config.json")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    login_openrouter_key(config)
    assert config.provider("openrouter")["api_key"] == "sk-env"
    with pytest.raises(LoginError):
        save_openrouter_key(config, "")


def test_refresh_codex_rotates_tokens(tmp_path, monkeypatch) -> None:
    config = Config(tmp_path / "config.json")
    config.update_provider(
        "codex",
        {
            "access_token": "old",
            "refresh_token": "refresh-old",
            "expires_at": time.time() - 10,
        },
    )

    class FakeResponse:
        status_code = 200

        def json(self) -> dict:
            return {
                "access_token": jwt({"exp": int(time.time()) + 3600}),
                "refresh_token": "refresh-new",
                "id_token": jwt(
                    {"https://api.openai.com/auth": {"chatgpt_account_id": "acct_9"}}
                ),
                "expires_in": 3600,
            }

    def fake_post(*args, **kwargs):
        assert kwargs["data"]["grant_type"] == "refresh_token"
        assert kwargs["data"]["refresh_token"] in {"refresh-old", "refresh-new"}
        return FakeResponse()

    monkeypatch.setattr(auth.httpx, "post", fake_post)
    values = ensure_codex_auth(config)
    assert values["refresh_token"] == "refresh-new"
    assert values["account_id"] == "acct_9"
    assert config.provider("codex")["access_token"] == values["access_token"]
    assert config.provider("codex")["expires_at"] > time.time()

    values = refresh_codex(config)
    assert values["access_token"]


def test_codex_device_login_uses_polled_verifier(tmp_path, monkeypatch) -> None:
    config = Config(tmp_path / "config.json")
    calls: list[tuple[str, dict]] = []

    class FakeResponse:
        def __init__(self, status: int, payload: dict) -> None:
            self.status_code = status
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self) -> dict:
            return self._payload

    def fake_post(url: str, **kwargs):
        calls.append((url, kwargs))
        if url == auth.CODEX_DEVICE_CODE_URL:
            return FakeResponse(
                200,
                {"device_auth_id": "dev-1", "user_code": "CODE", "interval": 0.01},
            )
        if url == auth.CODEX_DEVICE_TOKEN_URL:
            return FakeResponse(
                200,
                {"authorization_code": "auth-code", "code_verifier": "verifier-1"},
            )
        return FakeResponse(
            200,
            {
                "access_token": jwt({"exp": int(time.time()) + 3600}),
                "refresh_token": "refresh-device",
                "id_token": jwt(
                    {"https://api.openai.com/auth": {"chatgpt_account_id": "acct_device"}}
                ),
                "expires_in": 3600,
            },
        )

    monkeypatch.setattr(auth.httpx, "post", fake_post)
    path = auth.login_codex_device(config, timeout=5)
    assert path == config.path.as_posix()
    exchange = calls[-1][1]["data"]
    assert exchange["code_verifier"] == "verifier-1"
    assert exchange["redirect_uri"] == auth.CODEX_DEVICE_REDIRECT_URI
    saved = config.provider("codex")
    assert saved["refresh_token"] == "refresh-device"
    assert saved["account_id"] == "acct_device"


def test_codex_auth_without_login_errors(tmp_path) -> None:
    config = Config(tmp_path / "config.json")
    with pytest.raises(LoginError):
        ensure_codex_auth(config)
