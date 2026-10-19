"""OpenRouter and Codex authentication flows.

Credentials live only in the Osmia config file (mode 0600).  Tokens are
refreshed lazily by the proxy without mutating process environment variables.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx

from .proxy.config import Config


OPENROUTER_AUTH_URL = "https://openrouter.ai/auth"
OPENROUTER_KEYS_URL = "https://openrouter.ai/api/v1/auth/keys"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_DEVICE_CODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
CODEX_DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
CODEX_REDIRECT_URI = "http://localhost:1455/auth/callback"
CODEX_DEVICE_REDIRECT_URI = "https://auth.openai.com/deviceauth/callback"
CODEX_SCOPE = "openid profile email offline_access"


class LoginError(RuntimeError):
    pass


@dataclass
class Pkce:
    verifier: str
    challenge: str

    @classmethod
    def generate(cls) -> Pkce:
        verifier = secrets.token_urlsafe(64)
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        return cls(verifier=verifier, challenge=challenge)


class _CallbackServer:
    """One-shot loopback HTTP server for OAuth redirects."""

    def __init__(self, port: int, path: str, state: str | None = None) -> None:
        self.port = port
        self.path = path
        self.state = state
        self.result: dict[str, str] | None = None
        self.error: str | None = None
        self._event = threading.Event()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def redirect_uri(self) -> str:
        return f"http://localhost:{self.port}{self.path}"

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "OsmiaLogin/1.0"

            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ANN001
                return

            def do_GET(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path != outer.path:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                params = urllib.parse.parse_qs(parsed.query)
                if outer.state is not None and params.get("state", [None])[0] != outer.state:
                    outer.error = "OAuth state did not match"
                elif params.get("error"):
                    outer.error = params.get("error_description", params["error"][0])[0]
                else:
                    outer.result = {key: values[0] for key, values in params.items()}
                body = (
                    b"<html><body><h2>Osmia login complete</h2>"
                    b"<p>You can close this window.</p></body></html>"
                )
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                outer._event.set()

        try:
            self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        except OSError as exc:
            raise LoginError(
                f"Cannot listen on localhost:{self.port} for the OAuth callback: {exc}. "
                "Close the process using that port and retry."
            ) from exc
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def wait(self, timeout: float) -> dict[str, str]:
        if not self._event.wait(timeout):
            raise LoginError(f"Timed out waiting for the OAuth callback on {self.redirect_uri}")
        if self.error:
            raise LoginError(self.error)
        if self.result is None:
            raise LoginError("OAuth callback returned no parameters")
        return self.result

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)


def _jwt_claims(token: str) -> dict[str, Any]:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload)
        claims = json.loads(decoded)
        return claims if isinstance(claims, dict) else {}
    except Exception:
        return {}


def _account_id(token_data: dict[str, Any]) -> str | None:
    for token_name in ("id_token", "access_token"):
        token = token_data.get(token_name)
        if not isinstance(token, str):
            continue
        claims = _jwt_claims(token)
        auth = claims.get("https://api.openai.com/auth")
        if isinstance(auth, dict):
            account = auth.get("chatgpt_account_id")
            if account:
                return str(account)
        if claims.get("chatgpt_account_id"):
            return str(claims["chatgpt_account_id"])
        organizations = claims.get("organizations")
        if isinstance(organizations, list) and organizations:
            first = organizations[0]
            if isinstance(first, dict) and first.get("id"):
                return str(first["id"])
    return None


def _expiry(token_data: dict[str, Any]) -> float:
    expires_in = token_data.get("expires_in")
    if isinstance(expires_in, (int, float)):
        return time.time() + float(expires_in)
    for token_name in ("access_token", "id_token"):
        token = token_data.get(token_name)
        if isinstance(token, str):
            exp = _jwt_claims(token).get("exp")
            if isinstance(exp, (int, float)):
                return float(exp)
    return time.time() + 3600


def save_openrouter_key(config: Config, api_key: str) -> None:
    key = api_key.strip()
    if not key:
        raise LoginError("OpenRouter API key is empty")
    config.update_provider("openrouter", {"api_key": key, "logged_in_at": time.time()})


def login_openrouter_key(config: Config, api_key: str | None = None) -> str:
    key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise LoginError(
            "No OpenRouter API key supplied. Pass --api-key or set OPENROUTER_API_KEY."
        )
    save_openrouter_key(config, key)
    return config.path.as_posix()


def login_openrouter_oauth(
    config: Config, open_browser: bool = True, timeout: float = 300.0
) -> str:
    pkce = Pkce.generate()
    callback = _CallbackServer(port=0, path="/callback")
    callback.start()
    try:
        query = urllib.parse.urlencode(
            {
                "response_type": "code",
                "client_id": "osmia",
                "redirect_uri": callback.redirect_uri,
                "code_challenge": pkce.challenge,
                "code_challenge_method": "S256",
            }
        )
        url = f"{OPENROUTER_AUTH_URL}?{query}"
        print(f"Open this URL to authorize OpenRouter:\n{url}\n")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        result = callback.wait(timeout)
        code = result.get("code")
        if not code:
            raise LoginError("OpenRouter callback did not include an authorization code")
    finally:
        callback.close()

    response = httpx.post(
        OPENROUTER_KEYS_URL,
        json={
            "code": code,
            "code_verifier": pkce.verifier,
            "code_challenge_method": "S256",
        },
        timeout=30.0,
    )
    if response.status_code >= 400:
        raise LoginError(f"OpenRouter key exchange failed: {response.status_code} {response.text}")
    payload = response.json()
    key = payload.get("key") if isinstance(payload, dict) else None
    if not key:
        raise LoginError(f"OpenRouter key exchange returned no key: {payload}")
    save_openrouter_key(config, str(key))
    return config.path.as_posix()


def _decode_oauth_response(response: httpx.Response, expected: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise LoginError(f"Token endpoint returned non-JSON: {response.text}") from exc
    if response.status_code >= 400 or not isinstance(payload, dict):
        raise LoginError(f"Token endpoint failed: {response.status_code} {payload}")
    if payload.get("error"):
        raise LoginError(f"Token endpoint error: {payload}")
    if expected and not payload.get(expected):
        raise LoginError(f"Token response is missing {expected}: {payload}")
    return payload


def _exchange_codex_code(code: str, verifier: str, redirect_uri: str) -> dict[str, Any]:
    response = httpx.post(
        CODEX_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": CODEX_CLIENT_ID,
            "code_verifier": verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30.0,
    )
    return _decode_oauth_response(response, "access_token")


def _save_codex_token(config: Config, token_data: dict[str, Any]) -> dict[str, Any]:
    access_token = str(token_data["access_token"])
    refresh_token = token_data.get("refresh_token")
    account = _account_id(token_data)
    values = {
        "access_token": access_token,
        "refresh_token": str(refresh_token) if refresh_token else None,
        "expires_at": _expiry(token_data),
        "account_id": account,
        "logged_in_at": time.time(),
    }
    if not values["refresh_token"]:
        existing = config.provider("codex").get("refresh_token")
        values["refresh_token"] = existing
    config.update_provider("codex", values)
    return values


def login_codex_browser(config: Config, open_browser: bool = True, timeout: float = 300.0) -> str:
    pkce = Pkce.generate()
    state = secrets.token_urlsafe(24)
    callback = _CallbackServer(port=1455, path="/auth/callback", state=state)
    callback.start()
    try:
        query = urllib.parse.urlencode(
            {
                "response_type": "code",
                "client_id": CODEX_CLIENT_ID,
                "redirect_uri": CODEX_REDIRECT_URI,
                "scope": CODEX_SCOPE,
                "code_challenge": pkce.challenge,
                "code_challenge_method": "S256",
                "state": state,
                "id_token_add_organizations": "true",
                "codex_cli_simplified_flow": "true",
            }
        )
        url = f"{CODEX_AUTHORIZE_URL}?{query}"
        print(f"Open this URL to authorize Codex / ChatGPT:\n{url}\n")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        result = callback.wait(timeout)
        code = result.get("code")
        if not code:
            raise LoginError("Codex callback did not include an authorization code")
    finally:
        callback.close()

    token_data = _exchange_codex_code(code, pkce.verifier, CODEX_REDIRECT_URI)
    _save_codex_token(config, token_data)
    return config.path.as_posix()


def login_codex_device(config: Config, timeout: float = 900.0) -> str:
    start = httpx.post(
        CODEX_DEVICE_CODE_URL,
        json={"client_id": CODEX_CLIENT_ID},
        timeout=30.0,
    )
    if start.status_code >= 400:
        raise LoginError(f"Codex device authorization failed: {start.status_code} {start.text}")
    payload = start.json()
    if not isinstance(payload, dict):
        raise LoginError(f"Unexpected device authorization response: {payload}")
    device_auth_id = payload.get("device_auth_id") or payload.get("device_code")
    user_code = payload.get("user_code") or payload.get("usercode")
    verification_uri = (
        payload.get("verification_uri")
        or payload.get("verification_url")
        or "https://auth.openai.com/codex/device"
    )
    if not device_auth_id or not user_code:
        raise LoginError(f"Device authorization response missing fields: {payload}")
    interval = float(payload.get("interval") or 5)
    print(f"Open {verification_uri}\nand enter code: {user_code}\n")

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(interval)
        poll = httpx.post(
            CODEX_DEVICE_TOKEN_URL,
            json={
                "client_id": CODEX_CLIENT_ID,
                "device_auth_id": device_auth_id,
                "user_code": user_code,
            },
            timeout=30.0,
        )
        if poll.status_code == 200:
            token_payload = poll.json()
            if isinstance(token_payload, dict):
                code = token_payload.get("authorization_code") or token_payload.get("code")
                verifier = token_payload.get("code_verifier") or ""
                if code:
                    token_data = _exchange_codex_code(
                        code, verifier, CODEX_DEVICE_REDIRECT_URI
                    )
                    _save_codex_token(config, token_data)
                    return config.path.as_posix()
        try:
            error_payload = poll.json()
        except ValueError:
            error_payload = {}
        error = str(error_payload.get("error") or "") if isinstance(error_payload, dict) else ""
        if error in ("authorization_pending", "pending", "deviceauth_authorization_pending"):
            continue
        if error in ("slow_down", "slow_down_error"):
            interval = min(interval + 5, 30)
            continue
        if error in ("access_denied", "expired_token", "expired"):
            raise LoginError(f"Codex device authorization ended: {error}")
        if poll.status_code not in (200, 400, 403, 404, 428):
            poll.raise_for_status()
    raise LoginError("Timed out waiting for Codex device authorization")


def refresh_codex(config: Config) -> dict[str, Any]:
    """Refresh an expired Codex token and persist the rotated credentials."""
    values = config.provider("codex")
    refresh_token = values.get("refresh_token")
    if not refresh_token:
        raise LoginError(
            "Codex is not logged in. Run `osmia login codex` first."
        )
    response = httpx.post(
        CODEX_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CODEX_CLIENT_ID,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30.0,
    )
    token_data = _decode_oauth_response(response, "access_token")
    return _save_codex_token(config, token_data)


def ensure_codex_auth(config: Config) -> dict[str, Any]:
    values = config.provider("codex")
    access_token = values.get("access_token")
    expires_at = values.get("expires_at")
    try:
        expiry = float(expires_at) if expires_at else 0.0
    except (TypeError, ValueError):
        expiry = 0.0
    if not access_token or expiry <= time.time() + 120:
        return refresh_codex(config)
    return values


def redact(value: Any) -> str:
    if not value:
        return "<unset>"
    text = str(value)
    return text[:4] + "..." if len(text) > 8 else "<set>"
