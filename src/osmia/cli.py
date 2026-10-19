"""Command line interface for Osmia."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .auth import (
    LoginError,
    login_codex_browser,
    login_codex_device,
    login_openrouter_key,
    login_openrouter_oauth,
)
from .client import ProxyClient, ProxyError, default_proxy_url
from .proxy.config import Config
from .proxy.server import Proxy


def _proxy_url(args: argparse.Namespace) -> str:
    return args.proxy or default_proxy_url()


def cmd_serve(args: argparse.Namespace) -> int:
    config = Config.load(Path(args.config) if args.config else None)
    proxy = Proxy(config=config, host=args.host, port=args.port)
    proxy.start()
    print(f"Osmia proxy listening on {proxy.url}")
    print(f"Default model: {config.default_model}")
    print(f"Models: {', '.join(sorted(config.routes()))}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping Osmia proxy.")
    finally:
        proxy.stop()
    return 0


def cmd_login(args: argparse.Namespace) -> int:
    config = Config.load(Path(args.config) if args.config else None)
    try:
        if args.provider == "openrouter":
            if args.oauth:
                path = login_openrouter_oauth(
                    config, open_browser=not args.no_browser
                )
            else:
                key = args.api_key if args.api_key is not None else os.environ.get("OPENROUTER_API_KEY")
                if not key and sys.stdin.isatty():
                    key = getpass.getpass("OpenRouter API key: ")
                path = login_openrouter_key(config, key)
            print(f"OpenRouter credentials saved to {path}")
        elif args.provider == "codex":
            if args.device:
                path = login_codex_device(config)
            else:
                path = login_codex_browser(
                    config, open_browser=not args.no_browser
                )
            print(f"Codex credentials saved to {path}")
        else:
            raise LoginError(f"Unsupported provider {args.provider}")
    except LoginError as exc:
        print(f"Login failed: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_logout(args: argparse.Namespace) -> int:
    config = Config.load(Path(args.config) if args.config else None)
    config.clear_provider(args.provider)
    print(f"Cleared {args.provider} credentials in {config.path}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    config = Config.load(Path(args.config) if args.config else None)
    routes = config.routes()
    if args.json:
        print(
            json.dumps(
                {
                    slug: {
                        "provider": route.provider,
                        "model": route.model,
                        **({"options": route.options} if route.options else {}),
                    }
                    for slug, route in sorted(routes.items())
                },
                indent=2,
            )
        )
        return 0
    default = config.default_model
    for slug, route in sorted(routes.items()):
        marker = "*" if slug == default else " "
        print(f"{marker} {slug:<28} {route.provider:<12} {route.model}")
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    client = ProxyClient(_proxy_url(args))
    try:
        agents = client.agents()
    except ProxyError as exc:
        print(f"Cannot reach the Osmia proxy at {client.base_url}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(agents, indent=2, default=str))
        return 0
    if not agents:
        print(f"No agents registered at {client.base_url}.")
        return 0
    print(f"{'ID':<22} {'STATUS':<9} {'TURNS':<6} {'MODEL':<18} ATTACH")
    for agent in agents:
        status = str(agent.get("status") or "?")
        if agent.get("stale"):
            status += "*"
        print(
            f"{str(agent.get('id')):<22} {status:<9} {str(agent.get('turns', 0)):<6} "
            f"{str(agent.get('model', '')):<18} {agent.get('attach_url') or agent.get('url') or ''}"
        )
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    import httpx

    url = _proxy_url(args).rstrip("/") + "/v1/events"
    params: dict[str, Any] = {"after": args.after, "limit": args.limit}
    if args.agent:
        params["agent"] = args.agent
    try:
        response = httpx.get(url, params=params, timeout=10)
        if response.status_code >= 400:
            print(f"Proxy returned {response.status_code}: {response.text}", file=sys.stderr)
            return 1
        data = response.json().get("data") or []
    except Exception as exc:
        print(f"Cannot reach the Osmia proxy at {_proxy_url(args)}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        return 0
    for event in data:
        payload = event.get("payload") or {}
        summary = json.dumps(payload, ensure_ascii=False, default=str)
        if len(summary) > 160:
            summary = summary[:157] + "..."
        print(f"{event.get('seq'):>6} {event.get('kind'):<24} {summary}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    config = Config.load(Path(args.config) if args.config else None)
    if args.json:
        print(json.dumps(config.to_dict(redact=not args.show_secrets), indent=2))
    else:
        print(f"Config: {config.path}")
        print(f"Proxy: http://{config.host}:{config.port}")
        print(f"Default model: {config.default_model}")
        print("Providers:")
        for name in ("openrouter", "codex"):
            values = config.provider(name)
            active = any(value for key, value in values.items() if key in (
                "api_key", "access_token", "refresh_token"
            ))
            print(f"  {name:<12} {'logged in' if active else 'not logged in'}")
        if args.show_secrets:
            print(json.dumps(config.to_dict(redact=False), indent=2))
        else:
            print("Use `osmia config --show-secrets` to print credentials.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osmia",
        description="Minimal Python agent harness with a local model proxy.",
    )
    parser.add_argument("--version", action="version", version=f"osmia {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the local OpenAI-compatible proxy")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--config", default=None)
    serve.set_defaults(func=cmd_serve)

    login = sub.add_parser("login", help="Log in to a provider")
    login.add_argument("provider", choices=["openrouter", "codex"])
    login.add_argument("--api-key", default=None, help="OpenRouter API key")
    login.add_argument("--oauth", action="store_true", help="Use OpenRouter OAuth PKCE")
    login.add_argument("--device", action="store_true", help="Use the Codex device flow")
    login.add_argument("--no-browser", action="store_true")
    login.add_argument("--config", default=None)
    login.set_defaults(func=cmd_login)

    logout = sub.add_parser("logout", help="Clear saved provider credentials")
    logout.add_argument("provider", choices=["openrouter", "codex"])
    logout.add_argument("--config", default=None)
    logout.set_defaults(func=cmd_logout)

    models = sub.add_parser("models", help="List configured model slugs")
    models.add_argument("--config", default=None)
    models.add_argument("--json", action="store_true")
    models.set_defaults(func=cmd_models)

    agents = sub.add_parser("agents", help="List agents registered with the proxy")
    agents.add_argument("--proxy", default=None)
    agents.add_argument("--json", action="store_true")
    agents.set_defaults(func=cmd_agents)

    events = sub.add_parser("events", help="Read proxy or agent events")
    events.add_argument("--proxy", default=None)
    events.add_argument("--agent", default=None)
    events.add_argument("--after", type=int, default=0)
    events.add_argument("--limit", type=int, default=100)
    events.add_argument("--json", action="store_true")
    events.set_defaults(func=cmd_events)

    config = sub.add_parser("config", help="Show the config path and status")
    config.add_argument("--config", default=None)
    config.add_argument("--json", action="store_true")
    config.add_argument("--show-secrets", action="store_true")
    config.set_defaults(func=cmd_config)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ProxyError, LoginError, ValueError, KeyError) as exc:
        print(f"osmia: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
