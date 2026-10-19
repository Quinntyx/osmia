# Osmia Agent Guidance

## Project conventions

- Python 3.10+; `from __future__ import annotations` in every module.
- Runtime dependencies are intentionally limited to `httpx` and `websockets`.
- Standard-library servers (`http.server`) are used instead of a web framework.
- Keep the agent free of authentication code and provider credentials.
- The proxy is the only component that may read or write `~/.config/osmia/config.json`
  credentials.
- `Agent(prompt=...)` stores an initial user message; `Agent.run(prompt)` appends another.
- Prefer explicit dataclasses and small pure helpers over frameworks.
- Do not add a control-plane UI. `GET /v1/agents` and the agent's `/state` + `/events` are the
  contract.

## Commands

```bash
uv sync --extra dev
uv run pytest
uv run osmia --help
uv run osmia serve
```

## Manual checks

```bash
# Proxy health and model routing
uv run osmia serve --port 8765
curl -s http://127.0.0.1:8765/health
curl -s http://127.0.0.1:8765/v1/models

# Registry while an agent runs
curl -s http://127.0.0.1:8765/v1/agents

# Agent attach surface (port from the registry)
curl -s http://127.0.0.1:8766/state
curl -s 'http://127.0.0.1:8766/events.json?since=0'
curl -s -X POST http://127.0.0.1:8766/steer \
  -H 'content-type: application/json' \
  -d '{"text":"focus on tests"}'
```

## Constraints

- Codex transport must use WebSocket, not the HTTP Responses fallback. The reference
  implementation is `quinntyx/CLIProxyAPI` on `git.quinntyx.dev`; relevant files include
  `internal/runtime/executor/codex_websockets_connection.go`,
  `codex_websockets_request.go`, and `codex_websockets_stream.go`.
- Never accept a `provider` field from the agent request. Resolve it from the model slug.
- Never block tool execution on steering input.
- Do not write credentials to logs, event streams, or agent state.
- Tests must not require real credentials or network access.
