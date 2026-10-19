# Osmia Plan

Status as of 2026-10-19. The first working implementation is complete and tested.

## Milestones

### 1. Tool layer — complete

`src/osmia/tools.py` provides `Tool`, `ToolMetadata`, `tool()`, docstring/annotation schema
inference, schema overrides, and positional-only invocation. Covered by `tests/test_tools.py`.

### 2. Agent loop and terminal surface — complete

`src/osmia/agent.py`, `client.py`, `events.py`, and `tui.py` implement SSE accumulation,
synchronous tool execution, turn-boundary steering, stop handling, and TTY-gated rendering.
`Agent` accepts `prompt` at construction or `run()`. Covered by `tests/test_agent.py`.

### 3. Discovery — complete

`src/osmia/discovery.py` binds a port above the proxy, serves `/state`, `/events`,
`/events.json`, `/steer`, and `/stop`, and registers with the proxy every two seconds.
The proxy lists and controls agents through `/v1/agents`. Covered by `tests/test_agent.py`
and `tests/test_proxy.py`.

### 4. Proxy core — complete

`src/osmia/proxy/server.py` and `config.py` provide `POST /v1/chat/completions`,
`GET /v1/models`, `GET /health`, streaming SSE relay, non-stream aggregation, slug routing,
and the agent registry. Covered by `tests/test_proxy.py`.

### 5. Codex WebSocket transport — complete

`src/osmia/providers.py` translates Chat Completions to Responses, opens the Codex
`responses` WebSocket with OAuth headers, sends `response.create`, and translates Responses
events into Chat Completions chunks. A 401 handshake triggers one token refresh and reconnect.
Covered by `tests/test_providers.py` with a local fake WebSocket server.

### 6. Login and CLI — complete

`src/osmia/auth.py` implements Codex PKCE browser/device login, token refresh, OpenRouter
API-key and PKCE login, and atomic `0600` config writes. `src/osmia/cli.py` provides
`serve`, `login`, `logout`, `models`, `agents`, `events`, and `config`. Covered by
`tests/test_auth.py` and `tests/test_cli.py`.

### 7. Documentation and verification — mostly complete

README, ARCHITECTURE, AGENTS, and this plan are written. `uv run pytest` passes. A live
upstream call still requires the user's OpenRouter key or Codex login.

## Current status

- 30 local tests pass, including an end-to-end agent/tool loop through a live local proxy and
  a Codex WebSocket translation test.
- Packaging uses `hatchling`; dependencies are `httpx` and `websockets`.

## Remaining checks

- Manual live verification against OpenRouter and ChatGPT Codex with real credentials.
- Optional: an SSE attach endpoint if a control plane needs push rather than polling.

## Next action

Run a live proxy request after `osmia login codex` or `osmia login openrouter`.
