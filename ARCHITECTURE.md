# Osmia Architecture

## Purpose

Osmia is a small, synchronous Python harness for agents that use Python functions as tools.
Its defining constraint is that the agent process never handles provider authentication.
Credentials and provider routing live in a local OpenAI-compatible proxy, while agents remain
portable, inspectable processes.

## System boundaries

```
user program
   |
   v
osmia.Agent  --- HTTP/SSE (OpenAI Chat Completions) --->  osmia.Proxy
   |                                                          |
   | local HTTP + registry heartbeat                          +--> OpenRouter
   v                                                          |
control plane / attach client                                  +--> ChatGPT Codex
                                                                    (Responses over WebSocket)
```

- **Agent process** — owns the prompt, conversation, Python callables, steering queue, event
  log, and local attach server. It has no credential store and no provider SDK.
- **Proxy process** — owns provider credentials, model slug routing, token refresh, and
  upstream transports. It selects a provider by model slug only.
- **Control plane** — explicitly out of scope. Osmia exposes the registry and per-agent stream
  that makes one trivial to build.

## Agent

`Agent` accepts a prompt, optional system prompt, a list of Python callables or `Tool`
objects, an optional model slug, and an optional proxy URL. `Agent.run()` is synchronous and
drives a standard tool-calling loop:

1. Send the conversation and tool schemas to `POST /v1/chat/completions` with `stream=true`.
2. Accumulate assistant content and tool-call deltas from SSE.
3. If tool calls are present, execute each synchronously, append `role=tool` results, and
   start another turn.
4. If no tool calls are present, drain any queued steering messages. If steering exists,
   continue with those user messages; otherwise return the final assistant text.

Steering is cooperative: `Agent.steer(text)` and `POST /steer` queue a user message that is
delivered at the next safe boundary (after the current upstream response or after its tool
calls). This avoids interrupting tool execution or leaving a malformed transcript.

`tui=None` enables the terminal UI only when stdin and stdout are TTYs. The UI streams
assistant text, shows tool calls and results, and runs a `select`-based stdin reader for
steering. `tui=False` is quiet headless mode.

## Tools

`Tool` wraps a callable with an agent-facing specification. Plain callables are accepted and
converted automatically. `@tool` may override the name, description, or the entire parameter
JSON Schema. Nested types (`Optional`, `Literal`, `Enum`, lists, dicts, `Annotated`) are
inferred where practical; otherwise a parameter is omitted or described as an object.

Description precedence:

1. `@tool(description=...)`
2. callable docstring
3. `Call <name>.` fallback

Parameter descriptions come from the callable's annotations or from a complete
`@tool(schema=...)` override. Positional-only parameters are invoked positionally. Variadic
parameters are omitted from the schema rather than guessed at.

## Proxy

The proxy is an `http.server.ThreadingHTTPServer` so that its dependency surface stays small.

### Model routing

The proxy config maps a public slug to exactly one provider and upstream model. The agent may
send `model` as a slug, but provider selection is never accepted from the agent. An unknown
slug returns HTTP 404 with the configured slugs in the error body. A missing model uses
`default_model`.

### Upstream adapters

- **OpenRouter** — forwards Chat Completions requests to
  `https://openrouter.ai/api/v1/chat/completions` and relays its SSE. The configured model
  target replaces the public slug.
- **ChatGPT Codex** — translates Chat Completions to the Responses API and uses the Codex
  Responses WebSocket. The socket URL is the HTTP base URL with `ws`/`wss` substituted and
  `/responses` appended; the first text frame is the request body plus
  `{"type": "response.create"}`.

Codex WebSocket headers follow the Codex transport contract observed in CLIProxyAPI:

```
Authorization: Bearer <access token>
chatgpt-account-id: <account id>
originator: osmia
OpenAI-Beta: responses_websockets=2026-02-06
session_id: <agent/session id>
conversation_id: <agent/session id>
```

Codex OAuth uses the public Codex client id, PKCE, and the fixed callback
`http://localhost:1455/auth/callback`. A device-code flow is available for headless login.
Access tokens are refreshed from the stored refresh token before expiry and written back to
the config with mode `0600`.

### Event translation

Codex emits Responses events. The proxy translates:

- `response.output_text.delta` -> assistant `content` deltas
- `response.reasoning_summary_text.delta` / `response.reasoning_text.delta` ->
  `reasoning_content` deltas
- `response.output_item.added` (`function_call`) -> `tool_calls` delta with call id and name
- `response.function_call_arguments.delta` -> `tool_calls[].function.arguments` deltas
- `response.completed` / `response.incomplete` -> final `finish_reason`, usage, and `[DONE]`

A new WebSocket is opened for each proxy request. The session id remains stable per agent so
upstream prompt caching still works. Duplex steering over the Codex socket
(`response.steer`) is intentionally not used: Osmia's steering is turn-boundary based.

## Discovery

The agent chooses an open port by binding candidates from `proxy_port + 1` to
`proxy_port + agent_port_count - 1`. The bound server exposes state, events, steering, and stop.

Every two seconds the agent `PUT`s a compact snapshot to `/v1/agents/{id}` on the proxy.
The proxy keeps the registry in memory, adds a `last_seen` timestamp and a `stale` flag, and
serves `GET /v1/agents`. On shutdown the agent sends `DELETE`. Registry failures are ignored
by the agent beyond being recorded; they never fail an agent run.

The event log is a bounded in-memory list with a monotonically increasing sequence number.
`GET /events?after=N` (also `since=N`) returns JSON events for replay. Live control commands
are forwarded to the agent's local `/steer` and `/stop` endpoints.

## Invariants

- The agent never reads provider credentials or refreshes tokens.
- The agent cannot set the provider; it can only request a configured model slug.
- Tool execution and steering happen on the run thread, never concurrently with transcript
  mutation.
- The proxy only binds to loopback by default. Exposing it remotely requires an explicit
  `--host` and an external security boundary; Osmia deliberately has no proxy auth.
- Registry and attach surfaces are observability/control aids, not security boundaries.
