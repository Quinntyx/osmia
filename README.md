# Osmia

Osmia is a minimal, synchronous Python harness for building tool-using agents. It keeps
provider credentials and routing out of the agent process: a local OpenAI-compatible proxy
owns authentication, and agents only choose a configured model slug.

```python
import os
import osmia

osmia.Agent(
    prompt="Create a new project named Osmia and set up a simple coding harness there.",
    system_prompt=(
        "You are a coding agent. You have access to run bash code and "
        "evaluate Python expressions."
    ),
    tools=[os.system, eval],
).run()
```

`Agent.run()` returns the final assistant text. When stdin and stdout are a TTY it also
renders a small streaming terminal UI with cooperative steering. Pass `tui=False` for quiet,
headless execution, or call `agent.run("another prompt")` to continue a conversation.

## Install

```bash
uv tool install --editable .
# or
uv pip install -e '.[dev]'
```

The `osmia` CLI is installed alongside the library.

## First run

The proxy owns all credentials. The agent never reads provider secrets.

```bash
osmia login codex        # ChatGPT OAuth (browser flow); --device for headless
osmia login openrouter   # API key from --api-key or OPENROUTER_API_KEY; --oauth for PKCE
osmia serve              # default: http://127.0.0.1:8765
```

Then run an agent in another shell:

```python
import osmia

result = osmia.Agent(
    prompt="List the files in the current directory.",
    tools=[__import__("os").system],
).run()
print(result)
```

## Tools

Any Python callable can be a tool. Its docstring becomes the description and its type
annotations become the parameter schema. `@tool` can override the name and description, or
replace the whole parameter schema.

```python
import osmia

@osmia.tool(description="Read a UTF-8 text file.")
def read_file(path: str) -> str:
    """Read the file at `path`."""
    with open(path, encoding="utf-8") as handle:
        return handle.read()

@osmia.tool(name="search", schema={
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query"},
        "limit": {"type": "integer", "description": "Maximum results"},
    },
    "required": ["query"],
    "additionalProperties": False,
})
def search(query, limit=10):
    ...
```

## Proxy configuration

Configuration lives at `~/.config/osmia/config.json` (override with `OSMIA_CONFIG` or
`OSMIA_HOME`). The proxy maps public model slugs to providers. Agents may select a slug but
cannot select a provider.

```json
{
  "default_model": "gpt-5.1-codex",
  "models": {
    "gpt-5.1-codex": { "provider": "codex", "model": "gpt-5.1-codex" },
    "claude-sonnet-4.5": {
      "provider": "openrouter",
      "model": "anthropic/claude-sonnet-4.5"
    }
  }
}
```

ChatGPT Codex uses the Responses API over a WebSocket (`wss://.../backend-api/codex/responses`),
including ChatGPT OAuth token refresh. OpenRouter uses its Chat Completions API over SSE.
Credentials are written atomically with mode `0600`.

## Discovery

Every agent binds a local port in the range immediately above the proxy port (for example
`8766..8865` for a proxy on `8765`) and registers itself with the proxy every two seconds.
The agent's local HTTP surface is:

| Endpoint | Purpose |
| --- | --- |
| `GET /state` | Agent snapshot; `?messages=0` omits the conversation |
| `GET /events?after=N` | Replayable JSON event log (also accepts `since=N`) |
| `GET /events.json?after=N` | Alias for the JSON event log |
| `POST /steer` | `{"text": "..."}` queues a steering message |
| `POST /stop` | Requests a graceful stop |

The proxy exposes the live registry at `GET /v1/agents`, and can forward control with
`POST /v1/agents/{id}/steer` and `POST /v1/agents/{id}/stop`. A control plane that uses these
endpoints is intentionally out of scope.

## CLI

```bash
osmia serve                      # run the proxy
osmia login codex|openrouter     # store provider credentials
osmia logout codex|openrouter    # clear provider credentials
osmia models                     # list configured slugs
osmia agents --proxy URL         # list registered agents
osmia events --agent ID          # inspect the proxy event log
osmia config                     # show redacted configuration
```

## Layout

- `src/osmia/agent.py` — synchronous agent loop, steering, and discovery lifecycle.
- `src/osmia/tools.py` — `@tool`, schema inference, and tool execution.
- `src/osmia/client.py` — the agent's small proxy client.
- `src/osmia/discovery.py` — per-agent HTTP server and registry heartbeat.
- `src/osmia/proxy/` — proxy server, config, and provider construction.
- `src/osmia/providers.py` — OpenRouter and Codex adapters.
- `src/osmia/cli.py` — CLI entry points.

## Tests

```bash
uv run pytest
```

Tests use scripted providers and a local fake WebSocket server; they do not require real
credentials or network access.
