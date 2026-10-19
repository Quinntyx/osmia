"""Interactive chat with a bash-capable Osmia agent.

Start the proxy first (``osmia serve``), then open a session with ``uv run chat.py``.
"""

import subprocess

import osmia

TIMEOUT_SECONDS = 120.0


def bash(command: str) -> str:
    """Run a bash command and return its combined output.

    The command runs under `bash -lc` from the current working directory.
    stdout and stderr are captured together and returned with the exit code,
    so the model can see both the output and whether the command succeeded.
    """
    process = subprocess.Popen(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output, _ = process.communicate(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate()
        return f"exit code: timeout after {TIMEOUT_SECONDS:g}s\n{output}".rstrip()
    return f"exit code: {process.returncode}\n{output}".rstrip()


osmia.Agent(
    system_prompt=(
        "You are a coding agent named osmia-worker. Use the `bash` tool to inspect "
        "and change the local machine, then answer briefly."
    ),
    tools=[bash],
).run()
