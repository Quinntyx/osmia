"""Run a tiny Osmia agent with two Python tools."""

import os

import osmia


def read_file(path: str) -> str:
    """Read a UTF-8 text file."""
    with open(path, encoding="utf-8") as handle:
        return handle.read()


agent = osmia.Agent(
    prompt="List the files in the current directory and summarize what this project does.",
    system_prompt=(
        "You are a coding agent. Use the provided tools and keep the answer brief."
    ),
    tools=[os.system, read_file],
)

print(agent.run())
