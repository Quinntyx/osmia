from __future__ import annotations

import json
from typing import Any

import pytest

from osmia.tools import ToolSchemaError, as_tool, schema_for_callable, tool


def test_schema_docstring_and_annotations() -> None:
    def add(a: int, b: int = 2) -> int:
        """Add two integers."""
        return a + b

    name, description, schema = schema_for_callable(add)
    assert name == "add"
    assert description == "Add two integers."
    assert schema["properties"]["a"] == {"type": "integer"}
    assert schema["properties"]["b"] == {"type": "integer", "default": 2}
    assert schema["required"] == ["a"]
    assert schema["additionalProperties"] is False


def test_optional_union_annotation() -> None:
    def maybe(value: str | None = None) -> str:
        return value or ""

    _, _, schema = schema_for_callable(maybe)
    assert schema["properties"]["value"] == {"type": "string", "default": None}
    assert schema.get("required", []) == []


def test_decorator_precedence() -> None:
    @tool(name="renamed", description="Override", schema={"type": "object", "properties": {}})
    def original(x: int) -> int:
        """Original doc."""
        return x

    built = as_tool(original)
    assert built.name == "renamed"
    assert built.description == "Override"
    assert built.parameters == {"type": "object", "properties": {}}


def test_positional_only_invoke() -> None:
    def divide(a: int, /, b: int) -> float:
        return a / b

    built = as_tool(divide)
    assert built.invoke({"a": 10, "b": 4}) == "2.5"


def test_bad_arguments_and_lambda() -> None:
    def ok(value: int) -> int:
        return value

    with pytest.raises(ValueError):
        as_tool(ok).invoke("{not json")

    with pytest.raises(ToolSchemaError):
        as_tool(lambda: 1)


def test_result_serialization() -> None:
    def data() -> dict[str, Any]:
        return {"x": 1}

    assert json.loads(as_tool(data).invoke({})) == {"x": 1}
