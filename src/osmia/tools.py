"""Python callables exposed as agent tools."""

from __future__ import annotations

import inspect
import json
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ToolMetadata:
    name: str | None = None
    description: str | None = None
    schema: dict[str, Any] | None = None


class ToolSchemaError(ValueError):
    """Raised when a callable cannot be converted into a valid tool schema."""


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Signature.empty or annotation is Any:
        return {}
    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is dict or annotation is Mapping:
        return {"type": "object"}
    if annotation in (list, tuple, set):
        return {"type": "array"}

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Union or origin is types.UnionType:
        non_null = [item for item in args if item is not type(None)]
        if len(non_null) == 1:
            return _annotation_schema(non_null[0])
        return {}
    if origin in (list, tuple, set):
        if args:
            return {"type": "array", "items": _annotation_schema(args[0])}
        return {"type": "array"}
    if origin is dict or origin is Mapping:
        return {"type": "object"}
    if hasattr(annotation, "model_json_schema") and callable(annotation.model_json_schema):
        return annotation.model_json_schema()
    if isinstance(annotation, type):
        return {"type": "object"}
    return {}


def schema_for_callable(fn: Callable[..., Any]) -> tuple[str, str, dict[str, Any]]:
    """Return ``(name, description, parameters)`` for a Python callable."""
    if not callable(fn):
        raise ToolSchemaError(f"Tool {fn!r} is not callable")
    name = getattr(fn, "__name__", None)
    if not name or name == "<lambda>":
        raise ToolSchemaError(
            "Tools must have a stable name; lambdas need an explicit signature override"
        )
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise ToolSchemaError(f"Cannot inspect tool {name}: {exc}") from exc
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        entry = _annotation_schema(hints.get(parameter.name, parameter.annotation))
        if parameter.default is not inspect.Signature.empty:
            try:
                json.dumps(parameter.default)
                entry["default"] = parameter.default
            except (TypeError, ValueError):
                pass
        else:
            required.append(parameter.name)
        properties[parameter.name] = entry

    description = inspect.getdoc(fn) or f"Call {name}."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = required
    return name, description, parameters


@dataclass
class Tool:
    """A callable plus the JSON schema advertised to the model."""

    name: str
    description: str
    parameters: dict[str, Any]
    callable: Callable[..., Any]
    _positional_only: set[str] = field(default_factory=set, repr=False)

    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., Any],
        metadata: ToolMetadata | None = None,
    ) -> Tool:
        name, description, parameters = schema_for_callable(fn)
        positional_only: set[str] = set()
        try:
            for parameter in inspect.signature(fn).parameters.values():
                if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                    positional_only.add(parameter.name)
        except (TypeError, ValueError):
            pass
        if metadata:
            name = metadata.name or name
            description = metadata.description or description
            if metadata.schema is not None:
                parameters = json.loads(json.dumps(metadata.schema))
        return cls(
            name=name,
            description=description,
            parameters=parameters,
            callable=fn,
            _positional_only=positional_only,
        )

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def invoke(self, arguments: str | dict[str, Any] | None) -> str:
        """Invoke the tool and serialize the result as text."""
        if arguments is None or arguments == "":
            kwargs: dict[str, Any] = {}
        elif isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Arguments for {self.name} are not valid JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"Arguments for {self.name} must be a JSON object")
            kwargs = parsed
        elif isinstance(arguments, dict):
            kwargs = arguments
        else:
            raise ValueError(f"Arguments for {self.name} must be JSON text or a dict")

        if self._positional_only:
            ordered = [kwargs.pop(key) for key in self._positional_only if key in kwargs]
            result = self.callable(*ordered, **kwargs)
        else:
            result = self.callable(**kwargs)
        return serialize_tool_result(result)


def serialize_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if result is None:
        return "null"
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: dict[str, Any] | None = None,
) -> Any:
    """Decorator that attaches tool metadata to a callable."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        function.__osmia_tool__ = ToolMetadata(
            name=name, description=description, schema=schema
        )
        return function

    if fn is not None:
        return decorate(fn)
    return decorate


def as_tool(value: Callable[..., Any] | Tool) -> Tool:
    if isinstance(value, Tool):
        return value
    metadata = getattr(value, "__osmia_tool__", None)
    return Tool.from_callable(value, metadata)
