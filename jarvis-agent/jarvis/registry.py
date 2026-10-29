"""Tool registry: JSON schemas + safe dispatch with error capture."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from typing import Any, Callable

MAX_TOOL_CHARS = 20_000


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]
    dangerous: bool = False


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, name: str, description: str, parameters: dict, dangerous: bool = False):
        def decorator(func: Callable[..., Any]):
            self._tools[name] = Tool(name, description, parameters, func, dangerous)
            return func

        return decorator

    def tool(self, name: str, description: str, parameters: dict | None = None, dangerous: bool = False):
        """Decorator form: @registry.tool("name", "desc", {schema})."""
        schema = parameters or {"type": "object", "properties": {}}
        return self.add(name, description, schema, dangerous)

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def help_text(self) -> str:
        lines = []
        for name in self.names():
            tool = self._tools[name]
            first = (tool.description or "").strip().splitlines()[0]
            lines.append(f"  {name:<22} {first}")
        return "\n".join(lines)

    def call(self, name: str, arguments: dict | None = None) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool '{name}'. Available: {', '.join(self.names())}"
        args = dict(arguments or {})
        if "__raw__" in args:
            return f"ERROR: could not parse arguments as JSON: {args['__raw__']}"
        try:
            result = tool.func(**args)
        except TypeError as exc:
            return f"ERROR: bad arguments for '{name}': {exc}"
        except Exception as exc:  # noqa: BLE001 - surfaced to the model on purpose
            detail = traceback.format_exc(limit=2) if "--debug" in args else ""
            return f"ERROR: {name} raised {type(exc).__name__}: {exc}\n{detail}".strip()

        if result is None:
            return "ok (no output)"
        if isinstance(result, (dict, list)):
            text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        else:
            text = str(result)
        if len(text) > MAX_TOOL_CHARS:
            text = text[:MAX_TOOL_CHARS] + f"\n... [truncated, {len(text)} chars total]"
        return text
