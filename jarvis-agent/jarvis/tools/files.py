"""Workspace file tools. Every path is confined to the configured workspace."""

from __future__ import annotations

from pathlib import Path


class PathError(RuntimeError):
    pass


def _safe(workspace: Path, path: str) -> Path:
    root = Path(workspace).expanduser().resolve()
    target = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if root != target and root not in target.parents:
        raise PathError(f"path '{path}' escapes the workspace ({root})")
    return target


def write(workspace: Path, path: str, content: str, append: bool = False) -> dict:
    target = _safe(workspace, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(target, mode, encoding="utf-8") as handle:
        handle.write(content)
    return {"path": str(target), "bytes": len(content.encode("utf-8")), "appended": append}


def read(workspace: Path, path: str, max_chars: int = 20000, start_line: int = 1, end_line: int = 0) -> dict:
    target = _safe(workspace, path)
    if not target.exists():
        raise PathError(f"no such file: {target}")
    if target.is_dir():
        raise PathError(f"{target} is a directory; use list_files")
    text = target.read_text(encoding="utf-8", errors="replace")
    if end_line and end_line >= start_line:
        lines = text.splitlines()
        text = "\n".join(lines[start_line - 1 : end_line])
    return {
        "path": str(target),
        "chars": len(text),
        "content": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def list_files(workspace: Path, path: str = ".", pattern: str = "*", recursive: bool = False) -> dict:
    target = _safe(workspace, path)
    if not target.exists():
        raise PathError(f"no such path: {target}")
    if target.is_file():
        return {"path": str(target), "entries": [{"name": target.name, "size": target.stat().st_size}]}
    globber = target.rglob if recursive else target.glob
    entries = []
    for item in sorted(globber(pattern)):
        if any(part in {".git", "__pycache__", ".venv", "node_modules"} for part in item.parts):
            continue
        entries.append(
            {
                "name": item.name,
                "path": str(item.relative_to(Path(workspace).resolve())),
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
            }
        )
    return {"root": str(target), "count": len(entries), "entries": entries[:300]}


def register(registry, config) -> None:
    @registry.tool(
        "write_file",
        "Write (or append) a file in your workspace. Use this to build projects, then run them.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative path"},
                "content": {"type": "string"},
                "append": {"type": "boolean", "default": False},
            },
            "required": ["path", "content"],
        },
    )
    def write_file(path: str, content: str, append: bool = False):
        return write(config.workspace, path, content, append)

    @registry.tool(
        "read_file",
        "Read a file from the workspace, optionally a line range.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "default": 1},
                "end_line": {"type": "integer", "description": "0 = end of file", "default": 0},
            },
            "required": ["path"],
        },
    )
    def read_file(path: str, start_line: int = 1, end_line: int = 0):
        return read(config.workspace, path, start_line=start_line, end_line=end_line)

    @registry.tool(
        "list_files",
        "List files or directories in the workspace. Use recursive with a glob to map a project.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "pattern": {"type": "string", "default": "*"},
                "recursive": {"type": "boolean", "default": False},
            },
        },
    )
    def list_files(path: str = ".", pattern: str = "*", recursive: bool = False):
        return globals()["list_files"](config.workspace, path, pattern, recursive)
