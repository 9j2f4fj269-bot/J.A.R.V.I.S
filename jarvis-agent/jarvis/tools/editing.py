"""Code editing tools: surgical patches, workspace search, cleanup.

Editing an existing file by replacing an exact snippet is far safer than
rewriting a whole file from memory, so `apply_patch` is the preferred edit path.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

MAX_FILE_BYTES = 2_000_000
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".mypy_cache"}
TEXT_SUFFIXES = {
    ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".json", ".md", ".txt", ".yml", ".yaml",
    ".toml", ".cfg", ".ini", ".html", ".css", ".scss", ".sh", ".sql", ".go", ".rs", ".java", ".rb",
    ".php", ".c", ".h", ".cpp", ".env", ".gitignore", ".svg", ".xml",
}


def _safe_root(workspace: Path, path: str) -> Path:
    root = Path(workspace).expanduser().resolve()
    target = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"'{path}' escapes the workspace")
    return target


def apply_patch(
    workspace: Path,
    path: str,
    find: str,
    replace: str,
    all: bool = False,
    count: int = 1,
) -> dict:
    target = _safe_root(workspace, path)
    if not target.exists():
        raise FileNotFoundError(f"no such file: {path} (use write_file to create it)")
    original = target.read_text(encoding="utf-8")
    occurrences = original.count(find)
    if occurrences == 0:
        return {
            "ok": False,
            "error": "find-string not present; read the file and copy the exact text (whitespace matters)",
            "path": path,
        }
    if occurrences > 1 and not all:
        return {
            "ok": False,
            "error": f"'{find[:60]}' appears {occurrences} times; pass all=true or a longer unique snippet",
            "occurrences": occurrences,
            "path": path,
        }
    limit = -1 if all else max(1, count)
    updated = original.replace(find, replace, limit)
    target.write_text(updated, encoding="utf-8")
    return {
        "ok": True,
        "path": path,
        "replacements": occurrences if all else min(count, occurrences),
        "bytes_before": len(original.encode()),
        "bytes_after": len(updated.encode()),
    }


def grep_files(
    workspace: Path,
    pattern: str,
    path: str = ".",
    glob: str = "*",
    fixed: bool = False,
    ignore_case: bool = False,
    max_results: int = 40,
    context_lines: int = 0,
) -> dict:
    root = _safe_root(workspace, path)
    flags = re.IGNORECASE if ignore_case else 0
    regex = None
    if not fixed:
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return {"ok": False, "error": f"bad regex: {exc}"}

    files = [root] if root.is_file() else sorted(root.rglob(glob))
    matches: list[dict] = []
    scanned = 0
    for file in files:
        if len(matches) >= max_results:
            break
        if not file.is_file() or any(part in SKIP_DIRS for part in file.parts):
            continue
        if file.suffix.lower() not in TEXT_SUFFIXES and file.suffix != "":
            continue
        try:
            if file.stat().st_size > MAX_FILE_BYTES:
                continue
            lines = file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        scanned += 1
        for index, line in enumerate(lines):
            hit = (pattern in line) if fixed else bool(regex.search(line))
            if not hit:
                continue
            entry = {
                "file": str(file.relative_to(Path(workspace).resolve())),
                "line": index + 1,
                "text": line.strip()[:240],
            }
            if context_lines:
                start = max(0, index - context_lines)
                end = min(len(lines), index + context_lines + 1)
                entry["context"] = [l.strip()[:200] for l in lines[start:end]]
            matches.append(entry)
            if len(matches) >= max_results:
                break
    return {"ok": True, "pattern": pattern, "files_scanned": scanned, "matches": matches, "truncated": len(matches) >= max_results}


def delete_path(workspace: Path, path: str, recursive: bool = False) -> dict:
    target = _safe_root(workspace, path)
    if not target.exists():
        return {"ok": False, "error": f"no such path: {path}"}
    if target.is_dir():
        if not recursive:
            return {"ok": False, "error": "directory is not empty-safe: pass recursive=true to remove a tree"}
        shutil.rmtree(target)
        return {"ok": True, "removed": path, "type": "directory"}
    target.unlink()
    return {"ok": True, "removed": path, "type": "file"}


def move_path(workspace: Path, src: str, dest: str) -> dict:
    source = _safe_root(workspace, src)
    target = _safe_root(workspace, dest)
    if not source.exists():
        return {"ok": False, "error": f"no such path: {src}"}
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    return {"ok": True, "from": src, "to": dest}


def register(registry, config) -> None:
    @registry.tool(
        "apply_patch",
        "Edit a file by replacing an exact snippet. Preferred over rewriting whole files. "
        "Read the file first; the find-string must match exactly (indentation included).",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "find": {"type": "string", "description": "Exact text to replace"},
                "replace": {"type": "string", "description": "Replacement text ('' deletes it)"},
                "all": {"type": "boolean", "description": "Replace every occurrence", "default": False},
            },
            "required": ["path", "find", "replace"],
        },
    )
    def apply_patch_tool(path: str, find: str, replace: str, all: bool = False):
        return apply_patch(config.workspace, path, find, replace, all)

    @registry.tool(
        "grep_files",
        "Search file contents in the workspace (code search). Returns file, line number and "
        "matching text. Use this to locate code instead of reading every file.",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex, or literal text when fixed=true"},
                "path": {"type": "string", "default": "."},
                "glob": {"type": "string", "description": "e.g. '*.py' or '*.js'", "default": "*"},
                "fixed": {"type": "boolean", "default": False},
                "ignore_case": {"type": "boolean", "default": False},
                "max_results": {"type": "integer", "default": 40},
                "context_lines": {"type": "integer", "default": 0},
            },
            "required": ["pattern"],
        },
    )
    def grep_files_tool(
        pattern: str,
        path: str = ".",
        glob: str = "*",
        fixed: bool = False,
        ignore_case: bool = False,
        max_results: int = 40,
        context_lines: int = 0,
    ):
        return grep_files(config.workspace, pattern, path, glob, fixed, ignore_case, max_results, context_lines)

    @registry.tool(
        "delete_file",
        "Delete a file (or a directory tree with recursive=true) from the workspace.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean", "default": False},
            },
            "required": ["path"],
        },
        dangerous=True,
    )
    def delete_file(path: str, recursive: bool = False):
        return delete_path(config.workspace, path, recursive)

    @registry.tool(
        "move_file",
        "Move or rename a file/folder inside the workspace.",
        {
            "type": "object",
            "properties": {"src": {"type": "string"}, "dest": {"type": "string"}},
            "required": ["src", "dest"],
        },
    )
    def move_file(src: str, dest: str):
        return move_path(config.workspace, src, dest)
