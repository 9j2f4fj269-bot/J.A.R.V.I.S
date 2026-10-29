"""Code editing tools: surgical patches, workspace search, cleanup.

Editing an existing file by replacing an exact snippet is far safer than
rewriting a whole file from memory, so `apply_patch` is the preferred edit path.
"""

from __future__ import annotations

import difflib
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


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _reindent(replacement_lines: list[str], target_indent: str) -> list[str]:
    """Give an unindented snippet the indentation of the code it replaces."""
    filled = [line for line in replacement_lines if line.strip()]
    if not filled or not target_indent:
        return replacement_lines
    smallest = min(len(_indent(line)) for line in filled)
    if smallest:
        return replacement_lines
    return [target_indent + line if line.strip() else line for line in replacement_lines]


def _window_score(window: list[str], needle: list[str]) -> float:
    """Line-level similarity, upgraded to character-level so a one-word typo still scores high."""
    line_ratio = difflib.SequenceMatcher(None, window, needle).ratio()
    if line_ratio >= 0.99:
        return 1.0
    char_ratio = difflib.SequenceMatcher(None, "\n".join(window), "\n".join(needle)).ratio()
    return max(line_ratio, char_ratio)


def _best_window(haystack: list[str], needle: list[str]) -> tuple[int | None, float]:
    """Closest run of lines to the requested snippet, ignoring surrounding whitespace."""
    if not needle or len(haystack) < len(needle):
        return None, 0.0
    huge = len(haystack) > 4000  # keep a fuzzy scan bounded on very large files
    best_start, best_ratio = None, 0.0
    for start in range(0, len(haystack) - len(needle) + 1):
        window = haystack[start : start + len(needle)]
        if huge:
            ratio = difflib.SequenceMatcher(None, window, needle).ratio()
        else:
            ratio = _window_score(window, needle)
        if ratio > best_ratio:
            best_start, best_ratio = start, ratio
        if best_ratio == 1.0:
            break
    return best_start, best_ratio


def _unified_diff(before: str, after: str, path: str) -> str:
    diff = list(
        difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}", n=2,
        )
    )
    return "".join(diff[:200])


def apply_patch(
    workspace: Path,
    path: str,
    find: str,
    replace: str,
    all: bool = False,
    count: int = 1,
    dry_run: bool = False,
    fuzzy_threshold: float = 0.9,
) -> dict:
    target = _safe_root(workspace, path)
    if not target.exists():
        raise FileNotFoundError(f"no such file: {path} (use write_file to create it)")
    original = target.read_text(encoding="utf-8")
    occurrences = original.count(find)

    if occurrences:
        if occurrences > 1 and not all:
            return {
                "ok": False,
                "error": f"'{find[:60]}' appears {occurrences} times; pass all=true or a longer unique snippet",
                "occurrences": occurrences,
                "path": path,
            }
        limit = -1 if all else max(1, count)
        updated = original.replace(find, replace, limit)
        result = {
            "ok": True,
            "path": path,
            "match": "exact",
            "replacements": occurrences if all else min(count, occurrences),
            "bytes_before": len(original.encode()),
            "bytes_after": len(updated.encode()),
        }
        if dry_run:
            result["diff"] = _unified_diff(original, updated, path)
            result["dry_run"] = True
            return result
        target.write_text(updated, encoding="utf-8")
        return result

    # Nothing matched verbatim: try the same snippet with whitespace ignored, which is
    # the usual reason a correct-looking patch fails (indentation drift, CRLF, trailing spaces).
    haystack_lines = original.splitlines()
    haystack_norm = [line.strip() for line in haystack_lines]
    needle_lines = find.splitlines()
    needle_norm = [line.strip() for line in needle_lines]
    if not needle_lines:
        return {"ok": False, "path": path, "error": "the find-string is empty"}

    windows: list[int] = []
    width = len(needle_lines)
    for start in range(0, max(0, len(haystack_norm) - width + 1)):
        if haystack_norm[start : start + width] == needle_norm:
            windows.append(start)

    match_kind = "whitespace"
    similarity = 1.0
    chosen = windows
    if not chosen:
        best_start, similarity = _best_window(haystack_norm, needle_norm)
        if best_start is not None and similarity >= fuzzy_threshold:
            chosen = [best_start]
            match_kind = "fuzzy"

    if not chosen:
        closest = None
        best_start, similarity = _best_window(haystack_norm, needle_norm)
        if best_start is not None and similarity > 0.4:
            closest = {
                "line_start": best_start + 1,
                "line_end": best_start + width,
                "similarity": round(similarity, 2),
                "text": "\n".join(haystack_lines[best_start : best_start + width])[:600],
            }
        return {
            "ok": False,
            "path": path,
            "error": (
                "find-string not present, even ignoring whitespace"
                + (f"; the closest block is {closest['similarity']:.0%} similar" if closest else "")
            ),
            "closest_match": closest,
            "hint": "read the file again and copy the current text, or use closest_match to correct the snippet",
            "occurrences": 0,
        }

    if len(chosen) > 1 and not all:
        return {
            "ok": False,
            "path": path,
            "error": f"'{find[:60]}' matches {len(chosen)} locations once whitespace is ignored",
            "line_starts": [entry + 1 for entry in chosen],
            "hint": "pass all=true, or include more of the surrounding code to make the snippet unique",
        }

    replacement_lines = replace.splitlines() if replace.strip() else []
    selected = chosen if all else chosen[: max(1, count)]
    updated_lines = list(haystack_lines)
    # Replace from the bottom so earlier indices stay valid.
    for start in sorted(selected, reverse=True):
        block_indent = _indent(haystack_lines[start]) if haystack_lines else ""
        filler = _reindent(replacement_lines, block_indent)
        updated_lines[start : start + width] = filler

    updated = "\n".join(updated_lines)
    if original.endswith("\n"):
        updated += "\n"

    warnings: list[str] = []
    if match_kind == "whitespace":
        warnings.append("matched after ignoring whitespace, so indentation may differ from the snippet")
    if match_kind == "fuzzy":
        warnings.append(f"no exact match; applied to the closest block ({similarity:.0%} similar)")

    result = {
        "ok": True,
        "path": path,
        "match": match_kind,
        "similarity": round(similarity, 2),
        "replacements": len(selected),
        "line_starts": [entry + 1 for entry in selected],
        "bytes_before": len(original.encode()),
        "bytes_after": len(updated.encode()),
        "warnings": warnings,
    }
    if dry_run:
        result["diff"] = _unified_diff(original, updated, path)
        result["dry_run"] = True
        return result
    target.write_text(updated, encoding="utf-8")
    return result


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
        "Read the file first. Exact text is preferred, but whitespace-only differences and near "
        "matches above fuzzy_threshold still apply (with a warning), and a failed patch returns the "
        "closest block so you can correct it. Pass dry_run=true to see the diff before writing.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "find": {"type": "string", "description": "Text to replace"},
                "replace": {"type": "string", "description": "Replacement text ('' deletes it)"},
                "all": {"type": "boolean", "description": "Replace every occurrence", "default": False},
                "dry_run": {"type": "boolean", "description": "Return the diff without writing", "default": False},
                "fuzzy_threshold": {
                    "type": "number",
                    "description": "Similarity needed for a fuzzy match (0.9 default; 1.0 disables it)",
                    "default": 0.9,
                },
            },
            "required": ["path", "find", "replace"],
        },
    )
    def apply_patch_tool(
        path: str,
        find: str,
        replace: str,
        all: bool = False,
        dry_run: bool = False,
        fuzzy_threshold: float = 0.9,
    ):
        return apply_patch(
            config.workspace, path, find, replace, all=all, dry_run=dry_run, fuzzy_threshold=fuzzy_threshold
        )

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
