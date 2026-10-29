"""Pure helpers for Node.js error analysis.

No subprocesses here, just parsing: stack frames, source context, and plain-English
explanations for the errors Node actually throws. `node_debug` and the JavaScript
runner both use these.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

FRAME_RE = re.compile(
    r"^\s*at\s+(?:(?P<fn>.+?)\s+\()?"
    r"(?P<loc>(?:file://)?[^\s()]+?):(?P<line>\d+):(?P<col>\d+)\)?\s*$"
)
ERROR_LINE_RE = re.compile(r"^(?P<name>[A-Za-z_$][\w.$]*):\s?(?P<message>.*)$")
INTERNAL_MARKERS = ("node:internal", "node:fs", "internal/", "<anonymous>", "node:events")
MAX_SOURCE_SPAN = 3


def _clean_path(location: str) -> str:
    return location[7:] if location.startswith("file://") else location


def _relative(path: Path, root: Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def parse_stack(
    stack: str | None, root: Path | None = None, limit: int = 30, include_scratch: bool = False
) -> list[dict]:
    """Turn a V8 stack string into frames, marking which ones are the user's code."""
    if not stack:
        return []
    frames: list[dict] = []
    for line in stack.splitlines():
        match = FRAME_RE.match(line)
        if not match:
            continue
        location = _clean_path(match.group("loc"))
        if any(marker in location for marker in INTERNAL_MARKERS):
            continue
        basename = Path(location).name
        if basename.startswith((".jarvis_diag", ".jarvis_check", ".jarvis_prof")):
            continue  # our own harness, never the user's code
        if not include_scratch and basename.startswith(".jarvis_"):
            continue
        line_no = int(match.group("line"))
        col_no = int(match.group("col"))
        path = Path(location)
        absolute = path if path.is_absolute() else Path.cwd() / path
        in_app = bool(root) and (root == absolute.parent or root in absolute.parents)
        frames.append(
            {
                "function": (match.group("fn") or "<anonymous>").strip(),
                "file": _relative(absolute, root) if root else location,
                "absolute": str(absolute),
                "line": line_no,
                "column": col_no,
                "in_app": in_app,
            }
        )
        if len(frames) >= limit:
            break
    return frames


def source_context(file: str | Path, line: int, span: int = 2) -> dict | None:
    """Numbered source around `line`, plus a pointer line for the exact column."""
    path = Path(file)
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    if not lines:
        return None
    span = max(0, min(span, MAX_SOURCE_SPAN))
    index = max(1, min(line, len(lines))) - 1
    start = max(0, index - span)
    end = min(len(lines), index + span + 1)
    numbered = [f"{n + 1:>4} | {lines[n]}" for n in range(start, end)]
    return {
        "file": str(path),
        "line": index + 1,
        "text": lines[index],
        "snippet": "\n".join(numbered),
    }


def error_header(stderr: str) -> dict | None:
    """Last `Name: message` line before a stack, which is the real error."""
    for line in reversed((stderr or "").splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith("at "):
            continue
        match = ERROR_LINE_RE.match(stripped)
        if match and ("Error" in match.group("name") or match.group("name").endswith("Exception")):
            return {"name": match.group("name"), "message": match.group("message").strip()}
    return None


def _package_deps(project_dir: Path | None) -> dict[str, str]:
    if not project_dir:
        return {}
    manifest = project_dir / "package.json"
    if not manifest.is_file():
        return {}
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    deps: dict[str, str] = {}
    for key in ("dependencies", "devDependencies"):
        for name, version in (data.get(key) or {}).items():
            deps[name] = f"{key}: {version}"
    return deps


def _missing_module(message: str) -> str | None:
    for pattern in (r"Cannot find module '([^']+)'", r"Cannot find package '([^']+)'"):
        match = re.search(pattern, message or "")
        if match:
            return match.group(1)
    return None


def explain(errors: list[dict], frames: list[dict], project_dir: Path | None = None) -> list[str]:
    """Rule-based advice for the errors Node throws most often."""
    suggestions: list[str] = []
    deps = _package_deps(project_dir)

    for error in errors or []:
        name = (error.get("name") or "").strip()
        message = (error.get("message") or "").strip()
        code = (error.get("code") or "").strip()
        combined = f"{code} {name} {message}"
        app_frame = next((f for f in frames if f.get("in_app")), None)
        if app_frame:
            label = "<snippet>" if Path(app_frame["file"]).name.startswith(".jarvis_") else app_frame["file"]
            where = f" at {label}:{app_frame['line']}"
        else:
            where = ""

        missing = _missing_module(message)
        imported_from = re.search(r"imported from (?:file://)?(\S+)", message or "")
        if missing or "MODULE_NOT_FOUND" in combined or "ERR_MODULE_NOT_FOUND" in combined:
            if missing and missing.startswith((".", "/")) or "ERR_MODULE_NOT_FOUND" in code:
                shown = Path(missing).name if missing else "that path"
                origin = ""
                if imported_from:
                    origin_path = Path(imported_from.group(1))
                    origin = f" (imported from {origin_path.name})"
                suffix = "" if missing and Path(missing).suffix else " Add the file extension: ESM requires './mod.js', not './mod'."
                suggestions.append(
                    f"The import path {shown!r} is wrong{origin}.{suffix}"
                )
            elif missing:
                installed = missing.split("/")[0]
                if installed in deps:
                    suggestions.append(
                        f"'{installed}' is declared in package.json but not installed: run "
                        f"npm_install(cwd=...) so node_modules exists."
                    )
                else:
                    suggestions.append(
                        f"'{installed}' is not installed. Install it with npm_install(packages=['{installed}']) "
                        f"before requiring it."
                    )
            continue

        if "require is not defined in ES module scope" in message or "require is not defined" in message:
            suggestions.append(
                "This file is being treated as ESM, where `require` doesn't exist. Either rename it .cjs, "
                "or switch to `import x from 'x'` / `createRequire(import.meta.url)`."
            )
            continue

        if "Cannot use import statement outside a module" in message:
            suggestions.append(
                "Node is parsing ESM syntax as CommonJS. Rename the file .mjs, or add "
                "\"type\": \"module\" to package.json."
            )
            continue

        if "ERR_REQUIRE_ESM" in combined or "ERR_REQUIRE_ASYNC_MODULE" in combined:
            suggestions.append(
                "That package is ESM-only. Use dynamic `await import('pkg')` from CommonJS, or move the "
                "file to ESM."
            )
            continue

        if "is not a function" in message or "is not a constructor" in message:
            suggestions.append(
                f"A value is used as a function{where}. Check the export shape: default vs named export "
                "(`const x = require('pkg')` vs `const {{ x }} = require('pkg')`), and that the symbol "
                "isn't undefined."
            )
            continue

        if "Cannot read properties of undefined" in message or "Cannot read properties of null" in message:
            property_name = re.search(r"reading '([^']+)'", message)
            target = property_name.group(1) if property_name else "a property"
            suggestions.append(
                f"Something upstream returned undefined/null before '{target}' was read{where}. Log or "
                "guard that value (optional chaining `?.`), and check the await/return path that "
                "produced it."
            )
            continue

        if name == "ReferenceError" or "is not defined" in message:
            symbol = re.search(r"(\w+) is not defined", message)
            label = symbol.group(1) if symbol else "the symbol"
            suggestions.append(
                f"'{label}' is used before it exists{where}: typo, missing declaration, or a module "
                "import that returned undefined. Grep for it to confirm."
            )
            continue

        if "SyntaxError" in combined:
            suggestions.append(
                "Syntax error, so the file never ran. Run node_check on the file for the exact "
                "line/column before executing it again."
            )
            continue

        if "EADDRINUSE" in combined:
            suggestions.append(
                "Port already taken. Stop the other process or read the port from "
                "process.env.PORT with a fallback."
            )
            continue

        if "ERR_INVALID_ARG_TYPE" in combined or "ERR_INVALID_ARG_VALUE" in combined:
            suggestions.append(
                f"An argument has the wrong type{where}. Check the call signature and whether a "
                "string/number is arriving as undefined."
            )
            continue

        if name == "RangeError" and "call stack" in message:
            suggestions.append(
                "Infinite recursion. Check the recursion base case and any listener that re-triggers itself."
            )
            continue

        if "Maximum call stack" in message:
            suggestions.append("Infinite recursion or a cyclic listener; add a base case or a guard flag.")
            continue

        if "ERR_ASSERTION" in combined or error.get("phase") == "unhandledRejection":
            if error.get("phase") == "unhandledRejection":
                suggestions.append(
                    "An async function rejected and nothing caught it. Add `.catch()` or `try/await` "
                    "around the call, and check the awaited promise actually resolves to a value."
                )
            continue

        if name == "TimeoutError" or "timed out" in message:
            suggestions.append(
                "The process didn't exit. Open handles (servers, timers, sockets) keep Node alive; "
                "close them, call process.exit, or pass a shorter timeout."
            )
            continue

    if not suggestions and errors:
        suggestions.append(
            "No known pattern matched. Read the top user frame and the source lines above; reproduce "
            "with node_diagnose on a smaller snippet to isolate it."
        )
    return suggestions


def diagnostic_report(
    errors: list[dict],
    stderr: str,
    root: Path | None,
    project_dir: Path | None = None,
    span: int = 2,
    include_scratch: bool = False,
    source_text: str | None = None,
) -> dict:
    """Assemble the full ticket: errors, frames, source context, suggestions."""
    frames: list[dict] = []
    for error in errors or []:
        frames.extend(parse_stack(error.get("stack"), root, include_scratch=include_scratch))
    header = error_header(stderr)
    if header and not errors:
        errors = [{**header, "stack": stderr, "phase": "stderr"}]
        frames.extend(parse_stack(stderr, root, include_scratch=include_scratch))

    seen = set()
    unique: list[dict] = []
    for frame in frames:
        key = (frame["file"], frame["line"], frame["column"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(frame)

    code_lines = source_text.splitlines() if source_text else []
    for frame in unique[:5]:
        if not frame.get("in_app"):
            continue
        context = source_context(frame["absolute"], frame["line"], span)
        if context:
            frame["source"] = context["snippet"]
            frame["source_line"] = context["text"]
        elif code_lines and Path(frame["file"]).name.startswith(".jarvis_"):
            line_no = frame["line"]
            start = max(0, line_no - 1 - span)
            end = min(len(code_lines), line_no + span)
            frame["source"] = "\n".join(f"{n + 1:>4} | {code_lines[n]}" for n in range(start, end))
            frame["source_line"] = code_lines[line_no - 1] if 0 < line_no <= len(code_lines) else ""

    return {
        "errors": errors,
        "frames": unique[:15],
        "app_frames": [f for f in unique if f.get("in_app")][:5],
        "suggestions": explain(errors, unique, project_dir),
    }
