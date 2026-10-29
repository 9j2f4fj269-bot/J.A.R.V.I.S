"""Python diagnosis: a traceback is a stack, not a string.

`run_python` hands back raw stderr, which is fine for a one-line error and useless for
a four-frame traceback with a chained cause. This module turns the traceback into
frames with source context, points at the failing line, and adds a suggestion that is
grounded in the actual code (the closest real name, the real signature, the real class
body) instead of generic advice.

Nothing here executes arbitrary analysis: the script runs as it would under run_python,
and everything after that is parsing.
"""

from __future__ import annotations

import ast
import builtins
import difflib
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path

FRAME = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>.+?))?\s*$')
CHAIN_MARKERS = (
    "During handling of the above exception, another exception occurred:",
    "The above exception was the direct cause of the following exception:",
)
EXCEPTION_TAIL = re.compile(r"^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit|Interrupt|Stop))(?::\s*(?P<message>.*))?$")


def _is_app_path(path: str, workspace: Path) -> bool:
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    try:
        resolved.relative_to(workspace)
        return True
    except ValueError:
        pass
    stdlib = {str(Path(sys.prefix).resolve()), str(Path(sys.base_prefix).resolve())}
    return not any(str(resolved).startswith(prefix) for prefix in stdlib)


def parse_traceback(stderr: str, workspace: Path) -> tuple[list[dict], list[str]]:
    """Split a traceback into exception blocks and frames."""
    lines = stderr.splitlines()
    blocks: list[dict] = []
    frames: list[dict] = []
    chain: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        frame = FRAME.match(line)
        if frame:
            entry = {
                "file": frame.group("file"),
                "line": int(frame.group("line")),
                "function": (frame.group("func") or "<module>").strip(),
                "source": None,
                "is_app": _is_app_path(frame.group("file"), workspace),
            }
            lookahead = index + 1
            while lookahead < len(lines) and not lines[lookahead].strip():
                lookahead += 1
            if lookahead < len(lines):
                candidate = lines[lookahead]
                if candidate.startswith("    ") and not FRAME.match(candidate):
                    entry["source"] = candidate.strip()
                    index = lookahead
            frames.append(entry)
            index += 1
            continue

        if any(marker in line for marker in CHAIN_MARKERS):
            if frames:
                blocks.append({"frames": frames, "exception": None})
                frames = []
            index += 1
            continue

        tail = EXCEPTION_TAIL.match(line.strip()) if line and not line.startswith((" ", "\t")) else None
        if tail and (frames or blocks):
            blocks.append(
                {
                    "frames": frames,
                    "exception": {"type": tail.group("type"), "message": (tail.group("message") or "").strip()},
                }
            )
            frames = []
        index += 1

    if frames:
        blocks.append({"frames": frames, "exception": None})

    if not blocks:
        return [], []
    final = blocks[-1]
    for block in blocks[:-1]:
        exception = block.get("exception")
        if exception:
            chain.append(f"{exception['type']}: {exception['message']}")
    return final["frames"], chain


def syntax_error_report(source: str, filename: str) -> dict | None:
    """ast.parse catches SyntaxError without running anything."""
    try:
        ast.parse(source, filename=filename)
    except SyntaxError as exc:
        line = exc.lineno or 1
        offset = exc.offset or 1
        text = (exc.text or "").rstrip("\n")
        suggestions = _syntax_suggestions(exc.msg, text)
        return {
            "error_type": type(exc).__name__,
            "message": exc.msg,
            "line": line,
            "column": offset,
            "text": text,
            "caret": " " * max(0, offset - 1) + "^",
            "suggestions": suggestions,
        }
    return None


def _syntax_suggestions(message: str, text: str) -> list[str]:
    lowered = message.lower()
    hints: list[str] = []
    if "never closed" in lowered or "unexpected end of file" in lowered or "was never closed" in lowered:
        opener = "(" if "(" in text else "[" if "[" in text else "{" if "{" in text else None
        hints.append(
            f"an opening {opener} on this line is never closed; count the brackets on the statement"
            if opener
            else "a bracket opened earlier is never closed; count brackets on the statement"
        )
    if "expected ':'" in lowered:
        hints.append("a compound statement is missing its colon: if/for/while/def/class/try all need one")
    if "unexpected indent" in lowered:
        hints.append("this line is indented more than the block it belongs to; align it with the block above")
    if "expected an indented block" in lowered:
        hints.append("the line after a colon must be indented")
    if "tab" in lowered or "inconsistent use of tabs" in lowered:
        hints.append("tabs and spaces are mixed; convert the file to spaces")
    opens, closes = text.count("(") + text.count("[") + text.count("{"), text.count(")") + text.count("]") + text.count("}")
    if opens != closes:
        hints.append(f"this line has {opens} opening and {closes} closing brackets; balance them before anything else")
    if text.rstrip().endswith((",", "(", "[", "{", "+", "and", "or")):
        hints.append("the statement is incomplete: something must follow the trailing operator")
    if "invalid syntax" in lowered and not hints:
        hints.append("check the line for a missing comma, quote or operator just before the caret")
    return hints


def _collect_identifiers(source: str) -> list[str]:
    names: set[str] = set()
    for match in re.finditer(r"(?:^|\n)\s*(?:async\s+)?def\s+(\w+)", source):
        names.add(match.group(1))
    for match in re.finditer(r"(?:^|\n)\s*class\s+(\w+)", source):
        names.add(match.group(1))
    for match in re.finditer(r"(?:^|\n)\s*(\w+)\s*(?::[^=\n]+)?=", source):
        names.add(match.group(1))
    for match in re.finditer(r"(?:^|\n)\s*(?:from\s+[\w.]+\s+)?import\s+([\w, ]+)", source):
        for piece in match.group(1).split(","):
            candidate = piece.strip().split(" as ")[-1].strip()
            if candidate.isidentifier():
                names.add(candidate)
    for match in re.finditer(r"\bfor\s+(\w+)\s+in\b", source):
        names.add(match.group(1))
    for match in re.finditer(r"(?:^|\n)\s*with\s+.*?\bas\s+(\w+)", source):
        names.add(match.group(1))
    return sorted(names)


def _class_body(source: str, class_name: str) -> list[str]:
    pattern = re.compile(rf"(?:^|\n)(?P<indent>[ \t]*)class\s+{re.escape(class_name)}\b[^\n]*:\n(?P<body>(?:[ \t]+[^\n]*\n?|\n)*)")
    match = pattern.search(source)
    if not match:
        return []
    body = match.group("body") or ""
    return re.findall(r"(?:^|\n)\s*(?:async\s+)?def\s+(\w+)", body)


def _signature(source: str, function_name: str) -> str | None:
    pattern = re.compile(rf"(?:^|\n)\s*(?:async\s+)?def\s+{re.escape(function_name)}\s*\((?P<args>[^)]*)\)")
    match = pattern.search(source)
    if not match:
        return None
    return f"def {function_name}({' '.join(match.group('args').split())})"


def _suggest(error_type: str, message: str, source: str, frame_source: str | None, workspace: Path) -> list[str]:
    """Suggestions anchored in the real file, so they point at real names."""
    hints: list[str] = []
    identifiers = _collect_identifiers(source)

    if error_type == "NameError":
        missing = re.search(r"name '([^']+)' is not defined", message)
        if missing:
            name = missing.group(1)
            candidates = identifiers + [builtin for builtin in dir(builtins) if not builtin.startswith("_")]
            close = difflib.get_close_matches(name, candidates, n=3, cutoff=0.6)
            if close:
                hints.append(f"'{name}' is not defined; the closest name in this file is '{close[0]}'")
            else:
                hints.append(f"'{name}' is never assigned: define it, or import it if it comes from a module")
    elif error_type == "UnboundLocalError":
        name = re.search(r"local variable '([^']+)'", message)
        hints.append(
            f"'{name.group(1) if name else 'the variable'}' is only assigned inside a branch; "
            "set a default before the branch or return early"
        )
    elif error_type == "AttributeError":
        details = re.search(r"'([\w.]+)' object has no attribute '([^']+)'", message)
        if details:
            owner, attribute = details.group(1), details.group(2)
            if owner == "NoneType":
                hints.append(
                    f"something on this line returned None before '.{attribute}': guard it or default to an empty value"
                )
            else:
                members = _class_body(source, owner)
                close = difflib.get_close_matches(attribute, members, n=2, cutoff=0.5) if members else []
                if close:
                    hints.append(f"'{owner}' has no '{attribute}'; did you mean '{close[0]}'?")
                elif members:
                    hints.append(f"'{owner}' defines only {', '.join(members[:6])}; '{attribute}' is not one of them")
                else:
                    hints.append(f"check what '{owner}' actually exposes; the attribute name may be misspelled")
    elif error_type == "TypeError":
        missing = re.search(r"(\w+)\(\) missing \d+ required (?:positional |keyword-only )?argument[s]?: (.+)", message)
        if missing:
            signature = _signature(source, missing.group(1))
            detail = f" (defined as {signature})" if signature else ""
            hints.append(f"{missing.group(1)}() needs {missing.group(2)}{detail}; pass it or give the parameter a default")
        elif "takes" in message and "positional argument" in message:
            hints.append("the call passes too many positional arguments; check the definition's parameter list")
        elif "unsupported operand" in message:
            hints.append(
                "the two operands are different types: convert explicitly (int(), str()) or check the input with an isinstance guard"
            )
        elif "not callable" in message:
            hints.append("that name is a value, not a function: call the method on the object instead")
        elif "string indices must be integers" in message:
            hints.append("you are indexing a string as if it were a dict; check whether the value got parsed as JSON")
    elif error_type == "KeyError":
        key = message.strip().strip("'\"")
        hints.append(f"'{key}' is missing from the dict; use .get('{key}', default) or check membership first")
    elif error_type == "IndexError":
        hints.append("the index is past the end of the sequence; check len() or iterate instead of indexing")
    elif error_type == "ZeroDivisionError":
        hints.append("the divisor is zero; guard it before dividing, or filter empty inputs out earlier")
    elif error_type == "RecursionError":
        hints.append("the recursion has no reachable base case, or the input shrinks too slowly")
    elif error_type in {"ModuleNotFoundError", "ImportError"}:
        module = re.search(r"No module named '([\w.]+)'", message)
        if module:
            top = module.group(1).split(".")[0]
            local = list(workspace.rglob(f"{top}.py"))[:2]
            if local:
                hints.append(
                    f"a local file {local[0].name} shadows the import or is not importable from here; "
                    "check the working directory and package __init__.py"
                )
            else:
                hints.append(f"install it with 'pip install {top}' or remove the import")
    elif error_type == "FileNotFoundError":
        missing = re.search(r"No such file or directory: '([^']+)'", message)
        if missing:
            candidate = missing.group(1)
            nearby = [p.name for p in workspace.rglob(Path(candidate).name)][:3]
            hints.append(
                f"'{candidate}' does not exist relative to the working directory"
                + (f"; did you mean {', '.join(nearby)}?" if nearby else "")
            )
    elif error_type == "UnicodeDecodeError":
        hints.append("open the file with an explicit encoding (utf-8, or errors='replace')")
    elif error_type in {"IndentationError", "TabError"}:
        hints.append("fix the indentation before anything else; the parser never gets past it")
    elif error_type == "AssertionError" and frame_source:
        hints.append(f"the assertion that failed is '{frame_source}'; check the expected value against the real one")
    elif error_type == "StopIteration":
        hints.append("the iterator was already exhausted; create a fresh one or stop calling next() directly")

    if not hints:
        hints.append(f"read the frame above: the failing expression is {frame_source!r}" if frame_source else "read the traceback above for the failing expression")
    return hints


def _context(source_lines: list[str], line: int, span: int) -> list[dict]:
    start = max(1, line - span)
    end = min(len(source_lines), line + span)
    return [
        {"line": number, "text": source_lines[number - 1], "is_error": number == line}
        for number in range(start, end + 1)
    ]


def diagnose(
    workspace: Path,
    code: str | None = None,
    path: str | None = None,
    cwd: str = ".",
    timeout: int = 60,
    span: int = 2,
    args: list[str] | None = None,
    check: bool = False,
) -> dict:
    workspace = Path(workspace).resolve()
    base = (workspace / cwd).resolve() if cwd not in (".", "") else workspace
    created: Path | None = None
    try:
        if path:
            target = (base / path).resolve()
            if not target.is_file():
                return {"ok": False, "error": f"no such file: {path}"}
        elif code is not None:
            target = base / f".jarvis_py_{secrets.token_hex(4)}.py"
            target.write_text(code, encoding="utf-8")
            created = target
        else:
            return {"ok": False, "error": "pass either code or path"}
    except (ValueError, OSError) as exc:
        return {"ok": False, "error": str(exc)}

    source = target.read_text(encoding="utf-8", errors="replace")
    source_lines = source.splitlines()

    try:
        syntax = syntax_error_report(source, str(target))
        if check:
            return {
                "ok": syntax is None,
                "mode": "check",
                "target": str(target),
                "error_type": syntax["error_type"] if syntax else None,
                "message": syntax["message"] if syntax else "no syntax errors",
                "line": syntax["line"] if syntax else None,
                "context": _context(source_lines, syntax["line"], span) if syntax else [],
                "suggestions": syntax["suggestions"] if syntax else [],
            }
        if syntax is not None:
            return {
                "ok": False,
                "mode": "run",
                "target": str(target),
                "error_type": syntax["error_type"],
                "message": syntax["message"],
                "line": syntax["line"],
                "column": syntax["column"],
                "text": syntax["text"],
                "caret": syntax["caret"],
                "frames": [],
                "app_frames": [],
                "context": _context(source_lines, syntax["line"], span),
                "suggestions": syntax["suggestions"],
                "stdout": "",
                "stderr": f'{syntax["text"]}\n{syntax["caret"]}\n{syntax["error_type"]}: {syntax["message"]}',
            }

        env = {**os.environ, "PYTHONPATH": str(base), "PYTHONUNBUFFERED": "1"}
        process = subprocess.run(
            [sys.executable, str(target), *(args or [])],
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=max(5, min(timeout, 300)),
            env=env,
        )
        stdout, stderr, exit_code = process.stdout or "", process.stderr or "", process.returncode
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "mode": "run",
            "target": str(target),
            "error_type": "TimeoutError",
            "message": f"script did not exit within {timeout}s",
            "frames": [],
            "app_frames": [],
            "suggestions": ["the run is hanging: look for an unbounded loop or a blocking call such as input() or a socket read"],
            "stdout": "",
            "stderr": "",
        }
    finally:
        if created is not None:
            created.unlink(missing_ok=True)

    frames, chain = parse_traceback(stderr, base)
    for frame in frames:
        try:
            lines = Path(frame["file"]).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        if 1 <= frame["line"] <= len(lines):
            frame["context"] = _context(lines, frame["line"], min(span, 1))
            if not frame.get("source"):
                frame["source"] = lines[frame["line"] - 1].strip()

    app_frames = [frame for frame in frames if frame.get("is_app")]
    error_type = message = None
    suggestions: list[str] = []
    if frames or chain:
        last = re.findall(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit|Interrupt|Stop))(?::\s*(.*))?$", stderr, re.MULTILINE)
        if last:
            error_type, message = last[-1][0], (last[-1][1] or "").strip()
    if error_type:
        focus = app_frames[-1] if app_frames else (frames[-1] if frames else None)
        suggestions = _suggest(error_type, message or "", source, (focus or {}).get("source"), base)
        if chain and error_type not in {"KeyError", "NameError", "AttributeError", "TypeError"}:
            suggestions.insert(0, f"this is a wrapped error: the underlying cause was {chain[-1]}")

    return {
        "ok": exit_code == 0 and not error_type,
        "mode": "run",
        "target": str(target),
        "exit_code": exit_code,
        "error_type": error_type,
        "message": message,
        "chain": chain,
        "frames": frames,
        "app_frames": app_frames,
        "suggestions": suggestions,
        "stdout": stdout[-4000:],
        "stderr": stderr[-6000:],
    }


def register(registry, config) -> None:
    ws = config.workspace

    @registry.tool(
        "python_diagnose",
        "Run Python and explain the failure like a colleague would: parsed traceback frames with source "
        "context, the exception chain, and suggestions tied to the real name, signature or class in the "
        "file. Use it instead of re-running run_python when an error is not obvious. Pass check=true to "
        "only syntax-check with ast (nothing executes).",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Script to run (relative to cwd)"},
                "code": {"type": "string", "description": "Snippet to run instead of a file"},
                "cwd": {"type": "string", "default": "."},
                "check": {"type": "boolean", "default": False, "description": "Syntax check only, no execution"},
                "timeout": {"type": "integer", "default": 60},
                "span": {"type": "integer", "default": 2, "description": "Context lines around the failing line"},
                "args": {"type": "array", "items": {"type": "string"}, "description": "Arguments for the script"},
            },
        },
    )
    def python_diagnose_tool(
        path: str | None = None,
        code: str | None = None,
        cwd: str = ".",
        check: bool = False,
        timeout: int = 60,
        span: int = 2,
        args: list[str] | None = None,
    ):
        return diagnose(
            Path(ws), code=code, path=path, cwd=cwd, timeout=timeout, span=span, args=args, check=check
        )


UNUSED = difflib  # kept importable for tests that build suggestions directly
