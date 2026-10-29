"""JavaScript / Node.js runtime tools.

The agent codes in Python *and* JavaScript: write files, run them under Node,
read the real error, fix it. npm is available for dependencies.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import subprocess
from pathlib import Path

MAX_OUTPUT = 12_000
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}


def node_binary() -> str | None:
    return shutil.which("node")


def npm_binary() -> str | None:
    return shutil.which("npm")


def _node_version(node: str) -> str:
    try:
        return subprocess.run([node, "-v"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    half = MAX_OUTPUT // 2
    return text[:half] + f"\n... [{len(text) - MAX_OUTPUT} chars trimmed] ...\n" + text[-half:]


def pick_extension(code: str) -> str:
    """CommonJS vs ESM matters in Node; pick based on what the code uses."""
    has_require = bool(re.search(r"\brequire\s*\(", code))
    has_module_syntax = bool(re.search(r"^\s*(import|export)\s", code, re.M))
    if has_require and not has_module_syntax:
        return ".cjs"
    if has_module_syntax or re.search(r"^\s*await\s", code, re.M):
        return ".mjs"
    return ".js"


def run_javascript(
    code: str,
    workspace: Path,
    timeout: int = 60,
    filename: str | None = None,
    args: list[str] | None = None,
    cwd: str = ".",
) -> dict:
    node = node_binary()
    if not node:
        return {
            "ok": False,
            "error": "node is not installed. Install Node.js 18+ (https://nodejs.org) and retry.",
        }
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # Run from a real directory, not a throwaway subfolder: Node resolves relative
    # `require`/`import` paths against the script's own location.
    root = workspace.resolve()
    base = (root / cwd).resolve() if cwd and cwd != "." else root
    if root != base and root not in base.parents:
        return {"ok": False, "error": f"cwd '{cwd}' is outside the workspace"}
    base.mkdir(parents=True, exist_ok=True)

    extension = pick_extension(code)
    if filename and Path(filename).suffix in {".js", ".mjs", ".cjs"}:
        extension = Path(filename).suffix
    script = base / f".jarvis_run_{secrets.token_hex(4)}{extension}"
    script.write_text(code, encoding="utf-8")

    command = [node, "--enable-source-maps", str(script), *(args or [])]
    try:
        proc = subprocess.run(
            command,
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=max(1, min(timeout, 300)),
            env={**os.environ, "NODE_NO_WARNINGS": "1", "NO_COLOR": "1"},
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout}s", "runtime": "node"}
    except OSError as exc:
        return {"ok": False, "error": f"could not start node: {exc}", "runtime": "node"}
    finally:
        script.unlink(missing_ok=True)

    result = {
        "ok": proc.returncode == 0,
        "runtime": f"node {_node_version(node)}",
        "exit_code": proc.returncode,
        "stdout": _truncate(proc.stdout or ""),
        "stderr": _truncate(proc.stderr or ""),
    }
    if not result["ok"]:
        # Turn a raw stderr dump into something actionable: parsed frames with the
        # surrounding source lines, plus fix suggestions.
        from .node_diagnostics import diagnostic_report

        report = diagnostic_report(
            [],
            proc.stderr or "",
            root=workspace.resolve(),
            project_dir=Path(workspace),
            include_scratch=True,
            source_text=code,
        )
        if report["errors"] or report["frames"]:
            result["error_summary"] = report["errors"][0] if report["errors"] else None
            result["app_frames"] = report["app_frames"]
            result["suggestions"] = report["suggestions"]
    return result


def _project_dir(workspace: Path, cwd: str) -> Path:
    root = Path(workspace).resolve()
    target = (root / cwd).resolve() if not Path(cwd).is_absolute() else Path(cwd).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"'{cwd}' is outside the workspace")
    target.mkdir(parents=True, exist_ok=True)
    return target


def npm_install(
    workspace: Path,
    packages: list[str] | None = None,
    cwd: str = ".",
    dev: bool = False,
    timeout: int = 300,
) -> dict:
    npm = npm_binary()
    if not npm:
        return {"ok": False, "error": "npm is not installed (it ships with Node.js)"}
    project = _project_dir(workspace, cwd)

    steps: list[dict] = []
    if not (project / "package.json").exists():
        init = subprocess.run(
            [npm, "init", "-y"], cwd=str(project), capture_output=True, text=True, timeout=120
        )
        steps.append({"step": "npm init -y", "ok": init.returncode == 0})

    command = [npm, "install"]
    if packages:
        command += list(packages)
    if dev:
        command.append("--save-dev")
    try:
        proc = subprocess.run(
            command,
            cwd=str(project),
            capture_output=True,
            text=True,
            timeout=max(10, min(timeout, 600)),
            env={"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/tmp"),
                 "npm_config_yes": "true", "NO_COLOR": "1"},
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"npm install timed out after {timeout}s", "steps": steps}

    tree = []
    node_modules = project / "node_modules"
    if node_modules.is_dir():
        tree = sorted(p.name for p in node_modules.iterdir() if p.is_dir() and not p.name.startswith("."))[:40]
    return {
        "ok": proc.returncode == 0,
        "installed": packages or "package.json dependencies",
        "cwd": str(project),
        "steps": steps,
        "installed_modules": tree,
        "stdout": _truncate(proc.stdout or ""),
        "stderr": _truncate(proc.stderr or ""),
    }


def run_tests(
    workspace: Path, cwd: str = ".", command: str | None = None, timeout: int = 300
) -> dict:
    """Run a project's test suite with the right runner, or an explicit command."""
    project = _project_dir(workspace, cwd)
    node, npm = node_binary(), npm_binary()

    if command:
        runner = command
    elif (project / "package.json").exists():
        try:
            import json

            scripts = json.loads((project / "package.json").read_text(encoding="utf-8")).get("scripts") or {}
        except (ValueError, OSError):
            scripts = {}
        if "test" in scripts and npm:
            runner = "npm test --silent"
        elif (project / "node_modules" / ".bin" / "vitest").exists():
            runner = "npx vitest run"
        elif (project / "node_modules" / ".bin" / "jest").exists():
            runner = "npx jest --ci"
        else:
            runner = "node --test"
    elif any(project.glob("test_*.py")) or (project / "tests").is_dir() or (project / "pytest.ini").exists():
        runner = "python -m pytest -q" if shutil.which("pytest") else "python -m unittest discover -s tests -v"
    else:
        return {"ok": False, "error": "no test runner detected; pass an explicit command", "cwd": str(project)}

    if not node and "node" in runner:
        return {"ok": False, "error": "node is not installed", "command": runner}

    try:
        proc = subprocess.run(
            runner, shell=True, cwd=str(project), capture_output=True, text=True,
            timeout=max(10, min(timeout, 600)),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"tests timed out after {timeout}s", "command": runner}
    return {
        "ok": proc.returncode == 0,
        "command": runner,
        "cwd": str(project),
        "exit_code": proc.returncode,
        "stdout": _truncate(proc.stdout or ""),
        "stderr": _truncate(proc.stderr or ""),
    }


def register(registry, config) -> None:
    @registry.tool(
        "run_javascript",
        "Execute JavaScript/Node.js code and return stdout/stderr. Use it to build and verify "
        "JS: it gets the real stack trace, so it can debug its own code. CommonJS and ESM are "
        "auto-detected. Dependencies: run npm_install first.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Full JS source"},
                "timeout": {"type": "integer", "default": 60},
                "filename": {"type": "string", "description": "Only used to pick .js/.mjs/.cjs semantics"},
                "cwd": {"type": "string", "description": "Project folder to run inside, e.g. 'app'", "default": "."},
            },
            "required": ["code"],
        },
    )
    def run_javascript_tool(code: str, timeout: int = 60, filename: str | None = None, cwd: str = "."):
        return run_javascript(code, config.workspace, timeout, filename, None, cwd)

    @registry.tool(
        "npm_install",
        "Install npm packages into a project folder (creates package.json if missing). "
        "Use before running JS that imports third-party modules.",
        {
            "type": "object",
            "properties": {
                "packages": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['express']"},
                "cwd": {"type": "string", "description": "Project folder inside the workspace", "default": "."},
                "dev": {"type": "boolean", "default": False},
                "timeout": {"type": "integer", "default": 300},
            },
        },
        dangerous=True,
    )
    def npm_install_tool(packages: list | None = None, cwd: str = ".", dev: bool = False, timeout: int = 300):
        return npm_install(config.workspace, packages, cwd, dev, timeout)

    @registry.tool(
        "run_tests",
        "Detect and run a project's test suite (npm test, node --test, vitest/jest, pytest or "
        "unittest), or run an explicit test command. This is how you prove code works.",
        {
            "type": "object",
            "properties": {
                "cwd": {"type": "string", "default": "."},
                "command": {"type": "string", "description": "Explicit override, e.g. 'pytest -q'"},
                "timeout": {"type": "integer", "default": 300},
            },
        },
    )
    def run_tests_tool(cwd: str = ".", command: str | None = None, timeout: int = 300):
        return run_tests(config.workspace, cwd, command, timeout)
