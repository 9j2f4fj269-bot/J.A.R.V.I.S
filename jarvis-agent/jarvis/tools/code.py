"""Code execution tools: run Python snippets and (optionally) shell commands.

Processes run inside the configured workspace with a hard timeout. This is a
local convenience sandbox, not a security boundary: only enable shell when you
trust the agent's instructions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

MAX_OUTPUT = 12_000


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    half = MAX_OUTPUT // 2
    return text[:half] + f"\n... [{len(text) - MAX_OUTPUT} chars trimmed] ...\n" + text[-half:]


def run_python(code: str, workspace: Path, timeout: int = 60) -> dict:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="jarvis-", dir=str(workspace)))
    script = tmpdir / "snippet.py"
    script.write_text(textwrap.dedent(code), encoding="utf-8")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(
            [sys.executable, "-u", str(script)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=max(1, min(timeout, 300)),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout}s", "path": str(script)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": _truncate(proc.stdout or ""),
        "stderr": _truncate(proc.stderr or ""),
    }


def run_shell(command: str, workspace: Path, timeout: int = 60) -> dict:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=max(1, min(timeout, 300)),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout}s"}
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": _truncate(proc.stdout or ""),
        "stderr": _truncate(proc.stderr or ""),
    }


def register(registry, config) -> None:
    @registry.tool(
        "run_python",
        "Execute a Python snippet and return stdout/stderr. This is how you actually verify code "
        "you write: run it, read the error, fix it. Runs in the workspace with a timeout.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Full Python source to execute"},
                "timeout": {"type": "integer", "description": "Seconds (max 300)", "default": 60},
            },
            "required": ["code"],
        },
    )
    def run_python_tool(code: str, timeout: int = 60):
        return run_python(code, config.workspace, timeout)

    if config.allow_shell:

        @registry.tool(
            "run_shell",
            "Run a shell command in the workspace (git, pip, ls, curl, tests). Shell access is "
            "enabled for this install.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "default": 60},
                },
                "required": ["command"],
            },
            dangerous=True,
        )
        def run_shell_tool(command: str, timeout: int = 60):
            return run_shell(command, config.workspace, timeout)
