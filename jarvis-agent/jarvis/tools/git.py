"""Native git tools (no shell needed): clone, commit, push, branch, log, diff.

Works alongside the GitHub REST tools. Use REST for reviewing what is on GitHub,
git for real repository workflows (history, branches, merges, local tests).
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

MAX_OUTPUT = 12_000
SECRET = re.compile(r"(https://)[^@/\s]+@(github\.com|gitlab\.com|bitbucket\.org)")


def git_binary() -> str | None:
    return shutil.which("git")


def _sanitize(text: str) -> str:
    return SECRET.sub(r"\1***@\2", text or "")[:MAX_OUTPUT]


def _env(token: str | None) -> dict:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", "/tmp"),
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "echo",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    if token:
        env["GITHUB_TOKEN"] = token
    return env


def _run(args: list[str], cwd: Path, token: str | None, timeout: int = 180) -> dict:
    import subprocess

    git = git_binary()
    if not git:
        return {"ok": False, "error": "git is not installed on this machine"}
    try:
        proc = subprocess.run(
            [git, *args], cwd=str(cwd), capture_output=True, text=True,
            timeout=max(5, min(timeout, 900)), env=_env(token),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"git timed out after {timeout}s", "command": " ".join(args)}
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": _sanitize(proc.stdout),
        "stderr": _sanitize(proc.stderr),
    }


def _repo_dir(workspace: Path, path: str) -> Path:
    root = Path(workspace).resolve()
    target = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"'{path}' is outside the workspace")
    return target


def _is_repo(path: Path) -> bool:
    return (path / ".git").exists()


def _with_token(url: str, token: str | None) -> str:
    if not token or not url.startswith("https://"):
        return url
    for host in ("github.com", "gitlab.com", "bitbucket.org"):
        if f"://{host}" in url and "@" not in url.split("://", 1)[1].split("/", 1)[0]:
            return url.replace(f"://{host}", f"://x-access-token:{token}@{host}", 1)
    return url


def clone(token: str | None, workspace: Path, url: str, path: str | None = None, depth: int = 0) -> dict:
    name = path or re.sub(r"\.git$", "", url.rstrip("/").split("/")[-1])
    target = _repo_dir(workspace, name)
    if target.exists() and any(target.iterdir()):
        return {"ok": False, "error": f"{name} already exists and is not empty"}
    args = ["clone", _with_token(url, token), str(target)]
    if depth:
        args += ["--depth", str(depth)]
    result = _run(args, Path(workspace), token, timeout=300)
    result["path"] = str(target)
    return result


def status(workspace: Path, path: str = ".") -> dict:
    repo = _repo_dir(workspace, path)
    if not _is_repo(repo):
        return {"ok": False, "error": f"{repo} is not a git repository"}
    result = _run(["status", "--short", "--branch"], repo, None)
    return result


def diff(workspace: Path, path: str = ".", staged: bool = False, max_chars: int = 8000) -> dict:
    repo = _repo_dir(workspace, path)
    args = ["diff", "--staged"] if staged else ["diff"]
    result = _run([*args, "--stat", "--color=never"], repo, None)
    full = _run([*args, "--color=never"], repo, None)
    return {
        "ok": result.get("ok", False),
        "stat": result.get("stdout", ""),
        "diff": (full.get("stdout") or "")[:max_chars],
        "truncated": len(full.get("stdout") or "") > max_chars,
    }


def log(workspace: Path, path: str = ".", limit: int = 10) -> dict:
    repo = _repo_dir(workspace, path)
    result = _run(["log", f"-{max(1, min(limit, 50))}", "--pretty=%h | %an | %ar | %s"], repo, None)
    return {"ok": result.get("ok", False), "log": result.get("stdout", ""), "error": result.get("stderr")}


def init(workspace: Path, path: str = ".", branch: str = "main") -> dict:
    repo = _repo_dir(workspace, path)
    repo.mkdir(parents=True, exist_ok=True)
    result = _run(["init", "-b", branch], repo, None)
    return result


def commit(
    workspace: Path, path: str = ".", message: str = "update", add_all: bool = True
) -> dict:
    repo = _repo_dir(workspace, path)
    if not _is_repo(repo):
        return {"ok": False, "error": f"{repo} is not a git repository (run git_init or git_clone)"}
    if not message.strip():
        return {"ok": False, "error": "commit message is required"}
    add = _run(["add", "-A"] if add_all else ["add", "-u"], repo, None)
    if not add["ok"]:
        return add
    staged = _run(["diff", "--cached", "--name-only"], repo, None)
    if not (staged.get("stdout") or "").strip():
        return {"ok": True, "nothing_to_commit": True, "detail": "working tree already clean"}
    identity: list[str] = []
    if not (_run(["config", "user.name"], repo, None).get("stdout") or "").strip():
        identity += ["-c", "user.name=JARVIS Agent"]
    if not (_run(["config", "user.email"], repo, None).get("stdout") or "").strip():
        identity += ["-c", "user.email=jarvis@localhost"]
    result = _run([*identity, "commit", "-m", message], repo, None)
    sha = _run(["rev-parse", "--short", "HEAD"], repo, None)
    result["commit"] = (sha.get("stdout") or "").strip()
    return result


def push(
    token: str | None, workspace: Path, path: str = ".", remote: str = "origin", branch: str | None = None
) -> dict:
    repo = _repo_dir(workspace, path)
    if not _is_repo(repo):
        return {"ok": False, "error": f"{repo} is not a git repository"}
    if token:  # rewrite remote url for auth, then restore it
        current = _run(["remote", "get-url", remote], repo, None)
        url = (current.get("stdout") or "").strip()
        if url:
            authed = _with_token(url, token)
            _run(["remote", "set-url", remote, authed], repo, None)
            try:
                result = _run(["push", remote, branch or "HEAD"], repo, token, timeout=300)
            finally:
                _run(["remote", "set-url", remote, url], repo, None)
            return result
    return _run(["push", remote, branch or "HEAD"], repo, token, timeout=300)


def pull(token: str | None, workspace: Path, path: str = ".", remote: str = "origin") -> dict:
    repo = _repo_dir(workspace, path)
    return _run(["pull", "--ff-only", remote], repo, token, timeout=300)


def checkout(workspace: Path, path: str = ".", branch: str | None = None) -> dict:
    repo = _repo_dir(workspace, path)
    if not branch:
        return _run(["branch", "--show-current"], repo, None)
    existing = _run(["branch", "--list", branch], repo, None)
    if (existing.get("stdout") or "").strip():
        return _run(["checkout", branch], repo, None)
    return _run(["checkout", "-b", branch], repo, None)


def remotes(workspace: Path, path: str = ".") -> dict:
    repo = _repo_dir(workspace, path)
    return _run(["remote", "-v"], repo, None)


def register(registry, config) -> None:
    token = config.github_token
    ws = config.workspace

    @registry.tool(
        "git_clone",
        "Clone a repository into the workspace (token injected automatically for https URLs).",
        {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "https://github.com/owner/repo.git"},
                "path": {"type": "string", "description": "Destination folder name"},
                "depth": {"type": "integer", "description": "0 = full history", "default": 0},
            },
            "required": ["url"],
        },
    )
    def git_clone(url: str, path: str | None = None, depth: int = 0):
        return clone(token, ws, url, path, depth)

    @registry.tool(
        "git_status",
        "Show the working tree status (branch + changed files) of a repo in the workspace.",
        {"type": "object", "properties": {"path": {"type": "string", "default": "."}}},
    )
    def git_status(path: str = "."):
        return status(ws, path)

    @registry.tool(
        "git_diff",
        "Show unstaged or staged changes with a stat summary.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "staged": {"type": "boolean", "default": False},
            },
        },
    )
    def git_diff(path: str = ".", staged: bool = False):
        return diff(ws, path, staged)

    @registry.tool(
        "git_log",
        "Recent commit history of a repo in the workspace.",
        {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}, "limit": {"type": "integer", "default": 10}},
        },
    )
    def git_log(path: str = ".", limit: int = 10):
        return log(ws, path, limit)

    @registry.tool(
        "git_init",
        "Initialise a new git repository in a workspace folder.",
        {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}, "branch": {"type": "string", "default": "main"}},
        },
    )
    def git_init(path: str = ".", branch: str = "main"):
        return init(ws, path, branch)

    @registry.tool(
        "git_commit",
        "Stage everything and commit with a message. Returns the new short SHA.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "message": {"type": "string"},
                "add_all": {"type": "boolean", "default": True},
            },
            "required": ["message"],
        },
        dangerous=True,
    )
    def git_commit(message: str, path: str = ".", add_all: bool = True):
        return commit(ws, path, message, add_all)

    @registry.tool(
        "git_push",
        "Push the current branch to a remote, authenticating with GITHUB_TOKEN.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "remote": {"type": "string", "default": "origin"},
                "branch": {"type": "string", "description": "Defaults to the current HEAD"},
            },
        },
        dangerous=True,
    )
    def git_push(path: str = ".", remote: str = "origin", branch: str | None = None):
        return push(token, ws, path, remote, branch)

    @registry.tool(
        "git_pull",
        "Fast-forward pull from a remote.",
        {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}, "remote": {"type": "string", "default": "origin"}},
        },
        dangerous=True,
    )
    def git_pull(path: str = ".", remote: str = "origin"):
        return pull(token, ws, path, remote)

    @registry.tool(
        "git_branch",
        "Show the current branch, or switch to / create+switch to a branch.",
        {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}, "branch": {"type": "string"}},
        },
    )
    def git_branch(path: str = ".", branch: str | None = None):
        return checkout(ws, path, branch)

    @registry.tool(
        "git_remotes",
        "List configured remotes.",
        {"type": "object", "properties": {"path": {"type": "string", "default": "."}}},
    )
    def git_remotes(path: str = "."):
        return remotes(ws, path)
