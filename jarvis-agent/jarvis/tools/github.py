"""GitHub tools over the REST API.

Read tools work anonymously (60 req/h). Anything that writes needs GITHUB_TOKEN
with `repo` scope. The commit tool creates real atomic multi-file commits.
"""

from __future__ import annotations

import base64
import binascii
import os

import requests

from ..http_retry import request_with_retry

API = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")


class GitHubError(RuntimeError):
    pass


def _headers(token: str | None) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "jarvis-agent",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _api(method: str, path: str, token: str | None, retries: int = 3, **kwargs):
    extra_headers = kwargs.pop("headers", None) or {}
    kwargs.setdefault("timeout", 30)
    headers = {**_headers(token), **extra_headers}
    resp, _attempts = request_with_retry(
        method, f"{API}{path}", attempts=retries, headers=headers, **kwargs
    )
    if resp.status_code == 404:
        raise GitHubError(f"404 not found: {path}")
    if resp.status_code >= 400:
        message = ""
        try:
            message = resp.json().get("message", "")
        except ValueError:
            message = resp.text[:200]
        remaining = resp.headers.get("x-ratelimit-remaining")
        reset = resp.headers.get("x-ratelimit-reset")
        if resp.status_code in (401, 403) and not token:
            message += " (no GITHUB_TOKEN set: this endpoint needs authentication)"
        elif resp.status_code == 403 and "rate limit" in message.lower():
            message += f" (limit hit; remaining={remaining}; set GITHUB_TOKEN for 5000 req/h)"
        elif resp.status_code == 403:
            message += f" (forbidden; rate limit remaining={remaining})"
        if str(remaining) == "0" and reset:
            message += f" [resets at epoch {reset}]"
        raise GitHubError(f"HTTP {resp.status_code}: {message}")
    if resp.status_code == 204 or not resp.content:
        return {}
    if "json" in resp.headers.get("content-type", "").lower():
        try:
            return resp.json()
        except ValueError:
            return resp.text
    return resp.text


def _require_token(token: str | None, action: str) -> None:
    if not token:
        raise GitHubError(f"{action} needs a GITHUB_TOKEN (repo scope). Set it in .env.")


# --- read ------------------------------------------------------------
def search(token: str | None, query: str, kind: str = "repos", count: int = 8) -> dict:
    kind = (kind or "repos").lower()
    endpoint = {"repos": "repositories", "repositories": "repositories", "code": "code", "issues": "issues"}.get(kind)
    if not endpoint:
        raise GitHubError("kind must be one of: repos, code, issues")
    data = _api("GET", f"/search/{endpoint}", token, params={"q": query, "per_page": max(1, min(count, 30))})
    items = []
    for item in data.get("items", []):
        if endpoint == "repositories":
            items.append(
                {
                    "full_name": item.get("full_name"),
                    "url": item.get("html_url"),
                    "description": (item.get("description") or "")[:200],
                    "stars": item.get("stargazers_count"),
                    "language": item.get("language"),
                    "updated": item.get("updated_at"),
                }
            )
        elif endpoint == "code":
            items.append(
                {
                    "repo": (item.get("repository") or {}).get("full_name"),
                    "path": item.get("path"),
                    "url": item.get("html_url"),
                }
            )
        else:
            items.append(
                {
                    "repo": "/".join((item.get("repository_url") or "").split("/")[-2:]),
                    "number": item.get("number"),
                    "title": item.get("title"),
                    "state": item.get("state"),
                    "url": item.get("html_url"),
                }
            )
    return {"query": query, "kind": endpoint, "total": data.get("total_count"), "results": items}


def repo(token: str | None, repo: str) -> dict:
    data = _api("GET", f"/repos/{repo}", token)
    out = {
        key: data.get(key)
        for key in (
            "full_name",
            "description",
            "html_url",
            "default_branch",
            "language",
            "stargazers_count",
            "open_issues_count",
            "private",
            "pushed_at",
            "topics",
        )
    }
    try:
        readme = _api("GET", f"/repos/{repo}/readme", token, headers={"Accept": "application/vnd.github.raw"})
        if isinstance(readme, str):
            out["readme_excerpt"] = readme[:3000]
    except GitHubError:
        pass
    return out


def list_files(token: str | None, repo: str, path: str = "", ref: str | None = None, recursive: bool = False) -> dict:
    if recursive:
        branch = ref or _api("GET", f"/repos/{repo}", token).get("default_branch", "main")
        tree = _api("GET", f"/repos/{repo}/git/trees/{branch}", token, params={"recursive": "1"})
        entries = [
            {"path": e["path"], "type": e["type"], "size": e.get("size")}
            for e in tree.get("tree", [])
            if e.get("type") == "blob"
        ]
        return {"repo": repo, "ref": branch, "count": len(entries), "files": entries[:500]}
    params = {"ref": ref} if ref else {}
    data = _api("GET", f"/repos/{repo}/contents/{path.lstrip('/')}", token, params=params)
    if isinstance(data, dict):
        return {"repo": repo, "path": path, "type": data.get("type"), "size": data.get("size")}
    return {
        "repo": repo,
        "path": path,
        "entries": [{"name": e["name"], "path": e["path"], "type": e["type"], "size": e.get("size")} for e in data],
    }


def read_file(token: str | None, repo: str, path: str, ref: str | None = None, max_chars: int = 20000) -> dict:
    params = {"ref": ref} if ref else {}
    data = _api("GET", f"/repos/{repo}/contents/{path.lstrip('/')}", token, params=params)
    if isinstance(data, list):
        raise GitHubError(f"{path} is a directory; use github_list_files")
    if data.get("encoding") != "base64":
        raise GitHubError(f"unsupported encoding: {data.get('encoding')}")
    raw = base64.b64decode(data.get("content", "") + "=" * (-len(data.get("content", "")) % 4))
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = f"<binary file, {len(raw)} bytes>"
    return {
        "repo": repo,
        "path": path,
        "ref": ref or "default",
        "sha": data.get("sha"),
        "chars": len(text),
        "content": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def issues(token: str | None, repo: str, state: str = "open", count: int = 10) -> dict:
    data = _api(
        "GET",
        f"/repos/{repo}/issues",
        token,
        params={"state": state, "per_page": max(1, min(count, 50))},
    )
    return {
        "repo": repo,
        "state": state,
        "issues": [
            {
                "number": i.get("number"),
                "title": i.get("title"),
                "state": i.get("state"),
                "user": (i.get("user") or {}).get("login"),
                "labels": [l.get("name") for l in i.get("labels", [])],
                "is_pr": "pull_request" in i,
            }
            for i in data
        ],
    }


def branches(token: str | None, repo: str) -> dict:
    data = _api("GET", f"/repos/{repo}/branches", token, params={"per_page": 50})
    return {"repo": repo, "branches": [{"name": b.get("name"), "sha": (b.get("commit") or {}).get("sha")} for b in data]}


# --- write -----------------------------------------------------------
def write_file(
    token: str | None, repo: str, path: str, content: str, message: str, branch: str | None = None
) -> dict:
    _require_token(token, "github_write_file")
    path = path.lstrip("/")
    payload: dict = {"message": message, "content": base64.b64encode(content.encode("utf-8")).decode()}
    if branch:
        payload["branch"] = branch
    try:
        existing = _api("GET", f"/repos/{repo}/contents/{path}", token, params={"ref": branch} if branch else {})
        if isinstance(existing, dict) and existing.get("sha"):
            payload["sha"] = existing["sha"]
    except GitHubError:
        pass  # new file
    data = _api("PUT", f"/repos/{repo}/contents/{path}", token, json=payload)
    return {
        "repo": repo,
        "path": path,
        "commit": (data.get("commit") or {}).get("sha"),
        "url": (data.get("content") or {}).get("html_url"),
    }


def commit_files(
    token: str | None,
    repo: str,
    files: dict,
    message: str,
    branch: str | None = None,
    base_branch: str | None = None,
) -> dict:
    """Atomically commit many files to a branch (creates the branch if missing)."""
    _require_token(token, "github_commit_files")
    if not isinstance(files, dict) or not files:
        raise GitHubError("files must be an object of {path: content}")

    info = _api("GET", f"/repos/{repo}", token)
    base = base_branch or info.get("default_branch") or "main"
    target = branch or base

    base_ref = _api("GET", f"/repos/{repo}/git/ref/heads/{base}", token)
    base_sha = base_ref["object"]["sha"]

    created_branch = False
    if target != base:
        try:
            _api("GET", f"/repos/{repo}/git/ref/heads/{target}", token)
        except GitHubError:
            _api("POST", f"/repos/{repo}/git/refs", token, json={"ref": f"refs/heads/{target}", "sha": base_sha})
            created_branch = True

    tree = []
    for path, content in files.items():
        blob = _api(
            "POST",
            f"/repos/{repo}/git/blobs",
            token,
            json={"content": content, "encoding": "utf-8"},
        )
        tree.append({"path": path.lstrip("/"), "mode": "100644", "type": "blob", "sha": blob["sha"]})

    new_tree = _api("POST", f"/repos/{repo}/git/trees", token, json={"base_tree": base_sha, "tree": tree})
    commit = _api(
        "POST",
        f"/repos/{repo}/git/commits",
        token,
        json={"message": message, "tree": new_tree["sha"], "parents": [base_sha]},
    )
    _api("PATCH", f"/repos/{repo}/git/refs/heads/{target}", token, json={"sha": commit["sha"], "force": False})
    return {
        "repo": repo,
        "branch": target,
        "branch_created": created_branch,
        "commit": commit["sha"],
        "files": sorted(files),
        "url": f"https://github.com/{repo}/commit/{commit['sha']}",
    }


def create_issue(token: str | None, repo: str, title: str, body: str = "", labels: list | None = None) -> dict:
    _require_token(token, "github_create_issue")
    data = _api("POST", f"/repos/{repo}/issues", token, json={"title": title, "body": body, "labels": labels or []})
    return {"number": data.get("number"), "url": data.get("html_url"), "title": data.get("title")}


def create_pr(
    token: str | None, repo: str, title: str, head: str, base: str | None = None, body: str = "", draft: bool = False
) -> dict:
    _require_token(token, "github_create_pr")
    base = base or _api("GET", f"/repos/{repo}", token).get("default_branch", "main")
    data = _api(
        "POST",
        f"/repos/{repo}/pulls",
        token,
        json={"title": title, "head": head, "base": base, "body": body, "draft": draft},
    )
    return {"number": data.get("number"), "url": data.get("html_url"), "state": data.get("state")}


def whoami(token: str | None) -> dict:
    _require_token(token, "github_whoami")
    data = _api("GET", "/user", token)
    return {"login": data.get("login"), "name": data.get("name"), "public_repos": data.get("public_repos")}


def register(registry, config) -> None:
    token = config.github_token

    @registry.tool(
        "github_search",
        "Search GitHub repositories, code, or issues/PRs. Code search needs a token.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "GitHub search query, e.g. 'language:python agent'"},
                "kind": {"type": "string", "enum": ["repos", "code", "issues"], "default": "repos"},
                "count": {"type": "integer", "default": 8},
            },
            "required": ["query"],
        },
    )
    def github_search(query: str, kind: str = "repos", count: int = 8):
        return search(token, query, kind, count)

    @registry.tool(
        "github_repo",
        "Repository metadata plus a README excerpt. Use 'owner/name'.",
        {
            "type": "object",
            "properties": {"repo": {"type": "string", "description": "owner/name"}},
            "required": ["repo"],
        },
    )
    def github_repo(repo: str):
        return globals()["repo"](token, repo)

    @registry.tool(
        "github_list_files",
        "List files in a repository directory, or the whole tree with recursive=true.",
        {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "path": {"type": "string", "default": ""},
                "ref": {"type": "string", "description": "Branch, tag or commit"},
                "recursive": {"type": "boolean", "default": False},
            },
            "required": ["repo"],
        },
    )
    def github_list_files(repo: str, path: str = "", ref: str | None = None, recursive: bool = False):
        return list_files(token, repo, path, ref, recursive)

    @registry.tool(
        "github_read_file",
        "Read a file's text from a GitHub repository.",
        {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "path": {"type": "string"},
                "ref": {"type": "string"},
            },
            "required": ["repo", "path"],
        },
    )
    def github_read_file(repo: str, path: str, ref: str | None = None):
        return read_file(token, repo, path, ref)

    @registry.tool(
        "github_write_file",
        "Create or update one file in a repository (real commit). Needs GITHUB_TOKEN.",
        {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "path": {"type": "string"},
                "content": {"type": "string"},
                "message": {"type": "string", "description": "Commit message"},
                "branch": {"type": "string"},
            },
            "required": ["repo", "path", "content", "message"],
        },
    )
    def github_write_file(repo: str, path: str, content: str, message: str, branch: str | None = None):
        return write_file(token, repo, path, content, message, branch)

    @registry.tool(
        "github_commit_files",
        "Commit several files to GitHub in one atomic commit, optionally on a new branch. "
        "Best tool for shipping a whole project: pass {path: content} for every file.",
        {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/name"},
                "files": {"type": "object", "description": "Map of repo-relative path -> full file content"},
                "message": {"type": "string"},
                "branch": {"type": "string", "description": "Target branch (created if missing)"},
                "base_branch": {"type": "string"},
            },
            "required": ["repo", "files", "message"],
        },
        dangerous=True,
    )
    def github_commit_files(repo: str, files: dict, message: str, branch: str | None = None, base_branch: str | None = None):
        return commit_files(token, repo, files, message, branch, base_branch)

    @registry.tool(
        "github_issues",
        "List issues/PRs in a repository.",
        {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "state": {"type": "string", "enum": ["open", "closed", "all"], "default": "open"},
                "count": {"type": "integer", "default": 10},
            },
            "required": ["repo"],
        },
    )
    def github_issues(repo: str, state: str = "open", count: int = 10):
        return issues(token, repo, state, count)

    @registry.tool(
        "github_create_issue",
        "Open an issue on a repository. Needs GITHUB_TOKEN.",
        {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string"},
                "labels": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["repo", "title"],
        },
        dangerous=True,
    )
    def github_create_issue(repo: str, title: str, body: str = "", labels: list | None = None):
        return create_issue(token, repo, title, body, labels)

    @registry.tool(
        "github_create_pr",
        "Open a pull request from a head branch into a base branch. Needs GITHUB_TOKEN.",
        {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "title": {"type": "string"},
                "head": {"type": "string"},
                "base": {"type": "string"},
                "body": {"type": "string"},
                "draft": {"type": "boolean", "default": False},
            },
            "required": ["repo", "title", "head"],
        },
        dangerous=True,
    )
    def github_create_pr(repo: str, title: str, head: str, base: str | None = None, body: str = "", draft: bool = False):
        return create_pr(token, repo, title, head, base, body, draft)

    @registry.tool(
        "github_branches",
        "List branches in a repository.",
        {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
    )
    def github_branches(repo: str):
        return branches(token, repo)

    @registry.tool(
        "github_whoami",
        "Identify the authenticated GitHub user and confirm the token works.",
        {"type": "object", "properties": {}},
    )
    def github_whoami():
        return whoami(token)


__all__ = ["register", "GitHubError", "binascii"]
