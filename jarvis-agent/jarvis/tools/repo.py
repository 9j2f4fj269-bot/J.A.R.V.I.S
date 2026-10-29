"""Repository intelligence: read a whole repo cheaply, and search code on GitHub.

`repo_digest` is the gitingest idea (github.com/coderamp-labs/gitingest): turn a repo
into one LLM-friendly text with a tree, per-file contents and a token estimate, so the
agent can reason about a codebase in a single tool call instead of twenty HTTP reads.
Works on GitHub URLs (public repos, anonymous) and on local directories.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from ..http_retry import request_with_retry
from .files import read as read_file_tool

DEFAULT_MAX_FILES = 80
DEFAULT_MAX_CHARS = 60_000
PREVIEW_CHARS = 3_500

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", "target", "vendor", ".next", ".nuxt", "coverage", ".mypy_cache",
    ".pytest_cache", ".idea", ".vscode", "site-packages", ".profiles", ".digests",
}

SKIP_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".tiff", ".svg",
    ".mp3", ".mp4", ".wav", ".ogg", ".webm", ".mov", ".avi",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".jar", ".whl",
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".so", ".dylib", ".dll", ".exe", ".bin", ".o", ".a", ".class", ".pyc",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".lock", ".map", ".min.js", ".min.css", ".snap",
}

TEXT_EXT = {
    ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".json", ".md", ".txt",
    ".yml", ".yaml", ".toml", ".cfg", ".ini", ".html", ".css", ".scss", ".sh",
    ".sql", ".go", ".rs", ".java", ".kt", ".rb", ".php", ".c", ".h", ".cpp",
    ".hpp", ".cs", ".swift", ".lua", ".r", ".jl", ".ex", ".exs", ".vue", ".svelte",
    ".env.example", ".gitignore", ".dockerfile", ".tf",
    ".csv", ".tsv", ".xml", ".proto", ".graphql", ".gql", ".ini", ".conf", ".properties",
}

GITHUB_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/(?P<owner>[^/]+)/(?P<repo>[^/@#?]+)"
    r"(?:/tree/(?P<ref>[^/]+)(?:/(?P<subdir>.*?))?)?/?$"
)
SHORTHAND = re.compile(r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)$")


def parse_github(target: str) -> dict | None:
    """Accept a GitHub URL or 'owner/repo' shorthand."""
    text = (target or "").strip()
    if not text or text.startswith((".", "/", "~", "\\")):
        return None  # clearly a local path, not owner/repo
    match = GITHUB_URL.match(text)
    if not match:
        match = SHORTHAND.match(text)
    if not match:
        return None
    data = match.groupdict()
    return {
        "owner": data["owner"],
        "repo": re.sub(r"\.git$", "", data["repo"]),
        "ref": data.get("ref") or None,
        "subdir": (data.get("subdir") or "").strip("/") or None,
    }


def _language_for(name: str) -> str:
    return {
        ".py": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
        ".ts": "typescript", ".tsx": "tsx", ".jsx": "jsx", ".json": "json", ".md": "markdown",
        ".yml": "yaml", ".yaml": "yaml", ".sh": "bash", ".sql": "sql", ".go": "go",
        ".rs": "rust", ".java": "java", ".rb": "ruby", ".php": "php", ".c": "c",
        ".h": "c", ".cpp": "cpp", ".html": "html", ".css": "css",
    }.get(Path(name).suffix.lower(), "")


def _looks_textual(name: str) -> bool:
    suffix = Path(name).suffix.lower()
    if suffix in SKIP_EXT:
        return False
    if suffix in TEXT_EXT or suffix == "":
        return True
    return suffix in {".lock"} is False and suffix not in SKIP_EXT


def _skip_path(path: str) -> bool:
    parts = Path(path).parts
    if any(part in SKIP_DIRS for part in parts[:-1]):
        return True
    if any(fnmatch.fnmatch(path, pattern) for pattern in ("*.min.js", "*.min.css", "*.map")):
        return True
    return not _looks_textual(path)


def _match_filters(path: str, include: list[str] | None, exclude: list[str] | None) -> bool:
    if include and not any(fnmatch.fnmatch(path, pat) for pat in include):
        return False
    if exclude and any(fnmatch.fnmatch(path, pat) for pat in exclude):
        return False
    return True


def _tree_text(files: list[dict]) -> str:
    by_dir: dict[str, list[str]] = {}
    for entry in files:
        parent = str(Path(entry["path"]).parent)
        by_dir.setdefault("." if parent == "." else parent, []).append(Path(entry["path"]).name)
    lines = []
    for directory in sorted(by_dir):
        lines.append(f"{directory}/")
        for name in sorted(by_dir[directory]):
            lines.append(f"  {name}")
    return "\n".join(lines)


def _pack(sections: list[tuple[str, str]], max_chars: int) -> tuple[str, list[str], bool]:
    """Assemble the digest; report which files landed inside the budget."""
    chunks: list[str] = []
    included: list[str] = []
    used = 0
    truncated = False
    for path, body in sections:
        block = f"\n### {path}\n```{_language_for(path)}\n{body.rstrip()}\n```\n"
        if used + len(block) > max_chars:
            truncated = True
            continue
        chunks.append(block)
        included.append(path)
        used += len(block)
    return "".join(chunks), included, truncated


def _github_json(path: str, token: str | None) -> dict:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "jarvis-agent"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response, _attempts = request_with_retry(
        "GET", f"https://api.github.com{path}", attempts=3, headers=headers, timeout=30
    )
    if response.status_code >= 400:
        raise RuntimeError(f"GitHub HTTP {response.status_code} for {path}: {response.text[:180]}")
    return response.json()


def _fetch_raw(owner: str, repo: str, ref: str, path: str, token: str | None, max_bytes: int) -> str:
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
    headers = {"User-Agent": "jarvis-agent"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response, _attempts = request_with_retry("GET", url, attempts=3, headers=headers, timeout=30)
    if response.status_code >= 400:
        return ""
    if len(response.content) > max_bytes:
        return f"[skipped: {len(response.content)} bytes over limit]"
    return response.text


def digest_github(
    owner: str,
    repo: str,
    ref: str | None = None,
    subdir: str | None = None,
    token: str | None = None,
    max_files: int = DEFAULT_MAX_FILES,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_file_kb: int = 200,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    workspace: Path | None = None,
) -> dict:
    meta = _github_json(f"/repos/{owner}/{repo}", token)
    branch = ref or meta.get("default_branch") or "main"
    tree = _github_json(f"/repos/{owner}/{repo}/git/trees/{branch}?recursive=1", token)

    all_paths = [node.get("path", "") for node in tree.get("tree", []) if node.get("path")]
    entries = []
    for node in tree.get("tree", []):
        if node.get("type") not in {"file", "blob"}:
            continue
        path = node.get("path", "")
        if subdir and not path.startswith(subdir.rstrip("/") + "/"):
            continue
        if _skip_path(path) or not _match_filters(path, include, exclude):
            continue
        entries.append({"path": path, "size": node.get("size", 0)})

    if not entries:
        top_dirs = sorted({p.split("/")[0] for p in all_paths if "/" in p})[:25]
        return {
            "ok": False,
            "error": f"no files matched in {owner}/{repo}@{branch}"
            + (f" under subdir '{subdir}'" if subdir else "")
            + (" with those include/exclude globs" if include or exclude else ""),
            "top_level_entries": top_dirs,
            "hint": "retry with one of these paths, or drop subdir/include filters",
        }

    entries.sort(key=lambda e: e["path"])
    selected = entries[:max_files]
    skipped_count = max(0, len(entries) - len(selected))

    with ThreadPoolExecutor(max_workers=8) as pool:
        bodies = list(
            pool.map(
                lambda entry: _fetch_raw(owner, repo, branch, entry["path"], token, max_file_kb * 1024),
                selected,
            )
        )

    sections: list[tuple[str, str]] = []
    total_lines = 0
    for entry, body in zip(selected, bodies):
        if not body or body.startswith("[skipped:"):
            continue
        total_lines += body.count("\n") + (0 if body.endswith("\n") else 1)
        sections.append((entry["path"], body))

    packed, included, truncated = _pack(sections, max_chars)
    header = (
        f"# {owner}/{repo} @ {branch}\n"
        f"{meta.get('description') or 'no description'}\n"
        f"stars={meta.get('stargazers_count')} forks={meta.get('forks_count')} "
        f"language={meta.get('language')} license={(meta.get('license') or {}).get('spdx_id')}\n"
        f"files_in_repo={len(entries)} files_in_digest={len(included)} "
        f"lines={total_lines} est_tokens={round(len(packed) / 4)}\n"
    )
    tree_text = _tree_text(selected[:200])
    digest = f"{header}\n## Tree\n```\n{tree_text}\n```\n\n## Files\n{packed}"

    path = None
    if workspace is not None:
        directory = Path(workspace) / ".digests"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{owner}-{repo}-{str(branch).replace('/', '-')}.md"
        path.write_text(digest, encoding="utf-8")

    return {
        "ok": True,
        "source": f"github:{owner}/{repo}@{branch}",
        "ref": branch,
        "description": meta.get("description"),
        "stars": meta.get("stargazers_count"),
        "language": meta.get("language"),
        "files_in_repo": len(entries),
        "files_included": [entry["path"] for entry in selected if entry["path"] in included],
        "files_omitted": skipped_count + (len(sections) - len(included)),
        "lines": total_lines,
        "est_tokens": round(len(digest) / 4),
        "digest_path": str(Path(path).relative_to(Path(workspace))) if path else None,
        "truncated": truncated,
        "preview": digest[:PREVIEW_CHARS],
    }


def digest_local(
    directory: str,
    workspace: Path,
    max_files: int = DEFAULT_MAX_FILES,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_file_kb: int = 200,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict:
    root = Path(workspace).resolve()
    base = (root / directory).resolve() if not Path(directory).is_absolute() else Path(directory).resolve()
    if root != base and root not in base.parents:
        raise ValueError(f"'{directory}' is outside the workspace")
    if not base.exists():
        raise FileNotFoundError(f"no such directory: {directory}")

    entries: list[dict] = []
    if base.is_file():
        entries = [{"path": base.name, "absolute": base}]
    else:
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(base).as_posix()
            if _skip_path(relative) or not _match_filters(relative, include, exclude):
                continue
            entries.append({"path": relative, "absolute": path})

    selected = entries[:max_files]
    sections: list[tuple[str, str]] = []
    total_lines = 0
    for entry in selected:
        try:
            if entry["absolute"].stat().st_size > max_file_kb * 1024:
                continue
            body = read_file_tool(root, str(entry["absolute"]))["content"]
        except (OSError, KeyError, ValueError):
            continue
        total_lines += body.count("\n") + (0 if body.endswith("\n") else 1)
        sections.append((entry["path"], body))

    packed, included, truncated = _pack(sections, max_chars)
    header = (
        f"# local:{base.name}\n"
        f"files_in_dir={len(entries)} files_in_digest={len(included)} lines={total_lines}\n"
    )
    digest = f"{header}\n## Tree\n```\n{_tree_text(selected[:200])}\n```\n\n## Files\n{packed}"

    digests = root / ".digests"
    digests.mkdir(parents=True, exist_ok=True)
    target = digests / f"local-{base.name}.md"
    target.write_text(digest, encoding="utf-8")

    return {
        "ok": True,
        "source": f"local:{base}",
        "files_in_dir": len(entries),
        "files_included": [entry["path"] for entry in selected if entry["path"] in included],
        "files_omitted": max(0, len(entries) - len(included)),
        "lines": total_lines,
        "est_tokens": round(len(digest) / 4),
        "digest_path": str(target.relative_to(root)),
        "truncated": truncated,
        "preview": digest[:PREVIEW_CHARS],
    }


def github_code_search(
    query: str,
    token: str | None,
    language: str | None = None,
    repo: str | None = None,
    per_page: int = 10,
) -> dict:
    """Search code on GitHub. Requires a token; falls back to repository search."""
    per_page = max(1, min(per_page, 30))
    qualifiers = [query]
    if language:
        qualifiers.append(f"language:{language}")
    if repo:
        qualifiers.append(f"repo:{repo}")
    search = " ".join(qualifiers)

    if token:
        url = "https://api.github.com/search/code"
        response, attempts = request_with_retry(
            "GET",
            url,
            attempts=3,
            headers={
                "Accept": "application/vnd.github.text-match+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "jarvis-agent",
            },
            params={"q": search, "per_page": per_page},
            timeout=30,
        )
        if response.status_code < 400:
            data = response.json()
            return {
                "ok": True,
                "mode": "code",
                "total_count": data.get("total_count"),
                "attempts": attempts,
                "results": [
                    {
                        "repo": item.get("repository", {}).get("full_name"),
                        "path": item.get("path"),
                        "url": item.get("html_url"),
                        "fragments": [
                            (match.get("fragment") or "").strip()[:400]
                            for match in item.get("text_matches", [])
                        ],
                    }
                    for item in data.get("items", [])
                ],
            }
        detail = response.text[:200].replace("\n", " ")
        fallback_reason = f"code search unavailable ({response.status_code}: {detail})"
    else:
        fallback_reason = "code search needs GITHUB_TOKEN (unauthenticated code search is disabled by GitHub)"

    response, attempts = request_with_retry(
        "GET",
        "https://api.github.com/search/repositories",
        attempts=3,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "jarvis-agent"},
        params={"q": query, "per_page": per_page},
        timeout=30,
    )
    repos = []
    if response.status_code < 400:
        repos = [
            {
                "repo": item.get("full_name"),
                "stars": item.get("stargazers_count"),
                "description": item.get("description"),
                "url": item.get("html_url"),
                "language": item.get("language"),
            }
            for item in response.json().get("items", [])
        ]
    return {
        "ok": True,
        "mode": "repositories",
        "note": fallback_reason,
        "attempts": attempts,
        "results": repos,
        "next": "use repo_digest on any of these repos to read the code, "
        "or run_python/npm_install to try a library locally",
    }


def register(registry, config) -> None:
    ws = config.workspace
    token = config.github_token

    @registry.tool(
        "repo_digest",
        "Turn an entire repository into one readable digest: file tree, contents of the "
        "important files, line counts and a token estimate. Works on GitHub URLs (public, "
        "anonymous) or local folders. Use this instead of many github_read_file calls when "
        "you need to understand a codebase, look for code worth borrowing, or review a repo.",
        {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "GitHub URL, owner/repo, or local folder"},
                "ref": {"type": "string", "description": "Branch, tag or SHA (defaults to the repo default)"},
                "subdir": {"type": "string", "description": "Only include files under this path"},
                "include": {"type": "array", "items": {"type": "string"}, "description": "Glob allowlist, e.g. ['*.py']"},
                "exclude": {"type": "array", "items": {"type": "string"}, "description": "Glob denylist, e.g. ['tests/*']"},
                "max_files": {"type": "integer", "default": DEFAULT_MAX_FILES},
                "max_chars": {"type": "integer", "default": DEFAULT_MAX_CHARS},
            },
            "required": ["target"],
        },
    )
    def repo_digest(
        target: str,
        ref: str | None = None,
        subdir: str | None = None,
        include: list | None = None,
        exclude: list | None = None,
        max_files: int = DEFAULT_MAX_FILES,
        max_chars: int = DEFAULT_MAX_CHARS,
    ):
        parsed = parse_github(target)
        if parsed:
            return digest_github(
                parsed["owner"],
                parsed["repo"],
                ref or parsed["ref"],
                subdir or parsed["subdir"],
                token,
                max_files,
                max_chars,
                include=include,
                exclude=exclude,
                workspace=Path(ws),
            )
        return digest_local(target, Path(ws), max_files, max_chars, include=include, exclude=exclude)

    @registry.tool(
        "github_code_search",
        "Search GitHub code for a pattern or library usage (needs GITHUB_TOKEN). Without a "
        "token it falls back to repository search so results are still useful.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "language": {"type": "string", "description": "e.g. python, javascript"},
                "repo": {"type": "string", "description": "owner/repo to search inside"},
                "per_page": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    )
    def github_code_search_tool(
        query: str,
        language: str | None = None,
        repo: str | None = None,
        per_page: int = 10,
    ):
        return github_code_search(query, token, language, repo, per_page)
