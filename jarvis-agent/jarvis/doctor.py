"""System checks: what is wired up and what is actually reachable."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

OK = "[ ok ]"
BAD = "[fail]"
WARN = "[warn]"


def _line(mark: str, label: str, detail: str = "") -> str:
    return f"{mark} {label:<16} {detail}".rstrip()


def run_checks(config, deep: bool = False) -> str:
    lines: list[str] = ["JARVIS diagnostics", "=" * 48]

    lines.append(_line(OK, "python", f"{sys.version.split()[0]} at {sys.executable}"))
    for module in ("requests", "bs4"):
        try:
            __import__(module)
            lines.append(_line(OK, module, "importable"))
        except ImportError:
            lines.append(_line(BAD, module, f"missing -> pip install {module}"))

    if config.api_key or (config.base_url and ("localhost" in config.base_url or "127.0.0.1" in config.base_url)):
        detail = f"{config.provider}:{config.model} -> {config.base_url}"
        lines.append(_line(OK, "brain", detail))
    else:
        lines.append(_line(BAD, "brain", "no API key: set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env"))

    # GitHub
    if config.github_token:
        from .tools.github import GitHubError, whoami

        try:
            user = whoami(config.github_token)
            lines.append(_line(OK, "github", f"token valid as {user.get('login')}"))
        except GitHubError as exc:
            lines.append(_line(BAD, "github", str(exc)[:120]))
    else:
        import requests

        try:
            data = requests.get("https://api.github.com/rate_limit", timeout=12).json()
            left = data["resources"]["core"]["remaining"]
            lines.append(_line(WARN, "github", f"anonymous only, {left}/60 requests left this hour"))
        except Exception as exc:  # noqa: BLE001
            lines.append(_line(BAD, "github", f"unreachable: {exc}"))

    # Web search
    from .tools.web import search_web

    started = time.time()
    result = search_web("python asyncio", count=1, provider=config.search_provider, api_key=config.search_api_key)
    elapsed = int((time.time() - started) * 1000)
    if result.get("results"):
        first = result["results"][0]
        lines.append(_line(OK, "web search", f"{elapsed} ms -> {first['title'][:50]}"))
    else:
        lines.append(_line(BAD, "web search", str(result.get("error", "no results"))[:100]))

    # YouTube
    from .tools.youtube import search_youtube

    yt = search_youtube("iron man jarvis", count=1)
    if yt.get("results"):
        lines.append(_line(OK, "youtube", f"search ok -> {yt['results'][0]['title'][:50]}"))
    else:
        lines.append(_line(BAD, "youtube", str(yt.get("error", "no results"))[:100]))

    # Workspace
    try:
        config.workspace.mkdir(parents=True, exist_ok=True)
        probe = Path(config.workspace) / ".jarvis_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        lines.append(_line(OK, "workspace", str(config.workspace)))
    except Exception as exc:  # noqa: BLE001
        lines.append(_line(BAD, "workspace", f"not writable: {exc}"))

    # Memory
    try:
        from .memory import Memory

        mem = Memory(config.memory_path)
        stats = mem.stats()
        mem.close()
        lines.append(_line(OK, "memory", f"{stats['facts']} facts, {stats['messages']} messages"))
    except Exception as exc:  # noqa: BLE001
        lines.append(_line(BAD, "memory", str(exc)[:100]))

    node = shutil.which("node")
    if node:
        import subprocess

        version = subprocess.run([node, "-v"], capture_output=True, text=True, timeout=10).stdout.strip()
        npm = shutil.which("npm")
        debug_ready = subprocess.run(
            [node, "--cpu-prof", "--eval", "process.exit(0)"], capture_output=True, text=True, timeout=20
        ).returncode == 0
        lines.append(
            _line(
                OK,
                "javascript",
                f"{version} at {node}"
                + (", npm present" if npm else ", npm missing")
                + (", debugger ready" if debug_ready else ", profiler unavailable"),
            )
        )
    else:
        lines.append(_line(WARN, "javascript", "node not found: run_javascript unavailable"))

    try:
        import pypdf

        lines.append(_line(OK, "documents", f"pypdf {getattr(pypdf, '__version__', '?')} (PDF), docx/html/csv built in"))
    except ImportError:
        lines.append(_line(WARN, "documents", "text/html/csv/docx work; pip install pypdf for PDF"))

    git = shutil.which("git")
    lines.append(_line(OK if git else WARN, "git cli", git or "not found (REST tools still work)"))
    lines.append(_line(OK if config.allow_shell else WARN, "shell tool", "enabled" if config.allow_shell else "disabled (JARVIS_ALLOW_SHELL=1 to enable)"))

    try:
        from .memory import Memory as Mem
        from .tools import build_registry

        registry = build_registry(config, Mem(config.memory_path))
        lines.append(_line(OK, "tools", f"{len(registry.names())} registered"))
    except Exception as exc:  # noqa: BLE001
        lines.append(_line(BAD, "tools", str(exc)[:120]))

    if deep:
        lines.append(_line(WARN, "deep", "live LLM round-trip"))
        try:
            from .llm import LLM

            turn = LLM(config).chat([{"role": "user", "content": "Reply with the single word: ready"}], [])
            lines.append(_line(OK, "llm reply", (turn.text or "").strip()[:80]))
        except Exception as exc:  # noqa: BLE001
            lines.append(_line(BAD, "llm reply", str(exc)[:140]))

    return "\n".join(lines)
