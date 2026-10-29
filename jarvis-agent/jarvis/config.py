"""JARVIS configuration: environment variables, .env loading, sane defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_WORKSPACE = Path.home() / "jarvis-workspace"
DEFAULT_MEMORY = Path.home() / ".jarvis" / "memory.sqlite3"


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (no extra dependency). Existing env wins."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        val = os.environ.get(name)
        if val and val.strip():
            return val.strip()
    return default


def _flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    github_token: str | None = None
    search_provider: str = "bing"
    search_api_key: str | None = None
    workspace: Path = DEFAULT_WORKSPACE
    memory_path: Path = DEFAULT_MEMORY
    allow_shell: bool = False
    max_steps: int = 12
    context_budget: int = 24000  # approx tokens kept in the prompt before compaction
    temperature: float = 0.2
    verbose: bool = False
    extra: dict = field(default_factory=dict)

    # --- construction -------------------------------------------------
    @classmethod
    def from_env(cls, **overrides) -> "Config":
        load_dotenv()
        anthropic_key = _env("ANTHROPIC_API_KEY")
        openai_key = _env("OPENAI_API_KEY", "JARVIS_API_KEY")

        provider = _env("JARVIS_PROVIDER") or ("anthropic" if anthropic_key and not openai_key else "openai")
        provider = provider.lower()

        if provider == "anthropic":
            api_key = anthropic_key or openai_key
            base_url = _env("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
            model = _env("JARVIS_MODEL", "ANTHROPIC_MODEL", default="claude-sonnet-4-5")
        else:
            api_key = openai_key or anthropic_key
            base_url = _env("OPENAI_BASE_URL", "JARVIS_BASE_URL") or "https://api.openai.com/v1"
            model = _env("JARVIS_MODEL", "OPENAI_MODEL", default="gpt-4o-mini")

        config = cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            github_token=_env("GITHUB_TOKEN", "GH_TOKEN"),
            search_provider=(_env("JARVIS_SEARCH", default="bing") or "bing").lower(),
            search_api_key=_env("TAVILY_API_KEY", "BRAVE_API_KEY", "SEARCH_API_KEY"),
            workspace=Path(_env("JARVIS_WORKSPACE", default=str(DEFAULT_WORKSPACE))).expanduser(),
            memory_path=Path(_env("JARVIS_MEMORY", default=str(DEFAULT_MEMORY))).expanduser(),
            allow_shell=_flag("JARVIS_ALLOW_SHELL", False),
            max_steps=int(_env("JARVIS_MAX_STEPS", default="12") or 12),
            context_budget=int(_env("JARVIS_CONTEXT_BUDGET", default="24000") or 24000),
            temperature=float(_env("JARVIS_TEMPERATURE", default="0.2") or 0.2),
            verbose=_flag("JARVIS_VERBOSE", False),
        )
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        config.workspace.mkdir(parents=True, exist_ok=True)
        config.memory_path.parent.mkdir(parents=True, exist_ok=True)
        return config

    # --- helpers ------------------------------------------------------
    @property
    def has_llm(self) -> bool:
        """Local OpenAI-compatible servers (Ollama/LM Studio) need no key."""
        return bool(self.api_key) or "localhost" in (self.base_url or "") or "127.0.0.1" in (self.base_url or "")

    def summary(self) -> str:
        brain = f"{self.provider}:{self.model}" + ("" if self.api_key else " (no api key)")
        lines = [
            f"brain       {brain}",
            f"endpoint    {self.base_url}",
            f"github      {'token present' if self.github_token else 'anonymous (read-only, 60 req/h)'}",
            f"search      {self.search_provider}{' (key)' if self.search_api_key else ' (keyless)'}",
            f"workspace   {self.workspace}",
            f"shell       {'ENABLED' if self.allow_shell else 'disabled'}",
        ]
        return "\n".join(lines)
