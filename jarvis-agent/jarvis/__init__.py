"""JARVIS: a personal agent with real tools (web, YouTube, GitHub, code, memory)."""

from .agent import Agent
from .config import Config
from .memory import Memory
from .registry import Registry
from .tools import build_registry

__version__ = "1.0.0"
__all__ = ["Agent", "Config", "Memory", "Registry", "build_registry", "__version__"]
