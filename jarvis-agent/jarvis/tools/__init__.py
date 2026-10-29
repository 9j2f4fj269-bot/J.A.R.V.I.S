"""Tool package. `build_registry` wires every tool set onto one registry."""

from __future__ import annotations

from ..registry import Registry
from . import code, documents, editing, files, git, github, http, js, node_debug, node_diagnostics, node_memory, repo, web, youtube


def build_registry(config, memory=None) -> Registry:
    registry = Registry()
    web.register(registry, config)
    youtube.register(registry, config)
    github.register(registry, config)
    code.register(registry, config)
    js.register(registry, config)
    node_debug.register(registry, config)
    node_memory.register(registry, config)
    repo.register(registry, config)
    documents.register(registry, config)
    files.register(registry, config)
    editing.register(registry, config)
    http.register(registry, config)
    git.register(registry, config)
    if memory is not None:
        from .. import memory as memory_tools

        memory_tools.register(registry, config, memory)
    return registry


__all__ = ["build_registry", "web", "youtube", "github", "code", "js", "files", "editing", "http", "git"]
