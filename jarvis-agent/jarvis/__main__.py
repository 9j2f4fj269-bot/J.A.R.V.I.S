"""Command line entry point: `python -m jarvis`."""

from __future__ import annotations

import argparse
import sys

from .agent import Agent
from .config import Config

BANNER = r"""
     _   _   _  _ _   _ ___
    | | /_\ | \| | | | / __|
 _  | |/ _ \| .` | |_| \__ \
| |_| /_/ \_\_|\_|\___/|___/   personal agent
"""

HELP = """commands
  /help            this list
  /tools           every tool the agent can call
  /doctor          live connectivity checks
  /reset           clear conversation context (memory facts stay)
  /memory          memory stats
  /recall <text>   search saved facts
  /model <name>    switch model for this session
  /quit            exit
"""


def _print_event(verbose: bool):
    def handler(event: dict) -> None:
        kind = event.get("type")
        if kind == "tool":
            args = event.get("arguments") or {}
            preview = ", ".join(f"{k}={str(v)[:40]}" for k, v in list(args.items())[:3])
            print(f"\n  \033[36m→ {event['name']}\033[0m({preview})", flush=True)
        elif kind == "tool_result" and verbose:
            output = (event.get("output") or "").replace("\n", "\n    ")
            print(f"    \033[90m{output[:900]}\033[0m", flush=True)
        elif kind == "error":
            print(f"\033[31m{event.get('message')}\033[0m", flush=True)

    return handler


def build_agent(args) -> Agent:
    config = Config.from_env(
        model=args.model,
        provider=args.provider,
        workspace=args.workspace,
        allow_shell=True if args.allow_shell else None,
        verbose=args.verbose,
    )
    return Agent(config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis", description="A Jarvis-style personal agent with real tools.")
    parser.add_argument("prompt", nargs="*", help="One-shot request. Omit for interactive mode.")
    parser.add_argument("--model", help="Override the model name")
    parser.add_argument("--provider", choices=["openai", "anthropic"], help="Override the provider")
    parser.add_argument("--workspace", help="Workspace directory for code and files")
    parser.add_argument("--allow-shell", action="store_true", help="Expose the run_shell tool")
    parser.add_argument("--verbose", action="store_true", help="Show tool output")
    parser.add_argument("--voice", action="store_true", help="Speak replies aloud if a TTS binary exists")
    parser.add_argument("--doctor", action="store_true", help="Run diagnostics and exit")
    parser.add_argument("--deep", action="store_true", help="With --doctor: also test the LLM")
    args = parser.parse_args(argv)

    if args.doctor:
        from .config import Config as CFG
        from .doctor import run_checks

        print(run_checks(CFG.from_env(workspace=args.workspace), deep=args.deep))
        return 0

    # `tools` listing works without any credentials
    if args.prompt and args.prompt[0] == "tools":
        from .memory import Memory
        from .tools import build_registry

        config = Config.from_env(workspace=args.workspace)
        registry = build_registry(config, Memory(config.memory_path))
        print(f"{len(registry.names())} tools\n")
        print(registry.help_text())
        return 0

    try:
        agent = build_agent(args)
    except Exception as exc:  # noqa: BLE001
        print(f"startup failed: {exc}", file=sys.stderr)
        print("Run `python -m jarvis --doctor` to see what's missing.", file=sys.stderr)
        return 2

    speak = None
    if args.voice:
        from .voice import speak as tts

        speak = tts

    handler = _print_event(args.verbose)

    # --- one shot ---
    if args.prompt:
        answer = agent.run(" ".join(args.prompt), on_event=handler)
        print("\n" + answer)
        if speak:
            speak(answer)
        return 0

    # --- interactive ---
    print(BANNER)
    print(agent.config.summary())
    print("\nType /help for commands, /quit to exit.\n")

    while True:
        try:
            user_input = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if not user_input:
            continue
        if user_input in {"/quit", "/exit", "/q"}:
            print("Goodbye.")
            return 0
        if user_input == "/help":
            print(HELP)
            continue
        if user_input == "/tools":
            print(agent.registry.help_text())
            continue
        if user_input == "/doctor":
            print(agent.doctor())
            continue
        if user_input == "/reset":
            agent.reset()
            print("context cleared.")
            continue
        if user_input == "/memory":
            print(agent.memory.stats())
            continue
        if user_input.startswith("/recall"):
            query = user_input.removeprefix("/recall").strip()
            for fact in agent.memory.recall(query):
                print(f"  #{fact['id']} {fact['text']}")
            continue
        if user_input.startswith("/model"):
            new_model = user_input.removeprefix("/model").strip()
            if new_model:
                agent.config.model = new_model
                print(f"model -> {new_model}")
            else:
                print(f"model is {agent.config.model}")
            continue

        answer = agent.run(user_input, on_event=handler)
        if speak:
            speak(answer)


if __name__ == "__main__":
    raise SystemExit(main())
