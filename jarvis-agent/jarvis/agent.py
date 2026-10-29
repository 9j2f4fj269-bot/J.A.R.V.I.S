"""The agent loop: think -> call tools -> observe -> answer."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .config import Config
from .llm import LLM, LLMError
from .memory import Memory
from .registry import Registry
from .tools import build_registry

SYSTEM_PROMPT = """You are JARVIS, a personal AI agent running on {host}. You are not a chatbot: you finish \
tasks with tools, then report what actually happened.

You write real software in **Python and JavaScript/Node.js**, verify it by running it, and ship it \
(workspace, git, GitHub, or a PR).

How you work:
- Act, then report. If a request needs current facts, search. If it needs code, write files, run them, \
read the real error, fix it, run again. If it needs shipping, commit and push. Never claim something is \
done, tested or pushed unless a tool result proves it.
- Edit, don't rewrite: use apply_patch for changes to existing files (it tolerates whitespace drift and
  near matches, and dry_run shows the diff first), grep_files to locate code, \
run_tests to prove a suite passes.
- Debugging Python: python_diagnose turns a traceback into frames with source context plus a suggestion
  tied to the real name, signature or class, and check=true syntax-checks without running anything.
- Debugging JavaScript: when a JS error is not obvious, use node_diagnose (parsed stack + source \
context + fix hints) instead of re-running the same code; node_check catches syntax errors before a run; \
node_profile finds the slow function, and node_memory watches heap plus off-heap memory for leaks \
(verdict, MB/s slope, and the objects that keep growing). Read the suggestions before editing.
- Language choice: Python for scripts, data, automation and backends; JavaScript/Node for web front ends, \
npm packages and anything the user asks for in JS. npm_install before importing third-party modules.
- Borrow before you build: to add a capability, digest a promising repo (repo_digest) or search code \
(github_code_search), then implement only the small part you need and credit the source in a comment.
- Chain tools freely. A failing tool is information, not defeat; try a different route once before \
reporting a blocker, and say plainly what you could not verify.
- Prefer small verifiable steps. Evidence first: command output, HTTP status, file read, commit SHA.
- Use `remember` for durable facts worth keeping across sessions, and `recall` when the user refers to \
earlier context or their own setup.
- Never print or store secrets in files or commit messages. Tokens come from the environment.
- Keep replies tight and concrete. Lead with the answer. No filler, no restating the question.

Tool map: web_search / fetch_url (internet), youtube_search / youtube_transcript / youtube_info, \
github_* (REST: search, read, commit, issues, PRs), github_code_search (find code across GitHub), \
repo_digest (a whole repo as one digest: tree + key files + token estimate), read_document (PDF, DOCX, \
HTML, CSV, JSON), git_* (local clone/commit/push/branch/log), \
run_python / run_javascript / npm_install / run_tests (execution), python_diagnose (Python tracebacks), \
node_check / node_diagnose / \
node_profile / node_memory (JavaScript debugging and leak hunting), write_file / read_file / apply_patch / \
grep_files / list_files (code editing), http_request (any API), remember / recall (memory).

Tone: composed, dry wit, unflappable, like Tony Stark's JARVIS. Helpful first, stylish second. \
No emojis. Never invent results.
"""


# Tools that only read: safe to run concurrently when the model requests several at once.
# Everything else (writes, exec, git mutations, memory) stays sequential.
PARALLEL_SAFE = frozenset(
    {
        "web_search", "fetch_url", "http_request", "read_document", "repo_digest",
        "github_search", "github_code_search", "github_repo", "github_list_files",
        "github_read_file", "github_issues", "github_branches", "github_whoami",
        "youtube_search", "youtube_transcript", "youtube_info",
        "git_status", "git_diff", "git_log", "git_remotes",
        "list_files", "read_file", "grep_files", "recall",
        "node_check", "node_profile",
    }
)
COMPACT_NOTE = "[context compacted: earlier steps summarised to save room]"


def estimate_tokens(messages: list[dict]) -> int:
    """Cheap token estimate (~4 chars per token) over content and tool call arguments."""
    total = 0
    for message in messages:
        total += len(str(message.get("content") or ""))
        calls = message.get("tool_calls")
        if calls:
            total += len(json.dumps(calls, default=str))
        total += 12  # per-message overhead
    return total // 4


class Agent:
    def __init__(self, config: Config, llm: LLM | None = None, registry: Registry | None = None, memory: Memory | None = None):
        self.config = config
        self.memory = memory or Memory(config.memory_path)
        self.registry = registry or build_registry(config, self.memory)
        self.llm = llm or LLM(config)
        self.session = self.memory.new_session_id()
        self.history: list[dict] = []
        self.tool_calls_made = 0

    # --- public -------------------------------------------------------
    def run(self, user_input: str, on_event: Callable[[dict], None] | None = None, max_steps: int | None = None) -> str:
        emit = on_event or (lambda event: None)
        self.history.append({"role": "user", "content": user_input})
        self.memory.log(self.session, "user", user_input)

        steps = max_steps or self.config.max_steps
        schemas = self.registry.schemas()
        system = SYSTEM_PROMPT.format(host="a local sandbox" if self.config.allow_shell else "a sandboxed workspace")

        for step in range(1, steps + 1):
            try:
                turn = self.llm.chat(self._trimmed(), schemas, system=system)
            except LLMError as exc:
                message = f"I couldn't reach the model: {exc}"
                emit({"type": "error", "message": message})
                return message

            if not turn.wants_tools:
                text = (turn.text or "").strip() or "Done."
                self.history.append({"role": "assistant", "content": text})
                self.memory.log(self.session, "assistant", text)
                emit({"type": "text", "text": text})
                return text

            self.history.append(
                {
                    "role": "assistant",
                    "content": turn.text or "",
                    "tool_calls": [
                        {"id": c.id, "name": c.name, "arguments": c.arguments} for c in turn.tool_calls
                    ],
                }
            )
            if turn.text and turn.text.strip():
                emit({"type": "text", "text": turn.text.strip()})

            for call, output in self._run_tool_calls(turn.tool_calls, emit):
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": output,
                    }
                )

        notice = (
            f"I hit my step budget ({steps} steps) before finishing. "
            "Here's where I got to: the tool output above is what I have. Ask me to continue and I'll pick it up."
        )
        emit({"type": "text", "text": notice})
        self.history.append({"role": "assistant", "content": notice})
        self.memory.log(self.session, "assistant", notice)
        return notice

    # --- helpers ------------------------------------------------------
    def _run_tool_calls(self, calls: list, emit: Callable[[dict], None]) -> list[tuple]:
        """Execute a turn's tool calls: concurrently when every call is read-only."""
        if not calls:
            return []
        for call in calls:
            self.tool_calls_made += 1
            emit({"type": "tool", "name": call.name, "arguments": call.arguments})

        if len(calls) > 1 and all(call.name in PARALLEL_SAFE for call in calls):
            workers = min(4, len(calls))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                outputs = list(pool.map(lambda c: self.registry.call(c.name, c.arguments), calls))
            results = list(zip(calls, outputs))
        else:
            results = []
            for call in calls:
                results.append((call, self.registry.call(call.name, call.arguments)))

        for call, output in results:
            emit({"type": "tool_result", "name": call.name, "output": output[:1500]})
        return results

    def _summarise(self, messages: list[dict], budget_chars: int = 1500) -> str:
        """One-line-per-step digest of dropped history, newest last."""
        lines: list[str] = []
        for message in messages:
            role = message.get("role")
            if role == "tool":
                snippet = " ".join(str(message.get("content") or "").split())[:110]
                lines.append(f"- {message.get('name')} -> {snippet}")
            elif role == "assistant":
                calls = [c.get("name") for c in (message.get("tool_calls") or [])]
                text = " ".join(str(message.get("content") or "").split())[:80]
                if calls:
                    lines.append(f"- assistant called {', '.join(calls)} {text}".strip())
                elif text:
                    lines.append(f"- assistant said: {text}")
            elif role == "user":
                text = " ".join(str(message.get("content") or "").split())[:110]
                lines.append(f"- user asked: {text}")
        joined = "\n".join(lines)
        if len(joined) > budget_chars:
            joined = joined[:budget_chars] + "\n- (older steps omitted)"
        return f"{COMPACT_NOTE}\n{joined}"

    @staticmethod
    def _cap_tool_outputs(messages: list[dict], cap: int = 700) -> list[dict]:
        """Last resort when even a compact window is too big: trim bulky tool output."""
        capped = []
        for index, message in enumerate(messages):
            if index and message.get("role") == "tool" and len(str(message.get("content") or "")) > cap:
                message = {**message, "content": str(message["content"])[:cap] + " ... [truncated to fit context]"}
            capped.append(message)
        return capped

    def _trimmed(self, keep: int = 14) -> list[dict]:
        """Fit the history in the context budget without orphaning tool results.

        Under budget: unchanged. Over budget: keep the opening request, the most recent
        turns in full, and replace the middle with a compact summary of what happened.
        """
        budget = getattr(self.config, "context_budget", 24000) or 24000
        if estimate_tokens(self.history) <= budget:
            return self.history

        head = self.history[:1] if self.history and self.history[0].get("role") == "user" else []
        best = self.history
        for window in (keep, max(4, keep // 2), 4):
            tail = self.history[-window:] if len(self.history) > window else list(self.history)
            while tail and tail[0].get("role") == "tool":
                tail = tail[1:]
            middle = self.history[len(head) : len(self.history) - len(tail)]
            if not middle:
                continue
            # Give the summary whatever room the fixed parts leave behind.
            fixed = head + tail
            if estimate_tokens(fixed) > budget:
                fixed = self._cap_tool_outputs(fixed)
            room = max(150, (budget - estimate_tokens(fixed)) * 4)
            summary = {"role": "user", "content": self._summarise(middle, budget_chars=room)}
            candidate = head + [summary] + (fixed[len(head):] if fixed else [])
            if estimate_tokens(candidate) > budget:
                candidate = self._cap_tool_outputs(candidate)
            if estimate_tokens(candidate) < estimate_tokens(best):
                best = candidate
            if estimate_tokens(best) <= budget:
                break
        return best

    def context_stats(self) -> dict:
        return {
            "messages": len(self.history),
            "est_tokens": estimate_tokens(self.history),
            "budget": getattr(self.config, "context_budget", 24000),
            "steps": self.tool_calls_made,
        }

    def reset(self) -> None:
        self.history.clear()
        self.session = self.memory.new_session_id()

    def doctor(self) -> str:
        from .doctor import run_checks

        return run_checks(self.config)


def main_debug_dump(agent: Agent) -> None:  # pragma: no cover - debugging helper
    print(json.dumps(agent.history, indent=2, default=str), file=sys.stderr)
