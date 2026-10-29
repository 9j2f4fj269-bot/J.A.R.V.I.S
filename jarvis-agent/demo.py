"""Offline demonstration: a scripted 'brain' drives the real tools.

Run `python demo.py` with no API key and no network. Two beats, both real:

1. Debugging: a JS module crashes, `node_check` clears the syntax, `node_diagnose`
   returns the parsed stack, the source line and a concrete fix, then the patch is
   applied and verified.
2. Building: Python and JavaScript calculators, run for real, an actual bug found by
   comparing outputs, patched, re-run, then committed with git.

This is a harness for the tool layer, not the agent's intelligence.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from jarvis.agent import Agent
from jarvis.config import Config
from jarvis.llm import ToolCall, Turn

REPORT_JS_BUGGY = '''// builds an invoice summary (deliberately broken)
const account = undefined;

function summary() {
  const owner = account.owner;
  const total = account.total;
  return `${owner} owes ${total}`;
}

module.exports = { summary };
'''

REPORT_JS_FIXED = REPORT_JS_BUGGY.replace(
    "const account = undefined;",
    "const account = { owner: 'Ada', total: 128.5 };",
)

CALC_PY = '''"""Tiny calculator used by the demo."""


def add(a: float, b: float) -> float:
    return a + b


def mul(a: float, b: float) -> float:
    return a * b


print("python:", add(19, 23), mul(6, 7))
'''

CALC_JS_BUGGY = '''// Tiny calculator used by the demo (has a deliberate bug).
function add(a, b) {
  return a - b;
}

function mul(a, b) {
  return a * b;
}

module.exports = { add, mul };
'''

SCORES_CSV = "name,score\nAda,10\nGrace,9\nLinus,9\n"

RUN_JS = "const { add, mul } = require('./project/calc.js');\nconsole.log('js:', add(19, 23), mul(6, 7));"
REPORT_RUN = "const { summary } = require('./debug/report.js');\nconsole.log(summary());"


class ScriptedBrain:
    """Returns predetermined turns so the demo needs no network or API key."""

    def __init__(self) -> None:
        self.turns = [
            Turn(
                text="Beat 1: the invoice module. Writing it, then checking it properly.",
                tool_calls=[
                    ToolCall("d1", "write_file", {"path": "debug/report.js", "content": REPORT_JS_BUGGY}),
                    ToolCall("d2", "node_check", {"path": "debug/report.js"}),
                ],
            ),
            Turn(
                text="Syntax is clean, so the crash is at runtime. Diagnosing.",
                tool_calls=[ToolCall("d3", "node_diagnose", {"code": REPORT_RUN})],
            ),
            Turn(
                text="The diagnostic points at line 5: `account` is undefined. Fixing the data, not the symptom.",
                tool_calls=[
                    ToolCall(
                        "d4",
                        "apply_patch",
                        {
                            "path": "debug/report.js",
                            "find": "const account = undefined;",
                            "replace": "const account = { owner: 'Ada', total: 128.5 };",
                        },
                    )
                ],
            ),
            Turn(
                tool_calls=[
                    ToolCall("d5", "run_javascript", {"code": REPORT_RUN})
                ]
            ),
            Turn(
                text="Beat 2: both calculators, then I compare their numbers.",
                tool_calls=[
                    ToolCall("c1", "write_file", {"path": "project/calc.py", "content": CALC_PY}),
                    ToolCall("c2", "write_file", {"path": "project/calc.js", "content": CALC_JS_BUGGY}),
                    ToolCall("c3", "write_file", {"path": "project/scores.csv", "content": SCORES_CSV}),
                ],
            ),
            Turn(
                tool_calls=[
                    ToolCall("c4", "run_python", {"code": "exec(open('project/calc.py').read())"}),
                    ToolCall("c5", "run_javascript", {"code": RUN_JS}),
                ]
            ),
            Turn(
                text="JS says -4 where Python says 42: `add` subtracts. Patching it.",
                tool_calls=[
                    ToolCall(
                        "c6",
                        "apply_patch",
                        {"path": "project/calc.js", "find": "return a - b;", "replace": "return a + b;"},
                    )
                ],
            ),
            Turn(tool_calls=[ToolCall("c7", "run_javascript", {"code": RUN_JS})]),
            Turn(
                text="Both agree now. Checking the project surface with a digest, then committing.",
                tool_calls=[
                    ToolCall(
                        "c8b",
                        "node_profile",
                        {
                            "code": "function fib(n){ return n < 2 ? n : fib(n-1) + fib(n-2); }\nconsole.log(fib(31));",
                            "kind": "cpu",
                            "top": 3,
                        },
                    ),
                    ToolCall("c8", "read_document", {"path": "project/scores.csv", "as_table": True}),
                    ToolCall("c9", "repo_digest", {"target": "project", "max_files": 10}),
                    ToolCall("c10", "git_init", {"path": "project"}),
                    ToolCall("c11", "git_commit", {"path": "project", "message": "feat: calculator in python and js"}),
                ],
            ),
            Turn(
                tool_calls=[
                    ToolCall("c12", "remember", {"text": "User demoed JARVIS debugging Node and building Python + JS.", "tags": "demo"})
                ]
            ),
            Turn(
                text="Done. The invoice module runs, both calculators print 42, the JS bug is fixed, the CSV "
                "and digest both read cleanly, and the work is committed in project/."
            ),
        ]
        self.index = 0

    def chat(self, messages, tools, system=None):  # noqa: D401 - matches LLM.chat signature
        turn = self.turns[min(self.index, len(self.turns) - 1)]
        self.index += 1
        return turn


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="jarvis-demo-"))
    config = Config.from_env(workspace=workspace, memory_path=workspace / "memory.sqlite3")
    config.api_key = config.api_key or "demo"

    agent = Agent(config, llm=ScriptedBrain())

    def show(event: dict) -> None:
        if event["type"] == "tool":
            args = ", ".join(f"{k}={str(v)[:44]!r}" for k, v in (event.get("arguments") or {}).items())
            print(f"\n  -> {event['name']}({args})")
        elif event["type"] == "tool_result":
            out = (event.get("output") or "").strip().replace("\n", "\n     ")
            print(f"     {out[:520]}")
        elif event["type"] == "text":
            print(f"\n  jarvis: {event['text']}")

    print("JARVIS offline demo (scripted brain, real tools)")
    print(f"workspace: {workspace}")
    answer = agent.run(
        "Debug the invoice module, then build a calculator in Python and JavaScript, prove both work, commit it.",
        on_event=show,
    )
    print("\nfinal:", answer)
    print("files:", sorted(p.relative_to(workspace).as_posix() for p in workspace.rglob("*") if p.is_file() and ".git" not in p.parts and ".profiles" not in p.parts))
    if "--keep" not in sys.argv:
        shutil.rmtree(workspace, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
