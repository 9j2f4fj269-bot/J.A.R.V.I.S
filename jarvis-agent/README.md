# JARVIS

A local, Jarvis-style personal agent that actually does things. It talks, then acts
through real tools: searches the web, reads pages, searches YouTube and pulls
transcripts, reads and writes GitHub repositories, writes **Python and JavaScript**,
runs both to prove they work, **debugs what fails with real stack frames and source
context**, reads PDFs and DOCX files, digests whole repositories, and commits the result.

No agent framework, no SDK: ~2,600 lines of plain Python with two dependencies
(`requests`, `beautifulsoup4`). Bring any model.

```
you > build a tip calculator in python and javascript, test both, commit it

  -> write_file(path='tip/calc.py', ...)          write_file(path='tip/calc.js', ...)
  -> run_python(...)                              run_javascript(...)
     python: 42.0                                    js: 38.0     <- wrong, reads it
  -> apply_patch(path='tip/calc.js', find='return a - b', replace='return a + b')
  -> run_javascript(...)  ->  js: 42.0
  -> git_init('tip')      ->  git_commit('feat: tip calculator')
  Both calculators agree now. Committed as eff11e4.
```

Debugging is the part most agents fake. JARVIS parses the failure instead of re-running
blindly:

```
you > the invoice module crashes, find out why

  -> node_check(path='report.js')          syntax OK
  -> node_diagnose(path='report.js')
     TypeError: Cannot read properties of undefined (reading 'owner')
       at summary  report.js:5
        3 | function summary() {
        4 |   const owner = account.owner;
        5 |   const total = account.total;
     suggestion: something upstream returned undefined before 'owner' was read;
                 guard that value and check the path that produced it
  Fixed report.js:5 (account was never initialised), re-ran, prints "Ada owes 128.5".
```

## Quick start

```bash
pip install -r requirements.txt      # or: pip install -e .
cp .env.example .env                 # add OPENAI_API_KEY (or ANTHROPIC_API_KEY)
python -m jarvis --doctor            # live checks for every integration
python -m jarvis                      # interactive REPL
python -m jarvis "compare the top 3 python agent frameworks"
```

No key yet? `python demo.py` runs the whole loop offline with a scripted brain and
real tools: it writes Python and JS, runs both, catches a wrong result, patches the
file, re-runs, and commits with git.

## Commands

```bash
python -m jarvis                       # chat
python -m jarvis "one shot task"       # single request
python -m jarvis --doctor [--deep]     # diagnostics (deep = real LLM round-trip)
python -m jarvis tools                 # list every tool
python -m jarvis --allow-shell         # add run_shell this session
python -m jarvis --voice               # speak replies (say / espeak if installed)
python -m jarvis --workspace ~/code    # point the sandbox at a real folder
```

In chat: `/help`, `/tools`, `/doctor`, `/reset`, `/memory`, `/recall <text>`,
`/model <name>`, `/quit`.

## Tools (47)

| Area | Tools |
| --- | --- |
| Internet | `web_search`, `fetch_url`, `http_request` |
| Repositories | `repo_digest`, `github_code_search` |
| Documents | `read_document` (PDF, DOCX, HTML, CSV, JSON, text) |
| YouTube | `youtube_search`, `youtube_transcript`, `youtube_info` |
| GitHub (REST) | `github_search`, `github_repo`, `github_list_files`, `github_read_file`, `github_write_file`, `github_commit_files`, `github_issues`, `github_create_issue`, `github_create_pr`, `github_branches`, `github_whoami` |
| Git (local) | `git_clone`, `git_init`, `git_status`, `git_diff`, `git_log`, `git_commit`, `git_push`, `git_pull`, `git_branch`, `git_remotes` |
| Python | `run_python` |
| JavaScript | `run_javascript`, `npm_install` |
| JS debugging | `node_check`, `node_diagnose`, `node_profile`, `node_memory` |
| Verification | `run_tests` (npm/vitest/jest/node --test/pytest/unittest auto-detect) |
| Code editing | `write_file`, `read_file`, `apply_patch`, `grep_files`, `list_files`, `move_file`, `delete_file` |
| Memory | `remember`, `recall` |
| Shell | `run_shell` (opt-in via `JARVIS_ALLOW_SHELL=1`) |

Worth knowing:

- **`run_javascript`** auto-detects CommonJS vs ESM (`.cjs`/`.mjs`), keeps real stack
  traces, and runs from a real directory so `require('./lib/x.js')` resolves properly.
  Pass `cwd='app'` to work inside a sub-project.
- **`repo_digest`** collapses a whole repository into one digest: file tree, contents of
  the important files, line counts and a token estimate. One tool call instead of twenty
  `github_read_file` reads, and it works on a local folder too.
- **`github_code_search`** hunts for a pattern or library usage across GitHub. Without a
  `GITHUB_TOKEN` it falls back to repository search and says so instead of failing.
- **`read_document`** handles PDFs (via pypdf), DOCX (parsed with the standard library, no
  dependency), HTML, CSV/TSV as tables, JSON and plain text, locally or from a URL.
- **`node_diagnose`** runs JS under a harness that catches uncaught exceptions,
  unhandled promise rejections and warnings, then returns parsed frames with the
  surrounding source lines and rule-based fixes for the ~15 errors Node actually
  throws (missing module, ESM/CommonJS mixups, undefined property access, wrong
  argument types, port in use, infinite recursion, hung process...).
- **`node_check`** catches syntax errors without executing anything, so a bad patch
  never turns into a confusing runtime failure.
- **`node_profile`** records a CPU or heap profile and reports the hottest functions
  with percentages (`kind='cpu'|'heap'`), which turns "it's slow" into a name and a line.
- **`node_memory`** hunts leaks: a `--require` monitor samples `process.memoryUsage()` while the
  script runs, so growth shows up as a slope in MB/s, and two V8 heap snapshots (baseline plus
  mid-run) are diffed by constructor to name what keeps growing. It tracks off-heap memory too,
  because a Buffer leak looks perfectly flat if you only watch the V8 heap.
- **`run_tests`** figures out the runner and returns pass/fail plus output, so the agent
  stops guessing whether code works.
- **`apply_patch`** edits an exact snippet instead of rewriting a whole file, and
  refuses ambiguous matches (tell it `all=true` if you really mean every hit).
- **`github_commit_files`** ships a whole project in one atomic commit (blobs -> tree ->
  commit -> ref), optionally onto a new branch you can PR.
- **`git_*`** needs no shell and no token for local work; `git_push` authenticates with
  `GITHUB_TOKEN` and scrubs the token out of every log line it returns.

## Finding code worth borrowing

The fastest way to make the agent better is to steal well-tested code instead of
reinventing it. The intended loop:

```bash
python -m jarvis "search github for python libraries that chunk long documents, digest the most
promising one, then add a chunking helper to jarvis/tools that uses the same idea"
```

`github_code_search` finds candidates, `repo_digest` reads the winner in one call, and the
implementation cites what it borrowed. That is how these features got in:

| Borrowed idea | Source | Where it landed |
| --- | --- | --- |
| Repository-to-text digest, tree + token count | [gitingest](https://github.com/coderamp-labs/gitingest) | `repo_digest` |
| Exponential backoff, `Retry-After`, retry only idempotent writes | tenacity / httpx | `jarvis/http_retry.py` |
| Parallel tool calls and context compaction in agent loops | OpenHands, aider | `jarvis/agent.py` |
| PDF text extraction | [pypdf](https://github.com/py-pdf/pypdf) | `read_document` |
| DOCX as a zip of `word/document.xml` | ECMA-376 / python-docx | `read_document` |
| Post-mortem error diagnosis with source context | V8 stack semantics | `node_diagnose` |
| Heap snapshot diffing by constructor, sampled memory timeline | V8 `v8.writeHeapSnapshot` format | `node_memory` |

## Python + JavaScript

Both runtimes are first class:

```bash
python -m jarvis "add a /health endpoint to app/server.js with express, then curl it"
python -m jarvis "port this python script to node and make the outputs match"
python -m jarvis "write pytest tests for utils.py, run them, fix whatever fails"
```

`npm_install` creates `package.json` if needed and installs dependencies before
`run_javascript` imports them. Python needs no setup beyond the interpreter.

## GitHub

1. Create a classic PAT with `repo` scope (fine-grained: Contents, Issues, Pull
   requests - read & write).
2. Put it in `.env` as `GITHUB_TOKEN`.
3. `python -m jarvis "what repos do I have? then clone scratch and add a usage section"`.

Read-only browsing works with no token (60 requests/hour).

## Models

Any OpenAI-compatible endpoint works, plus Anthropic:

```bash
# OpenAI
OPENAI_API_KEY=sk-...
JARVIS_MODEL=gpt-4o-mini

# Local, free, no key (Ollama)
OPENAI_BASE_URL=http://localhost:11434/v1
JARVIS_MODEL=qwen2.5-coder:7b

# Anthropic
JARVIS_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Tool-calling quality matters: pick a model that is good at function calling. Small
local models work but need more nudging.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | - | the brain |
| `JARVIS_PROVIDER` | auto | `openai` or `anthropic` |
| `JARVIS_MODEL` / `OPENAI_BASE_URL` | `gpt-4o-mini` / OpenAI | model + endpoint |
| `GITHUB_TOKEN` | - | repo write, push, higher rate limits |
| `JARVIS_SEARCH` + `TAVILY_API_KEY`/`BRAVE_API_KEY` | `bing` (keyless) | search backend |
| `JARVIS_WORKSPACE` | `~/jarvis-workspace` | where code lives |
| `JARVIS_MEMORY` | `~/.jarvis/memory.sqlite3` | facts + history |
| `JARVIS_ALLOW_SHELL` | off | expose `run_shell` |
| `JARVIS_MAX_STEPS` | 12 | tool-call budget per turn |
| `JARVIS_CONTEXT_BUDGET` | 24000 | approx prompt tokens before compaction |
| `HTTPS_PROXY` | - | route requests (helps YouTube captions) |

## How it works

```
jarvis/
  __main__.py   CLI: chat, one-shot, doctor, tools
  agent.py      the loop: model -> tool calls -> observations -> answer
  llm.py        OpenAI-compatible + Anthropic adapters, retries, tool schemas
  registry.py   tool schemas + dispatch; tool errors become observations, not crashes
  memory.py     SQLite durable facts + conversation history
  http_retry.py retry with backoff, Retry-After, write-safety
  doctor.py     live connectivity checks
  voice.py      optional TTS
  tools/
    web.py      Bing/Tavily/Brave search, HTML -> text
    youtube.py  search, oEmbed metadata, caption extraction
    github.py   REST client, atomic multi-file commits
    git.py      native git: clone/commit/push/branch/diff/log
    repo.py     repo_digest (gitingest-style), GitHub code search
    documents.py  PDF/DOCX/HTML/CSV/JSON readers
    code.py     python subprocess runner
    js.py       node runner, npm install, test runner detection
    node_debug.py        diagnosis harness, syntax check, CPU/heap profiles
    node_diagnostics.py  stack parsing, source context, error explanations
    node_memory.py       leak timeline, heap snapshot diffing
    editing.py  apply_patch, grep, move, delete
    http.py     generic API calls
    files.py    workspace-confined file I/O
```

The loop is deliberately boring: send messages + tool schemas, run every requested
tool, append the result, repeat until it answers or hits the step budget. Failed tools
return an `ERROR:` string the model can read and recover from.

## Tests

```bash
python -m unittest discover -s tests -v     # 110 tests, no network, no API key
```

Coverage: agent loop and step budget, message persistence, both LLM adapter payload
shapes, tool dispatch and error capture, workspace path escapes, Python timeouts, JS
CommonJS/ESM execution and relative-require resolution, apply_patch ambiguity rules,
grep, npm-driven test detection, git commit cycle and token scrubbing, memory dedupe,
Bing redirect unwrapping, YouTube parsing, GitHub commit tree builder, HTTP client
success and failure paths, plus the debugger: stack parsing, harness frame filtering,
every explanation rule, uncaught exception and unhandled rejection capture, ESM
detection, timeouts, CPU/heap profiling and scratch-file cleanup, retry policy (backoff,
`Retry-After`, no duplicate POSTs), repo digest parsing and globs, document extraction for
every supported format, leak detection on heap and Buffer workloads, snapshot diffing and
internals filtering, and the agent loop's parallel execution plus compaction.

## Known limits (measured, not guessed)

- **YouTube captions from datacenter IPs**: YouTube often answers `LOGIN_REQUIRED`,
  which blocks transcripts. The tool tries the watch page then the InnerTube API and
  says plainly when it is an IP block. Search and oEmbed metadata keep working; a
  residential connection or `HTTPS_PROXY` fixes the transcript path.
- **Keyless search** scrapes Bing; good results, but HTML can change. Add Tavily or
  Brave for a contract instead of a scrape.
- **Not a security sandbox.** `run_python`, `run_javascript` and `npm_install` execute
  code on your machine. `run_shell` stays off until you enable it. Treat the model's
  instructions like a script from a stranger, because that is what they are.
- **Truncation**: tool output is capped (12-20k chars) to keep context manageable, and the
  prompt itself is compacted to `JARVIS_CONTEXT_BUDGET` (default 24k tokens) with a summary
  of the dropped middle.
- **Anonymous GitHub is rate limit 60 req/h**, and `github_code_search` needs a token at all.
  `repo_digest` uses raw.githubusercontent (no API limit) for file bodies, so one digest is
  usually cheaper than many reads.
- **Digests are capped** at `max_files` (80) and `max_chars` (60k) so a giant repo does not
  flood the context; the rest is listed as omitted.
- **Leak verdicts need a window**: a script that finishes in under a second has no timeline, so
  `node_memory` reports "no trend" instead of guessing. Anything long-running is stopped at
  `seconds=` with a final sample.
- **Heap snapshots under-count Buffer backing stores**, so for Buffer leaks read `count_delta`
  rather than `retained_kb`; the tool says so when it reports `growth_area=buffers/external`.
- **Profilers need bounded scripts**: `node_profile` only writes a sample file when the
  process exits, so profiling a server means wrapping the work in a script that ends.
- **Breakpoints are not supported yet.** Diagnosis is post-mortem (capture, parse,
  explain, suggest) rather than interactive stepping, which is a deliberate trade for
  zero dependencies.

## Adding a tool

```python
@registry.tool("weather", "Current weather for a city", {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
})
def weather(city: str):
    return http_request(f"https://api.example.com/weather?q={city}")
```

Register it in `jarvis/tools/__init__.py`; the model can use it immediately. Bad
arguments and exceptions come back as observations instead of crashing the loop.
