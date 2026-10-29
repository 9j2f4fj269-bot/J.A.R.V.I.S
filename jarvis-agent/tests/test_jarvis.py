"""Test suite: `python -m unittest discover -s tests -v` (no network, no API key)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis.agent import Agent
from jarvis.config import Config
from jarvis.llm import LLM, ToolCall, Turn
from jarvis.memory import Memory
from jarvis.registry import Registry
from jarvis.tools import build_registry
from jarvis.tools import code as code_tools
from jarvis.tools import files as file_tools
from jarvis.tools import editing as edit_tools
from jarvis.tools import git as git_tools
from jarvis.tools import github as gh
from jarvis.tools import http as http_tools
from jarvis.tools import js as js_tools
from jarvis.tools import node_debug as node_debug_tools
from jarvis.tools import node_diagnostics as diag_tools
from jarvis.tools import node_memory as clone_tools
from jarvis import http_retry as retry_tools
from jarvis.tools import documents as doc_tools
from jarvis.tools import repo as repo_tools
from jarvis.tools import web as web_tools
from jarvis.tools import youtube as yt_tools


def temp_config(**kw) -> Config:
    tmp = Path(tempfile.mkdtemp(prefix="jarvis-test-"))
    return Config.from_env(workspace=tmp, memory_path=tmp / "mem.sqlite3", **kw)


class TestRegistry(unittest.TestCase):
    def test_unknown_tool_is_an_error_string(self):
        reg = Registry()
        self.assertIn("unknown tool", reg.call("nope", {}))

    def test_exceptions_are_captured_not_raised(self):
        reg = Registry()

        @reg.tool("boom", "explodes", {"type": "object", "properties": {}})
        def boom():
            raise ValueError("kaboom")

        self.assertIn("kaboom", reg.call("boom", {}))

    def test_schema_shape(self):
        reg = build_registry(temp_config(), Memory(tempfile.mkdtemp() + "/m.sqlite3"))
        schema = reg.schemas()[0]
        self.assertEqual(schema["type"], "function")
        self.assertIn("name", schema["function"])
        self.assertIn("parameters", schema["function"])
        for expected in ("web_search", "youtube_transcript", "github_commit_files", "run_python", "remember"):
            self.assertIn(expected, reg.names())


class TestFilesAndCode(unittest.TestCase):
    def test_sandbox_escape_blocked(self):
        with self.assertRaises(file_tools.PathError):
            file_tools.write(Path("/tmp/jarvis-root"), "../escaped.txt", "nope")

    def test_roundtrip(self):
        root = Path(tempfile.mkdtemp())
        file_tools.write(root, "pkg/app.py", "print('hi')\n")
        out = file_tools.read(root, "pkg/app.py")
        self.assertIn("print", out["content"])
        listing = file_tools.list_files(root, ".", "*", recursive=True)
        paths = {e["path"] for e in listing["entries"]}
        self.assertIn("pkg/app.py", paths)

    def test_run_python_success(self):
        result = code_tools.run_python("print(6*7)", Path(tempfile.mkdtemp()))
        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"].strip(), "42")

    def test_run_python_reports_traceback(self):
        result = code_tools.run_python("1/0", Path(tempfile.mkdtemp()))
        self.assertFalse(result["ok"])
        self.assertIn("ZeroDivisionError", result["stderr"])

    def test_run_python_times_out(self):
        result = code_tools.run_python("import time\ntime.sleep(5)", Path(tempfile.mkdtemp()), timeout=1)
        self.assertFalse(result["ok"])
        self.assertIn("timed out", result["error"])


class TestMemory(unittest.TestCase):
    def test_remember_recall_forget(self):
        mem = Memory(Path(tempfile.mkdtemp()) / "m.sqlite3")
        mem.remember("Prefers Python over JavaScript", "prefs")
        mem.remember("Building a Jarvis agent", "projects")
        hits = mem.recall("javascript")
        self.assertEqual(len(hits), 1)
        self.assertIn("Python", hits[0]["text"])
        mem.forget(hits[0]["id"])
        self.assertEqual(mem.recall("javascript"), [])

    def test_duplicate_facts_are_not_doubled(self):
        mem = Memory(Path(tempfile.mkdtemp()) / "m.sqlite3")
        mem.remember("likes tea")
        second = mem.remember("likes tea")
        self.assertTrue(second.get("duplicate"))
        self.assertEqual(mem.stats()["facts"], 1)


class TestParsers(unittest.TestCase):
    def test_bing_redirect_unwrap(self):
        import base64

        target = "https://example.com/article"
        token = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
        href = f"https://www.bing.com/ck/a?!&&p=abc&u=a1{token}&ntb=1"
        self.assertEqual(web_tools._decode_bing_url(href), target)

    def test_fetch_page_extracts_text(self):
        class FakeResponse:
            url = "https://example.com"
            status_code = 200
            headers = {"content-type": "text/html; charset=utf-8"}
            text = "<html><head><title>T</title><script>bad()</script></head><body><p>Hello world</p></body></html>"

        class FakeSession:
            def get(self, *a, **kw):
                return FakeResponse()

            headers: dict = {}

        original = web_tools._session
        web_tools._session = lambda: FakeSession()
        try:
            page = web_tools.fetch_page("https://example.com")
        finally:
            web_tools._session = original
        self.assertEqual(page["title"], "T")
        self.assertIn("Hello world", page["text"])
        self.assertNotIn("bad()", page["text"])

    def test_youtube_search_parsing(self):
        payload = {
            "contents": {
                "sectionListRenderer": {
                    "contents": [
                        {
                            "videoRenderer": {
                                "videoId": "dQw4w9WgXcQ",
                                "title": {"runs": [{"text": "Jarvis demo"}]},
                                "ownerText": {"runs": [{"text": "Channel X"}]},
                                "lengthText": {"simpleText": "3:21"},
                                "viewCountText": {"simpleText": "1,000 views"},
                            }
                        }
                    ]
                }
            }
        }
        html = f"<script>var ytInitialData = {json.dumps(payload)};</script>"
        data = yt_tools._initial_data(html)
        found: list = []
        yt_tools._walk(data, "videoRenderer", found)
        self.assertEqual(found[0]["videoId"], "dQw4w9WgXcQ")
        self.assertEqual(yt_tools._runs_text(found[0]["title"]), "Jarvis demo")

    def test_video_id_resolution(self):
        for value in (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "dQw4w9WgXcQ",
        ):
            self.assertEqual(yt_tools.video_id_of(value), "dQw4w9WgXcQ", value)

    def test_caption_json3(self):
        payload = json.dumps({"events": [{"segs": [{"utf8": "hello "}, {"utf8": "world"}]}, {"segs": [{"utf8": "again"}]}]})
        self.assertEqual(yt_tools._parse_json3(payload), "hello world again")

    def test_caption_tracks_extraction(self):
        html = 'x={"captionTracks":[{"baseUrl":"https://yt/api?x=1","languageCode":"en","kind":"asr"}]}'
        tracks = yt_tools._caption_tracks(html)
        self.assertEqual(tracks[0]["languageCode"], "en")


class TestGitHub(unittest.TestCase):
    def test_read_file_decodes_base64(self):
        import base64

        content = base64.b64encode(b"print('from github')\n").decode()

        def fake_api(method, path, token, **kwargs):
            return {"type": "file", "encoding": "base64", "content": content, "sha": "abc"}

        original = gh._api
        gh._api = fake_api
        try:
            result = gh.read_file("token", "me/repo", "main.py")
        finally:
            gh._api = original
        self.assertIn("from github", result["content"])

    def test_write_requires_token(self):
        with self.assertRaises(gh.GitHubError):
            gh.write_file(None, "me/repo", "a.py", "x", "msg")

    def test_commit_files_builds_tree(self):
        calls: list[tuple] = []
        blobs = iter(["sha1", "sha2"])

        def fake_api(method, path, token, **kwargs):
            calls.append((method, path, kwargs.get("json")))
            if path == "/repos/me/repo":
                return {"default_branch": "main"}
            if path == "/repos/me/repo/git/ref/heads/main":
                return {"object": {"sha": "basesha"}}
            if path.endswith("/git/blobs"):
                return {"sha": next(blobs)}
            if path.endswith("/git/trees"):
                return {"sha": "treesha"}
            if path.endswith("/git/commits"):
                return {"sha": "commitsha"}
            return {}

        original = gh._api
        gh._api = fake_api
        try:
            result = gh.commit_files("token", "me/repo", {"a.py": "1", "b.py": "2"}, "add files")
        finally:
            gh._api = original
        self.assertEqual(result["commit"], "commitsha")
        self.assertEqual(result["files"], ["a.py", "b.py"])
        tree_call = next(c for c in calls if c[1].endswith("/git/trees"))
        self.assertEqual(len(tree_call[2]["tree"]), 2)


class FakeLLM:
    def __init__(self, turns):
        self.turns = list(turns)
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools, system=None):
        self.seen.append([dict(m) for m in messages])
        return self.turns.pop(0)


class TestAgentLoop(unittest.TestCase):
    def test_tool_then_answer(self):
        config = temp_config()
        brain = FakeLLM(
            [
                Turn(tool_calls=[ToolCall("1", "run_python", {"code": "print('loop works')"})]),
                Turn(text="Verified: the snippet printed 'loop works'."),
            ]
        )
        agent = Agent(config, llm=brain)
        events: list[dict] = []
        answer = agent.run("prove the loop works", on_event=events.append)

        self.assertIn("loop works", answer)
        self.assertIn("tool", [e["type"] for e in events])
        roles = [m["role"] for m in agent.history]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])
        self.assertIn("loop works", agent.history[2]["content"])
        # second call must include the tool result
        self.assertTrue(any(m["role"] == "tool" for m in brain.seen[1]))

    def test_step_budget_stops_runaway_loops(self):
        config = temp_config(max_steps=3)
        brain = FakeLLM([Turn(tool_calls=[ToolCall(str(i), "run_python", {"code": "print(1)"})]) for i in range(5)])
        agent = Agent(config, llm=brain)
        answer = agent.run("loop forever")
        self.assertIn("step budget", answer)

    def test_history_is_persisted(self):
        config = temp_config()
        agent = Agent(config, llm=FakeLLM([Turn(text="hi there")]))
        agent.run("hello")
        stored = Memory(config.memory_path).history(agent.session)
        self.assertEqual([m["role"] for m in stored], ["user", "assistant"])


class TestLLMAdapters(unittest.TestCase):
    def test_openai_payload(self):
        config = temp_config()
        config.provider, config.model, config.api_key = "openai", "gpt-test", "sk-test"
        config.base_url = "https://api.openai.com/v1"
        llm = LLM(config)
        captured: dict = {}

        class FakeHTTP:
            def post(self, url, headers, payload):
                captured.update({"url": url, "headers": headers, "payload": payload})
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "calling a tool",
                                "tool_calls": [
                                    {"id": "call_1", "function": {"name": "run_python", "arguments": '{"code": "print(1)"}'}}
                                ],
                            }
                        }
                    ],
                    "usage": {"total_tokens": 10},
                }

        llm.http = FakeHTTP()
        turn = llm.chat([{"role": "user", "content": "hi"}], [{"type": "function", "function": {"name": "x"}}], "sys")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(captured["payload"]["messages"][0]["role"], "system")
        self.assertEqual(turn.tool_calls[0].name, "run_python")
        self.assertEqual(turn.tool_calls[0].arguments, {"code": "print(1)"})

    def test_anthropic_payload_groups_tool_results(self):
        config = temp_config()
        config.provider, config.model, config.api_key = "anthropic", "claude-test", "sk-ant"
        config.base_url = "https://api.anthropic.com"
        llm = LLM(config)
        captured: dict = {}

        class FakeHTTP:
            def post(self, url, headers, payload):
                captured.update({"url": url, "payload": payload})
                return {
                    "content": [
                        {"type": "text", "text": "done"},
                        {"type": "tool_use", "id": "tu_1", "name": "run_python", "input": {"code": "1"}},
                    ]
                }

        llm.http = FakeHTTP()
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tu_1", "name": "run_python", "arguments": {"code": "1"}}],
            },
            {"role": "tool", "tool_call_id": "tu_1", "name": "run_python", "content": "1"},
        ]
        turn = llm.chat(messages, [{"type": "function", "function": {"name": "run_python", "parameters": {}}}], "sys")
        self.assertEqual(captured["payload"]["system"], "sys")
        self.assertEqual(captured["payload"]["tools"][0]["input_schema"], {"type": "object", "properties": {}})
        self.assertEqual(captured["payload"]["messages"][-1]["role"], "user")
        self.assertEqual(turn.tool_calls[0].name, "run_python")
        self.assertEqual(turn.text, "done")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestJavaScript(unittest.TestCase):
    @unittest.skipUnless(js_tools.node_binary(), "node not installed")
    def test_commonjs(self):
        result = js_tools.run_javascript("console.log([1,2,3].map(n => n * 2).join(','))", Path(tempfile.mkdtemp()))
        self.assertTrue(result["ok"], result.get("stderr"))
        self.assertEqual(result["stdout"].strip(), "2,4,6")

    @unittest.skipUnless(js_tools.node_binary(), "node not installed")
    def test_esm_top_level_await(self):
        code = "const value = await Promise.resolve('esm ok');\nconsole.log(value);"
        result = js_tools.run_javascript(code, Path(tempfile.mkdtemp()))
        self.assertTrue(result["ok"], result.get("stderr"))
        self.assertIn("esm ok", result["stdout"])

    @unittest.skipUnless(js_tools.node_binary(), "node not installed")
    def test_real_stack_trace_is_returned(self):
        result = js_tools.run_javascript("throw new Error('boom');", Path(tempfile.mkdtemp()))
        self.assertFalse(result["ok"])
        self.assertNotEqual(result["exit_code"], 0)
        self.assertIn("boom", result["stderr"])

    def test_extension_detection(self):
        self.assertEqual(js_tools.pick_extension("const x = require('fs')"), ".cjs")
        self.assertEqual(js_tools.pick_extension("import fs from 'node:fs'"), ".mjs")
        self.assertEqual(js_tools.pick_extension("console.log(1)"), ".js")

    @unittest.skipUnless(js_tools.node_binary(), "node not installed")
    def test_relative_require_resolves_from_workspace(self):
        ws = Path(tempfile.mkdtemp())
        file_tools.write(ws, "lib/math.js", "module.exports = { add: (a, b) => a + b };\n")
        result = js_tools.run_javascript(
            "const { add } = require('./lib/math.js');\nconsole.log(add(2, 3));", ws
        )
        self.assertTrue(result["ok"], result.get("stderr"))
        self.assertIn("5", result["stdout"])

    @unittest.skipUnless(js_tools.node_binary(), "node not installed")
    def test_cwd_option_runs_inside_project(self):
        ws = Path(tempfile.mkdtemp())
        file_tools.write(ws, "app/helper.js", "module.exports = 'from app';\n")
        result = js_tools.run_javascript("console.log(require('./helper.js'));", ws, cwd="app")
        self.assertTrue(result["ok"], result.get("stderr"))
        self.assertIn("from app", result["stdout"])

    def test_cwd_cannot_escape_workspace(self):
        result = js_tools.run_javascript("console.log(1)", Path(tempfile.mkdtemp()), cwd="../../etc")
        self.assertFalse(result["ok"])
        self.assertIn("outside the workspace", result["error"])

    def test_snippet_file_is_cleaned_up(self):
        ws = Path(tempfile.mkdtemp())
        js_tools.run_javascript("console.log('x')", ws)
        leftovers = [p.name for p in ws.iterdir() if p.name.startswith(".jarvis_run_")]
        self.assertEqual(leftovers, [])

    def test_missing_runtime_is_reported_cleanly(self):
        original = js_tools.node_binary
        js_tools.node_binary = lambda: None
        try:
            result = js_tools.run_javascript("console.log(1)", Path(tempfile.mkdtemp()))
        finally:
            js_tools.node_binary = original
        self.assertFalse(result["ok"])
        self.assertIn("node is not installed", result["error"])


class TestEditing(unittest.TestCase):
    def setUp(self):
        self.ws = Path(tempfile.mkdtemp())
        file_tools.write(self.ws, "src/app.py", "def add(a, b):\n    return a - b\n")

    def test_patch_replaces_exact_snippet(self):
        result = edit_tools.apply_patch(self.ws, "src/app.py", "return a - b", "return a + b")
        self.assertTrue(result["ok"])
        self.assertIn("return a + b", file_tools.read(self.ws, "src/app.py")["content"])

    def test_patch_reports_missing_snippet(self):
        result = edit_tools.apply_patch(self.ws, "src/app.py", "not there", "x")
        self.assertFalse(result["ok"])
        self.assertIn("not present", result["error"])

    def test_patch_refuses_ambiguous_match(self):
        file_tools.write(self.ws, "dup.js", "x++;\nx++;\n")
        result = edit_tools.apply_patch(self.ws, "dup.js", "x++;", "x+=2;")
        self.assertFalse(result["ok"])
        self.assertEqual(result["occurrences"], 2)
        ok = edit_tools.apply_patch(self.ws, "dup.js", "x++;", "x+=2;", all=True)
        self.assertTrue(ok["ok"])
        self.assertEqual(ok["replacements"], 2)

    def test_patch_cannot_escape_workspace(self):
        with self.assertRaises(ValueError):
            edit_tools.apply_patch(self.ws, "../outside.py", "a", "b")

    def test_grep_finds_lines_and_skips_node_modules(self):
        file_tools.write(self.ws, "node_modules/pkg/index.js", "return a - b\n")
        hits = edit_tools.grep_files(self.ws, r"return a - b")
        self.assertEqual(len(hits["matches"]), 1)
        self.assertEqual(hits["matches"][0]["line"], 2)

    def test_move_and_delete(self):
        self.assertTrue(edit_tools.move_path(self.ws, "src/app.py", "src/main.py")["ok"])
        self.assertTrue(edit_tools.delete_path(self.ws, "src/main.py")["ok"])
        self.assertFalse(Path(self.ws, "src/main.py").exists())


class TestGit(unittest.TestCase):
    def test_commit_cycle(self):
        ws = Path(tempfile.mkdtemp())
        self.assertTrue(git_tools.init(ws, "repo")["ok"])
        file_tools.write(ws, "repo/readme.md", "# hi\n")
        commit = git_tools.commit(ws, "repo", "init")
        self.assertTrue(commit["ok"], commit.get("stderr"))
        self.assertTrue(commit["commit"])
        self.assertIn("init", git_tools.log(ws, "repo")["log"])
        again = git_tools.commit(ws, "repo", "noop")
        self.assertTrue(again.get("nothing_to_commit"))

    def test_status_reports_branch(self):
        ws = Path(tempfile.mkdtemp())
        git_tools.init(ws, "repo")
        status = git_tools.status(ws, "repo")
        self.assertTrue(status["ok"])
        self.assertIn("main", status["stdout"])

    def test_not_a_repo_errors_cleanly(self):
        result = git_tools.commit(Path(tempfile.mkdtemp()), ".", "x")
        self.assertFalse(result["ok"])
        self.assertIn("not a git repository", result["error"])

    def test_tokens_are_sanitized(self):
        self.assertEqual(
            git_tools._sanitize("fatal: https://x-access-token:ghp_secretvalue@github.com/o/r.git denied"),
            "fatal: https://***@github.com/o/r.git denied",
        )

    def test_token_injected_into_https_only(self):
        self.assertIn(
            "x-access-token:abc@github.com",
            git_tools._with_token("https://github.com/o/r.git", "abc"),
        )
        self.assertEqual(git_tools._with_token("git@github.com:o/r.git", "abc"), "git@github.com:o/r.git")


class TestHttpTool(unittest.TestCase):
    def test_json_round_trip(self):
        from unittest.mock import MagicMock, patch

        response = MagicMock()
        response.ok = True
        response.status_code = 201
        response.url = "https://api.test/v1/items"
        response.headers = {"content-type": "application/json"}
        response.text = '{"id": 7}'
        response.json.return_value = {"id": 7}
        response.elapsed.total_seconds.return_value = 0.12

        with patch("jarvis.tools.http.requests.request", return_value=response) as request:
            result = http_tools.http_request(
                "https://api.test/v1/items", "POST", headers={"Authorization": "Bearer t"}, json_body={"name": "x"}
            )

        self.assertEqual(result["status"], 201)
        self.assertEqual(result["json"], {"id": 7})
        args, kwargs = request.call_args
        self.assertEqual(args[0], "POST")
        self.assertEqual(args[1], "https://api.test/v1/items")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer t")
        self.assertEqual(kwargs["json"], {"name": "x"})

    def test_network_error_is_reported(self):
        from unittest.mock import patch
        import requests

        with patch("jarvis.tools.http.requests.request", side_effect=requests.ConnectionError("dns fail")):
            result = http_tools.http_request("https://nope.invalid")
        self.assertFalse(result["ok"])
        self.assertIn("dns fail", result["error"])


class TestFullToolset(unittest.TestCase):
    def test_all_expected_tools_registered(self):
        config = temp_config()
        registry = build_registry(config, Memory(config.memory_path))
        names = set(registry.names())
        for expected in (
            "run_javascript", "npm_install", "run_tests", "apply_patch", "grep_files",
            "git_clone", "git_commit", "git_push", "http_request", "delete_file",
        ):
            self.assertIn(expected, names)
        self.assertGreaterEqual(len(names), 38)
        # every tool must have a usable schema
        for schema in registry.schemas():
            self.assertEqual(schema["function"]["parameters"]["type"], "object")


NODE = js_tools.node_binary()
NODE_REASON = "node not installed"


class TestStackParsing(unittest.TestCase):
    SAMPLE = """TypeError: Cannot read properties of undefined (reading 'name')
    at Object.<anonymous> (/app/ws/src/app.js:12:15)
    at Module._compile (node:internal/modules/cjs/loader:1929:14)
    at /app/ws/src/helper.js:4:9
"""

    def test_frames_are_parsed_and_internal_dropped(self):
        frames = diag_tools.parse_stack(self.SAMPLE, Path("/app/ws"))
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0]["file"], "src/app.js")
        self.assertEqual((frames[0]["line"], frames[0]["column"]), (12, 15))
        self.assertTrue(all(f["in_app"] for f in frames))

    def test_harness_frames_are_always_hidden(self):
        stack = "    at x (/app/ws/.jarvis_diag_ab12.mjs:64:5)\n    at y (/app/ws/src/a.js:2:1)\n"
        frames = diag_tools.parse_stack(stack, Path("/app/ws"), include_scratch=True)
        self.assertEqual([f["file"] for f in frames], ["src/a.js"])

    def test_source_context_numbers_lines(self):
        ws = Path(tempfile.mkdtemp())
        file_tools.write(ws, "a.js", "one\ntwo\nthree\nfour\n")
        context = diag_tools.source_context(ws / "a.js", 3, span=1)
        self.assertIn("> | three".replace(">", "3"), context["snippet"])
        self.assertEqual(context["text"], "three")

    def test_error_header_extraction(self):
        header = diag_tools.error_header(self.SAMPLE)
        self.assertEqual(header["name"], "TypeError")


class TestExplanations(unittest.TestCase):
    def explain(self, name, message, code=None, frames=None):
        return diag_tools.explain(
            [{"name": name, "message": message, "code": code}], frames or [], Path("/app/ws")
        )

    def test_missing_package_suggests_install(self):
        text = " ".join(self.explain("Error", "Cannot find module 'axios'", "MODULE_NOT_FOUND"))
        self.assertIn("npm_install", text)

    def test_declared_but_missing_dependency(self):
        ws = Path(tempfile.mkdtemp())
        file_tools.write(ws, "package.json", '{"dependencies": {"zustand": "^4.0.0"}}')
        text = " ".join(diag_tools.explain(
            [{"name": "Error", "message": "Cannot find module 'zustand'", "code": "MODULE_NOT_FOUND"}],
            [], ws,
        ))
        self.assertIn("not installed", text)

    def test_esm_require_explained(self):
        text = " ".join(self.explain("ReferenceError", "require is not defined in ES module scope, you can use import instead"))
        self.assertIn(".cjs", text)

    def test_syntax_error_points_at_node_check(self):
        text = " ".join(self.explain("SyntaxError", "Unexpected token '}'"))
        self.assertIn("node_check", text)

    def test_reference_error_names_symbol(self):
        text = " ".join(self.explain("ReferenceError", "total is not defined"))
        self.assertIn("total", text)

    def test_unhandled_rejection_advice(self):
        text = " ".join(diag_tools.explain(
            [{"name": "Error", "message": "boom", "phase": "unhandledRejection"}], [], None
        ))
        self.assertIn("catch", text.lower())

    def test_unknown_error_gets_generic_advice(self):
        text = " ".join(self.explain("WeirdError", "something odd"))
        self.assertIn("node_diagnose", text)


@unittest.skipUnless(NODE, NODE_REASON)
class TestNodeDiagnose(unittest.TestCase):
    def test_type_error_gets_frame_source_and_advice(self):
        ws = Path(tempfile.mkdtemp())
        file_tools.write(
            ws, "src/app.js",
            "function load() { return undefined; }\nconst data = load();\nconsole.log(data.items);\n",
        )
        result = node_debug_tools.run_diagnosed(ws, path="src/app.js")
        self.assertFalse(result["ok"])
        self.assertEqual(result["errors"][0]["name"], "TypeError")
        self.assertTrue(result["app_frames"])
        frame = result["app_frames"][0]
        self.assertEqual(frame["file"], "src/app.js")
        self.assertEqual(frame["line"], 3)
        self.assertIn("data.items", frame["source"])
        self.assertIn("'items'", result["suggestions"][0])
        self.assertIn("undefined", result["suggestions"][0])

    def test_unhandled_rejection_is_captured(self):
        ws = Path(tempfile.mkdtemp())
        result = node_debug_tools.run_diagnosed(
            ws, code="async function go(){ throw new Error('async boom') }\ngo();\nsetTimeout(()=>{}, 10);\n"
        )
        self.assertFalse(result["ok"])
        phases = [e["phase"] for e in result["errors"]]
        self.assertIn("unhandledRejection", phases)
        self.assertIn("async boom", result["errors"][0]["message"])

    def test_missing_module_reports_code_and_fix(self):
        ws = Path(tempfile.mkdtemp())
        result = node_debug_tools.run_diagnosed(ws, code="require('definitely-not-installed-pkg');")
        self.assertFalse(result["ok"])
        self.assertEqual(result["errors"][0].get("code"), "MODULE_NOT_FOUND")
        self.assertIn("npm_install", " ".join(result["suggestions"]))

    def test_clean_run_reports_ok(self):
        ws = Path(tempfile.mkdtemp())
        result = node_debug_tools.run_diagnosed(ws, code="console.log('fine');")
        self.assertTrue(result["ok"], result.get("errors"))
        self.assertEqual(result["stdout"].strip(), "fine")
        self.assertEqual(result["errors"], [])

    def test_esm_detection_from_package_json(self):
        ws = Path(tempfile.mkdtemp())
        file_tools.write(ws, "app/package.json", '{"type": "module"}')
        file_tools.write(ws, "app/mod.js", "export const value = 7;\nconsole.log('esm', value);\n")
        result = node_debug_tools.run_diagnosed(ws, path="app/mod.js")
        self.assertEqual(result["module_type"], "esm")
        self.assertTrue(result["ok"], result.get("errors"))

    def test_timeout_is_reported_with_advice(self):
        ws = Path(tempfile.mkdtemp())
        result = node_debug_tools.run_diagnosed(ws, code="setInterval(() => {}, 1000);", timeout=2)
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["ok"])
        self.assertIn("didn't exit", result["suggestions"][0])

    def test_scratch_files_are_removed(self):
        ws = Path(tempfile.mkdtemp())
        node_debug_tools.run_diagnosed(ws, code="console.log(1)")
        node_debug_tools.syntax_check(ws, code="const a = 1;")
        leftovers = [p.name for p in ws.rglob(".jarvis_*")]
        self.assertEqual(leftovers, [])


@unittest.skipUnless(NODE, NODE_REASON)
class TestNodeCheck(unittest.TestCase):
    def test_valid_snippet_passes(self):
        result = node_debug_tools.syntax_check(Path(tempfile.mkdtemp()), code="const a = 1; console.log(a);")
        self.assertTrue(result["ok"])

    def test_broken_syntax_reports_line(self):
        ws = Path(tempfile.mkdtemp())
        file_tools.write(ws, "broken.js", "function oops( {\n  return 1;\n}\n")
        result = node_debug_tools.syntax_check(ws, path="broken.js")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["name"], "SyntaxError")
        self.assertEqual(result["location"]["line"], 2)
        self.assertIn("^", result["stderr"])

    def test_missing_file_is_a_clean_error(self):
        result = node_debug_tools.syntax_check(Path(tempfile.mkdtemp()), path="nope.js")
        self.assertFalse(result["ok"])
        self.assertIn("no such file", result["error"])


@unittest.skipUnless(NODE, NODE_REASON)
class TestNodeProfiler(unittest.TestCase):
    def test_cpu_profile_names_the_hot_function(self):
        ws = Path(tempfile.mkdtemp())
        code = (
            "function churn(n){ let s=0; for(let i=0;i<n;i++) s += Math.sqrt(i); return s; }\n"
            "let acc=0; for (let k=0;k<150;k++) acc += churn(40000);\n"
            "console.log('acc', Math.round(acc));\n"
        )
        result = node_debug_tools.cpu_profile(ws, code=code, top=5)
        self.assertTrue(result["ok"], result.get("error"))
        self.assertGreater(result["samples"], 0)
        self.assertTrue(result["hottest"])
        self.assertEqual(result["hottest"][0]["function"], "churn")
        self.assertTrue((ws / result["profile_path"]).is_file())

    def test_heap_profile_returns_allocations(self):
        ws = Path(tempfile.mkdtemp())
        result = node_debug_tools.cpu_profile(
            ws,
            code="const keep=[]; for (let i=0;i<3000;i++) keep.push({i, s:'x'.repeat(40)}); console.log(keep.length);",
            kind="heap",
            top=5,
        )
        self.assertTrue(result["ok"], result.get("error"))
        self.assertGreater(result["total_bytes"], 0)


class TestRunJavaScriptErrorEnrichment(unittest.TestCase):
    @unittest.skipUnless(NODE, NODE_REASON)
    def test_failed_run_includes_suggestions_and_frames(self):
        ws = Path(tempfile.mkdtemp())
        result = js_tools.run_javascript("const cfg = null;\nconsole.log(cfg.port);\n", ws)
        self.assertFalse(result["ok"])
        self.assertIn("port", " ".join(result["suggestions"]))
        self.assertTrue(result["app_frames"])
        self.assertIn("cfg.port", result["app_frames"][0]["source"])
        self.assertIn("cfg.port", (result.get("error_summary") or {}).get("stack", ""))

    @unittest.skipUnless(NODE, NODE_REASON)
    def test_successful_run_has_no_diagnostic_noise(self):
        result = js_tools.run_javascript("console.log('ok')", Path(tempfile.mkdtemp()))
        self.assertTrue(result["ok"])
        self.assertNotIn("suggestions", result)


class TestDebugToolsRegistered(unittest.TestCase):
    def test_debugging_tools_are_wired(self):
        config = temp_config()
        registry = build_registry(config, Memory(config.memory_path))
        names = set(registry.names())
        for expected in ("node_check", "node_diagnose", "node_profile"):
            self.assertIn(expected, names)
        self.assertGreaterEqual(len(names), 43)


class TestRetry(unittest.TestCase):
    def _response(self, status, headers=None):
        from unittest.mock import MagicMock

        response = MagicMock()
        response.status_code = status
        response.headers = headers or {}
        response.text = "body"
        return response

    def test_retries_then_succeeds(self):
        from unittest.mock import patch

        with patch("jarvis.http_retry.time.sleep"), patch(
            "jarvis.http_retry.requests.request", side_effect=[self._response(503), self._response(200)]
        ) as request:
            response, attempts = retry_tools.request_with_retry("GET", "https://x.test")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(attempts, 2)
        self.assertEqual(request.call_count, 2)

    def test_gives_up_after_attempts(self):
        from unittest.mock import patch

        with patch("jarvis.http_retry.time.sleep"), patch(
            "jarvis.http_retry.requests.request", return_value=self._response(500)
        ) as request:
            response, attempts = retry_tools.request_with_retry("GET", "https://x.test", attempts=3)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(attempts, 3)
        self.assertEqual(request.call_count, 3)

    def test_honors_retry_after(self):
        from unittest.mock import patch

        delays = []
        with patch("jarvis.http_retry.time.sleep", side_effect=delays.append), patch(
            "jarvis.http_retry.requests.request",
            side_effect=[self._response(429, {"Retry-After": "2"}), self._response(200)],
        ):
            retry_tools.request_with_retry("GET", "https://x.test")
        self.assertEqual(delays, [2.0])

    def test_post_not_retried_on_server_error(self):
        from unittest.mock import patch

        with patch("jarvis.http_retry.time.sleep") as sleep, patch(
            "jarvis.http_retry.requests.request", return_value=self._response(500)
        ) as request:
            _, attempts = retry_tools.request_with_retry("POST", "https://x.test", attempts=3)
        self.assertEqual(attempts, 1)  # a 5xx write may already have landed
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

    def test_post_retried_on_rate_limit(self):
        from unittest.mock import patch

        with patch("jarvis.http_retry.time.sleep"), patch(
            "jarvis.http_retry.requests.request",
            side_effect=[self._response(429), self._response(201)],
        ):
            response, attempts = retry_tools.request_with_retry("POST", "https://x.test")
        self.assertEqual((response.status_code, attempts), (201, 2))

    def test_network_error_is_retried_then_raised(self):
        import requests as requests_module
        from unittest.mock import patch

        with patch("jarvis.http_retry.time.sleep"), patch(
            "jarvis.http_retry.requests.request",
            side_effect=[requests_module.ConnectionError("dns"), self._response(200)],
        ):
            response, attempts = retry_tools.request_with_retry("GET", "https://x.test")
        self.assertEqual((response.status_code, attempts), (200, 2))

        with patch("jarvis.http_retry.time.sleep"), patch(
            "jarvis.http_retry.requests.request", side_effect=requests_module.ConnectionError("dns")
        ):
            with self.assertRaises(requests_module.ConnectionError):
                retry_tools.request_with_retry("GET", "https://x.test", attempts=2)


class TestRepoDigest(unittest.TestCase):
    def test_parse_github_variants(self):
        self.assertEqual(
            repo_tools.parse_github("https://github.com/psf/requests"),
            {"owner": "psf", "repo": "requests", "ref": None, "subdir": None},
        )
        parsed = repo_tools.parse_github("https://github.com/psf/requests/tree/v2.31.0/src/requests")
        self.assertEqual(parsed["ref"], "v2.31.0")
        self.assertEqual(parsed["subdir"], "src/requests")
        self.assertEqual(repo_tools.parse_github("owner/repo")["repo"], "repo")
        self.assertIsNone(repo_tools.parse_github("./local-folder"))

    def test_skip_rules(self):
        self.assertTrue(repo_tools._skip_path("node_modules/react/index.js"))
        self.assertTrue(repo_tools._skip_path("assets/logo.png"))
        self.assertTrue(repo_tools._skip_path("dist/app.min.js"))
        self.assertFalse(repo_tools._skip_path("src/app.py"))

    def test_local_digest_packs_text_files_only(self):
        ws = Path(tempfile.mkdtemp())
        file_tools.write(ws, "proj/a.py", "def add(a, b):\n    return a + b\n")
        file_tools.write(ws, "proj/notes.md", "# notes\n")
        file_tools.write(ws, "proj/scores.csv", "name,score\nAda,10\n")
        file_tools.write(ws, "proj/node_modules/junk.js", "junk\n")
        result = repo_tools.digest_local("proj", ws)
        self.assertTrue(result["ok"])
        self.assertEqual(sorted(result["files_included"]), ["a.py", "notes.md", "scores.csv"])
        self.assertIn("return a + b", (ws / result["digest_path"]).read_text())

    def test_local_digest_respects_globs(self):
        ws = Path(tempfile.mkdtemp())
        file_tools.write(ws, "proj/a.py", "x = 1\n")
        file_tools.write(ws, "proj/b.md", "# b\n")
        result = repo_tools.digest_local("proj", ws, include=["*.md"])
        self.assertEqual(result["files_included"], ["b.md"])

    def test_github_digest_builds_tree_and_report(self):
        from unittest.mock import patch

        ws = Path(tempfile.mkdtemp())
        meta = {"description": "demo", "default_branch": "main", "stargazers_count": 5,
                "forks_count": 1, "language": "Python", "license": {"spdx_id": "MIT"}}
        tree = {"tree": [
            {"path": "pkg/core.py", "type": "blob", "size": 20},
            {"path": "pkg/util.py", "type": "blob", "size": 20},
            {"path": "node_modules/x/index.js", "type": "blob", "size": 20},
            {"path": "logo.png", "type": "blob", "size": 40},
            {"path": "pkg", "type": "tree"},
        ]}

        def fake_json(path, token):
            return meta if path.startswith("/repos/o/r") and "git/trees" not in path else tree

        with patch.object(repo_tools, "_github_json", side_effect=fake_json), patch.object(
            repo_tools, "_fetch_raw", side_effect=lambda o, r, ref, path, token, limit: f"# {path}\n"
        ):
            result = repo_tools.digest_github("o", "r", workspace=ws)

        self.assertTrue(result["ok"])
        self.assertEqual(result["files_included"], ["pkg/core.py", "pkg/util.py"])
        self.assertEqual(result["source"], "github:o/r@main")
        self.assertIn("pkg/", result["preview"])
        self.assertTrue((ws / result["digest_path"]).is_file())

    def test_github_digest_explains_an_empty_subdir(self):
        from unittest.mock import patch

        ws = Path(tempfile.mkdtemp())
        with patch.object(
            repo_tools,
            "_github_json",
            side_effect=[
                {"default_branch": "main"},
                {"tree": [{"path": "src/a.py", "type": "blob"}, {"path": "docs/b.md", "type": "blob"}]},
            ],
        ):
            result = repo_tools.digest_github("o", "r", subdir="nope", workspace=ws)
        self.assertFalse(result["ok"])
        self.assertIn("no files matched", result["error"])
        self.assertIn("src", result["top_level_entries"])

    def test_code_search_falls_back_without_token(self):
        from unittest.mock import patch

        response = type("R", (), {"status_code": 200, "headers": {}, "text": "",
                                  "json": staticmethod(lambda: {"items": [{"full_name": "a/b", "stargazers_count": 9,
                                                                           "description": "d", "html_url": "u", "language": "Python"}]})})()
        with patch.object(repo_tools, "request_with_retry", return_value=(response, 1)):
            result = repo_tools.github_code_search("query", None)
        self.assertEqual(result["mode"], "repositories")
        self.assertIn("GITHUB_TOKEN", result["note"])
        self.assertEqual(result["results"][0]["repo"], "a/b")


class TestDocumentTools(unittest.TestCase):
    def setUp(self):
        self.ws = Path(tempfile.mkdtemp())

    def test_csv_as_table(self):
        file_tools.write(self.ws, "scores.csv", "name,score\nAda,10\nGrace,9\n")
        result = doc_tools.read_document(self.ws, path="scores.csv", as_table=True)
        self.assertEqual((result["rows"], result["columns"]), (3, 2))
        self.assertEqual(result["header"], ["name", "score"])

    def test_json_is_pretty_printed(self):
        file_tools.write(self.ws, "cfg.json", '{"a": 1, "b": [1, 2]}')
        result = doc_tools.read_document(self.ws, path="cfg.json")
        self.assertEqual(result["kind"], "json")
        self.assertIn('"b": [', result["text"])

    def test_html_is_stripped(self):
        file_tools.write(self.ws, "p.html", "<html><body><h1>Title</h1><script>x()</script><p>Hi</p></body></html>")
        result = doc_tools.read_document(self.ws, path="p.html")
        self.assertEqual(result["kind"], "html")
        self.assertIn("Title", result["text"])
        self.assertNotIn("x()", result["text"])

    def test_docx_paragraphs(self):
        import zipfile

        with zipfile.ZipFile(self.ws / "m.docx", "w") as archive:
            archive.writestr(
                "word/document.xml",
                '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body><w:p><w:r><w:t>First</w:t></w:r></w:p><w:p><w:r><w:t>Second</w:t></w:r></w:p></w:body></w:document>",
            )
        result = doc_tools.read_document(self.ws, path="m.docx")
        self.assertEqual(result["kind"], "docx")
        self.assertEqual(result["paragraphs"], 2)
        self.assertEqual(result["text"], "First\nSecond")

    def test_pdf_pages_and_ranges(self):
        from pypdf import PdfWriter

        writer = PdfWriter()
        for _ in range(3):
            writer.add_blank_page(width=200, height=200)
        with open(self.ws / "doc.pdf", "wb") as handle:
            writer.write(handle)
        full = doc_tools.read_document(self.ws, path="doc.pdf")
        self.assertEqual((full["kind"], full["pages"]), ("pdf", 3))
        partial = doc_tools.read_document(self.ws, path="doc.pdf", pages="2-3")
        self.assertEqual(partial["pages_read"], "2-3")

    def test_page_range_parsing(self):
        self.assertEqual(doc_tools._page_range("3", 10), (2, 3))
        self.assertEqual(doc_tools._page_range("2-5", 10), (1, 5))
        self.assertEqual(doc_tools._page_range("4-", 10), (3, 10))
        self.assertEqual(doc_tools._page_range(None, 10), (0, 10))

    def test_truncation_reported(self):
        file_tools.write(self.ws, "long.txt", "a" * 500)
        result = doc_tools.read_document(self.ws, path="long.txt", max_chars=100)
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["text"]), 100)
        self.assertIn("note", result)

    def test_missing_and_escaping_paths(self):
        self.assertFalse(doc_tools.read_document(self.ws, path="nope.pdf")["ok"])
        self.assertFalse(doc_tools.read_document(self.ws, path="../../etc/passwd")["ok"])


class TestAgentUpgrades(unittest.TestCase):
    def _agent(self, **config):
        ws = Path(tempfile.mkdtemp())
        conf = temp_config()
        conf.workspace = ws
        conf.api_key = "test"
        for key, value in config.items():
            setattr(conf, key, value)
        from jarvis.agent import Agent

        return Agent(conf)

    def test_token_estimate_grows_with_content(self):
        from jarvis.agent import estimate_tokens

        small = [{"role": "user", "content": "hi"}]
        big = [{"role": "user", "content": "hi" * 500}]
        self.assertLess(estimate_tokens(small), estimate_tokens(big))

    def test_history_untouched_under_budget(self):
        agent = self._agent(context_budget=50_000)
        agent.history = [{"role": "user", "content": "task"}] + [
            {"role": "tool", "tool_call_id": "1", "name": "read_file", "content": "x" * 100}
        ]
        self.assertEqual(agent._trimmed(), agent.history)

    def test_compaction_keeps_opening_request_and_recent_steps(self):
        agent = self._agent(context_budget=1500)
        agent.history = [{"role": "user", "content": "original objective"}]
        for index in range(50):
            agent.history.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{index}", "name": "read_file", "arguments": {"path": "a.py"}}]})
            agent.history.append({"role": "tool", "tool_call_id": f"c{index}", "name": "read_file", "content": "z" * 400})

        from jarvis.agent import estimate_tokens

        trimmed = agent._trimmed()
        self.assertLess(estimate_tokens(trimmed), estimate_tokens(agent.history))
        self.assertEqual(trimmed[0]["content"], "original objective")
        self.assertIn("compacted", trimmed[1]["content"])
        self.assertEqual(trimmed[-1]["name"], "read_file")
        self.assertNotEqual(trimmed[2]["role"], "tool")

    def test_context_stats_shape(self):
        agent = self._agent()
        stats = agent.context_stats()
        self.assertEqual(set(stats), {"messages", "est_tokens", "budget", "steps"})

    def _timed_calls(self, tools_safe=True):
        import time
        from jarvis.registry import Registry

        registry = Registry()
        name = "web_search" if tools_safe else "write_file"

        @registry.tool(name, "slow", {"type": "object", "properties": {}})
        def slow(**_kwargs):
            time.sleep(0.2)
            return "ok"

        agent = self._agent()
        agent.registry = registry
        agent.llm = object()
        calls = [type("C", (), {"id": f"i{n}", "name": name, "arguments": {}})() for n in range(3)]
        start = time.monotonic()
        results = agent._run_tool_calls(calls, lambda event: None)
        return time.monotonic() - start, results

    def test_read_only_tools_run_concurrently(self):
        elapsed, results = self._timed_calls(tools_safe=True)
        self.assertLess(elapsed, 0.45)
        self.assertEqual([out for _, out in results], ["ok", "ok", "ok"])

    def test_mutating_tools_stay_sequential(self):
        elapsed, results = self._timed_calls(tools_safe=False)
        self.assertGreaterEqual(elapsed, 0.55)
        self.assertEqual(len(results), 3)


class TestNewToolsRegistered(unittest.TestCase):
    def test_repo_and_document_tools_are_wired(self):
        config = temp_config()
        registry = build_registry(config, Memory(config.memory_path))
        names = set(registry.names())
        for expected in ("repo_digest", "github_code_search", "read_document"):
            self.assertIn(expected, names)
        self.assertGreaterEqual(len(names), 46)


class TestNodeMemory(unittest.TestCase):
    def _samples(self, sizes, external=0.0):
        offheap = external if isinstance(external, list) else [external] * len(sizes)
        return [
            {"t": index * 100, "heapUsed": size * 1024 * 1024, "rss": (size + 40) * 1024 * 1024,
             "heapTotal": (size + 5) * 1024 * 1024, "external": offheap[index] * 1024 * 1024,
             "arrayBuffers": 0}
            for index, size in enumerate(sizes)
        ]

    def test_climbing_heap_is_a_leak(self):
        result = clone_tools.analyse(self._samples([10, 12, 14, 16, 18, 20, 22, 24]))
        self.assertEqual(result["verdict"], "leak suspected")
        self.assertEqual(result["growth_area"], "heap")
        self.assertGreater(result["slope_mb_per_s"], 1)

    def test_climbing_buffers_are_caught_off_heap(self):
        result = clone_tools.analyse(self._samples([10] * 8, external=[2, 6, 12, 18, 24, 30, 36, 42]))
        self.assertEqual(result["verdict"], "leak suspected")
        self.assertEqual(result["growth_area"], "buffers/external")
        self.assertIn("Buffer", result["reason"])

    def test_flat_memory_is_stable(self):
        result = clone_tools.analyse(self._samples([10, 10.1, 9.9, 10.05, 10.02, 10.1, 10.0, 10.05]))
        self.assertEqual(result["verdict"], "stable")

    def test_short_run_reports_no_trend(self):
        result = clone_tools.analyse(self._samples([10, 11]))
        self.assertEqual(result["verdict"], "no trend")
        self.assertIn("too short", result["reason"])

    def _snapshot_file(self, path, entries):
        """entries: list of (type, name, bytes)"""
        meta = {
            "node_fields": ["type", "name", "id", "self_size", "edge_count", "trace_node_id", "detachedness"],
            "node_types": [["hidden", "array", "string", "object", "code", "closure", "regexp", "number",
                            "native", "synthetic", "concatenated string", "sliced string", "symbol", "bigint"],
                           "string", "number", "number", "number", "number", "number"],
        }
        strings = ["(root)"]
        index = {}
        nodes = []
        for position, (type_name, name, size) in enumerate(entries):
            if name not in index:
                index[name] = len(strings)
                strings.append(name)
            type_index = meta["node_types"][0].index(type_name)
            nodes.extend([type_index, index[name], position + 1, size, 0, 0, 0])
        Path(path).write_text(json.dumps({"snapshot": {"meta": meta}, "nodes": nodes, "strings": strings}))
        return Path(path)

    def test_snapshot_diff_names_the_growing_objects(self):
        import tempfile as temp

        directory = Path(temp.mkdtemp())
        first = self._snapshot_file(directory / "a.heapsnapshot", [("object", "Object", 100)])
        last = self._snapshot_file(
            directory / "b.heapsnapshot",
            [("object", "Object", 100)] + [("object", "LeakyRow", 4096)] * 5 + [("code", "system / Map", 90000)],
        )
        comparison = clone_tools.compare_snapshots(
            clone_tools.load_snapshot(first), clone_tools.load_snapshot(last)
        )
        self.assertTrue(comparison["compared"])
        self.assertEqual(comparison["suspects"][0]["name"], "LeakyRow")
        self.assertEqual(comparison["suspects"][0]["count_delta"], 5)
        self.assertGreater(comparison["retained_growth_kb"], 15)
        self.assertEqual(comparison["internals_hidden"], 1)  # V8 internals kept out of the way

    def test_long_names_are_compacted(self):
        import tempfile as temp

        directory = Path(temp.mkdtemp())
        huge = "x" * 5000
        snapshot = clone_tools.load_snapshot(
            self._snapshot_file(directory / "c.heapsnapshot", [("string", huge, 5000)])
        )
        name = next(iter(snapshot["by_name"].values()))["name"]
        self.assertLessEqual(len(name), 83)
        self.assertTrue(name.endswith("..."))

    def _profile(self, code, **kwargs):
        workspace = Path(tempfile.mkdtemp())
        result = clone_tools.memory_profile(workspace, code=code, **kwargs)
        leftovers = [p.name for p in workspace.iterdir() if p.name.startswith(".jarvis_mem")]
        return result, leftovers, workspace

    def test_profile_detects_a_running_leak_and_cleans_up(self):
        code = (
            "const stash = [];\n"
            "setInterval(() => { for (let i = 0; i < 300; i++) stash.push({ i, pad: 'x'.repeat(1024) }); }, 10);\n"
        )
        result, leftovers, _ = self._profile(code, seconds=2, snapshots=False)
        self.assertEqual(leftovers, [])
        self.assertEqual(result["verdict"], "leak suspected")
        self.assertGreater(result["growth_mb"], 0.5)
        self.assertTrue(result["stopped_by_limit"])

    def test_profile_keeps_buffer_leaks_visible(self):
        code = (
            "const cache = new Map();\n"
            "let i = 0;\n"
            "setInterval(() => { cache.set(i++, Buffer.alloc(4096)); }, 5);\n"
        )
        result, leftovers, _ = self._profile(code, seconds=2, snapshots=True)
        self.assertEqual(leftovers, [])
        self.assertEqual(result["verdict"], "leak suspected")
        self.assertEqual(result["growth_area"], "buffers/external")
        self.assertTrue(result["snapshots"]["compared"])
        self.assertTrue(result["snapshots"]["suspects"])

    def test_profile_stable_script_is_not_flagged(self):
        code = (
            "let total = 0;\n"
            "const timer = setInterval(() => { total += Math.random(); }, 20);\n"
            "setTimeout(() => clearInterval(timer), 1500);\n"
        )
        result, leftovers, _ = self._profile(code, seconds=3, snapshots=False)
        self.assertEqual(leftovers, [])
        self.assertNotEqual(result["verdict"], "leak suspected")
