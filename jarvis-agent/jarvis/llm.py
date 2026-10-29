"""Provider-agnostic LLM client.

Works with any OpenAI-compatible endpoint (OpenAI, Groq, Together, OpenRouter,
Ollama, LM Studio) and with Anthropic. No SDKs required, just `requests`.

Portable message format used internally:
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [{"id", "name", "arguments"}]}
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import requests


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class _HTTP:
    """Shared POST with retries on 429/5xx."""

    def __init__(self, timeout: int = 120, retries: int = 2):
        self.timeout = timeout
        self.retries = retries

    def post(self, url: str, headers: dict, payload: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:  # network level
                last = exc
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                last = LLMError(f"{resp.status_code}: {resp.text[:300]}")
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code >= 400:
                raise LLMError(f"LLM HTTP {resp.status_code}: {resp.text[:600]}")
            try:
                return resp.json()
            except ValueError as exc:
                raise LLMError(f"LLM returned non-JSON: {resp.text[:300]}") from exc
        raise LLMError(f"LLM request failed after retries: {last}")


def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"__raw__": str(raw)}


class LLM:
    """Unified chat interface. `tools` uses the OpenAI function-schema shape."""

    def __init__(self, config):
        self.config = config
        self.http = _HTTP()
        if not config.has_llm:
            raise LLMError(
                "No LLM credential found. Set OPENAI_API_KEY (or ANTHROPIC_API_KEY), "
                "or point OPENAI_BASE_URL at a local server such as Ollama."
            )

    # -- public --------------------------------------------------------
    def chat(self, messages: list[dict], tools: list[dict], system: str | None = None) -> Turn:
        if self.config.provider == "anthropic":
            return self._anthropic(messages, tools, system)
        return self._openai(messages, tools, system)

    # -- OpenAI compatible --------------------------------------------
    def _openai(self, messages: list[dict], tools: list[dict], system: str | None) -> Turn:
        payload_messages: list[dict] = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        for msg in messages:
            role = msg["role"]
            if role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
                calls = msg.get("tool_calls") or []
                if calls:
                    entry["content"] = msg.get("content") or None
                    entry["tool_calls"] = [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
                        }
                        for c in calls
                    ]
                payload_messages.append(entry)
            elif role == "tool":
                payload_messages.append(
                    {"role": "tool", "tool_call_id": msg.get("tool_call_id") or "", "content": msg.get("content", "")}
                )
            else:
                payload_messages.append({"role": "user", "content": msg.get("content", "")})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": payload_messages,
            "temperature": self.config.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        data = self.http.post(url, headers, payload)

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = [
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=(tc.get("function") or {}).get("name", ""),
                arguments=_parse_args((tc.get("function") or {}).get("arguments")),
            )
            for i, tc in enumerate(message.get("tool_calls") or [])
        ]
        return Turn(text=message.get("content") or "", tool_calls=calls, usage=data.get("usage") or {})

    # -- Anthropic -----------------------------------------------------
    def _anthropic(self, messages: list[dict], tools: list[dict], system: str | None) -> Turn:
        converted: list[dict] = []
        for msg in messages:
            role = msg["role"]
            if role == "assistant":
                blocks: list[dict] = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": msg["content"]})
                for call in msg.get("tool_calls") or []:
                    blocks.append(
                        {"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["arguments"]}
                    )
                if blocks:
                    converted.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id") or "",
                    "content": msg.get("content", ""),
                }
                if converted and converted[-1]["role"] == "user" and isinstance(converted[-1]["content"], list):
                    converted[-1]["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})
            else:
                converted.append({"role": "user", "content": [{"type": "text", "text": msg.get("content", "")}]})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": converted,
            "max_tokens": 4096,
            "temperature": self.config.temperature,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}},
                }
                for t in tools
            ]

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key or "",
            "anthropic-version": "2023-06-01",
        }
        url = self.config.base_url.rstrip("/") + "/v1/messages"
        data = self.http.post(url, headers, payload)

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(id=block.get("id", ""), name=block.get("name", ""), arguments=block.get("input") or {}))
        return Turn(text="".join(text_parts), tool_calls=calls, usage=data.get("usage") or {})
