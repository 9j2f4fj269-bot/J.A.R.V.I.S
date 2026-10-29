"""Generic HTTP client tool: call any API, not just the ones with dedicated tools."""

from __future__ import annotations

import json

import requests

from ..http_retry import request_with_retry

UA = "jarvis-agent/1.0"


def http_request(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    json_body: dict | list | None = None,
    data: str | None = None,
    params: dict | None = None,
    timeout: int = 30,
    max_chars: int = 12000,
    follow_redirects: bool = True,
    retries: int = 3,
) -> dict:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    method = method.upper()
    send_headers = {"User-Agent": UA, "Accept": "application/json, text/plain, */*"}
    send_headers.update(headers or {})
    if json_body is not None and "Content-Type" not in send_headers:
        send_headers["Content-Type"] = "application/json"

    try:
        resp, attempts = request_with_retry(
            method,
            url,
            attempts=retries,
            headers=send_headers,
            params=params,
            json=json_body,
            data=data,
            timeout=max(1, min(timeout, 120)),
            allow_redirects=follow_redirects,
        )
    except requests.RequestException as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "url": url, "method": method}

    ctype = resp.headers.get("content-type", "")
    out: dict = {
        "ok": resp.ok,
        "status": resp.status_code,
        "url": resp.url,
        "method": method,
        "content_type": ctype.split(";")[0],
        "elapsed_ms": int(resp.elapsed.total_seconds() * 1000),
        "attempts": attempts,
    }
    retry_after = resp.headers.get("retry-after")
    if retry_after:
        out["retry_after"] = retry_after
    remaining = resp.headers.get("x-ratelimit-remaining")
    if remaining is not None:
        out["rate_limit_remaining"] = remaining
    if "application/json" in ctype:
        try:
            out["json"] = resp.json()
            out["body"] = json.dumps(out["json"])[:max_chars]
            return out
        except ValueError:
            pass
    body = resp.text or ""
    out["body"] = body[:max_chars]
    out["truncated"] = len(body) > max_chars
    if not resp.ok:
        out["hint"] = "non-2xx: check auth header, payload shape, or endpoint"
    return out


def register(registry, config) -> None:
    @registry.tool(
        "http_request",
        "Make an authenticated HTTP request to any API (REST, GraphQL, webhooks): GET/POST/PUT/"
        "PATCH/DELETE with headers, query params and a JSON body. Returns status plus parsed JSON "
        "or text. Use this when no dedicated tool covers the service.",
        {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"], "default": "GET"},
                "headers": {"type": "object", "description": "e.g. {\"Authorization\": \"Bearer ...\"}"},
                "json_body": {"type": "object", "description": "JSON request body"},
                "data": {"type": "string", "description": "Raw body when not JSON"},
                "params": {"type": "object", "description": "Query string parameters"},
                "timeout": {"type": "integer", "default": 30},
                "max_chars": {"type": "integer", "default": 12000},
                "retries": {"type": "integer", "default": 3, "description": "Retries on 429/5xx with backoff"},
            },
            "required": ["url"],
        },
    )
    def http_request_tool(
        url: str,
        method: str = "GET",
        headers: dict | None = None,
        json_body: dict | list | None = None,
        data: str | None = None,
        params: dict | None = None,
        timeout: int = 30,
        max_chars: int = 12000,
        retries: int = 3,
    ):
        return http_request(url, method, headers, json_body, data, params, timeout, max_chars, retries)
