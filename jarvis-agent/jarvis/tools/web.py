"""Web tools: keyless search (Bing/Tavily/Brave) and page reading."""

from __future__ import annotations

import base64
import html as htmlmod
import re
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _decode_bing_url(href: str) -> str:
    """Unwrap Bing's /ck/a?...&u=a1<base64> redirector."""
    if not href:
        return href
    if "bing.com/ck/a" not in href:
        return href
    match = re.search(r"[?&]u=a1([A-Za-z0-9_\-=]+)", href)
    if not match:
        return href
    token = match.group(1)
    padded = token + "=" * (-len(token) % 4)
    try:
        return base64.urlsafe_b64decode(padded).decode("utf-8", "ignore")
    except Exception:
        return href


def _clean(text: str) -> str:
    text = htmlmod.unescape(text or "")
    return re.sub(r"\s+", " ", text).strip()


def _bing_search(session: requests.Session, query: str, count: int) -> list[dict]:
    resp = session.get(
        "https://www.bing.com/search",
        params={"q": query, "count": max(count, 10), "setlang": "en"},
        timeout=20,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[dict] = []
    for li in soup.select("li.b_algo"):
        anchor = li.select_one("h2 a") or li.find("a")
        if not anchor:
            continue
        url = _decode_bing_url(anchor.get("href", ""))
        if not url.startswith("http"):
            cite = li.select_one("cite")
            if cite:
                url = "https://" + _clean(cite.get_text()).split(" ")[0]
        snippet_el = li.select_one("p") or li.select_one(".b_caption")
        results.append(
            {
                "title": _clean(anchor.get_text()),
                "url": url,
                "snippet": _clean(snippet_el.get_text() if snippet_el else ""),
            }
        )
        if len(results) >= count:
            break
    return results


def _tavily_search(api_key: str, query: str, count: int) -> list[dict]:
    resp = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": count, "search_depth": "basic"},
        timeout=25,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")[:400]}
        for r in data.get("results", [])
    ]


def _brave_search(api_key: str, query: str, count: int) -> list[dict]:
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": count},
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        timeout=25,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "title": htmlmod.unescape(r.get("title", "")),
            "url": r.get("url", ""),
            "snippet": _clean(BeautifulSoup(r.get("description", ""), "html.parser").get_text()),
        }
        for r in (data.get("web", {}) or {}).get("results", [])
    ]


def search_web(query: str, count: int = 6, provider: str = "bing", api_key: str | None = None) -> dict:
    session = _session()
    try:
        if provider == "tavily" and api_key:
            results = _tavily_search(api_key, query, count)
        elif provider == "brave" and api_key:
            results = _brave_search(api_key, query, count)
        else:
            results = _bing_search(session, query, count)
            if not results:  # one retry with the lite endpoint
                resp = session.get("https://lite.duckduckgo.com/lite/", params={"q": query}, timeout=20)
                soup = BeautifulSoup(resp.text, "html.parser")
                results = [
                    {
                        "title": _clean(a.get_text()),
                        "url": _decode_bing_url(a.get("href", "")),
                        "snippet": _clean(a.find_parent("tr").get_text() if a.find_parent("tr") else ""),
                    }
                    for a in soup.select("a.result-link")[:count]
                ]
    except requests.RequestException as exc:
        return {"query": query, "error": f"search failed: {exc}", "results": []}
    return {"query": query, "count": len(results), "results": results}


def html_to_text(html: str, max_chars: int = 20000) -> str:
    """Strip a page down to readable text (used by fetch_url and read_document)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "svg", "iframe"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text("\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return re.sub(r"\n{3,}", "\n\n", text)[:max_chars]


def fetch_page(url: str, max_chars: int = 8000, raw: bool = False) -> dict:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    session = _session()
    try:
        resp = session.get(url, timeout=25, allow_redirects=True)
    except requests.RequestException as exc:
        return {"url": url, "error": f"fetch failed: {exc}"}

    ctype = resp.headers.get("content-type", "")
    out: dict = {
        "url": resp.url,
        "status": resp.status_code,
        "content_type": ctype.split(";")[0],
    }
    if resp.status_code >= 400:
        out["error"] = f"HTTP {resp.status_code}"

    if "html" in ctype or resp.text.lstrip().startswith("<"):
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "svg", "iframe"]):
            tag.decompose()
        out["title"] = _clean(soup.title.get_text()) if soup.title else ""
        main = soup.find("article") or soup.find("main") or soup.body or soup
        text = main.get_text("\n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        text = re.sub(r"\n{3,}", "\n\n", text)
        if raw:
            out["html"] = resp.text[:max_chars]
        out["text"] = text[:max_chars]
        out["truncated"] = len(text) > max_chars
    else:
        body = resp.text if "text" in ctype or "json" in ctype or "xml" in ctype else f"<{ctype} binary>"
        out["text"] = body[:max_chars]
        out["truncated"] = len(body) > max_chars
    return out


def register(registry, config) -> None:
    @registry.tool(
        "web_search",
        "Search the live web. Returns titles, URLs and snippets. Use this whenever the "
        "answer depends on current information, then read promising results with fetch_url.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query. Supports operators like site:github.com"},
                "count": {"type": "integer", "description": "Number of results (1-10)", "default": 6},
            },
            "required": ["query"],
        },
    )
    def web_search(query: str, count: int = 6):
        return search_web(query, count, config.search_provider, config.search_api_key)

    @registry.tool(
        "fetch_url",
        "Fetch a URL and return its readable text (HTML is stripped to content). "
        "Use for documentation, articles, GitHub pages, raw files and APIs.",
        {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute http(s) URL"},
                "max_chars": {"type": "integer", "description": "Text cap", "default": 8000},
            },
            "required": ["url"],
        },
    )
    def fetch_url(url: str, max_chars: int = 8000):
        return fetch_page(url, max_chars)


__all__ = ["register", "search_web", "fetch_page", "urlparse", "unquote"]
