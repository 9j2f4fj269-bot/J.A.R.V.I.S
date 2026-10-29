"""YouTube tools: search, video metadata, and transcripts (no API key needed)."""

from __future__ import annotations

import json
import re
from urllib.parse import parse_qs, urlparse

import requests

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Accept-Language": "en-US,en;q=0.9",
    "Cookie": "CONSENT=YES+cb; SOCS=CAI",
}
VIDEO_ID_RE = re.compile(r"(?:v=|/shorts/|youtu\.be/|/embed/|/live/)([A-Za-z0-9_-]{11})")


def video_id_of(value: str) -> str:
    value = (value or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    match = VIDEO_ID_RE.search(value)
    if match:
        return match.group(1)
    parsed = urlparse(value if "//" in value else "https://" + value)
    if parsed.query:
        v = parse_qs(parsed.query).get("v")
        if v:
            return v[0]
    return ""


def _initial_data(html: str) -> dict:
    match = re.search(r"var ytInitialData\s*=\s*(\{.+?\});</script>", html, re.S)
    if not match:
        match = re.search(r'window\["ytInitialData"\]\s*=\s*(\{.+?\});', html, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def _walk(node, key: str, found: list, limit: int = 200) -> None:
    if len(found) >= limit:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key and isinstance(v, dict):
                found.append(v)
            else:
                _walk(v, key, found, limit)
    elif isinstance(node, list):
        for item in node:
            _walk(item, key, found, limit)


def _runs_text(node) -> str:
    if not isinstance(node, dict):
        return ""
    if "simpleText" in node:
        return node["simpleText"]
    return "".join(run.get("text", "") for run in node.get("runs", []) if isinstance(run, dict))


def search_youtube(query: str, count: int = 8) -> dict:
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        resp = session.get(
            "https://www.youtube.com/results",
            params={"search_query": query, "hl": "en", "gl": "US"},
            timeout=25,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        return {"query": query, "error": f"youtube search failed: {exc}", "results": []}

    data = _initial_data(resp.text)
    found: list[dict] = []
    _walk(data, "videoRenderer", found)

    results = []
    for item in found:
        vid = item.get("videoId")
        if not vid:
            continue
        title = _runs_text(item.get("title", {}))
        channel = _runs_text((item.get("ownerText") or item.get("longBylineText") or {}))
        results.append(
            {
                "title": title,
                "video_id": vid,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "channel": channel,
                "duration": (item.get("lengthText") or {}).get("simpleText", ""),
                "views": (item.get("viewCountText") or {}).get("simpleText", ""),
                "published": (item.get("publishedTimeText") or {}).get("simpleText", ""),
            }
        )
        if len(results) >= count:
            break
    return {"query": query, "count": len(results), "results": results}


def oembed(video_id: str) -> dict:
    """Public metadata endpoint that survives most IP blocks."""
    vid = video_id_of(video_id)
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={vid}", "format": "json"},
            headers={"User-Agent": UA},
            timeout=15,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
    except (requests.RequestException, ValueError):
        return {}
    return {
        "title": data.get("title", ""),
        "channel": data.get("author_name", ""),
        "channel_url": data.get("author_url", ""),
        "thumbnail": data.get("thumbnail_url", ""),
    }


def video_info(video_id: str) -> dict:
    vid = video_id_of(video_id)
    if not vid:
        return {"error": "could not resolve a video id"}

    info: dict = {"video_id": vid, "url": f"https://www.youtube.com/watch?v={vid}"}
    info.update(oembed(vid))

    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        resp = session.get(f"https://www.youtube.com/watch?v={vid}", timeout=25)
    except requests.RequestException as exc:
        info["note"] = f"watch page unreachable ({exc}); metadata from oEmbed only"
        return info

    html = resp.text
    for label, pattern in (
        ("description", r'"shortDescription":"(.*?)","isCrawlable"'),
        ("length_seconds", r'"lengthSeconds":"(\d+)"'),
        ("view_count", r'"viewCount":"(\d+)"'),
        ("keywords", r'"keywords":\[(.*?)\]'),
    ):
        match = re.search(pattern, html, re.S)
        if match:
            value = match.group(1)
            try:
                value = json.loads(f'"{value}"')
            except json.JSONDecodeError:
                pass
            info[label] = value

    if "description" not in info:
        match = re.search(r'<meta name="description" content="([^"]*)"', html)
        if match:
            info["description"] = match.group(1)
        else:
            info["note"] = (
                "YouTube served a reduced page to this IP, so description/view counts were unavailable "
                "(title and channel came from oEmbed)."
            )

    tracks = _caption_tracks(html)
    info["captions_available"] = bool(tracks)
    if tracks:
        info["caption_languages"] = sorted({t.get("languageCode", "") for t in tracks if t.get("languageCode")})
    return info


def _caption_tracks(html: str) -> list[dict]:
    match = re.search(r'"captionTracks":(\[.*?\])', html, re.S)
    if not match:
        return []
    try:
        tracks = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return [t for t in tracks if isinstance(t, dict) and t.get("baseUrl")]


def _parse_json3(payload: str) -> str:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ""
    chunks: list[str] = []
    for event in data.get("events") or []:
        segs = event.get("segs") or []
        text = "".join(seg.get("utf8", "") for seg in segs if isinstance(seg, dict))
        if text.strip():
            chunks.append(text.strip())
    return " ".join(chunks)


def _parse_xml(payload: str) -> str:
    texts = re.findall(r"<text[^>]*>(.*?)</text>", payload, re.S)
    cleaned = [re.sub(r"<[^>]+>", "", t) for t in texts]
    import html as htmlmod

    return " ".join(htmlmod.unescape(t).strip() for t in cleaned if t.strip())


INNERTUBE_KEY = "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"


def _innertube_tracks(session, vid: str) -> tuple[list[dict], str]:
    """Second route to captions: the InnerTube player API."""
    payload = {
        "context": {
            "client": {"clientName": "WEB", "clientVersion": "2.20241010.00.00", "hl": "en", "gl": "US"}
        },
        "videoId": vid,
        "contentCheckOk": True,
        "racyCheckOk": True,
    }
    try:
        resp = session.post(
            "https://www.youtube.com/youtubei/v1/player",
            params={"key": INNERTUBE_KEY},
            json=payload,
            timeout=20,
        )
        if resp.status_code != 200:
            return [], f"http {resp.status_code}"
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    status = (data.get("playabilityStatus") or {}).get("status") or "unknown"
    renderer = (data.get("captions") or {}).get("playerCaptionsTracklistRenderer") or {}
    return [t for t in (renderer.get("captionTracks") or []) if t.get("baseUrl")], status


def _pick_track(tracks: list[dict], lang: str | None) -> dict:
    if lang:
        match = next((t for t in tracks if t.get("languageCode", "").startswith(lang)), None)
        if match:
            return match
    manual = [t for t in tracks if t.get("kind") != "asr"]
    return (manual or tracks)[0]


def transcript(video_id: str, lang: str | None = None, max_chars: int = 12000) -> dict:
    vid = video_id_of(video_id)
    if not vid:
        return {"error": "could not resolve a video id"}

    session = requests.Session()
    session.headers.update(HEADERS)
    tracks: list[dict] = []
    status = "unknown"

    try:
        resp = session.get(f"https://www.youtube.com/watch?v={vid}", timeout=25)
        if resp.status_code == 200:
            tracks = _caption_tracks(resp.text)
            match = re.search(r'"playabilityStatus":\{"status":"(\w+)"', resp.text)
            if match:
                status = match.group(1)
    except requests.RequestException as exc:
        status = f"watch page error: {exc}"

    if not tracks:
        tracks, inner_status = _innertube_tracks(session, vid)
        if inner_status != "unknown":
            status = inner_status

    if not tracks:
        blocked = status in {"LOGIN_REQUIRED", "ERROR", "UNPLAYABLE"}
        hint = (
            "YouTube is refusing to expose captions to this IP (bot check / LOGIN_REQUIRED). "
            "This is an IP reputation limit, not a code bug: it usually works from a residential "
            "connection, or set HTTPS_PROXY to route through any proxy. Meanwhile use "
            "youtube_info for the title and description, or youtube_search to find the video."
            if blocked
            else "No caption tracks are published for this video (captions disabled by the uploader)."
        )
        return {"video_id": vid, "url": f"https://www.youtube.com/watch?v={vid}", "error": hint,
                "playability_status": status, "captions_available": False}

    chosen = _pick_track(tracks, lang)
    base_url = chosen["baseUrl"]
    joiner = "&" if "?" in base_url else "?"
    text = ""
    for fmt in ("json3", "srv1"):
        try:
            cap = session.get(f"{base_url}{joiner}fmt={fmt}", timeout=25)
            if cap.status_code != 200 or not cap.text.strip():
                continue
        except requests.RequestException:
            continue
        text = _parse_json3(cap.text) if fmt == "json3" else _parse_xml(cap.text)
        if text:
            break

    if not text:
        return {"video_id": vid, "error": "caption track fetched but came back empty",
                "language": chosen.get("languageCode"), "captions_available": True}

    available = sorted({t.get("languageCode", "") for t in tracks if t.get("languageCode")})
    return {
        "video_id": vid,
        "url": f"https://www.youtube.com/watch?v={vid}",
        "language": chosen.get("languageCode"),
        "auto_generated": chosen.get("kind") == "asr",
        "languages_available": available,
        "chars": len(text),
        "transcript": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def register(registry, config) -> None:
    @registry.tool(
        "youtube_search",
        "Search YouTube. Returns video ids, titles, channels, duration and view counts.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms"},
                "count": {"type": "integer", "description": "Results (1-20)", "default": 8},
            },
            "required": ["query"],
        },
    )
    def youtube_search(query: str, count: int = 8):
        return search_youtube(query, count)

    @registry.tool(
        "youtube_transcript",
        "Get the spoken transcript of a YouTube video so you can summarize, quote or reason about "
        "its content. Accepts a video id or any YouTube URL.",
        {
            "type": "object",
            "properties": {
                "video": {"type": "string", "description": "Video id or URL"},
                "lang": {"type": "string", "description": "Preferred language code, e.g. en"},
                "max_chars": {"type": "integer", "description": "Transcript cap", "default": 12000},
            },
            "required": ["video"],
        },
    )
    def youtube_transcript(video: str, lang: str | None = None, max_chars: int = 12000):
        return transcript(video, lang, max_chars)

    @registry.tool(
        "youtube_info",
        "Metadata for one YouTube video: title, channel, description, duration, views.",
        {
            "type": "object",
            "properties": {"video": {"type": "string", "description": "Video id or URL"}},
            "required": ["video"],
        },
    )
    def youtube_info(video: str):
        return video_info(video)
