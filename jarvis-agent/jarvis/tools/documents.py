"""Document reading: PDF, DOCX, HTML, CSV/TSV, JSON, Markdown and plain text.

Most real tasks arrive as files, not as web pages. This turns them into text the
model can actually reason about. PDF support uses pypdf when installed (optional);
DOCX is parsed with the standard library so it works with zero extra dependencies.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import requests

from ..http_retry import request_with_retry

DOC_EXT = {".pdf", ".docx", ".html", ".htm", ".csv", ".tsv", ".json", ".md", ".txt", ".log", ".rst"}
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _wire(path: Path) -> str:
    return path.as_posix()


def _safe(workspace: Path, path: str) -> Path:
    root = Path(workspace).resolve()
    target = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"'{path}' is outside the workspace")
    return target


def _page_range(pages: str | None, total: int) -> tuple[int, int]:
    """Accept '3', '2-5', or '2-' and return zero-indexed (start, end) bounds."""
    if not pages:
        return 0, total
    text = str(pages).strip()
    if "-" in text:
        start_s, _, end_s = text.partition("-")
        start = max(1, int(start_s or 1))
        end = total if not end_s.strip() else min(total, int(end_s))
    else:
        start = end = max(1, int(text))
    return start - 1, max(start, end)


def pdf_text(data: bytes, pages: str | None = None) -> dict:
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - dependency is optional
        return {
            "ok": False,
            "error": "pypdf is not installed; run: pip install pypdf",
            "hint": "or convert with a tool that outputs text and read that instead",
        }
    reader = PdfReader(io.BytesIO(data))
    total = len(reader.pages)
    start, end = _page_range(pages, total)
    chunks = []
    for index in range(start, end):
        try:
            chunks.append(f"\n--- page {index + 1} ---\n{reader.pages[index].extract_text() or ''}")
        except Exception as exc:  # noqa: BLE001 - a single bad page shouldn't kill the read
            chunks.append(f"\n--- page {index + 1} ---\n[unreadable: {exc}]")
    meta = {}
    try:
        meta = dict(reader.metadata or {})
    except Exception:  # noqa: BLE001
        meta = {}
    return {
        "ok": True,
        "kind": "pdf",
        "pages": total,
        "pages_read": f"{start + 1}-{end}",
        "title": str(meta.get("/Title") or "").strip() or None,
        "text": "".join(chunks).strip(),
    }


def docx_text(data: bytes) -> dict:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        return {"ok": False, "error": f"not a readable .docx: {exc}"}
    root = ElementTree.fromstring(xml)
    paragraphs = []
    for para in root.iter(f"{WORD_NS}p"):
        text = "".join(node.text or "" for node in para.iter(f"{WORD_NS}t"))
        if text.strip():
            paragraphs.append(text)
    return {"ok": True, "kind": "docx", "paragraphs": len(paragraphs), "text": "\n".join(paragraphs)}


def html_text(html: str) -> dict:
    from .web import html_to_text  # reuse the stripper used by fetch_url

    return {"ok": True, "kind": "html", "text": html_to_text(html)}


def table_text(raw: str, delimiter: str) -> dict:
    rows = list(csv.reader(io.StringIO(raw), delimiter=delimiter))
    if not rows:
        return {"ok": True, "kind": "table", "rows": 0, "columns": 0, "text": ""}
    width = max(len(row) for row in rows)
    lines = [" | ".join(cell.strip() for cell in row) for row in rows[:400]]
    return {
        "ok": True,
        "kind": "table",
        "rows": len(rows),
        "columns": width,
        "header": rows[0] if rows else [],
        "text": "\n".join(lines),
    }


def read_document(
    workspace: Path,
    path: str | None = None,
    url: str | None = None,
    pages: str | None = None,
    max_chars: int = 20000,
    as_table: bool = False,
) -> dict:
    if not path and not url:
        return {"ok": False, "error": "pass either path or url"}

    source = path or url or ""
    data: bytes | None = None
    suffix = ""

    if path:
        try:
            target = _safe(workspace, path)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if not target.is_file():
            return {"ok": False, "error": f"no such file: {path}"}
        suffix = target.suffix.lower()
        if suffix in {".pdf", ".docx"}:
            data = target.read_bytes()
        else:
            try:
                raw = target.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                return {"ok": False, "error": str(exc)}
    else:
        assert url is not None
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            response, _attempts = request_with_retry("GET", url, attempts=3, timeout=45)
        except requests.RequestException as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "url": url}
        if response.status_code >= 400:
            return {"ok": False, "error": f"HTTP {response.status_code}", "url": url}
        suffix = Path(url.split("?")[0]).suffix.lower()
        ctype = response.headers.get("content-type", "").lower()
        if suffix in {".pdf", ".docx"} or "pdf" in ctype or "wordprocessingml" in ctype:
            data = response.content
            if not suffix:
                suffix = ".pdf" if "pdf" in ctype else ".docx"
        else:
            raw = response.text

    if data is not None:
        if suffix == ".pdf":
            result = pdf_text(data, pages)
        elif suffix == ".docx":
            result = docx_text(data)
        else:
            result = {"ok": False, "error": f"unsupported binary type: {suffix}"}
    elif suffix in {".html", ".htm"}:
        result = html_text(raw)
    elif suffix in {".csv"} or (as_table and suffix not in {".json"}):
        result = table_text(raw, ",")
    elif suffix == ".tsv":
        result = table_text(raw, "\t")
    elif suffix == ".json":
        try:
            parsed = json.loads(raw)
            result = {"ok": True, "kind": "json", "text": json.dumps(parsed, indent=2, ensure_ascii=False)[:max_chars]}
        except ValueError:
            result = {"ok": True, "kind": "json", "text": raw}
    else:
        result = {"ok": True, "kind": "text", "text": raw}

    if not result.get("ok"):
        result["source"] = source
        return result

    text = result.pop("text", "")
    total = len(text)
    result.update(
        {
            "ok": True,
            "source": source,
            "chars": total,
            "est_tokens": round(total / 4),
            "truncated": total > max_chars,
            "text": text[:max_chars],
        }
    )
    if result["truncated"]:
        result["note"] = f"showing first {max_chars} of {total} characters; pass pages= to narrow a PDF"
    return result


def register(registry, config) -> None:
    ws = config.workspace

    @registry.tool(
        "read_document",
        "Read a PDF, DOCX, HTML, CSV/TSV, JSON, Markdown or text file (local path or URL) and "
        "return its text. Use pages='1-5' to narrow a long PDF. Check est_tokens before "
        "pulling a huge document into context.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace path"},
                "url": {"type": "string", "description": "Public URL instead of a path"},
                "pages": {"type": "string", "description": "PDF pages, e.g. '2-6'"},
                "max_chars": {"type": "integer", "default": 20000},
                "as_table": {"type": "boolean", "description": "Parse as CSV and show rows/columns"},
            },
        },
    )
    def read_document_tool(
        path: str | None = None,
        url: str | None = None,
        pages: str | None = None,
        max_chars: int = 20000,
        as_table: bool = False,
    ):
        return read_document(Path(ws), path=path, url=url, pages=pages, max_chars=max_chars, as_table=as_table)
