"""Persistent memory: durable facts plus conversation history (SQLite, stdlib only)."""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    tags TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session, id);
"""


class Memory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # --- facts --------------------------------------------------------
    def remember(self, text: str, tags: str = "") -> dict:
        text = text.strip()
        if not text:
            return {"ok": False, "error": "empty fact"}
        existing = self.conn.execute("SELECT id FROM facts WHERE lower(text) = lower(?)", (text,)).fetchone()
        if existing:
            return {"ok": True, "id": existing["id"], "text": text, "duplicate": True}
        cur = self.conn.execute("INSERT INTO facts (text, tags) VALUES (?, ?)", (text, tags))
        self.conn.commit()
        return {"ok": True, "id": cur.lastrowid, "text": text}

    def recall(self, query: str = "", limit: int = 20) -> list[dict]:
        terms = [t for t in re.findall(r"[A-Za-z0-9_'-]{2,}", query or "")][:8]
        if not terms:
            rows = self.conn.execute(
                "SELECT id, text, tags, created_at FROM facts ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            where = " OR ".join(["text LIKE ?"] * len(terms))
            params = [f"%{t}%" for t in terms] + [limit]
            rows = self.conn.execute(
                f"SELECT id, text, tags, created_at FROM facts WHERE {where} ORDER BY id DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def forget(self, fact_id: int) -> dict:
        cur = self.conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        self.conn.commit()
        return {"ok": cur.rowcount > 0, "deleted": fact_id}

    # --- history ------------------------------------------------------
    def log(self, session: str, role: str, content: str) -> None:
        if not content:
            return
        self.conn.execute(
            "INSERT INTO messages (session, role, content) VALUES (?, ?, ?)", (session, role, content[:8000])
        )
        self.conn.commit()

    def history(self, session: str, limit: int = 12) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE session = ? ORDER BY id DESC LIMIT ?", (session, limit)
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def new_session_id(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S")

    def stats(self) -> dict:
        facts = self.conn.execute("SELECT COUNT(*) c FROM facts").fetchone()["c"]
        messages = self.conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
        return {"facts": facts, "messages": messages, "path": str(self.path)}

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


def register(registry, config, memory: Memory) -> None:
    @registry.tool(
        "remember",
        "Save a durable fact about the user, their preferences, projects or decisions so it "
        "survives restarts. Use for things worth recalling later, not chatter.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "One self-contained fact"},
                "tags": {"type": "string", "description": "Optional comma-separated tags"},
            },
            "required": ["text"],
        },
    )
    def remember(text: str, tags: str = ""):
        return memory.remember(text, tags)

    @registry.tool(
        "recall",
        "Search long-term memory. Call this before answering questions about the user, their "
        "projects, or anything from previous sessions.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords; empty returns recent facts"},
                "limit": {"type": "integer", "default": 20},
            },
        },
    )
    def recall(query: str = "", limit: int = 20):
        return {"facts": memory.recall(query, limit)}
