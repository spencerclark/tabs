import hashlib
import sqlite3
from datetime import datetime, timezone

import requests


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_article_text(url: str, http_get=requests.get) -> str:
    response = http_get(url, timeout=10)
    response.raise_for_status()
    return response.text


def store_article(
    conn: sqlite3.Connection,
    source_id: int,
    url: str,
    title: str,
    published_at: str | None,
    full_text: str,
) -> int:
    """Insert a new article, or a new version if content changed since the last fetch."""
    content_hash = _hash_content(full_text)
    retrieved_at = datetime.now(timezone.utc).isoformat()

    existing = conn.execute(
        "SELECT id, content_hash FROM articles WHERE url = ? ORDER BY id DESC LIMIT 1",
        (url,),
    ).fetchone()

    if existing is not None and existing["content_hash"] == content_hash:
        return existing["id"]

    previous_version_id = existing["id"] if existing is not None else None

    cursor = conn.execute(
        """
        INSERT INTO articles
            (source_id, url, title, full_text, content_hash,
             published_at, retrieved_at, previous_version_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_id, url, title, full_text, content_hash,
            published_at, retrieved_at, previous_version_id,
        ),
    )
    conn.commit()
    return cursor.lastrowid
