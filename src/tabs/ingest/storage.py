import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from html import unescape

import requests

# Minimal, dependency-free HTML -> text extraction. This is deliberately not
# publication-quality extraction; its job is to remove the dominant sources of
# spurious content-hash diffs (scripts, style blocks, markup attributes with
# per-request tokens, whitespace formatting) so the 14-day re-check window
# detects real edits/retractions instead of boilerplate churn (SPEC §5.3).
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?(?:</\1\s*>|\Z)", re.IGNORECASE | re.DOTALL
)
_TAG_RE = re.compile(r"<[^>]*>")
_WHITESPACE_RE = re.compile(r"\s+")


def _extract_text(html: str) -> str:
    """Reduce an HTML document to normalized visible text."""
    text = _COMMENT_RE.sub(" ", html)
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = unescape(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_article_text(url: str, http_get=requests.get) -> str:
    response = http_get(url, timeout=10)
    response.raise_for_status()
    return _extract_text(response.text)


def store_article(
    conn: sqlite3.Connection,
    source_id: int,
    url: str,
    title: str,
    published_at: str | None,
    full_text: str,
) -> tuple[int, bool]:
    """Insert a new article, or a new version if content changed since the last fetch.

    Returns ``(article_id, created)`` where ``created`` is True only when a row was
    actually inserted (first-time ingest or a real content change), and False when the
    existing unchanged row was returned as-is.
    """
    content_hash = _hash_content(full_text)
    retrieved_at = datetime.now(timezone.utc).isoformat()

    existing = conn.execute(
        "SELECT id, content_hash FROM articles WHERE url = ? ORDER BY id DESC LIMIT 1",
        (url,),
    ).fetchone()

    if existing is not None and existing["content_hash"] == content_hash:
        return existing["id"], False

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
    return cursor.lastrowid, True
