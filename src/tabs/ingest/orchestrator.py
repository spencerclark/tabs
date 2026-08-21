import sqlite3
from datetime import datetime, timedelta, timezone

from tabs.ingest.fetch import FeedFetchError, fetch_feed
from tabs.ingest.storage import fetch_article_text, store_article

RECENT_ARTICLE_WINDOW_DAYS = 14


def run_ingest(conn: sqlite3.Connection) -> dict:
    """Run one ingestion pass over every allowlisted source. Returns a summary dict."""
    summary = {"sources_ok": 0, "sources_failed": 0, "articles_stored": 0}

    sources = conn.execute("SELECT id, feed_url FROM sources").fetchall()
    for source in sources:
        try:
            entries = fetch_feed(source["feed_url"])
        except FeedFetchError as exc:
            _record_failure(conn, source["id"], str(exc))
            summary["sources_failed"] += 1
            continue

        recheck_urls = _urls_in_recheck_window(conn, source["id"])
        for entry in entries:
            already_seen = _already_ingested(conn, entry.url)
            if already_seen and entry.url not in recheck_urls:
                continue  # older article, already ingested, outside the re-check window

            try:
                full_text = fetch_article_text(entry.url)
            except Exception as exc:  # noqa: BLE001 — one bad article must not kill the run
                _log_run(conn, source["id"], "error", f"article fetch failed: {entry.url}: {exc}")
                continue

            try:
                store_article(conn, source["id"], entry.url, entry.title, entry.published_at, full_text)
            except Exception as exc:  # noqa: BLE001 — one bad article must not kill the run
                _log_run(conn, source["id"], "error", f"article store failed: {entry.url}: {exc}")
                continue
            summary["articles_stored"] += 1

        _record_success(conn, source["id"])
        summary["sources_ok"] += 1

    return summary


def _urls_in_recheck_window(conn: sqlite3.Connection, source_id: int) -> set[str]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RECENT_ARTICLE_WINDOW_DAYS)).isoformat()
    rows = conn.execute(
        "SELECT DISTINCT url FROM articles WHERE source_id = ? AND retrieved_at >= ?",
        (source_id, cutoff),
    ).fetchall()
    return {row["url"] for row in rows}


def _already_ingested(conn: sqlite3.Connection, url: str) -> bool:
    row = conn.execute("SELECT 1 FROM articles WHERE url = ? LIMIT 1", (url,)).fetchone()
    return row is not None


def _record_failure(conn: sqlite3.Connection, source_id: int, message: str) -> None:
    conn.execute(
        "UPDATE sources SET consecutive_failures = consecutive_failures + 1 WHERE id = ?",
        (source_id,),
    )
    conn.commit()
    _log_run(conn, source_id, "error", message)


def _record_success(conn: sqlite3.Connection, source_id: int) -> None:
    conn.execute(
        "UPDATE sources SET consecutive_failures = 0, last_successful_fetch_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), source_id),
    )
    conn.commit()


def _log_run(conn: sqlite3.Connection, source_id: int, status: str, message: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO run_log (run_started_at, run_finished_at, source_id, status, message) "
        "VALUES (?, ?, ?, ?, ?)",
        (now, now, source_id, status, message),
    )
    conn.commit()
