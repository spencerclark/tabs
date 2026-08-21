import sqlite3
import time
from datetime import datetime, timedelta, timezone

from tabs.curate.extraction import extract_claims_and_perspectives
from tabs.curate.storage import store_extraction_result
from tabs.curate.triage import triage_article
from tabs.ingest.fetch import REQUEST_DELAY_SECONDS, FeedFetchError, fetch_feed
from tabs.ingest.storage import fetch_article_text, store_article

RECENT_ARTICLE_WINDOW_DAYS = 14
# SPEC §5.2: requests are rate-limited with a small delay between them. A feed yields
# many article fetches, so the per-entry loop needs the same delay fetch_feed applies.
# This delay is scoped to fetching the allowlisted news site's article body — it does
# not apply to triage/extraction, which call Anthropic's API, not the news site, and
# already get retry/backoff from the SDK's own client.
ARTICLE_REQUEST_DELAY_SECONDS = REQUEST_DELAY_SECONDS


def run_ingest(conn: sqlite3.Connection, client, sleep=time.sleep) -> dict:
    """Run one ingestion + curation pass over every allowlisted source. Returns a summary dict."""
    summary = {
        "sources_ok": 0,
        "sources_failed": 0,
        "articles_stored": 0,
        "articles_out_of_scope": 0,
        "claims_extracted": 0,
        "perspectives_extracted": 0,
    }
    any_article_fetched = False
    run_started_at = datetime.now(timezone.utc).isoformat()

    sources = conn.execute("SELECT id, name, category, feed_url FROM sources").fetchall()
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
                triage_result = triage_article(client, entry.title, entry.summary, source["category"])
            except Exception as exc:  # noqa: BLE001 — one bad article must not kill the run
                _log_run(
                    conn, source["id"], "error",
                    f"triage failed: {entry.url}: {type(exc).__name__}: {exc}",
                )
                continue

            if not triage_result.in_scope:
                summary["articles_out_of_scope"] += 1
                continue

            if any_article_fetched:  # no delay before the very first request of the run
                sleep(ARTICLE_REQUEST_DELAY_SECONDS)
            any_article_fetched = True

            try:
                full_text = fetch_article_text(entry.url)
            except Exception as exc:  # noqa: BLE001 — one bad article must not kill the run
                _log_run(
                    conn, source["id"], "error",
                    f"article fetch failed: {entry.url}: {type(exc).__name__}: {exc}",
                )
                continue

            try:
                article_id, created = store_article(
                    conn, source["id"], entry.url, entry.title, entry.published_at, full_text
                )
            except Exception as exc:  # noqa: BLE001 — one bad article must not kill the run
                _log_run(
                    conn, source["id"], "error",
                    f"article store failed: {entry.url}: {type(exc).__name__}: {exc}",
                )
                continue

            if not created:
                continue  # unchanged content: already curated on a previous run
            summary["articles_stored"] += 1

            try:
                extraction_result = extract_claims_and_perspectives(client, full_text, source["name"])
            except Exception as exc:  # noqa: BLE001 — one bad article must not kill the run
                _log_run(
                    conn, source["id"], "error",
                    f"extraction failed: {entry.url}: {type(exc).__name__}: {exc}",
                )
                continue

            try:
                counts = store_extraction_result(
                    conn, article_id, source["id"], entry.published_at,
                    datetime.now(timezone.utc).isoformat(), extraction_result,
                )
            except Exception as exc:  # noqa: BLE001 — one bad article must not kill the run
                _log_run(
                    conn, source["id"], "error",
                    f"curation store failed: {entry.url}: {type(exc).__name__}: {exc}",
                )
                continue
            summary["claims_extracted"] += counts["claims_created"]
            summary["perspectives_extracted"] += counts["perspectives_created"]

        _record_success(conn, source["id"])
        summary["sources_ok"] += 1

    _record_run_completed(conn, run_started_at, summary)
    return summary


def _record_run_completed(
    conn: sqlite3.Connection, run_started_at: str, summary: dict
) -> None:
    """Write the one run-scoped row that proves the run actually happened.

    Without it a dead cron job (never fires) and a healthy one (runs, zero errors)
    leave identical traces in run_log. Per-source error rows remain in addition.
    """
    conn.execute(
        "INSERT INTO run_log (run_started_at, run_finished_at, source_id, status, message) "
        "VALUES (?, ?, NULL, 'success', ?)",
        (
            run_started_at,
            datetime.now(timezone.utc).isoformat(),
            f"ingest run complete: {summary}",
        ),
    )
    conn.commit()


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
