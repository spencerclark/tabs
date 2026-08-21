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
        # Articles stored (content_hash committed) whose curation produced nothing. These
        # are permanently skipped on later runs — store_article will report the unchanged
        # content as already-curated — so without this counter the loss is invisible. A
        # real fix (a curation_status column and a retry path) is deferred; this is the
        # visibility half. Out-of-scope articles are NOT counted here: that is an
        # intentional skip, already reported as articles_out_of_scope.
        "articles_uncurated": 0,
        "claims_extracted": 0,
        "perspectives_extracted": 0,
    }
    any_article_fetched = False
    # Tracked outside `summary` (which is the operator-facing report) purely to detect a
    # wholly broken Anthropic client at the end of the run — see the check after the loop.
    # This counts EVERY Anthropic API call this run makes, not just triage: an already-seen
    # re-check-window entry skips triage entirely (see below) and goes straight to
    # extraction, so on many runs extraction is not merely a second call site but can be
    # the *only* Anthropic call made. Counting triage alone both false-positives (a failed
    # triage call plus a successful extraction call trips "every call failed") and
    # false-negatives (zero triage calls plus a failing extraction call never trips the
    # check at all) — see the check after the loop for the full rationale.
    llm_attempts = 0
    llm_failures = 0
    first_llm_error = None
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

            # Triage only genuinely new entries. An already-seen entry reaches here only
            # because it is inside the re-check window, and it was already judged in-scope
            # when first ingested. Re-triaging it burns a Haiku call per re-check per day,
            # and — worse — a nondeterministic flip to in_scope=False would skip the body
            # re-fetch below, silently defeating SPEC §5.3's edit/retraction detection for
            # exactly the articles the window exists to watch.
            if not already_seen:
                llm_attempts += 1
                try:
                    triage_result = triage_article(
                        client, entry.title, entry.summary, source["category"]
                    )
                except Exception as exc:  # noqa: BLE001 — one bad article must not kill the run
                    llm_failures += 1
                    if first_llm_error is None:
                        first_llm_error = f"{type(exc).__name__}: {exc}"
                    _log_run(
                        conn, source["id"], "error",
                        f"triage failed: {entry.url}: {type(exc).__name__}: {exc}",
                    )
                    continue

                # parsed_output is None when the model refuses (a real possibility for this
                # corpus of exploit/malware/CVE news). That is not an exception, so it slips
                # past the guard above — check it before dereferencing .in_scope.
                if triage_result is None:
                    llm_failures += 1
                    if first_llm_error is None:
                        first_llm_error = "no parsed output (refusal)"
                    _log_run(
                        conn, source["id"], "error",
                        f"triage returned no parsed output (likely a model refusal): {entry.url}",
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
            except ValueError as exc:
                # a local pre-flight rejection (article text over MAX_EXTRACTION_CHARS) —
                # the request was never sent, so it must not count toward the run-health
                # check's attempt/failure tally, or an oversized article could make a quiet
                # run falsely report "every Anthropic API call failed".
                _log_run(
                    conn, source["id"], "error",
                    f"extraction input rejected: {entry.url}: {exc}",
                )
                summary["articles_uncurated"] += 1
                continue
            except Exception as exc:  # noqa: BLE001 — one bad article must not kill the run
                llm_attempts += 1
                llm_failures += 1
                if first_llm_error is None:
                    first_llm_error = f"{type(exc).__name__}: {exc}"
                _log_run(
                    conn, source["id"], "error",
                    f"extraction failed: {entry.url}: {type(exc).__name__}: {exc}",
                )
                summary["articles_uncurated"] += 1
                continue

            llm_attempts += 1
            # as with triage: a refusal yields None, not an exception. Without this check
            # it reaches store_extraction_result and is mislabeled "curation store failed".
            if extraction_result is None:
                llm_failures += 1
                if first_llm_error is None:
                    first_llm_error = "no parsed output (refusal)"
                _log_run(
                    conn, source["id"], "error",
                    f"extraction returned no parsed output (likely a model refusal): {entry.url}",
                )
                summary["articles_uncurated"] += 1
                continue

            try:
                counts = store_extraction_result(
                    conn, article_id, source["id"], entry.published_at,
                    datetime.now(timezone.utc).isoformat(), extraction_result,
                )
            except Exception as exc:  # noqa: BLE001 — one bad article must not kill the run
                # store_extraction_result INSERTs each item before its single trailing
                # commit, so a mid-loop failure leaves pending INSERTs on the connection.
                # Discard them first: _log_run commits, and would otherwise persist those
                # orphaned rows — rows no summary counter ever accounted for.
                conn.rollback()
                _log_run(
                    conn, source["id"], "error",
                    f"curation store failed: {entry.url}: {type(exc).__name__}: {exc}",
                )
                summary["articles_uncurated"] += 1
                continue
            summary["claims_extracted"] += counts["claims_created"]
            summary["perspectives_extracted"] += counts["perspectives_created"]

        _record_success(conn, source["id"])
        summary["sources_ok"] += 1

    # Recorded before the all-failed check below so the audit trail still shows the run
    # happened — the per-article error rows need that run-scoped row for context.
    _record_run_completed(conn, run_started_at, summary)

    # anthropic.Anthropic() validates nothing at construction time, only at first request.
    # A missing or invalid key therefore fails every Anthropic call individually, each one
    # caught and logged by the per-article guard, leaving a run that exits 0 reporting
    # articles_stored=0 — a cron job would report success forever while doing nothing.
    # A total failure streak is the signal that the client itself, not the content, is
    # broken; a partial one is normal and must not trip this. This counts triage AND
    # extraction calls together (see llm_attempts' definition above) rather than triage
    # alone, since an already-seen re-check-window entry skips triage and goes straight to
    # extraction — on such a run extraction may be the only Anthropic call made at all.
    if llm_attempts > 0 and llm_failures == llm_attempts:
        raise RuntimeError(
            f"every Anthropic API call failed this run ({llm_failures}/{llm_attempts}); "
            f"first error: {first_llm_error} — "
            "check ANTHROPIC_API_KEY and Anthropic API status"
        )

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
