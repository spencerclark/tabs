import sqlite3
from datetime import datetime, timezone

import tabs.ingest.orchestrator as orchestrator_module
from tabs.curate.models import ExtractedItem, ExtractionResult, TriageResult
from tabs.db import get_connection, init_db
from tabs.ingest.fetch import FeedFetchError, FetchedEntry
from tabs.ingest.orchestrator import run_ingest
from tabs.ingest.storage import _hash_content


def _insert_source(conn, name, feed_url):
    conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES (?, ?, 'AppSec', 2, 2)",
        (name, feed_url),
    )
    conn.commit()


def _no_sleep(seconds):
    """Stand-in for the injected rate-limit delay, so tests never actually sleep."""


def _always_in_scope(client, title, summary, source_category):
    """Stand-in triage_article that lets everything through as AppSec, for tests that
    aren't specifically exercising triage gating."""
    return TriageResult(in_scope=True, category="AppSec")


def _no_extraction(client, full_text, source_name):
    """Stand-in extract_claims_and_perspectives that extracts nothing, for tests that
    aren't specifically exercising extraction."""
    return ExtractionResult()


def _install_default_curation_stubs(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "triage_article", _always_in_scope)
    monkeypatch.setattr(orchestrator_module, "extract_claims_and_perspectives", _no_extraction)


DEFAULT_SUMMARY_EXTRAS = {
    "articles_out_of_scope": 0, "articles_uncurated": 0,
    "claims_extracted": 0, "perspectives_extracted": 0,
}


def test_run_ingest_delays_between_article_fetches_but_not_before_the_first(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source A", "https://a.example/feed")
    _insert_source(conn, "Source B", "https://b.example/feed")

    entries_a = [
        FetchedEntry(url="https://a.example/1", title="1", published_at=None, summary="s"),
        FetchedEntry(url="https://a.example/2", title="2", published_at=None, summary="s"),
    ]
    entry_b = FetchedEntry(url="https://b.example/1", title="1", published_at=None, summary="s")

    monkeypatch.setattr(
        orchestrator_module,
        "fetch_feed",
        lambda feed_url: entries_a if feed_url == "https://a.example/feed" else [entry_b],
    )
    _install_default_curation_stubs(monkeypatch)

    calls = []
    monkeypatch.setattr(
        orchestrator_module,
        "fetch_article_text",
        lambda url: calls.append(("fetch", url)) or "text for " + url,
    )

    run_ingest(conn, client=None, sleep=lambda seconds: calls.append(("sleep", seconds)))

    # three article fetches, two delays: never before the first request of the run
    assert calls == [
        ("fetch", "https://a.example/1"),
        ("sleep", orchestrator_module.ARTICLE_REQUEST_DELAY_SECONDS),
        ("fetch", "https://a.example/2"),
        ("sleep", orchestrator_module.ARTICLE_REQUEST_DELAY_SECONDS),
        ("fetch", "https://b.example/1"),
    ]
    conn.close()


def test_run_ingest_records_a_run_scoped_success_row_even_with_zero_errors(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source", "https://s.example/feed")

    entry = FetchedEntry(url="https://s.example/a", title="A", published_at=None, summary="s")
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [entry])
    _install_default_curation_stubs(monkeypatch)
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    summary = run_ingest(conn, client=None, sleep=_no_sleep)

    row = conn.execute(
        "SELECT run_started_at, run_finished_at, source_id, status, message "
        "FROM run_log WHERE source_id IS NULL"
    ).fetchone()
    assert row is not None
    assert row["status"] == "success"
    assert row["run_started_at"] < row["run_finished_at"]
    assert str(summary["articles_stored"]) in row["message"]
    conn.close()


def test_run_ingest_records_the_run_scoped_row_even_when_a_source_fails(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Bad Source", "https://bad.example/feed")

    def fake_fetch_feed(feed_url):
        raise FeedFetchError("boom")

    monkeypatch.setattr(orchestrator_module, "fetch_feed", fake_fetch_feed)

    run_ingest(conn, client=None, sleep=_no_sleep)

    run_rows = conn.execute("SELECT status FROM run_log WHERE source_id IS NULL").fetchall()
    assert len(run_rows) == 1
    assert run_rows[0]["status"] == "success"
    error_rows = conn.execute(
        "SELECT status FROM run_log WHERE source_id IS NOT NULL AND status = 'error'"
    ).fetchall()
    assert len(error_rows) == 1
    conn.close()


def test_run_ingest_stores_articles_and_records_success(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Good Source", "https://good.example/feed")

    entry = FetchedEntry(url="https://good.example/a", title="A", published_at=None, summary="s")
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [entry])
    _install_default_curation_stubs(monkeypatch)
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    summary = run_ingest(conn, client=None, sleep=_no_sleep)

    assert summary == {
        "sources_ok": 1, "sources_failed": 0, "articles_stored": 1, **DEFAULT_SUMMARY_EXTRAS,
    }
    source_row = conn.execute("SELECT consecutive_failures, last_successful_fetch_at FROM sources").fetchone()
    assert source_row["consecutive_failures"] == 0
    assert source_row["last_successful_fetch_at"] is not None
    article_row = conn.execute("SELECT COUNT(*) AS n FROM articles").fetchone()
    assert article_row["n"] == 1
    conn.close()


def test_run_ingest_skips_failing_source_and_continues(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Bad Source", "https://bad.example/feed")
    _insert_source(conn, "Good Source", "https://good.example/feed")

    good_entry = FetchedEntry(url="https://good.example/a", title="A", published_at=None, summary="s")

    def fake_fetch_feed(feed_url):
        if feed_url == "https://bad.example/feed":
            raise FeedFetchError("boom")
        return [good_entry]

    monkeypatch.setattr(orchestrator_module, "fetch_feed", fake_fetch_feed)
    _install_default_curation_stubs(monkeypatch)
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    summary = run_ingest(conn, client=None, sleep=_no_sleep)

    assert summary == {
        "sources_ok": 1, "sources_failed": 1, "articles_stored": 1, **DEFAULT_SUMMARY_EXTRAS,
    }
    bad_row = conn.execute(
        "SELECT consecutive_failures FROM sources WHERE name = 'Bad Source'"
    ).fetchone()
    assert bad_row["consecutive_failures"] == 1
    run_log_rows = conn.execute("SELECT status FROM run_log WHERE status = 'error'").fetchall()
    assert len(run_log_rows) >= 1
    conn.close()


def test_run_ingest_skips_previously_ingested_urls_outside_recheck_window(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source", "https://s.example/feed")
    conn.execute(
        "INSERT INTO articles (source_id, url, title, full_text, content_hash, "
        "published_at, retrieved_at, previous_version_id) "
        "VALUES (1, 'https://s.example/old', 'Old', 'text', 'hash', NULL, '2000-01-01T00:00:00+00:00', NULL)"
    )
    conn.commit()

    old_entry = FetchedEntry(url="https://s.example/old", title="Old", published_at=None, summary="s")
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [old_entry])
    _install_default_curation_stubs(monkeypatch)

    fetch_calls = []
    monkeypatch.setattr(
        orchestrator_module,
        "fetch_article_text",
        lambda url: fetch_calls.append(url) or "text",
    )

    summary = run_ingest(conn, client=None, sleep=_no_sleep)

    assert fetch_calls == []  # old article, outside the 14-day re-check window: not re-fetched
    assert summary["articles_stored"] == 0
    conn.close()


def test_run_ingest_refetches_previously_ingested_urls_inside_recheck_window(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source", "https://s.example/feed")
    recent_retrieved_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO articles (source_id, url, title, full_text, content_hash, "
        "published_at, retrieved_at, previous_version_id) "
        "VALUES (1, 'https://s.example/recent', 'Recent', 'text', ?, NULL, ?, NULL)",
        (_hash_content("text"), recent_retrieved_at),
    )
    conn.commit()

    recent_entry = FetchedEntry(url="https://s.example/recent", title="Recent", published_at=None, summary="s")
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [recent_entry])
    _install_default_curation_stubs(monkeypatch)

    fetch_calls = []
    monkeypatch.setattr(
        orchestrator_module,
        "fetch_article_text",
        lambda url: fetch_calls.append(url) or "text",
    )

    summary = run_ingest(conn, client=None, sleep=_no_sleep)

    # recent article, inside the 14-day re-check window: must be re-fetched to detect edits/retractions
    assert fetch_calls == ["https://s.example/recent"]
    # ...but the content is unchanged, so nothing new is stored and nothing is counted
    assert summary["articles_stored"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM articles").fetchone()["n"] == 1
    conn.close()


def test_run_ingest_stores_new_version_when_recheck_finds_changed_content(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source", "https://s.example/feed")
    conn.execute(
        "INSERT INTO articles (source_id, url, title, full_text, content_hash, "
        "published_at, retrieved_at, previous_version_id) "
        "VALUES (1, 'https://s.example/recent', 'Recent', 'old text', ?, NULL, ?, NULL)",
        (_hash_content("old text"), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()

    recent_entry = FetchedEntry(url="https://s.example/recent", title="Recent", published_at=None, summary="s")
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [recent_entry])
    _install_default_curation_stubs(monkeypatch)
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "new text")

    summary = run_ingest(conn, client=None, sleep=_no_sleep)

    assert summary["articles_stored"] == 1
    rows = conn.execute("SELECT id, previous_version_id FROM articles ORDER BY id").fetchall()
    assert len(rows) == 2
    assert rows[1]["previous_version_id"] == rows[0]["id"]
    conn.close()


def test_run_ingest_reports_zero_stored_on_second_run_over_unchanged_content(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source", "https://s.example/feed")

    entry = FetchedEntry(url="https://s.example/a", title="A", published_at=None, summary="s")
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [entry])
    _install_default_curation_stubs(monkeypatch)
    monkeypatch.setattr(
        orchestrator_module,
        "fetch_article_text",
        lambda url: "<html><body><p>Same   article body.</p></body></html>",
    )

    first = run_ingest(conn, client=None, sleep=_no_sleep)
    second = run_ingest(conn, client=None, sleep=_no_sleep)

    assert first["articles_stored"] == 1
    # the article is inside the re-check window and is re-fetched, but the content is
    # unchanged, so no row is inserted and nothing may be counted as stored
    assert second["articles_stored"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM articles").fetchone()["n"] == 1
    conn.close()


def test_run_ingest_continues_when_store_article_fails_for_one_entry(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source A", "https://a.example/feed")
    _insert_source(conn, "Source B", "https://b.example/feed")

    bad_entry = FetchedEntry(url="https://a.example/bad", title="Bad", published_at=None, summary="s")
    good_entry_a = FetchedEntry(url="https://a.example/good", title="Good", published_at=None, summary="s")
    good_entry_b = FetchedEntry(url="https://b.example/good", title="Good", published_at=None, summary="s")

    def fake_fetch_feed(feed_url):
        if feed_url == "https://a.example/feed":
            return [bad_entry, good_entry_a]
        return [good_entry_b]

    monkeypatch.setattr(orchestrator_module, "fetch_feed", fake_fetch_feed)
    _install_default_curation_stubs(monkeypatch)
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    real_store_article = orchestrator_module.store_article
    store_calls = []

    def fake_store_article(conn, source_id, url, title, published_at, full_text):
        store_calls.append(url)
        if url == "https://a.example/bad":
            raise sqlite3.OperationalError("simulated store failure")
        return real_store_article(conn, source_id, url, title, published_at, full_text)

    monkeypatch.setattr(orchestrator_module, "store_article", fake_store_article)

    summary = run_ingest(conn, client=None, sleep=_no_sleep)

    # the failing store must not abort the run: the second entry in source A and all of
    # source B must still be processed.
    assert store_calls == [
        "https://a.example/bad",
        "https://a.example/good",
        "https://b.example/good",
    ]
    assert summary == {
        "sources_ok": 2, "sources_failed": 0, "articles_stored": 2, **DEFAULT_SUMMARY_EXTRAS,
    }
    error_rows = conn.execute(
        "SELECT message FROM run_log WHERE status = 'error'"
    ).fetchall()
    assert any("https://a.example/bad" in row["message"] for row in error_rows)
    article_row = conn.execute("SELECT COUNT(*) AS n FROM articles").fetchone()
    assert article_row["n"] == 2
    conn.close()


def test_run_ingest_skips_out_of_scope_articles_without_fetching_them(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source", "https://s.example/feed")

    entry = FetchedEntry(url="https://s.example/off-topic", title="Off Topic", published_at=None, summary="s")
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [entry])
    monkeypatch.setattr(
        orchestrator_module,
        "triage_article",
        lambda client, title, summary, source_category: TriageResult(in_scope=False),
    )

    fetch_calls = []
    monkeypatch.setattr(
        orchestrator_module,
        "fetch_article_text",
        lambda url: fetch_calls.append(url) or "text",
    )

    summary = run_ingest(conn, client=None, sleep=_no_sleep)

    # out-of-scope articles are never fetched or stored — triage runs on the cheap
    # feed-level title/summary before any HTTP fetch of the article body
    assert fetch_calls == []
    assert summary == {
        "sources_ok": 1, "sources_failed": 0, "articles_stored": 0,
        "articles_out_of_scope": 1, "articles_uncurated": 0,
        "claims_extracted": 0, "perspectives_extracted": 0,
    }
    assert conn.execute("SELECT COUNT(*) AS n FROM articles").fetchone()["n"] == 0
    conn.close()


def test_run_ingest_continues_when_triage_fails_for_one_entry(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source", "https://s.example/feed")

    bad_entry = FetchedEntry(url="https://s.example/bad", title="Bad", published_at=None, summary="s")
    good_entry = FetchedEntry(url="https://s.example/good", title="Good", published_at=None, summary="s")
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [bad_entry, good_entry])

    def fake_triage(client, title, summary, source_category):
        if title == "Bad":
            raise RuntimeError("simulated triage failure")
        return TriageResult(in_scope=True, category="AppSec")

    monkeypatch.setattr(orchestrator_module, "triage_article", fake_triage)
    monkeypatch.setattr(orchestrator_module, "extract_claims_and_perspectives", _no_extraction)
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    summary = run_ingest(conn, client=None, sleep=_no_sleep)

    # the failing triage call must not abort the run: the second entry must still be processed
    assert summary["articles_stored"] == 1
    error_rows = conn.execute(
        "SELECT message FROM run_log WHERE status = 'error'"
    ).fetchall()
    assert any("triage failed" in row["message"] for row in error_rows)
    conn.close()


def test_run_ingest_continues_when_triage_returns_no_parsed_output(tmp_path, monkeypatch):
    """`client.messages.parse(...).parsed_output` is None on a model refusal — not an
    exception, so the try/except around the triage call does not catch it. Dereferencing
    `.in_scope` on it must not abort the run (SPEC §5.4)."""
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source A", "https://a.example/feed")
    _insert_source(conn, "Source B", "https://b.example/feed")

    refused_entry = FetchedEntry(url="https://a.example/refused", title="Refused", published_at=None, summary="s")
    good_entry_a = FetchedEntry(url="https://a.example/good", title="Good", published_at=None, summary="s")
    good_entry_b = FetchedEntry(url="https://b.example/good", title="Good", published_at=None, summary="s")

    def fake_fetch_feed(feed_url):
        if feed_url == "https://a.example/feed":
            return [refused_entry, good_entry_a]
        return [good_entry_b]

    monkeypatch.setattr(orchestrator_module, "fetch_feed", fake_fetch_feed)

    def refusing_triage(client, title, summary, source_category):
        if title == "Refused":
            return None  # what a safety refusal looks like: no parsed output, no exception
        return TriageResult(in_scope=True, category="AppSec")

    monkeypatch.setattr(orchestrator_module, "triage_article", refusing_triage)
    monkeypatch.setattr(orchestrator_module, "extract_claims_and_perspectives", _no_extraction)
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    summary = run_ingest(conn, client=None, sleep=_no_sleep)

    # the refused entry is skipped, but the rest of source A and all of source B run
    assert summary["articles_stored"] == 2
    assert summary["sources_ok"] == 2
    error_rows = conn.execute("SELECT message FROM run_log WHERE status = 'error'").fetchall()
    assert any(
        "triage returned no parsed output" in row["message"]
        and "https://a.example/refused" in row["message"]
        for row in error_rows
    )
    # the run-completion row is still written — the run was not aborted mid-flight
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM run_log WHERE source_id IS NULL"
    ).fetchone()["n"] == 1
    conn.close()


def test_run_ingest_continues_when_extraction_returns_no_parsed_output(tmp_path, monkeypatch):
    """A None extraction result is a model refusal, not a storage fault — it must be
    logged as such rather than surfacing as a misleading 'curation store failed'."""
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source", "https://s.example/feed")

    entry = FetchedEntry(url="https://s.example/a", title="A", published_at=None, summary="s")
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [entry])
    monkeypatch.setattr(orchestrator_module, "triage_article", _always_in_scope)
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")
    monkeypatch.setattr(
        orchestrator_module,
        "extract_claims_and_perspectives",
        lambda client, full_text, source_name: None,
    )

    summary = run_ingest(conn, client=None, sleep=_no_sleep)

    # the article itself is still stored — only its curation was refused
    assert summary["articles_stored"] == 1
    assert summary["claims_extracted"] == 0
    assert summary["perspectives_extracted"] == 0
    assert summary["articles_uncurated"] == 1
    messages = [
        row["message"]
        for row in conn.execute("SELECT message FROM run_log WHERE status = 'error'").fetchall()
    ]
    assert any("extraction returned no parsed output" in message for message in messages)
    assert not any("curation store failed" in message for message in messages)
    conn.close()


def test_run_ingest_extracts_claims_and_perspectives_for_a_stored_article(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source", "https://s.example/feed")

    entry = FetchedEntry(
        url="https://s.example/a", title="A", published_at="2026-08-01", summary="s"
    )
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [entry])
    monkeypatch.setattr(orchestrator_module, "triage_article", _always_in_scope)
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    extraction_result = ExtractionResult(
        items=[
            ExtractedItem(
                text="A critical flaw was patched.", supporting_excerpt="patched today",
                item_type="factual", category="AppSec", sub_tags=["Patch"],
                llm_certainty=0.9, author="Jane Doe",
            ),
            ExtractedItem(
                text="This will likely be exploited within a week.",
                supporting_excerpt="likely be exploited", item_type="prediction",
                category="AppSec", sub_tags=[], llm_certainty=0.4,
            ),
            ExtractedItem(
                text="The fix was rushed and poorly tested.",
                supporting_excerpt="rushed and poorly tested", item_type="opinion",
                category="AppSec", sub_tags=["Opinion"], llm_certainty=0.7,
            ),
        ],
    )
    monkeypatch.setattr(
        orchestrator_module,
        "extract_claims_and_perspectives",
        lambda client, full_text, source_name: extraction_result,
    )

    summary = run_ingest(conn, client=None, sleep=_no_sleep)

    assert summary["claims_extracted"] == 2
    assert summary["perspectives_extracted"] == 1
    claim_rows = conn.execute(
        "SELECT claim_text, claim_type, author, published_at FROM claims ORDER BY id"
    ).fetchall()
    assert [row["claim_text"] for row in claim_rows] == [
        "A critical flaw was patched.", "This will likely be exploited within a week.",
    ]
    assert claim_rows[0]["claim_type"] == "factual"
    assert claim_rows[0]["author"] == "Jane Doe"
    assert claim_rows[0]["published_at"] == "2026-08-01"
    perspective_row = conn.execute("SELECT perspective_text FROM perspectives").fetchone()
    assert perspective_row["perspective_text"] == "The fix was rushed and poorly tested."
    conn.close()


def test_run_ingest_skips_extraction_when_content_is_unchanged(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source", "https://s.example/feed")

    entry = FetchedEntry(url="https://s.example/a", title="A", published_at=None, summary="s")
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [entry])
    monkeypatch.setattr(orchestrator_module, "triage_article", _always_in_scope)
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    extraction_calls = []

    def tracking_extraction(client, full_text, source_name):
        extraction_calls.append(full_text)
        return ExtractionResult()

    monkeypatch.setattr(orchestrator_module, "extract_claims_and_perspectives", tracking_extraction)

    run_ingest(conn, client=None, sleep=_no_sleep)  # first run: content is new
    run_ingest(conn, client=None, sleep=_no_sleep)  # second run: unchanged content

    # extraction must run once, not twice — re-curating unchanged content wastes API calls
    assert extraction_calls == ["full text"]
    conn.close()


def test_run_ingest_continues_when_extraction_fails_for_one_entry(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source", "https://s.example/feed")

    entry = FetchedEntry(url="https://s.example/a", title="A", published_at=None, summary="s")
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [entry])
    monkeypatch.setattr(orchestrator_module, "triage_article", _always_in_scope)
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    def failing_extraction(client, full_text, source_name):
        raise RuntimeError("simulated extraction failure")

    monkeypatch.setattr(orchestrator_module, "extract_claims_and_perspectives", failing_extraction)

    summary = run_ingest(conn, client=None, sleep=_no_sleep)

    # the article itself is still stored — only its curation failed
    assert summary["articles_stored"] == 1
    assert summary["claims_extracted"] == 0
    # ...and that silent permanent skip is reported rather than left invisible
    assert summary["articles_uncurated"] == 1
    error_rows = conn.execute(
        "SELECT message FROM run_log WHERE status = 'error'"
    ).fetchall()
    assert any("extraction failed" in row["message"] for row in error_rows)
    conn.close()


def test_run_ingest_rolls_back_partial_curation_writes_when_the_store_fails(tmp_path, monkeypatch):
    """store_extraction_result INSERTs each item then commits once at the end. If it
    raises partway through, the pending INSERTs are still on the connection — and
    _log_run's own commit would otherwise commit those orphaned rows alongside the
    error entry, even though no summary counter was ever incremented for them."""
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source", "https://s.example/feed")

    bad_entry = FetchedEntry(url="https://s.example/bad", title="Bad", published_at=None, summary="s")
    good_entry = FetchedEntry(url="https://s.example/good", title="Good", published_at=None, summary="s")
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [bad_entry, good_entry])
    _install_default_curation_stubs(monkeypatch)
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text " + url)

    store_calls = []

    def half_written_store(conn, article_id, source_id, published_at, retrieved_at, extraction):
        store_calls.append(article_id)
        if len(store_calls) == 1:
            # simulate failing partway through a multi-item insert loop
            conn.execute(
                "INSERT INTO claims (article_id, source_id, claim_text, supporting_excerpt, "
                "claim_type, category, sub_tags, llm_certainty, retrieved_at, created_at) "
                "VALUES (?, ?, 'partial', 'x', 'factual', 'AppSec', '[]', 0.5, ?, ?)",
                (article_id, source_id, retrieved_at, retrieved_at),
            )
            raise sqlite3.OperationalError("database is locked")
        return {"claims_created": 0, "perspectives_created": 0}

    monkeypatch.setattr(orchestrator_module, "store_extraction_result", half_written_store)

    summary = run_ingest(conn, client=None, sleep=_no_sleep)

    # the half-written claim must not survive: it was rolled back, not committed by the
    # error-logging path that follows it
    assert conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"] == 0
    assert summary["claims_extracted"] == 0
    assert summary["articles_uncurated"] == 1
    # ...and the run still continued through the second entry
    assert len(store_calls) == 2
    assert summary["articles_stored"] == 2
    error_rows = conn.execute("SELECT message FROM run_log WHERE status = 'error'").fetchall()
    assert any("curation store failed" in row["message"] for row in error_rows)
    conn.close()


def test_run_ingest_continues_when_curation_store_fails_for_one_entry(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source A", "https://a.example/feed")
    _insert_source(conn, "Source B", "https://b.example/feed")

    bad_entry = FetchedEntry(url="https://a.example/bad", title="Bad", published_at=None, summary="s")
    good_entry_a = FetchedEntry(url="https://a.example/good", title="Good", published_at=None, summary="s")
    good_entry_b = FetchedEntry(url="https://b.example/good", title="Good", published_at=None, summary="s")

    def fake_fetch_feed(feed_url):
        if feed_url == "https://a.example/feed":
            return [bad_entry, good_entry_a]
        return [good_entry_b]

    monkeypatch.setattr(orchestrator_module, "fetch_feed", fake_fetch_feed)
    _install_default_curation_stubs(monkeypatch)
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    real_store_extraction_result = orchestrator_module.store_extraction_result
    store_calls = []

    def fake_store_extraction_result(conn, article_id, source_id, published_at, retrieved_at, extraction):
        store_calls.append(article_id)
        if len(store_calls) == 1:
            # bad_entry is processed first; simulate the storage-layer failure a locked
            # sqlite db (e.g. an overlapping cron run) would raise here.
            raise sqlite3.OperationalError("database is locked")
        return real_store_extraction_result(conn, article_id, source_id, published_at, retrieved_at, extraction)

    monkeypatch.setattr(orchestrator_module, "store_extraction_result", fake_store_extraction_result)

    summary = run_ingest(conn, client=None, sleep=_no_sleep)

    # the failing curation store must not abort the run: the second entry in source A and
    # all of source B must still be processed.
    assert len(store_calls) == 3
    # the article itself is still stored for all three entries — only bad_entry's
    # curation-storage step failed, so claims/perspectives stay at 0 for it (as they do
    # for the others, since _no_extraction never produces any).
    assert summary == {
        "sources_ok": 2, "sources_failed": 0, "articles_stored": 3,
        **DEFAULT_SUMMARY_EXTRAS,
        # bad_entry was stored but never curated — it will look "already curated,
        # unchanged" on every future run, so it is surfaced here
        "articles_uncurated": 1,
    }
    error_rows = conn.execute(
        "SELECT message FROM run_log WHERE status = 'error'"
    ).fetchall()
    assert any(
        "curation store failed" in row["message"] and "https://a.example/bad" in row["message"]
        for row in error_rows
    )
    article_row = conn.execute("SELECT COUNT(*) AS n FROM articles").fetchone()
    assert article_row["n"] == 3
    conn.close()
