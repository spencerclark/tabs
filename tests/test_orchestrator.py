import sqlite3
from datetime import datetime, timezone

import tabs.ingest.orchestrator as orchestrator_module
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

    calls = []
    monkeypatch.setattr(
        orchestrator_module,
        "fetch_article_text",
        lambda url: calls.append(("fetch", url)) or "text for " + url,
    )

    run_ingest(conn, sleep=lambda seconds: calls.append(("sleep", seconds)))

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
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    summary = run_ingest(conn, sleep=_no_sleep)

    # a healthy run must leave a trace, otherwise it is indistinguishable from a cron
    # job that never fired at all
    row = conn.execute(
        "SELECT run_started_at, run_finished_at, source_id, status, message "
        "FROM run_log WHERE source_id IS NULL"
    ).fetchone()
    assert row is not None
    assert row["status"] == "success"
    assert row["run_started_at"] < row["run_finished_at"]  # not the same timestamp twice
    assert str(summary["articles_stored"]) in row["message"]
    conn.close()


def test_run_ingest_records_the_run_scoped_row_even_when_a_source_fails(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Bad Source", "https://bad.example/feed")

    def fake_fetch_feed(feed_url):
        raise FeedFetchError("boom")

    monkeypatch.setattr(orchestrator_module, "fetch_feed", fake_fetch_feed)

    run_ingest(conn, sleep=_no_sleep)

    run_rows = conn.execute("SELECT status FROM run_log WHERE source_id IS NULL").fetchall()
    assert len(run_rows) == 1
    assert run_rows[0]["status"] == "success"
    # the per-source error row is additive, not replaced
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
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    summary = run_ingest(conn, sleep=_no_sleep)

    assert summary == {"sources_ok": 1, "sources_failed": 0, "articles_stored": 1}
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
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    summary = run_ingest(conn, sleep=_no_sleep)

    assert summary == {"sources_ok": 1, "sources_failed": 1, "articles_stored": 1}
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

    fetch_calls = []
    monkeypatch.setattr(
        orchestrator_module,
        "fetch_article_text",
        lambda url: fetch_calls.append(url) or "text",
    )

    summary = run_ingest(conn, sleep=_no_sleep)

    assert fetch_calls == []  # old article, outside the 14-day re-check window: not re-fetched
    assert summary["articles_stored"] == 0
    conn.close()


def test_run_ingest_refetches_previously_ingested_urls_inside_recheck_window(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source", "https://s.example/feed")
    recent_retrieved_at = datetime.now(timezone.utc).isoformat()
    # seed the real hash of the text the fetch will return, so the unchanged-content
    # path is genuinely exercised (a literal placeholder hash never matches and would
    # mask the double-counting bug)
    conn.execute(
        "INSERT INTO articles (source_id, url, title, full_text, content_hash, "
        "published_at, retrieved_at, previous_version_id) "
        "VALUES (1, 'https://s.example/recent', 'Recent', 'text', ?, NULL, ?, NULL)",
        (_hash_content("text"), recent_retrieved_at),
    )
    conn.commit()

    recent_entry = FetchedEntry(url="https://s.example/recent", title="Recent", published_at=None, summary="s")
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [recent_entry])

    fetch_calls = []
    monkeypatch.setattr(
        orchestrator_module,
        "fetch_article_text",
        lambda url: fetch_calls.append(url) or "text",
    )

    summary = run_ingest(conn, sleep=_no_sleep)

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
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "new text")

    summary = run_ingest(conn, sleep=_no_sleep)

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
    monkeypatch.setattr(
        orchestrator_module,
        "fetch_article_text",
        lambda url: "<html><body><p>Same   article body.</p></body></html>",
    )

    first = run_ingest(conn, sleep=_no_sleep)
    second = run_ingest(conn, sleep=_no_sleep)

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
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    real_store_article = orchestrator_module.store_article
    store_calls = []

    def fake_store_article(conn, source_id, url, title, published_at, full_text):
        store_calls.append(url)
        if url == "https://a.example/bad":
            raise sqlite3.OperationalError("simulated store failure")
        return real_store_article(conn, source_id, url, title, published_at, full_text)

    monkeypatch.setattr(orchestrator_module, "store_article", fake_store_article)

    summary = run_ingest(conn, sleep=_no_sleep)

    # the failing store must not abort the run: the second entry in source A and all of
    # source B must still be processed.
    assert store_calls == [
        "https://a.example/bad",
        "https://a.example/good",
        "https://b.example/good",
    ]
    assert summary == {"sources_ok": 2, "sources_failed": 0, "articles_stored": 2}
    error_rows = conn.execute(
        "SELECT message FROM run_log WHERE status = 'error'"
    ).fetchall()
    assert any("https://a.example/bad" in row["message"] for row in error_rows)
    article_row = conn.execute("SELECT COUNT(*) AS n FROM articles").fetchone()
    assert article_row["n"] == 2
    conn.close()
