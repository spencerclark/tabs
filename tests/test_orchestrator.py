import sqlite3
from datetime import datetime, timezone

import tabs.ingest.orchestrator as orchestrator_module
from tabs.db import get_connection, init_db
from tabs.ingest.fetch import FeedFetchError, FetchedEntry
from tabs.ingest.orchestrator import run_ingest


def _insert_source(conn, name, feed_url):
    conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES (?, ?, 'AppSec', 2, 2)",
        (name, feed_url),
    )
    conn.commit()


def test_run_ingest_stores_articles_and_records_success(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Good Source", "https://good.example/feed")

    entry = FetchedEntry(url="https://good.example/a", title="A", published_at=None, summary="s")
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [entry])
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    summary = run_ingest(conn)

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

    summary = run_ingest(conn)

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

    summary = run_ingest(conn)

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
        "VALUES (1, 'https://s.example/recent', 'Recent', 'text', 'hash', NULL, ?, NULL)",
        (recent_retrieved_at,),
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

    summary = run_ingest(conn)

    # recent article, inside the 14-day re-check window: must be re-fetched to detect edits/retractions
    assert fetch_calls == ["https://s.example/recent"]
    assert summary["articles_stored"] == 1
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

    summary = run_ingest(conn)

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
