from tabs.db import get_connection, init_db
from tabs.ingest.storage import store_article


def test_store_article_inserts_new_article(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES ('S', 'https://s.example/feed', 'AppSec', 2, 2)"
    )
    conn.commit()

    article_id = store_article(
        conn, source_id=1, url="https://s.example/a", title="A",
        published_at="2026-08-01", full_text="original text",
    )

    row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
    assert row["url"] == "https://s.example/a"
    assert row["previous_version_id"] is None
    conn.close()


def test_store_article_returns_same_id_when_content_unchanged(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES ('S', 'https://s.example/feed', 'AppSec', 2, 2)"
    )
    conn.commit()
    first_id = store_article(
        conn, source_id=1, url="https://s.example/a", title="A",
        published_at="2026-08-01", full_text="original text",
    )

    second_id = store_article(
        conn, source_id=1, url="https://s.example/a", title="A",
        published_at="2026-08-01", full_text="original text",
    )

    assert first_id == second_id
    count = conn.execute("SELECT COUNT(*) AS n FROM articles").fetchone()["n"]
    assert count == 1
    conn.close()


def test_store_article_creates_new_version_when_content_changes(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES ('S', 'https://s.example/feed', 'AppSec', 2, 2)"
    )
    conn.commit()
    first_id = store_article(
        conn, source_id=1, url="https://s.example/a", title="A",
        published_at="2026-08-01", full_text="original text",
    )

    second_id = store_article(
        conn, source_id=1, url="https://s.example/a", title="A",
        published_at="2026-08-01", full_text="updated text",
    )

    assert second_id != first_id
    row = conn.execute("SELECT previous_version_id FROM articles WHERE id = ?", (second_id,)).fetchone()
    assert row["previous_version_id"] == first_id
    conn.close()
