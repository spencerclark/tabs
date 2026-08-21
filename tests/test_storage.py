from tabs.db import get_connection, init_db
from tabs.ingest.storage import _extract_text, store_article


def test_extract_text_strips_script_and_style_blocks_with_their_content():
    html = (
        "<html><head><style>body { color: red; }</style>"
        "<script>var token = 'abc123';</script></head>"
        "<body><p>Real content.</p></body></html>"
    )

    assert _extract_text(html) == "Real content."


def test_extract_text_strips_tags_and_unescapes_entities():
    html = "<div class='x'><h1>Title</h1><p>A &amp; B &lt;ok&gt;</p></div>"

    assert _extract_text(html) == "Title A & B <ok>"


def test_extract_text_collapses_whitespace_and_strips_edges():
    html = "\n  <p>one   two\n\n\tthree</p>  \n"

    assert _extract_text(html) == "one two three"


def test_extract_text_ignores_boilerplate_churn_but_keeps_real_changes():
    page_a = (
        "<html><body><nav>Home | About</nav>"
        "<script>var adSlot = 'req-1111';</script>"
        "<article>The vulnerability was patched.</article>"
        "<div class='ad' data-nonce='aaaa'></div></body></html>"
    )
    page_b = (
        "<html><body><nav>Home | About</nav>"
        "<script>var adSlot = 'req-2222';</script>"
        "<article>The vulnerability was patched.</article>"
        "<div class='ad' data-nonce='bbbb'></div></body></html>"
    )
    page_changed = page_a.replace(
        "The vulnerability was patched.", "The vulnerability was NOT patched."
    )

    assert _extract_text(page_a) == _extract_text(page_b)
    assert _extract_text(page_a) != _extract_text(page_changed)


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
