import pytest

from tabs.db import get_connection, init_db
from tabs.ingest.storage import (
    MAX_ARTICLE_BYTES,
    USER_AGENT,
    _extract_text,
    fetch_article_text,
    store_article,
)


class FakeResponse:
    """Minimal stand-in for requests.Response as fetch_article_text uses it."""

    def __init__(self, chunks: list[bytes], encoding: str = "utf-8", status_error=None):
        self._chunks = chunks
        self.encoding = encoding
        self._status_error = status_error
        self.closed = False

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error

    def iter_content(self, chunk_size: int):
        yield from self._chunks

    def close(self):
        self.closed = True


def test_fetch_article_text_returns_extracted_text_and_sends_a_user_agent():
    captured = {}
    body = b"<html><body><script>x=1</script><p>Hello  world</p></body></html>"

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse([body])

    assert fetch_article_text("https://s.example/a", http_get=fake_get) == "Hello world"
    assert captured["url"] == "https://s.example/a"
    assert captured["kwargs"]["headers"]["User-Agent"] == USER_AGENT
    assert captured["kwargs"]["stream"] is True
    assert captured["kwargs"]["timeout"] == 10


def test_fetch_article_text_rejects_non_http_schemes():
    def fake_get(url, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError(f"must not fetch {url}")

    for url in ("file:///etc/passwd", "ftp://s.example/a", "gopher://s.example/a", "not-a-url"):
        with pytest.raises(ValueError):
            fetch_article_text(url, http_get=fake_get)


def test_fetch_article_text_rejects_bodies_over_the_size_cap():
    oversized = [b"a" * 65536] * ((MAX_ARTICLE_BYTES // 65536) + 2)
    response = FakeResponse(oversized)

    with pytest.raises(ValueError, match="exceeds"):
        fetch_article_text("https://s.example/big", http_get=lambda url, **kw: response)

    assert response.closed is True


def test_fetch_article_text_accepts_a_body_at_the_size_cap():
    chunks = [b"x" * 1024] * ((MAX_ARTICLE_BYTES // 1024) - 1)

    text = fetch_article_text(
        "https://s.example/ok", http_get=lambda url, **kw: FakeResponse(chunks)
    )

    assert len(text) == MAX_ARTICLE_BYTES - 1024


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

    article_id, created = store_article(
        conn, source_id=1, url="https://s.example/a", title="A",
        published_at="2026-08-01", full_text="original text",
    )

    assert created is True
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
    first_id, _ = store_article(
        conn, source_id=1, url="https://s.example/a", title="A",
        published_at="2026-08-01", full_text="original text",
    )

    second_id, created = store_article(
        conn, source_id=1, url="https://s.example/a", title="A",
        published_at="2026-08-01", full_text="original text",
    )

    assert first_id == second_id
    assert created is False
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
    first_id, _ = store_article(
        conn, source_id=1, url="https://s.example/a", title="A",
        published_at="2026-08-01", full_text="original text",
    )

    second_id, created = store_article(
        conn, source_id=1, url="https://s.example/a", title="A",
        published_at="2026-08-01", full_text="updated text",
    )

    assert second_id != first_id
    assert created is True
    row = conn.execute("SELECT previous_version_id FROM articles WHERE id = ?", (second_id,)).fetchone()
    assert row["previous_version_id"] == first_id
    conn.close()
