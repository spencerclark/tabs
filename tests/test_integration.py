"""End-to-end ingest: CLI -> sync_sources -> run_ingest -> fetch_feed -> store_article
-> triage/extraction -> claims/perspectives.

Every other test in the suite stubs at a module boundary, so the real contracts between
these layers (FetchedEntry, the extracted-text hashing path, the CLI wiring, the
triage/extraction routing into claims vs perspectives) are never exercised together.
This test stubs only the outermost boundaries: feedparser.parse, requests.get, and the
Anthropic client construction (anthropic.Anthropic itself is never invoked for real).
"""

import feedparser
import requests
from click.testing import CliRunner

import tabs.commands.ingest_cmd as ingest_cmd_module
import tabs.ingest.fetch as fetch_module
import tabs.ingest.orchestrator as orchestrator_module
from tabs.cli import main
from tabs.curate.models import ExtractedItem, ExtractionResult, TriageResult
from tabs.db import get_connection
from tabs.ingest.storage import _extract_text, _hash_content

FEED_URL = "https://sec.example/feed"
ARTICLE_A = "https://sec.example/posts/rce"
ARTICLE_B = "https://sec.example/posts/policy"


def _page(body: str, nonce: str) -> bytes:
    """An article page wrapped in the boilerplate a real news site would ship."""
    return (
        "<html><head><title>Sec Example</title>"
        f"<script>window.adSlot='{nonce}';</script>"
        "<style>.ad { display: block; }</style></head>"
        "<body><nav>Home | Archive | Subscribe</nav>"
        f"<article>\n  {body}\n</article>"
        f"<div class='ad' data-request-id='{nonce}'>Ad</div>"
        "<footer>&copy; Sec Example</footer></body></html>"
    ).encode("utf-8")


class _Response:
    def __init__(self, body: bytes):
        self._body = body
        self.encoding = "utf-8"

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]

    def close(self):
        return None


def _parsed_feed():
    parsed = feedparser.FeedParserDict()
    parsed["bozo"] = False
    parsed["entries"] = [
        {"link": ARTICLE_A, "title": "Critical RCE", "published": "2026-08-19", "summary": "s"},
        {"link": ARTICLE_B, "title": "New Guidance", "published": "2026-08-20", "summary": "s"},
    ]
    return parsed


class _FakeParseResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


class _FakeMessages:
    """Stands in for client.messages — routes on output_format, like the real API
    would route on the caller's requested schema."""

    def parse(self, *, model, max_tokens, system, messages, output_format):
        if output_format is TriageResult:
            return _FakeParseResponse(TriageResult(in_scope=True, category="AppSec"))
        if output_format is ExtractionResult:
            return _FakeParseResponse(
                ExtractionResult(
                    items=[
                        ExtractedItem(
                            text="A vulnerability was disclosed and patched.",
                            supporting_excerpt="patched", item_type="factual",
                            category="AppSec", sub_tags=["Patch"], llm_certainty=0.85,
                        ),
                    ]
                )
            )
        raise AssertionError(f"unexpected output_format: {output_format}")


class _FakeAnthropicClient:
    def __init__(self):
        self.messages = _FakeMessages()


def _install_stubs(monkeypatch, pages: dict[str, bytes], fetched: list[str]):
    monkeypatch.setattr(feedparser, "parse", lambda url: _parsed_feed())

    def fake_get(url, **kwargs):
        fetched.append(url)
        return _Response(pages[url])

    monkeypatch.setattr(requests, "get", fake_get)
    # keep the real code paths, just don't actually wait out the rate-limit delays
    monkeypatch.setattr(fetch_module, "REQUEST_DELAY_SECONDS", 0)
    monkeypatch.setattr(orchestrator_module, "ARTICLE_REQUEST_DELAY_SECONDS", 0)
    # never make a real Anthropic API call
    monkeypatch.setattr(ingest_cmd_module.anthropic, "Anthropic", _FakeAnthropicClient)


def _write_sources_yaml(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(
        "- name: Sec Example\n"
        f"  feed_url: {FEED_URL}\n"
        "  category: AppSec\n"
        "  institutional_tier: 2\n"
    )
    return path


def test_ingest_command_end_to_end_stores_extracted_articles(tmp_path, monkeypatch):
    db_path = tmp_path / "tabs.db"
    sources_path = _write_sources_yaml(tmp_path)
    pages = {
        ARTICLE_A: _page("A critical RCE was patched today.", nonce="req-1111"),
        ARTICLE_B: _page("New guidance was published today.", nonce="req-1111"),
    }
    fetched: list[str] = []
    _install_stubs(monkeypatch, pages, fetched)

    result = CliRunner().invoke(
        main, ["--db-path", str(db_path), "ingest", "--sources-path", str(sources_path)]
    )

    assert result.exit_code == 0, result.output
    assert "sources_ok=1 sources_failed=0 articles_stored=2" in result.output
    assert "claims_extracted=2" in result.output
    assert fetched == [ARTICLE_A, ARTICLE_B]

    conn = get_connection(db_path)
    source = conn.execute("SELECT id, name, last_successful_fetch_at FROM sources").fetchone()
    assert source["name"] == "Sec Example"
    assert source["last_successful_fetch_at"] is not None

    rows = conn.execute(
        "SELECT source_id, url, title, full_text, content_hash, published_at FROM articles ORDER BY url"
    ).fetchall()
    assert [row["url"] for row in rows] == [ARTICLE_B, ARTICLE_A]
    by_url = {row["url"]: row for row in rows}

    stored = by_url[ARTICLE_A]
    assert stored["source_id"] == source["id"]
    assert stored["title"] == "Critical RCE"
    assert stored["published_at"] == "2026-08-19"
    # what lands in the DB is normalized visible text, not the raw HTML
    assert stored["full_text"] == _extract_text(pages[ARTICLE_A].decode("utf-8"))
    assert "A critical RCE was patched today." in stored["full_text"]
    assert "<script>" not in stored["full_text"]
    assert "window.adSlot" not in stored["full_text"]
    assert stored["content_hash"] == _hash_content(stored["full_text"])

    # one factual claim extracted per article, via the fake client's canned response
    claim_rows = conn.execute("SELECT article_id, claim_text, status FROM claims").fetchall()
    assert len(claim_rows) == 2
    assert all(row["status"] == "unverified" for row in claim_rows)  # not scored yet — Phase 2b's job

    run_row = conn.execute(
        "SELECT status, message FROM run_log WHERE source_id IS NULL"
    ).fetchone()
    assert run_row["status"] == "success"
    conn.close()


def test_ingest_command_end_to_end_is_idempotent_across_boilerplate_churn(tmp_path, monkeypatch):
    db_path = tmp_path / "tabs.db"
    sources_path = _write_sources_yaml(tmp_path)
    runner = CliRunner()
    argv = ["--db-path", str(db_path), "ingest", "--sources-path", str(sources_path)]

    first_pages = {
        ARTICLE_A: _page("A critical RCE was patched today.", nonce="req-1111"),
        ARTICLE_B: _page("New guidance was published today.", nonce="req-1111"),
    }
    _install_stubs(monkeypatch, first_pages, [])
    first = runner.invoke(main, argv)
    assert first.exit_code == 0, first.output
    assert "articles_stored=2" in first.output
    assert "claims_extracted=2" in first.output

    # second run: same article text, different ad/script tokens and whitespace, plus one
    # genuine edit to article B
    second_pages = {
        ARTICLE_A: _page("A   critical RCE was patched today.\n", nonce="req-9999"),
        ARTICLE_B: _page("New guidance was withdrawn today.", nonce="req-9999"),
    }
    refetched: list[str] = []
    _install_stubs(monkeypatch, second_pages, refetched)
    second = runner.invoke(main, argv)

    assert second.exit_code == 0, second.output
    # both are inside the 14-day re-check window, so both are re-fetched...
    assert refetched == [ARTICLE_A, ARTICLE_B]
    # ...but only the genuinely edited one is stored as a new version...
    assert "articles_stored=1" in second.output
    # ...and only that one is re-curated
    assert "claims_extracted=1" in second.output

    conn = get_connection(db_path)
    a_rows = conn.execute(
        "SELECT id FROM articles WHERE url = ? ORDER BY id", (ARTICLE_A,)
    ).fetchall()
    assert len(a_rows) == 1  # boilerplate churn alone must not create a version

    b_rows = conn.execute(
        "SELECT id, full_text, previous_version_id FROM articles WHERE url = ? ORDER BY id",
        (ARTICLE_B,),
    ).fetchall()
    assert len(b_rows) == 2
    assert b_rows[1]["previous_version_id"] == b_rows[0]["id"]
    assert "withdrawn" in b_rows[1]["full_text"]

    # 2 claims from the first run (one per article) + 1 from the second run's single
    # re-curated article (B) — A's boilerplate churn must not trigger re-extraction
    claim_count = conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"]
    assert claim_count == 3
    conn.close()
