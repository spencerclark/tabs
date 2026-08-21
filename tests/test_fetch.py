import feedparser
import pytest

from tabs.ingest.fetch import FeedFetchError, fetch_feed


def _make_parsed(bozo: bool, entries: list[dict]) -> feedparser.FeedParserDict:
    parsed = feedparser.FeedParserDict()
    parsed["bozo"] = bozo
    parsed["entries"] = entries
    return parsed


def test_fetch_feed_returns_entries_on_success(monkeypatch):
    entry = {
        "link": "https://example.com/a",
        "title": "A",
        "published": "2026-08-01",
        "summary": "summary text",
    }
    monkeypatch.setattr(feedparser, "parse", lambda url: _make_parsed(False, [entry]))

    entries = fetch_feed("https://example.com/feed.xml", sleep=lambda s: None)

    assert len(entries) == 1
    assert entries[0].url == "https://example.com/a"
    assert entries[0].title == "A"


def test_fetch_feed_retries_then_succeeds(monkeypatch):
    calls = {"count": 0}
    sleep_calls = []

    def fake_parse(url):
        calls["count"] += 1
        if calls["count"] < 2:
            return _make_parsed(True, [])
        return _make_parsed(
            False, [{"link": "u", "title": "t", "published": None, "summary": "s"}]
        )

    monkeypatch.setattr(feedparser, "parse", fake_parse)

    entries = fetch_feed("https://example.com/feed.xml", sleep=sleep_calls.append)

    assert calls["count"] == 2
    assert len(entries) == 1
    # Verify backoff values: first retry should sleep 2 * (2**0) = 2 seconds
    assert sleep_calls[0] == 2  # Backoff on first failure
    assert sleep_calls[1] == 1  # REQUEST_DELAY_SECONDS on success


def test_fetch_feed_raises_after_exhausting_retries(monkeypatch):
    calls = {"count": 0}

    def fake_parse(url):
        calls["count"] += 1
        return _make_parsed(True, [])

    monkeypatch.setattr(feedparser, "parse", fake_parse)

    with pytest.raises(FeedFetchError):
        fetch_feed("https://example.com/feed.xml", sleep=lambda s: None)

    # Verify exactly 3 calls to parse were made (MAX_RETRIES)
    assert calls["count"] == 3
