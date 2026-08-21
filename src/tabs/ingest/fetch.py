import time
from dataclasses import dataclass

import feedparser

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2
REQUEST_DELAY_SECONDS = 1


class FeedFetchError(Exception):
    """Raised when a feed cannot be fetched after all retries are exhausted."""


@dataclass
class FetchedEntry:
    url: str
    title: str
    published_at: str | None
    summary: str


def fetch_feed(feed_url: str, sleep=time.sleep) -> list[FetchedEntry]:
    """Fetch and parse a feed, retrying transient failures with backoff."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        parsed = feedparser.parse(feed_url)
        if parsed.get("bozo") and not parsed.entries:
            last_error = parsed.get("bozo_exception")
            if attempt < MAX_RETRIES - 1:  # no point backing off after the last attempt
                sleep(BACKOFF_BASE_SECONDS * (2**attempt))
            continue
        sleep(REQUEST_DELAY_SECONDS)
        return [
            FetchedEntry(
                url=entry.get("link", ""),
                title=entry.get("title", ""),
                published_at=entry.get("published"),
                summary=entry.get("summary", ""),
            )
            for entry in parsed.entries
        ]
    raise FeedFetchError(
        f"Failed to fetch {feed_url} after {MAX_RETRIES} attempts: {last_error}"
    )
