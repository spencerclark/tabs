import sqlite3
from pathlib import Path

import yaml

from tabs.models import Source


def load_sources_yaml(path: Path) -> list[Source]:
    with open(path) as f:
        raw = yaml.safe_load(f) or []
    return [
        Source(
            name=entry["name"],
            feed_url=entry["feed_url"],
            category=entry["category"],
            institutional_tier=entry["institutional_tier"],
        )
        for entry in raw
    ]


def sync_sources(conn: sqlite3.Connection, sources: list[Source]) -> None:
    for source in sources:
        existing = conn.execute(
            "SELECT id FROM sources WHERE feed_url = ?", (source.feed_url,)
        ).fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO sources
                    (name, feed_url, category, institutional_tier, earned_tier)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    source.name,
                    source.feed_url,
                    source.category,
                    source.institutional_tier,
                    source.institutional_tier,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE sources
                SET name = ?, category = ?, institutional_tier = ?
                WHERE feed_url = ?
                """,
                (source.name, source.category, source.institutional_tier, source.feed_url),
            )
    conn.commit()
