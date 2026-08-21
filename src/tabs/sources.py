import sqlite3
from pathlib import Path

import yaml

from tabs.models import Source


REQUIRED_TEXT_FIELDS = ("name", "feed_url", "category")


def load_sources_yaml(path: Path) -> list[Source]:
    """Load and validate the sources allowlist.

    Raises ValueError with a message identifying the offending entry and field, so a
    malformed sources.yaml surfaces as a clean CLI error rather than a raw traceback.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}: expected a list of source entries, got {type(raw).__name__}"
        )

    sources = []
    for index, entry in enumerate(raw, start=1):
        label = f"{path}: entry {index}"
        if not isinstance(entry, dict):
            raise ValueError(
                f"{label}: expected a mapping of source fields, got {type(entry).__name__}"
            )
        if isinstance(entry.get("name"), str) and entry["name"].strip():
            label = f"{label} ({entry['name']})"

        for field in REQUIRED_TEXT_FIELDS:
            if field not in entry:
                raise ValueError(f"{label}: missing required field '{field}'")
            value = entry[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{label}: field '{field}' must be a non-empty string, got {value!r}"
                )

        if "institutional_tier" not in entry:
            raise ValueError(f"{label}: missing required field 'institutional_tier'")
        tier = entry["institutional_tier"]
        if isinstance(tier, bool) or not isinstance(tier, int):
            raise ValueError(
                f"{label}: field 'institutional_tier' must be an integer, got {tier!r}"
            )

        sources.append(
            Source(
                name=entry["name"],
                feed_url=entry["feed_url"],
                category=entry["category"],
                institutional_tier=tier,
            )
        )
    return sources


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
