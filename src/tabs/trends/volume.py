import sqlite3


def category_volume(conn: sqlite3.Connection, start: str, end: str) -> dict[str, int]:
    """Count of claims + perspectives retrieved in [start, end), grouped by top-level
    category.

    Claims with status='misinformation' are excluded (SPEC §7 — a debunked claim must
    not inflate a category's volume). Perspectives are never truth-gated (SPEC §4.1) and
    have no status column, so every perspective in the window counts.
    """
    counts: dict[str, int] = {}
    _accumulate(counts, conn.execute(
        "SELECT category, COUNT(*) AS n FROM claims "
        "WHERE status != 'misinformation' AND retrieved_at >= ? AND retrieved_at < ? "
        "GROUP BY category",
        (start, end),
    ).fetchall())
    _accumulate(counts, conn.execute(
        "SELECT category, COUNT(*) AS n FROM perspectives "
        "WHERE retrieved_at >= ? AND retrieved_at < ? GROUP BY category",
        (start, end),
    ).fetchall())
    return counts


def sub_tag_volume(conn: sqlite3.Connection, start: str, end: str) -> dict[tuple[str, str], int]:
    """Count of claims + perspectives retrieved in [start, end), grouped by
    (category, sub_tag).

    sub_tags is a JSON array column; json_each expands it so an item tagged with several
    sub_tags contributes to each one's count. The CASE guards both a NULL sub_tags value
    (the column is nullable) and an actually-invalid JSON value — json_each(NULL) or
    json_each(<malformed JSON>) would otherwise raise sqlite3.OperationalError and abort
    the whole `tabs trends` command for one bad row. Same misinformation exclusion as
    category_volume.
    """
    counts: dict[tuple[str, str], int] = {}
    _accumulate_tags(counts, conn.execute(
        "SELECT category, je.value AS sub_tag, COUNT(*) AS n "
        "FROM claims, json_each(CASE WHEN json_valid(claims.sub_tags) THEN claims.sub_tags "
        "ELSE '[]' END) AS je "
        "WHERE status != 'misinformation' AND retrieved_at >= ? AND retrieved_at < ? "
        "GROUP BY category, je.value",
        (start, end),
    ).fetchall())
    _accumulate_tags(counts, conn.execute(
        "SELECT category, je.value AS sub_tag, COUNT(*) AS n "
        "FROM perspectives, json_each(CASE WHEN json_valid(perspectives.sub_tags) "
        "THEN perspectives.sub_tags ELSE '[]' END) AS je "
        "WHERE retrieved_at >= ? AND retrieved_at < ? GROUP BY category, je.value",
        (start, end),
    ).fetchall())
    return counts


def _accumulate(counts: dict[str, int], rows) -> None:
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + row["n"]


def _accumulate_tags(counts: dict[tuple[str, str], int], rows) -> None:
    for row in rows:
        key = (row["category"], row["sub_tag"])
        counts[key] = counts.get(key, 0) + row["n"]
