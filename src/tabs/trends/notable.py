import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class NotableStory:
    story_cluster_id: int
    category: str
    corroboration_count: int
    most_recent_retrieved_at: str
    sample_claim_text: str


def notable_stories(
    conn: sqlite3.Connection, since_days: int, limit: int = 10,
) -> list[NotableStory]:
    """Story clusters with at least one non-misinformation claim retrieved within the
    last `since_days` days, ranked by corroboration count then recency (SPEC §7).

    corroboration_count is read as-is from claims.corroboration_count — maintained by
    Phase 2b's score/storage.py._join_story_cluster — rather than recomputed here, so
    there stays one existing definition of "how corroborated is this cluster," not two
    that could drift apart. This function does not filter or adjust that value: the
    column currently counts ALL cluster members, including any with
    status='misinformation' (Phase 2b's _join_story_cluster does not exclude them from
    the count). So a cluster containing some debunked claims may display a
    corroboration_count slightly higher than its number of currently-non-misinformation
    members. This is a known, documented limitation carried over from Phase 2b, not a
    bug in this function.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
    rows = conn.execute(
        """
        SELECT sc.id AS story_cluster_id, sc.category,
               MAX(c.corroboration_count) AS corroboration_count,
               MAX(c.retrieved_at) AS most_recent_retrieved_at
        FROM story_clusters sc
        JOIN claims c ON c.story_cluster_id = sc.id
        WHERE c.status != 'misinformation' AND c.retrieved_at >= ?
        GROUP BY sc.id
        ORDER BY corroboration_count DESC, most_recent_retrieved_at DESC
        LIMIT ?
        """,
        (cutoff, limit),
    ).fetchall()

    stories = []
    for row in rows:
        sample = conn.execute(
            "SELECT claim_text FROM claims "
            "WHERE story_cluster_id = ? AND status != 'misinformation' "
            "ORDER BY retrieved_at DESC LIMIT 1",
            (row["story_cluster_id"],),
        ).fetchone()
        stories.append(NotableStory(
            story_cluster_id=row["story_cluster_id"],
            category=row["category"],
            corroboration_count=row["corroboration_count"],
            most_recent_retrieved_at=row["most_recent_retrieved_at"],
            sample_claim_text=sample["claim_text"],
        ))
    return stories
