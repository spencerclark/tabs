import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

from tabs.score.embeddings import cosine_similarity

# SPEC §6.3: "compared via vector similarity against recent claims in the same category."
# None of the window, similarity floor, or candidate count is fixed by SPEC — all three
# are tunable starting points, not final values.
CORROBORATION_WINDOW_DAYS = 30
SIMILARITY_THRESHOLD = 0.75
MAX_CANDIDATES = 5


class Candidate:
    """One existing claim retrieved as a plausible match for a new claim."""

    def __init__(
        self, claim_id: int, claim_text: str, source_id: int, effective_tier: int,
        retrieved_at: str, corroboration_count: int, story_cluster_id: Optional[int],
        similarity: float,
    ):
        self.claim_id = claim_id
        self.claim_text = claim_text
        self.source_id = source_id
        self.effective_tier = effective_tier
        self.retrieved_at = retrieved_at
        self.corroboration_count = corroboration_count
        self.story_cluster_id = story_cluster_id
        self.similarity = similarity


def find_candidate_claims(
    conn: sqlite3.Connection, claim_id: int, article_id: int, source_id: int,
    category: str, embedding: list[float],
) -> list[Candidate]:
    """Find up to MAX_CANDIDATES existing claims plausibly related to a new claim.

    Scoped to the same category, within the last CORROBORATION_WINDOW_DAYS, excluding
    other claims from the same article (extraction can produce several claims from one
    article — those aren't independent corroboration), the same source (SPEC §6.3 defines
    corroboration as "same underlying claim, different source" — same-outlet repeat
    coverage isn't independent confirmation either), and the claim itself. Candidates are
    ranked by cosine similarity and filtered to SIMILARITY_THRESHOLD before being
    returned, so an LLM judgment call is only spent on plausible matches, not the whole
    category — deliberately plain Python, not a vector-index extension: this only ever
    ranks a small, bounded candidate set, never the whole (unbounded) claims table.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CORROBORATION_WINDOW_DAYS)).isoformat()
    rows = conn.execute(
        """
        SELECT c.id, c.claim_text, c.source_id, c.retrieved_at, c.corroboration_count,
               c.story_cluster_id, c.embedding, s.institutional_tier, s.earned_tier
        FROM claims c
        JOIN sources s ON s.id = c.source_id
        WHERE c.category = ?
          AND c.article_id != ?
          AND c.source_id != ?
          AND c.id != ?
          AND c.embedding IS NOT NULL
          AND c.retrieved_at >= ?
        """,
        (category, article_id, source_id, claim_id, cutoff),
    ).fetchall()

    scored = []
    for row in rows:
        candidate_embedding = json.loads(row["embedding"])
        similarity = cosine_similarity(embedding, candidate_embedding)
        if similarity < SIMILARITY_THRESHOLD:
            continue
        scored.append(
            Candidate(
                claim_id=row["id"], claim_text=row["claim_text"], source_id=row["source_id"],
                effective_tier=max(row["institutional_tier"], row["earned_tier"]),
                retrieved_at=row["retrieved_at"], corroboration_count=row["corroboration_count"],
                story_cluster_id=row["story_cluster_id"], similarity=similarity,
            )
        )

    scored.sort(key=lambda c: c.similarity, reverse=True)
    return scored[:MAX_CANDIDATES]
