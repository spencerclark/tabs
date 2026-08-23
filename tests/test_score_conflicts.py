from datetime import datetime, timedelta, timezone

from tabs.score.conflicts import RECENCY_WINDOW_DAYS, ConflictCandidate, resolve_conflict


def _candidate(claim_id, tier, retrieved_at, corroboration_count=0):
    return ConflictCandidate(
        claim_id=claim_id, effective_tier=tier, retrieved_at=retrieved_at,
        corroboration_count=corroboration_count,
    )


def test_resolve_conflict_prefers_the_higher_tier_claim_and_marks_the_loser_misinformation():
    same_time = datetime(2026, 8, 20, tzinfo=timezone.utc).isoformat()
    a = _candidate(1, tier=3, retrieved_at=same_time)
    b = _candidate(2, tier=1, retrieved_at=same_time)

    resolution, winning_claim_id, misinformation_claim_id = resolve_conflict(a, b)

    assert resolution == "auto-resolved"
    assert winning_claim_id == 1
    assert misinformation_claim_id == 2


def test_resolve_conflict_by_tier_is_symmetric_regardless_of_argument_order():
    same_time = datetime(2026, 8, 20, tzinfo=timezone.utc).isoformat()
    a = _candidate(1, tier=1, retrieved_at=same_time)
    b = _candidate(2, tier=3, retrieved_at=same_time)

    resolution, winning_claim_id, misinformation_claim_id = resolve_conflict(a, b)

    assert winning_claim_id == 2
    assert misinformation_claim_id == 1


def test_resolve_conflict_by_recency_does_not_mark_misinformation():
    newer = datetime(2026, 8, 20, tzinfo=timezone.utc)
    older = newer - timedelta(days=RECENCY_WINDOW_DAYS + 1)
    a = _candidate(1, tier=2, retrieved_at=newer.isoformat())
    b = _candidate(2, tier=2, retrieved_at=older.isoformat())

    resolution, winning_claim_id, misinformation_claim_id = resolve_conflict(a, b)

    assert resolution == "auto-resolved"
    assert winning_claim_id == 1  # the more recent claim
    assert misinformation_claim_id is None


def test_resolve_conflict_within_the_recency_window_falls_through_to_corroboration_count():
    newer = datetime(2026, 8, 20, tzinfo=timezone.utc)
    slightly_older = newer - timedelta(days=1)
    a = _candidate(1, tier=2, retrieved_at=newer.isoformat(), corroboration_count=3)
    b = _candidate(2, tier=2, retrieved_at=slightly_older.isoformat(), corroboration_count=0)

    resolution, winning_claim_id, misinformation_claim_id = resolve_conflict(a, b)

    assert resolution == "auto-resolved"
    assert winning_claim_id == 1
    assert misinformation_claim_id is None


def test_resolve_conflict_needs_review_when_every_tiebreaker_is_tied():
    newer = datetime(2026, 8, 20, tzinfo=timezone.utc)
    slightly_older = newer - timedelta(days=1)
    a = _candidate(1, tier=2, retrieved_at=newer.isoformat(), corroboration_count=1)
    b = _candidate(2, tier=2, retrieved_at=slightly_older.isoformat(), corroboration_count=1)

    resolution, winning_claim_id, misinformation_claim_id = resolve_conflict(a, b)

    assert resolution == "needs-review"
    assert winning_claim_id is None
    assert misinformation_claim_id is None


def test_resolve_conflict_treats_a_missing_retrieved_at_as_not_recency_decisive():
    a = _candidate(1, tier=2, retrieved_at=None, corroboration_count=2)
    b = _candidate(
        2, tier=2,
        retrieved_at=datetime(2026, 8, 1, tzinfo=timezone.utc).isoformat(),
        corroboration_count=0,
    )

    resolution, winning_claim_id, misinformation_claim_id = resolve_conflict(a, b)

    assert resolution == "auto-resolved"
    assert winning_claim_id == 1  # falls through to corroboration count
    assert misinformation_claim_id is None
