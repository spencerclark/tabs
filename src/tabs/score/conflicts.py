from datetime import datetime
from typing import Optional

# SPEC §4.2/§6.4: two claims of similar tier and recency are "too close... to resolve
# automatically" and go to needs-review. Two retrieved_at timestamps within this many
# days of each other are treated as "similar recency"; a tunable starting point, not
# fixed by SPEC.
RECENCY_WINDOW_DAYS = 7


class ConflictCandidate:
    """The minimal fields resolve_conflict needs from one side of a conflict."""

    def __init__(
        self, claim_id: int, effective_tier: int, retrieved_at: Optional[str],
        corroboration_count: int,
    ):
        self.claim_id = claim_id
        self.effective_tier = effective_tier
        self.retrieved_at = retrieved_at
        self.corroboration_count = corroboration_count


def resolve_conflict(
    a: ConflictCandidate, b: ConflictCandidate,
) -> tuple[str, Optional[int], Optional[int]]:
    """Decide how a conflict between two claims resolves.

    Returns (resolution, winning_claim_id, misinformation_claim_id):
    - resolution is "auto-resolved" or "needs-review".
    - winning_claim_id is set whenever the cascade (tier, then recency, then
      corroboration count) is decisive — audit/display information on the conflicts
      record, independent of whether any claim's status is forced.
    - misinformation_claim_id is set ONLY when a tier difference was the decisive
      factor: SPEC §6.4 reserves the misinformation status for being "contradicted by a
      higher-effective-tier source" specifically. A recency or corroboration-count win
      between equal-tier sources is real audit information, but it is not the strong,
      source-authority-specific signal SPEC authorizes for that label — it leaves both
      claims' status to be determined independently by their own confidence scores.
    """
    if a.effective_tier != b.effective_tier:
        winner, loser = (a, b) if a.effective_tier > b.effective_tier else (b, a)
        return "auto-resolved", winner.claim_id, loser.claim_id

    recency_winner = _decisive_by_recency(a, b)
    if recency_winner is not None:
        return "auto-resolved", recency_winner.claim_id, None

    if a.corroboration_count != b.corroboration_count:
        winner = a if a.corroboration_count > b.corroboration_count else b
        return "auto-resolved", winner.claim_id, None

    return "needs-review", None, None


def _decisive_by_recency(
    a: ConflictCandidate, b: ConflictCandidate,
) -> Optional[ConflictCandidate]:
    if a.retrieved_at is None or b.retrieved_at is None:
        return None
    date_a = datetime.fromisoformat(a.retrieved_at)
    date_b = datetime.fromisoformat(b.retrieved_at)
    if abs((date_a - date_b).days) <= RECENCY_WINDOW_DAYS:
        return None
    return a if date_a > date_b else b
