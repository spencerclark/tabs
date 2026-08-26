from typing import Literal

# SPEC §6.4: "The exact weights and admission threshold are not fixed by this spec —
# they're empirically tuned during implementation against the golden set... rather than
# chosen up front." These are defensible, documented starting values, not final ones —
# expect to revisit once real curated data exists to tune against. Kept as named module
# constants (not buried in an expression) so they're easy to find and adjust without a
# schema change, per SPEC's explicit requirement.
TIER_WEIGHT = 1.0
CORROBORATION_WEIGHT = 1.5
CERTAINTY_WEIGHT = 2.0
CLAIM_TYPE_WEIGHTS = {"factual": 1.0, "prediction": 0.3}
VERIFICATION_THRESHOLD = 4.0


def compute_confidence_score(
    effective_tier: int,
    corroboration_count: int,
    llm_certainty: float,
    claim_type: Literal["factual", "prediction"],
) -> float:
    """Composite confidence score per SPEC §6.4: source tier + corroboration count + LLM
    certainty + claim-type weight, each independently weighted."""
    return (
        TIER_WEIGHT * effective_tier
        + CORROBORATION_WEIGHT * corroboration_count
        + CERTAINTY_WEIGHT * llm_certainty
        + CLAIM_TYPE_WEIGHTS[claim_type]
    )


def gate_status(score: float, contradicted_by_higher_tier: bool) -> str:
    """Map a confidence score (+ whether a higher-tier source contradicts this claim) to
    a claims.status value, per SPEC §6.4's three-way admission rule.

    contradicted_by_higher_tier always wins regardless of score — SPEC: "Contradicted by
    a higher-effective-tier source → misinformation, regardless of the contradicted
    claim's own score."
    """
    if contradicted_by_higher_tier:
        return "misinformation"
    return "verified" if score >= VERIFICATION_THRESHOLD else "unverified"
