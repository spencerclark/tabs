from tabs.score.scoring import (
    CLAIM_TYPE_WEIGHTS,
    CORROBORATION_WEIGHT,
    TIER_WEIGHT,
    CERTAINTY_WEIGHT,
    VERIFICATION_THRESHOLD,
    compute_confidence_score,
    gate_status,
)


def test_compute_confidence_score_combines_all_four_weighted_factors():
    score = compute_confidence_score(
        effective_tier=3, corroboration_count=2, llm_certainty=0.8, claim_type="factual",
    )
    expected = (
        TIER_WEIGHT * 3 + CORROBORATION_WEIGHT * 2 + CERTAINTY_WEIGHT * 0.8
        + CLAIM_TYPE_WEIGHTS["factual"]
    )
    assert score == expected


def test_compute_confidence_score_weighs_predictions_lower_than_factual_claims():
    factual = compute_confidence_score(
        effective_tier=2, corroboration_count=0, llm_certainty=0.5, claim_type="factual",
    )
    prediction = compute_confidence_score(
        effective_tier=2, corroboration_count=0, llm_certainty=0.5, claim_type="prediction",
    )
    assert prediction < factual


def test_gate_status_returns_misinformation_when_contradicted_regardless_of_score():
    status = gate_status(score=1000.0, contradicted_by_higher_tier=True)
    assert status == "misinformation"


def test_gate_status_returns_verified_when_score_clears_the_threshold():
    status = gate_status(score=VERIFICATION_THRESHOLD, contradicted_by_higher_tier=False)
    assert status == "verified"


def test_gate_status_returns_unverified_when_score_is_below_the_threshold():
    status = gate_status(score=VERIFICATION_THRESHOLD - 0.01, contradicted_by_higher_tier=False)
    assert status == "unverified"
