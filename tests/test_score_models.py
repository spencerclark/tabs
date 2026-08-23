import pytest
from pydantic import ValidationError

from tabs.score.models import CandidateJudgment, MatchJudgments


def test_candidate_judgment_rejects_an_invalid_relationship():
    with pytest.raises(ValidationError):
        CandidateJudgment(candidate_claim_id=1, relationship="agrees")


def test_candidate_judgment_accepts_each_valid_relationship():
    for relationship in ("corroborating", "conflicting", "unrelated"):
        judgment = CandidateJudgment(candidate_claim_id=1, relationship=relationship)
        assert judgment.relationship == relationship


def test_candidate_judgment_carries_the_candidate_id():
    judgment = CandidateJudgment(candidate_claim_id=42, relationship="unrelated")
    assert judgment.candidate_claim_id == 42


def test_match_judgments_defaults_to_no_judgments():
    result = MatchJudgments()
    assert result.judgments == []
