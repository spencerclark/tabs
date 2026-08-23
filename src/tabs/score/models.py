from typing import Literal

from pydantic import BaseModel, Field


class CandidateJudgment(BaseModel):
    """The model's relationship judgment for one specific candidate claim.

    candidate_claim_id is echoed back explicitly rather than relying on list-position
    order, since structured-output list length/order from the model isn't a hard
    guarantee — matching judgments back to candidates by id is robust to either.
    """

    candidate_claim_id: int
    relationship: Literal["corroborating", "conflicting", "unrelated"]


class MatchJudgments(BaseModel):
    """Output schema for the Claude Sonnet 5 corroboration/conflict judgment call."""

    judgments: list[CandidateJudgment] = Field(default_factory=list)
