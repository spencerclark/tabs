from typing import Literal, Optional

from pydantic import BaseModel, Field

Category = Literal["AppSec", "AI Security", "Policy & Industry"]


class TriageResult(BaseModel):
    """Output schema for the cheap relevance/category pass over a feed entry."""

    in_scope: bool
    category: Optional[Category] = None


class ExtractedItem(BaseModel):
    """A single claim or perspective pulled from an article's full text."""

    text: str
    supporting_excerpt: str
    item_type: Literal["factual", "prediction", "opinion"]
    category: Category
    sub_tags: list[str] = Field(default_factory=list)
    llm_certainty: float = Field(ge=0.0, le=1.0)
    author: Optional[str] = None


class ExtractionResult(BaseModel):
    """Output schema for the full-text extraction pass over one article."""

    items: list[ExtractedItem] = Field(default_factory=list)
    injection_anomaly: Optional[str] = None
