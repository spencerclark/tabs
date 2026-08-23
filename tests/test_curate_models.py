import pytest
from pydantic import ValidationError

from tabs.curate.models import ExtractedItem, ExtractionResult, TriageResult


def test_triage_result_defaults_category_to_none():
    result = TriageResult(in_scope=False)
    assert result.category is None


def test_triage_result_accepts_a_valid_category():
    result = TriageResult(in_scope=True, category="AI Security")
    assert result.category == "AI Security"


def test_triage_result_rejects_an_invalid_category():
    with pytest.raises(ValidationError):
        TriageResult(in_scope=True, category="Not A Category")


def test_extracted_item_defaults_sub_tags_to_empty_list_and_author_to_none():
    item = ExtractedItem(
        text="Claim text", supporting_excerpt="quote", item_type="factual",
        category="AppSec", llm_certainty=0.8,
    )
    assert item.sub_tags == []
    assert item.author is None


def test_extracted_item_rejects_llm_certainty_out_of_range():
    with pytest.raises(ValidationError):
        ExtractedItem(
            text="Claim text", supporting_excerpt="quote", item_type="factual",
            category="AppSec", llm_certainty=1.5,
        )


def test_extracted_item_rejects_an_invalid_item_type():
    with pytest.raises(ValidationError):
        ExtractedItem(
            text="Claim text", supporting_excerpt="quote", item_type="rumor",
            category="AppSec", llm_certainty=0.5,
        )


def test_extraction_result_defaults_to_no_items_and_no_anomaly():
    result = ExtractionResult()
    assert result.items == []
    assert result.injection_anomaly is None
