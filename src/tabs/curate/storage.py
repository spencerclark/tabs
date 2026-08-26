import json
import sqlite3
from datetime import datetime, timezone

from tabs.curate.models import ExtractionResult


def store_extraction_result(
    conn: sqlite3.Connection,
    article_id: int,
    source_id: int,
    published_at: str | None,
    retrieved_at: str,
    extraction: ExtractionResult,
) -> dict:
    """Persist an ExtractionResult's items into claims/perspectives, and any injection anomaly.

    Factual and prediction items go to `claims` (left at the schema's default
    `status='unverified'` — confidence scoring and gating are Phase 2b's job, once
    corroboration counts are computable). Opinion items go to `perspectives`, which has
    no status/confidence columns at all — perspectives are never truth-gated (SPEC §4.1).

    Returns claim_ids so callers can run corroboration/scoring on exactly the claims just
    created.
    """
    created_at = datetime.now(timezone.utc).isoformat()
    claims_created = 0
    perspectives_created = 0
    claim_ids = []

    for item in extraction.items:
        sub_tags_json = json.dumps(item.sub_tags)
        if item.item_type == "opinion":
            conn.execute(
                """
                INSERT INTO perspectives
                    (article_id, source_id, perspective_text, supporting_excerpt,
                     category, sub_tags, author, published_at, retrieved_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article_id, source_id, item.text, item.supporting_excerpt,
                    item.category, sub_tags_json, item.author,
                    published_at, retrieved_at, created_at,
                ),
            )
            perspectives_created += 1
        else:
            cursor = conn.execute(
                """
                INSERT INTO claims
                    (article_id, source_id, claim_text, supporting_excerpt, claim_type,
                     category, sub_tags, llm_certainty, author, published_at,
                     retrieved_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article_id, source_id, item.text, item.supporting_excerpt,
                    item.item_type, item.category, sub_tags_json, item.llm_certainty,
                    item.author, published_at, retrieved_at, created_at,
                ),
            )
            claim_ids.append(cursor.lastrowid)
            claims_created += 1

    if extraction.injection_anomaly:
        conn.execute(
            "INSERT INTO anomaly_flags (article_id, reason, created_at) VALUES (?, ?, ?)",
            (article_id, extraction.injection_anomaly, created_at),
        )

    conn.commit()
    return {
        "claims_created": claims_created,
        "perspectives_created": perspectives_created,
        "claim_ids": claim_ids,
    }
