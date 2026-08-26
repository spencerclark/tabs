from tabs.curate.models import ExtractedItem, ExtractionResult
from tabs.curate.storage import store_extraction_result
from tabs.db import get_connection, init_db


def _insert_source_and_article(conn):
    conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES ('Source', 'https://s.example/feed', 'AppSec', 2, 2)"
    )
    conn.execute(
        "INSERT INTO articles (source_id, url, title, full_text, content_hash, "
        "published_at, retrieved_at, previous_version_id) "
        "VALUES (1, 'https://s.example/a', 'A', 'text', 'hash', '2026-08-01', "
        "'2026-08-01T00:00:00+00:00', NULL)"
    )
    conn.commit()


def test_store_extraction_result_routes_factual_and_prediction_items_to_claims(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source_and_article(conn)

    extraction = ExtractionResult(
        items=[
            ExtractedItem(
                text="Factual claim", supporting_excerpt="quote1", item_type="factual",
                category="AppSec", sub_tags=["Tag1"], llm_certainty=0.9, author="Jane",
            ),
            ExtractedItem(
                text="Predicted claim", supporting_excerpt="quote2", item_type="prediction",
                category="AI Security", sub_tags=[], llm_certainty=0.3,
            ),
        ]
    )

    counts = store_extraction_result(
        conn, article_id=1, source_id=1,
        published_at="2026-08-01", retrieved_at="2026-08-01T00:00:00+00:00",
        extraction=extraction,
    )

    assert counts["claims_created"] == 2
    assert counts["perspectives_created"] == 0
    assert len(counts["claim_ids"]) == 2
    db_claim_ids = {
        row["id"] for row in conn.execute("SELECT id FROM claims").fetchall()
    }
    assert set(counts["claim_ids"]) == db_claim_ids
    rows = conn.execute(
        "SELECT claim_text, claim_type, category, sub_tags, llm_certainty, author, "
        "status, published_at, retrieved_at, article_id, source_id, supporting_excerpt "
        "FROM claims ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["claim_text"] == "Factual claim"
    assert rows[0]["claim_type"] == "factual"
    assert rows[0]["category"] == "AppSec"
    assert rows[0]["sub_tags"] == '["Tag1"]'
    assert rows[0]["llm_certainty"] == 0.9
    assert rows[0]["author"] == "Jane"
    assert rows[0]["status"] == "unverified"  # default — Phase 2b scores/gates it
    assert rows[0]["published_at"] == "2026-08-01"
    assert rows[0]["retrieved_at"] == "2026-08-01T00:00:00+00:00"
    assert rows[0]["article_id"] == 1
    assert rows[0]["source_id"] == 1
    assert rows[0]["supporting_excerpt"] == "quote1"
    assert rows[1]["claim_type"] == "prediction"
    assert rows[1]["author"] is None
    assert rows[1]["supporting_excerpt"] == "quote2"
    conn.close()


def test_store_extraction_result_routes_opinion_items_to_perspectives(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source_and_article(conn)

    extraction = ExtractionResult(
        items=[
            ExtractedItem(
                text="An opinion", supporting_excerpt="quote", item_type="opinion",
                category="Policy & Industry", sub_tags=["Take"], llm_certainty=0.5,
                author="John",
            ),
        ]
    )

    counts = store_extraction_result(
        conn, article_id=1, source_id=1,
        published_at="2026-08-01", retrieved_at="2026-08-01T00:00:00+00:00",
        extraction=extraction,
    )

    assert counts == {"claims_created": 0, "perspectives_created": 1, "claim_ids": []}
    row = conn.execute(
        "SELECT perspective_text, category, sub_tags, author, article_id, source_id, "
        "supporting_excerpt, published_at, retrieved_at "
        "FROM perspectives"
    ).fetchone()
    assert row["perspective_text"] == "An opinion"
    assert row["category"] == "Policy & Industry"
    assert row["sub_tags"] == '["Take"]'
    assert row["author"] == "John"
    assert row["article_id"] == 1
    assert row["source_id"] == 1
    assert row["supporting_excerpt"] == "quote"
    assert row["published_at"] == "2026-08-01"
    assert row["retrieved_at"] == "2026-08-01T00:00:00+00:00"
    assert conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"] == 0
    conn.close()


def test_store_extraction_result_flags_an_injection_anomaly(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source_and_article(conn)

    extraction = ExtractionResult(
        items=[], injection_anomaly="Article text contained 'ignore previous instructions'",
    )

    store_extraction_result(
        conn, article_id=1, source_id=1,
        published_at=None, retrieved_at="2026-08-01T00:00:00+00:00",
        extraction=extraction,
    )

    row = conn.execute("SELECT article_id, reason, reviewed FROM anomaly_flags").fetchone()
    assert row["article_id"] == 1
    assert "ignore previous instructions" in row["reason"]
    assert row["reviewed"] == 0
    conn.close()


def test_store_extraction_result_does_not_flag_an_anomaly_when_none_detected(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source_and_article(conn)

    extraction = ExtractionResult(items=[])

    store_extraction_result(
        conn, article_id=1, source_id=1,
        published_at=None, retrieved_at="2026-08-01T00:00:00+00:00",
        extraction=extraction,
    )

    assert conn.execute("SELECT COUNT(*) AS n FROM anomaly_flags").fetchone()["n"] == 0
    conn.close()
