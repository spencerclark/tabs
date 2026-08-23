import json
from datetime import datetime, timedelta, timezone

from tabs.db import get_connection, init_db
from tabs.score.matching import (
    CORROBORATION_WINDOW_DAYS,
    MAX_CANDIDATES,
    find_candidate_claims,
)


def _insert_source(conn, name, institutional_tier=2, earned_tier=2):
    cursor = conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES (?, ?, 'AppSec', ?, ?)",
        (name, f"https://{name}.example/feed", institutional_tier, earned_tier),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_article(conn, source_id, url):
    cursor = conn.execute(
        "INSERT INTO articles (source_id, url, title, full_text, content_hash, "
        "published_at, retrieved_at, previous_version_id) "
        "VALUES (?, ?, 'T', 'text', 'hash', NULL, ?, NULL)",
        (source_id, url, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_claim(
    conn, article_id, source_id, claim_text, category="AppSec", embedding=None,
    retrieved_at=None, corroboration_count=0, story_cluster_id=None,
):
    retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO claims (article_id, source_id, claim_text, supporting_excerpt, "
        "claim_type, category, sub_tags, llm_certainty, corroboration_count, "
        "story_cluster_id, retrieved_at, created_at, embedding) "
        "VALUES (?, ?, ?, 'excerpt', 'factual', ?, '[]', 0.5, ?, ?, ?, ?, ?)",
        (
            article_id, source_id, claim_text, category, corroboration_count,
            story_cluster_id, retrieved_at, retrieved_at,
            json.dumps(embedding) if embedding is not None else None,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def test_find_candidate_claims_returns_similar_claims_in_the_same_category(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn, "source")
    article_a = _insert_article(conn, source_id, "https://source.example/a")
    article_b = _insert_article(conn, source_id, "https://source.example/b")
    _insert_claim(conn, article_a, source_id, "New claim's own article", embedding=[1.0, 0.0])
    matching_id = _insert_claim(
        conn, article_b, source_id, "A similar claim", embedding=[0.99, 0.01],
    )

    candidates = find_candidate_claims(
        conn, claim_id=999999, article_id=article_a, category="AppSec", embedding=[1.0, 0.0],
    )

    assert [c.claim_id for c in candidates] == [matching_id]
    conn.close()


def test_find_candidate_claims_excludes_claims_from_the_same_article(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn, "source")
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    new_claim_id = _insert_claim(conn, article_id, source_id, "New claim", embedding=[1.0, 0.0])
    _insert_claim(conn, article_id, source_id, "Sibling claim, same article", embedding=[1.0, 0.0])

    candidates = find_candidate_claims(
        conn, claim_id=new_claim_id, article_id=article_id, category="AppSec",
        embedding=[1.0, 0.0],
    )

    assert candidates == []
    conn.close()


def test_find_candidate_claims_excludes_claims_below_the_similarity_threshold(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn, "source")
    article_a = _insert_article(conn, source_id, "https://source.example/a")
    article_b = _insert_article(conn, source_id, "https://source.example/b")
    _insert_claim(conn, article_b, source_id, "Unrelated claim", embedding=[0.0, 1.0])

    candidates = find_candidate_claims(
        conn, claim_id=999999, article_id=article_a, category="AppSec", embedding=[1.0, 0.0],
    )

    assert candidates == []
    conn.close()


def test_find_candidate_claims_excludes_claims_outside_the_recheck_window(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn, "source")
    article_a = _insert_article(conn, source_id, "https://source.example/a")
    article_b = _insert_article(conn, source_id, "https://source.example/b")
    old_retrieved_at = (
        datetime.now(timezone.utc) - timedelta(days=CORROBORATION_WINDOW_DAYS + 1)
    ).isoformat()
    _insert_claim(
        conn, article_b, source_id, "Old similar claim", embedding=[1.0, 0.0],
        retrieved_at=old_retrieved_at,
    )

    candidates = find_candidate_claims(
        conn, claim_id=999999, article_id=article_a, category="AppSec", embedding=[1.0, 0.0],
    )

    assert candidates == []
    conn.close()


def test_find_candidate_claims_excludes_a_different_category(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn, "source")
    article_a = _insert_article(conn, source_id, "https://source.example/a")
    article_b = _insert_article(conn, source_id, "https://source.example/b")
    _insert_claim(
        conn, article_b, source_id, "Similar but different category",
        category="AI Security", embedding=[1.0, 0.0],
    )

    candidates = find_candidate_claims(
        conn, claim_id=999999, article_id=article_a, category="AppSec", embedding=[1.0, 0.0],
    )

    assert candidates == []
    conn.close()


def test_find_candidate_claims_ranks_by_similarity_and_caps_at_max_candidates(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn, "source")
    article_a = _insert_article(conn, source_id, "https://source.example/a")
    expected_order = []
    for i in range(MAX_CANDIDATES + 2):
        article = _insert_article(conn, source_id, f"https://source.example/b{i}")
        # decreasing similarity: [1.0, i*0.01] moves further from [1.0, 0.0] as i grows
        claim_id = _insert_claim(
            conn, article, source_id, f"Claim {i}", embedding=[1.0, i * 0.01],
        )
        expected_order.append(claim_id)

    candidates = find_candidate_claims(
        conn, claim_id=999999, article_id=article_a, category="AppSec", embedding=[1.0, 0.0],
    )

    assert len(candidates) == MAX_CANDIDATES
    assert [c.claim_id for c in candidates] == expected_order[:MAX_CANDIDATES]
    conn.close()


def test_find_candidate_claims_computes_effective_tier_as_the_max(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn, "source", institutional_tier=1, earned_tier=3)
    article_a = _insert_article(conn, source_id, "https://source.example/a")
    article_b = _insert_article(conn, source_id, "https://source.example/b")
    _insert_claim(conn, article_b, source_id, "Claim", embedding=[1.0, 0.0])

    candidates = find_candidate_claims(
        conn, claim_id=999999, article_id=article_a, category="AppSec", embedding=[1.0, 0.0],
    )

    assert candidates[0].effective_tier == 3
    conn.close()
