import json
from datetime import datetime, timedelta, timezone

from tabs.db import get_connection, init_db
from tabs.score.models import CandidateJudgment, MatchJudgments
from tabs.score.scoring import CLAIM_TYPE_WEIGHTS, CERTAINTY_WEIGHT, TIER_WEIGHT
from tabs.score.storage import score_and_corroborate_claim


class _FakeEmbeddingsResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _FakeVoyageClient:
    def __init__(self, embedding=None, raises=False):
        self._embedding = embedding or [1.0, 0.0]
        self._raises = raises

    def embed(self, **kwargs):
        if self._raises:
            raise RuntimeError("simulated Voyage failure")
        return _FakeEmbeddingsResult([self._embedding])


class _FakeParseResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


class _FakeMessages:
    def __init__(self, result=None, raises=False):
        self._result = result
        self._raises = raises

    def parse(self, **kwargs):
        if self._raises:
            raise RuntimeError("simulated Anthropic failure")
        return _FakeParseResponse(self._result)


class _FakeClient:
    def __init__(self, result=None, raises=False):
        self.messages = _FakeMessages(result, raises)


def _insert_source(conn, name="source", institutional_tier=2, earned_tier=2):
    cursor = conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES (?, ?, 'AppSec', ?, ?)",
        (name, f"https://{name}.example/feed", institutional_tier, earned_tier),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_article(conn, source_id, url="https://source.example/a"):
    cursor = conn.execute(
        "INSERT INTO articles (source_id, url, title, full_text, content_hash, "
        "published_at, retrieved_at, previous_version_id) "
        "VALUES (?, ?, 'T', 'text', 'hash', NULL, ?, NULL)",
        (source_id, url, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_claim(
    conn, article_id, source_id, claim_text="A claim", claim_type="factual",
    category="AppSec", llm_certainty=0.5, embedding=None, retrieved_at=None,
    corroboration_count=0, story_cluster_id=None, status="unverified",
):
    retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO claims (article_id, source_id, claim_text, supporting_excerpt, "
        "claim_type, category, sub_tags, llm_certainty, corroboration_count, "
        "story_cluster_id, retrieved_at, created_at, embedding, status) "
        "VALUES (?, ?, ?, 'excerpt', ?, ?, '[]', ?, ?, ?, ?, ?, ?, ?)",
        (
            article_id, source_id, claim_text, claim_type, category, llm_certainty,
            corroboration_count, story_cluster_id, retrieved_at, retrieved_at,
            json.dumps(embedding) if embedding is not None else None, status,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def test_score_and_corroborate_claim_stores_the_embedding(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id)
    claim_id = _insert_claim(conn, article_id, source_id)
    voyage_client = _FakeVoyageClient(embedding=[0.5, 0.5])

    score_and_corroborate_claim(conn, _FakeClient(), voyage_client, claim_id)

    row = conn.execute("SELECT embedding FROM claims WHERE id = ?", (claim_id,)).fetchone()
    assert json.loads(row["embedding"]) == [0.5, 0.5]
    conn.close()


def test_score_and_corroborate_claim_scores_a_claim_with_no_candidates_on_its_own_merits(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn, institutional_tier=3, earned_tier=3)
    article_id = _insert_article(conn, source_id)
    claim_id = _insert_claim(conn, article_id, source_id, llm_certainty=0.9, claim_type="factual")

    result = score_and_corroborate_claim(
        conn, _FakeClient(), _FakeVoyageClient(), claim_id,
    )

    row = conn.execute(
        "SELECT confidence_score, corroboration_count, status FROM claims WHERE id = ?",
        (claim_id,),
    ).fetchone()
    expected_score = TIER_WEIGHT * 3 + CERTAINTY_WEIGHT * 0.9 + CLAIM_TYPE_WEIGHTS["factual"]
    assert row["confidence_score"] == expected_score
    assert row["corroboration_count"] == 0
    assert result.status == row["status"]
    assert result.embedding_failed is False
    assert result.judgment_failed is False
    conn.close()


def test_score_and_corroborate_claim_creates_a_story_cluster_on_first_corroboration(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    other_source_id = _insert_source(conn, "other")
    article_a = _insert_article(conn, source_id, "https://source.example/a")
    article_b = _insert_article(conn, other_source_id, "https://other.example/b")
    existing_id = _insert_claim(
        conn, article_b, other_source_id, claim_text="Existing claim", embedding=[1.0, 0.0],
    )
    new_id = _insert_claim(conn, article_a, source_id, claim_text="New claim")
    client = _FakeClient(
        MatchJudgments(
            judgments=[CandidateJudgment(candidate_claim_id=existing_id, relationship="corroborating")]
        )
    )

    result = score_and_corroborate_claim(conn, client, _FakeVoyageClient(), new_id)

    new_row = conn.execute(
        "SELECT story_cluster_id, corroboration_count FROM claims WHERE id = ?", (new_id,),
    ).fetchone()
    existing_row = conn.execute(
        "SELECT story_cluster_id, corroboration_count, status FROM claims WHERE id = ?",
        (existing_id,),
    ).fetchone()
    assert new_row["story_cluster_id"] is not None
    assert new_row["story_cluster_id"] == existing_row["story_cluster_id"]
    assert new_row["corroboration_count"] == 1
    assert existing_row["corroboration_count"] == 1  # rescored: cluster now has 2 members
    assert result.status in ("verified", "unverified")
    conn.close()


def test_score_and_corroborate_claim_joins_every_corroborating_candidate_not_just_the_best_match(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn, "source")
    other_source_id = _insert_source(conn, "other")
    third_source_id = _insert_source(conn, "third")
    article_a = _insert_article(conn, source_id, "https://source.example/a")
    article_b = _insert_article(conn, other_source_id, "https://other.example/b")
    article_c = _insert_article(conn, third_source_id, "https://third.example/c")
    existing_id_1 = _insert_claim(
        conn, article_b, other_source_id, claim_text="Existing claim one", embedding=[1.0, 0.0],
    )
    existing_id_2 = _insert_claim(
        conn, article_c, third_source_id, claim_text="Existing claim two", embedding=[0.99, 0.01],
    )
    new_id = _insert_claim(conn, article_a, source_id, claim_text="New claim")
    client = _FakeClient(
        MatchJudgments(
            judgments=[
                CandidateJudgment(candidate_claim_id=existing_id_1, relationship="corroborating"),
                CandidateJudgment(candidate_claim_id=existing_id_2, relationship="corroborating"),
            ]
        )
    )

    score_and_corroborate_claim(conn, client, _FakeVoyageClient(), new_id)

    rows = {
        row["id"]: row for row in conn.execute(
            "SELECT id, story_cluster_id, corroboration_count FROM claims"
        ).fetchall()
    }
    cluster_ids = {rows[new_id]["story_cluster_id"], rows[existing_id_1]["story_cluster_id"],
                   rows[existing_id_2]["story_cluster_id"]}
    assert None not in cluster_ids
    assert len(cluster_ids) == 1  # all three joined the SAME cluster
    assert rows[new_id]["corroboration_count"] == 2
    assert rows[existing_id_1]["corroboration_count"] == 2
    assert rows[existing_id_2]["corroboration_count"] == 2
    conn.close()


def test_score_and_corroborate_claim_joins_an_existing_story_cluster(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    other_source_id = _insert_source(conn, "other")
    article_a = _insert_article(conn, source_id, "https://source.example/a")
    article_b = _insert_article(conn, other_source_id, "https://other.example/b")
    conn.execute(
        "INSERT INTO story_clusters (category, summary, created_at) VALUES ('AppSec', NULL, ?)",
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.commit()
    cluster_id = conn.execute("SELECT id FROM story_clusters").fetchone()["id"]
    existing_id = _insert_claim(
        conn, article_b, other_source_id, claim_text="Existing claim", embedding=[1.0, 0.0],
        story_cluster_id=cluster_id, corroboration_count=0,
    )
    new_id = _insert_claim(conn, article_a, source_id, claim_text="New claim")
    client = _FakeClient(
        MatchJudgments(
            judgments=[CandidateJudgment(candidate_claim_id=existing_id, relationship="corroborating")]
        )
    )

    score_and_corroborate_claim(conn, client, _FakeVoyageClient(), new_id)

    new_row = conn.execute("SELECT story_cluster_id FROM claims WHERE id = ?", (new_id,)).fetchone()
    assert new_row["story_cluster_id"] == cluster_id
    cluster_count = conn.execute(
        "SELECT COUNT(*) AS n FROM story_clusters",
    ).fetchone()["n"]
    assert cluster_count == 1  # no new cluster created — joined the existing one
    conn.close()


def test_score_and_corroborate_claim_records_a_conflict_and_marks_the_lower_tier_claim_as_misinformation(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    low_tier_source = _insert_source(conn, "low", institutional_tier=1, earned_tier=1)
    high_tier_source = _insert_source(conn, "high", institutional_tier=3, earned_tier=3)
    article_low = _insert_article(conn, low_tier_source, "https://low.example/a")
    article_high = _insert_article(conn, high_tier_source, "https://high.example/a")
    low_claim_id = _insert_claim(
        conn, article_low, low_tier_source, claim_text="Low tier claim", embedding=[1.0, 0.0],
    )
    high_claim_id = _insert_claim(conn, article_high, high_tier_source, claim_text="High tier claim")
    client = _FakeClient(
        MatchJudgments(
            judgments=[CandidateJudgment(candidate_claim_id=low_claim_id, relationship="conflicting")]
        )
    )

    score_and_corroborate_claim(conn, client, _FakeVoyageClient(), high_claim_id)

    low_row = conn.execute("SELECT status FROM claims WHERE id = ?", (low_claim_id,)).fetchone()
    assert low_row["status"] == "misinformation"
    conflict_row = conn.execute("SELECT resolution, winning_claim_id FROM conflicts").fetchone()
    assert conflict_row["resolution"] == "auto-resolved"
    assert conflict_row["winning_claim_id"] == high_claim_id
    conn.close()


def test_score_and_corroborate_claim_needs_review_conflict_does_not_force_misinformation(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn, institutional_tier=2, earned_tier=2)
    other_source_id = _insert_source(conn, "other", institutional_tier=2, earned_tier=2)
    article_a = _insert_article(conn, source_id, "https://source.example/a")
    article_b = _insert_article(conn, other_source_id, "https://other.example/b")
    same_time = datetime.now(timezone.utc).isoformat()
    existing_id = _insert_claim(
        conn, article_b, other_source_id, claim_text="Existing claim", embedding=[1.0, 0.0],
        retrieved_at=same_time, corroboration_count=0,
    )
    new_id = _insert_claim(
        conn, article_a, source_id, claim_text="New claim", retrieved_at=same_time,
        corroboration_count=0,
    )
    client = _FakeClient(
        MatchJudgments(
            judgments=[CandidateJudgment(candidate_claim_id=existing_id, relationship="conflicting")]
        )
    )

    score_and_corroborate_claim(conn, client, _FakeVoyageClient(), new_id)

    existing_row = conn.execute("SELECT status FROM claims WHERE id = ?", (existing_id,)).fetchone()
    new_row = conn.execute("SELECT status FROM claims WHERE id = ?", (new_id,)).fetchone()
    conflict_row = conn.execute("SELECT resolution FROM conflicts").fetchone()
    assert conflict_row["resolution"] == "needs-review"
    assert existing_row["status"] != "misinformation"
    assert new_row["status"] != "misinformation"
    conn.close()


def test_score_and_corroborate_claim_ignores_a_judgment_for_an_unoffered_candidate_id(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    other_source_id = _insert_source(conn, "other")
    article_a = _insert_article(conn, source_id, "https://source.example/a")
    article_b = _insert_article(conn, other_source_id, "https://other.example/b")
    existing_id = _insert_claim(
        conn, article_b, other_source_id, claim_text="Existing claim", embedding=[1.0, 0.0],
    )
    new_id = _insert_claim(conn, article_a, source_id, claim_text="New claim")
    client = _FakeClient(
        MatchJudgments(
            judgments=[CandidateJudgment(candidate_claim_id=999999, relationship="corroborating")]
        )
    )

    result = score_and_corroborate_claim(conn, client, _FakeVoyageClient(), new_id)

    new_row = conn.execute("SELECT story_cluster_id FROM claims WHERE id = ?", (new_id,)).fetchone()
    assert new_row["story_cluster_id"] is None  # bogus id ignored, not guessed at
    assert result.judgment_failed is False
    conn.close()


def test_score_and_corroborate_claim_degrades_gracefully_when_embedding_fails(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id)
    claim_id = _insert_claim(conn, article_id, source_id)

    result = score_and_corroborate_claim(
        conn, _FakeClient(), _FakeVoyageClient(raises=True), claim_id,
    )

    row = conn.execute(
        "SELECT confidence_score, status, embedding FROM claims WHERE id = ?", (claim_id,),
    ).fetchone()
    assert row["embedding"] is None
    assert row["confidence_score"] is not None  # still scored, on tier/certainty/type alone
    assert result.embedding_failed is True
    assert result.judgment_failed is False  # never reached — no embedding to match with
    conn.close()


def test_score_and_corroborate_claim_degrades_gracefully_when_judgment_fails(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    other_source_id = _insert_source(conn, "other")
    article_a = _insert_article(conn, source_id, "https://source.example/a")
    article_b = _insert_article(conn, other_source_id, "https://other.example/b")
    _insert_claim(conn, article_b, other_source_id, claim_text="Existing claim", embedding=[1.0, 0.0])
    new_id = _insert_claim(conn, article_a, source_id, claim_text="New claim")

    result = score_and_corroborate_claim(
        conn, _FakeClient(raises=True), _FakeVoyageClient(), new_id,
    )

    row = conn.execute("SELECT confidence_score, status FROM claims WHERE id = ?", (new_id,)).fetchone()
    assert row["confidence_score"] is not None
    assert result.embedding_failed is False
    assert result.judgment_failed is True
    conn.close()


def test_score_and_corroborate_claim_leaves_an_existing_misinformation_status_untouched(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn, institutional_tier=3, earned_tier=3)
    article_id = _insert_article(conn, source_id)
    claim_id = _insert_claim(
        conn, article_id, source_id, llm_certainty=0.99, status="misinformation",
    )

    result = score_and_corroborate_claim(conn, _FakeClient(), _FakeVoyageClient(), claim_id)

    row = conn.execute("SELECT status FROM claims WHERE id = ?", (claim_id,)).fetchone()
    assert row["status"] == "misinformation"  # sticky — never re-verified by a high score
    assert result.status == "misinformation"
    conn.close()
