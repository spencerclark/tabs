from datetime import datetime, timedelta, timezone

from tabs.db import get_connection, init_db
from tabs.trends.notable import notable_stories


def _insert_source(conn, name="source"):
    cursor = conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES (?, ?, 'AppSec', 2, 2)",
        (name, f"https://{name}.example/feed"),
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


def _insert_cluster(conn, category="AppSec"):
    cursor = conn.execute(
        "INSERT INTO story_clusters (category, summary, created_at) VALUES (?, NULL, ?)",
        (category, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_claim(
    conn, article_id, source_id, category, claim_text, story_cluster_id,
    corroboration_count=0, status="verified", days_ago=0,
):
    retrieved_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    cursor = conn.execute(
        "INSERT INTO claims (article_id, source_id, claim_text, supporting_excerpt, "
        "claim_type, category, sub_tags, status, llm_certainty, corroboration_count, "
        "story_cluster_id, retrieved_at, created_at) "
        "VALUES (?, ?, ?, 'excerpt', 'factual', ?, '[]', ?, 0.5, ?, ?, ?, ?)",
        (
            article_id, source_id, claim_text, category, status, corroboration_count,
            story_cluster_id, retrieved_at, retrieved_at,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def test_notable_stories_ranks_by_corroboration_count_descending(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    low = _insert_cluster(conn)
    high = _insert_cluster(conn)
    _insert_claim(
        conn, article_id, source_id, "AppSec", "Low corroboration", low, corroboration_count=1,
    )
    _insert_claim(
        conn, article_id, source_id, "AppSec", "High corroboration", high, corroboration_count=5,
    )

    stories = notable_stories(conn, since_days=7)

    assert [s.story_cluster_id for s in stories] == [high, low]
    conn.close()


def test_notable_stories_excludes_clusters_with_no_activity_in_the_window(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    cluster_id = _insert_cluster(conn)
    _insert_claim(
        conn, article_id, source_id, "AppSec", "Stale claim", cluster_id, days_ago=30,
    )

    stories = notable_stories(conn, since_days=7)

    assert stories == []
    conn.close()


def test_notable_stories_excludes_misinformation_claims(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    cluster_id = _insert_cluster(conn)
    _insert_claim(
        conn, article_id, source_id, "AppSec", "Debunked", cluster_id, status="misinformation",
    )

    stories = notable_stories(conn, since_days=7)

    assert stories == []
    conn.close()


def test_notable_stories_includes_the_most_recent_claim_text_as_a_sample(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    cluster_id = _insert_cluster(conn)
    _insert_claim(
        conn, article_id, source_id, "AppSec", "Older claim text", cluster_id, days_ago=2,
    )
    _insert_claim(
        conn, article_id, source_id, "AppSec", "Newest claim text", cluster_id, days_ago=0,
    )

    stories = notable_stories(conn, since_days=7)

    assert stories[0].sample_claim_text == "Newest claim text"
    conn.close()


def test_notable_stories_respects_the_limit(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    for i in range(3):
        cluster_id = _insert_cluster(conn)
        _insert_claim(
            conn, article_id, source_id, "AppSec", f"Claim {i}", cluster_id,
            corroboration_count=i,
        )

    stories = notable_stories(conn, since_days=7, limit=2)

    assert len(stories) == 2
    conn.close()
