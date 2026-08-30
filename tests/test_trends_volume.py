import json
from datetime import datetime, timedelta, timezone

from tabs.db import get_connection, init_db
from tabs.trends.volume import category_volume, sub_tag_volume


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


def _insert_claim(
    conn, article_id, source_id, category="AppSec", sub_tags=None, status="verified",
    retrieved_at=None,
):
    retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO claims (article_id, source_id, claim_text, supporting_excerpt, "
        "claim_type, category, sub_tags, status, llm_certainty, retrieved_at, created_at) "
        "VALUES (?, ?, 'claim', 'excerpt', 'factual', ?, ?, ?, 0.5, ?, ?)",
        (
            article_id, source_id, category, json.dumps(sub_tags or []), status,
            retrieved_at, retrieved_at,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_perspective(
    conn, article_id, source_id, category="AppSec", sub_tags=None, retrieved_at=None,
):
    retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO perspectives (article_id, source_id, perspective_text, "
        "supporting_excerpt, category, sub_tags, retrieved_at, created_at) "
        "VALUES (?, ?, 'perspective', 'excerpt', ?, ?, ?, ?)",
        (article_id, source_id, category, json.dumps(sub_tags or []), retrieved_at, retrieved_at),
    )
    conn.commit()
    return cursor.lastrowid


def _window():
    now = datetime.now(timezone.utc)
    return (now - timedelta(days=1)).isoformat(), (now + timedelta(days=1)).isoformat()


def test_category_volume_counts_claims_and_perspectives_together(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    _insert_claim(conn, article_id, source_id, category="AppSec")
    _insert_perspective(conn, article_id, source_id, category="AppSec")
    start, end = _window()

    counts = category_volume(conn, start, end)

    assert counts == {"AppSec": 2}
    conn.close()


def test_category_volume_excludes_misinformation_claims(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    _insert_claim(conn, article_id, source_id, category="AppSec", status="verified")
    _insert_claim(conn, article_id, source_id, category="AppSec", status="misinformation")
    start, end = _window()

    counts = category_volume(conn, start, end)

    assert counts == {"AppSec": 1}
    conn.close()


def test_category_volume_excludes_items_outside_the_window(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    _insert_claim(conn, article_id, source_id, category="AppSec", retrieved_at=old)
    start, end = _window()

    counts = category_volume(conn, start, end)

    assert counts == {}
    conn.close()


def test_category_volume_window_is_half_open_at_the_end_boundary(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    start, end = _window()
    _insert_claim(conn, article_id, source_id, category="AppSec", retrieved_at=start)
    _insert_claim(conn, article_id, source_id, category="AppSec", retrieved_at=end)

    counts = category_volume(conn, start, end)

    assert counts == {"AppSec": 1}
    conn.close()


def test_sub_tag_volume_counts_each_tag_on_a_multi_tagged_item(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    _insert_claim(
        conn, article_id, source_id, category="AppSec",
        sub_tags=["Supply Chain", "AuthN/AuthZ"],
    )
    start, end = _window()

    counts = sub_tag_volume(conn, start, end)

    assert counts == {("AppSec", "Supply Chain"): 1, ("AppSec", "AuthN/AuthZ"): 1}
    conn.close()


def test_sub_tag_volume_ignores_items_with_no_sub_tags(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    _insert_claim(conn, article_id, source_id, category="AppSec", sub_tags=[])
    start, end = _window()

    counts = sub_tag_volume(conn, start, end)

    assert counts == {}
    conn.close()


def test_sub_tag_volume_excludes_misinformation_claims(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    _insert_claim(
        conn, article_id, source_id, category="AppSec", sub_tags=["Supply Chain"],
        status="misinformation",
    )
    start, end = _window()

    counts = sub_tag_volume(conn, start, end)

    assert counts == {}
    conn.close()


def test_sub_tag_volume_counts_claims_and_perspectives_together(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    _insert_claim(conn, article_id, source_id, category="AppSec", sub_tags=["Prompt Injection"])
    _insert_perspective(
        conn, article_id, source_id, category="AppSec", sub_tags=["Prompt Injection"],
    )
    start, end = _window()

    counts = sub_tag_volume(conn, start, end)

    assert counts == {("AppSec", "Prompt Injection"): 2}
    conn.close()
