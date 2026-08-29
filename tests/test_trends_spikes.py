import json
from datetime import datetime, timedelta, timezone

from tabs.db import get_connection, init_db
from tabs.trends.spikes import (
    BASELINE_MULTIPLIER,
    MIN_VOLUME_GUARD,
    SPIKE_THRESHOLD_PCT,
    detect_spikes,
)


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


def _insert_claim_days_ago(conn, article_id, source_id, category, days_ago, sub_tags=None):
    retrieved_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    conn.execute(
        "INSERT INTO claims (article_id, source_id, claim_text, supporting_excerpt, "
        "claim_type, category, sub_tags, status, llm_certainty, retrieved_at, created_at) "
        "VALUES (?, ?, 'claim', 'excerpt', 'factual', ?, ?, 'verified', 0.5, ?, ?)",
        (article_id, source_id, category, json.dumps(sub_tags or []), retrieved_at, retrieved_at),
    )
    conn.commit()


def _setup(conn):
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    return source_id, article_id


def _fill_baseline(conn, article_id, source_id, category, count_per_window, since_days=7):
    for window in range(BASELINE_MULTIPLIER):
        days_ago = since_days + window * since_days + 1
        for _ in range(count_per_window):
            _insert_claim_days_ago(conn, article_id, source_id, category, days_ago=days_ago)


def test_detect_spikes_flags_a_category_with_no_baseline_history_as_new(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id, article_id = _setup(conn)
    for _ in range(MIN_VOLUME_GUARD):
        _insert_claim_days_ago(conn, article_id, source_id, "AppSec", days_ago=1)

    spikes = detect_spikes(conn, since_days=7)

    matching = [s for s in spikes if s.category == "AppSec" and s.sub_tag is None]
    assert len(matching) == 1
    assert matching[0].pct_change is None
    assert matching[0].current_volume == MIN_VOLUME_GUARD
    conn.close()


def test_detect_spikes_ignores_a_category_below_the_minimum_volume_guard(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id, article_id = _setup(conn)
    for _ in range(MIN_VOLUME_GUARD - 1):
        _insert_claim_days_ago(conn, article_id, source_id, "AppSec", days_ago=1)

    spikes = detect_spikes(conn, since_days=7)

    assert spikes == []
    conn.close()


def test_detect_spikes_ignores_a_category_whose_change_is_below_the_threshold(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id, article_id = _setup(conn)
    _fill_baseline(conn, article_id, source_id, "AppSec", count_per_window=10)
    for _ in range(10):  # matches the baseline average exactly: 0% change
        _insert_claim_days_ago(conn, article_id, source_id, "AppSec", days_ago=1)

    spikes = detect_spikes(conn, since_days=7)

    assert spikes == []
    conn.close()


def test_detect_spikes_flags_a_category_whose_volume_exceeds_the_threshold(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id, article_id = _setup(conn)
    _fill_baseline(conn, article_id, source_id, "AppSec", count_per_window=4)
    # baseline_avg = (4*4)/4 = 4; comfortably above the 4 * (1 + SPIKE_THRESHOLD_PCT) floor
    current_volume = int(4 * (1 + SPIKE_THRESHOLD_PCT)) + 3
    for _ in range(current_volume):
        _insert_claim_days_ago(conn, article_id, source_id, "AppSec", days_ago=1)

    spikes = detect_spikes(conn, since_days=7)

    matching = [s for s in spikes if s.category == "AppSec" and s.sub_tag is None]
    assert len(matching) == 1
    assert matching[0].current_volume == current_volume
    assert matching[0].baseline_avg == 4.0
    assert matching[0].pct_change >= SPIKE_THRESHOLD_PCT
    conn.close()


def test_detect_spikes_flags_a_sub_tag_spike_independently_of_its_category(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id, article_id = _setup(conn)
    for _ in range(MIN_VOLUME_GUARD):
        _insert_claim_days_ago(
            conn, article_id, source_id, "AppSec", days_ago=1, sub_tags=["Supply Chain"],
        )

    spikes = detect_spikes(conn, since_days=7)

    assert any(s.sub_tag == "Supply Chain" for s in spikes)
    conn.close()


def test_detect_spikes_sorts_new_tags_before_tags_ranked_by_pct_change_descending(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id, article_id = _setup(conn)
    _fill_baseline(conn, article_id, source_id, "AI Security", count_per_window=4)
    for _ in range(20):
        _insert_claim_days_ago(conn, article_id, source_id, "AI Security", days_ago=1)
    for _ in range(MIN_VOLUME_GUARD):
        _insert_claim_days_ago(conn, article_id, source_id, "AppSec", days_ago=1)

    spikes = detect_spikes(conn, since_days=7)

    assert spikes[0].category == "AppSec"
    assert spikes[0].pct_change is None
    conn.close()
