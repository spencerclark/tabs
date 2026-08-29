from datetime import datetime, timezone

import click
import pytest
from click.testing import CliRunner

from tabs.cli import main
from tabs.commands.trends_cmd import parse_since
from tabs.db import get_connection, init_db


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


def _insert_claim(conn, article_id, source_id, category, claim_text, story_cluster_id, corroboration_count):
    retrieved_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO claims (article_id, source_id, claim_text, supporting_excerpt, "
        "claim_type, category, sub_tags, status, llm_certainty, corroboration_count, "
        "story_cluster_id, retrieved_at, created_at) "
        "VALUES (?, ?, ?, 'excerpt', 'factual', ?, '[]', 'verified', 0.5, ?, ?, ?, ?)",
        (
            article_id, source_id, claim_text, category, corroboration_count,
            story_cluster_id, retrieved_at, retrieved_at,
        ),
    )
    conn.commit()


def test_parse_since_parses_a_day_count():
    assert parse_since("30d") == 30


def test_parse_since_rejects_a_malformed_value():
    with pytest.raises(click.BadParameter):
        parse_since("banana")


def test_trends_command_reports_notable_stories_and_spikes(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    cluster_id = _insert_cluster(conn)
    _insert_claim(
        conn, article_id, source_id, "AppSec", "A notable claim", cluster_id,
        corroboration_count=3,
    )
    conn.close()

    result = CliRunner().invoke(main, ["--db-path", str(db_path), "trends"])

    assert result.exit_code == 0, result.output
    assert "Notable Stories" in result.output
    assert "A notable claim" in result.output
    assert "Trending Topics" in result.output


def test_trends_command_reports_no_activity_cleanly(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    init_db(conn)
    conn.close()

    result = CliRunner().invoke(main, ["--db-path", str(db_path), "trends"])

    assert result.exit_code == 0, result.output
    assert "(none)" in result.output


def test_trends_command_accepts_a_since_option(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    init_db(conn)
    conn.close()

    result = CliRunner().invoke(main, ["--db-path", str(db_path), "trends", "--since", "30d"])

    assert result.exit_code == 0, result.output


def test_trends_command_rejects_a_malformed_since_value_cleanly(tmp_path):
    result = CliRunner().invoke(
        main, ["--db-path", str(tmp_path / "test.db"), "trends", "--since", "banana"],
    )

    assert result.exit_code != 0
    assert "Traceback" not in result.output
