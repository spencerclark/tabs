import sqlite3

import pytest
from click.testing import CliRunner

import tabs.commands.ingest_cmd as ingest_cmd_module
from tabs.cli import main
from tabs.db import get_connection


def test_ingest_command_syncs_sources_and_runs_ingest(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(
        "- name: Test Source\n"
        "  feed_url: https://test.example/feed\n"
        "  category: AppSec\n"
        "  institutional_tier: 2\n"
    )

    monkeypatch.setattr(
        ingest_cmd_module,
        "run_ingest",
        lambda conn: {"sources_ok": 1, "sources_failed": 0, "articles_stored": 3},
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--db-path", str(db_path), "ingest", "--sources-path", str(sources_yaml)],
    )

    assert result.exit_code == 0
    assert "articles_stored=3" in result.output

    # Verify that the source was actually synced to the database
    conn = get_connection(db_path)
    source_row = conn.execute("SELECT COUNT(*) AS n FROM sources").fetchone()
    assert source_row["n"] == 1
    synced_source = conn.execute(
        "SELECT name, feed_url, category, institutional_tier FROM sources WHERE feed_url = ?",
        ("https://test.example/feed",),
    ).fetchone()
    assert synced_source is not None
    assert synced_source["name"] == "Test Source"
    assert synced_source["feed_url"] == "https://test.example/feed"
    assert synced_source["category"] == "AppSec"
    assert synced_source["institutional_tier"] == 2
    conn.close()


def test_ingest_command_reports_a_missing_sources_file_cleanly(tmp_path):
    result = CliRunner().invoke(
        main,
        [
            "--db-path", str(tmp_path / "test.db"),
            "ingest", "--sources-path", str(tmp_path / "nope.yaml"),
        ],
    )

    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    assert "nope.yaml" in result.output


def test_ingest_command_reports_malformed_sources_yaml_cleanly(tmp_path):
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(
        "- name: Broken Source\n  feed_url: https://broken.example/feed\n"
    )  # no category, no institutional_tier

    result = CliRunner().invoke(
        main,
        ["--db-path", str(tmp_path / "test.db"), "ingest", "--sources-path", str(sources_yaml)],
    )

    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    assert "category" in result.output
    assert "Broken Source" in result.output


def test_ingest_command_closes_the_connection_when_the_command_body_raises(tmp_path, monkeypatch):
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(
        "- name: Test Source\n"
        "  feed_url: https://test.example/feed\n"
        "  category: AppSec\n"
        "  institutional_tier: 2\n"
    )

    opened = []
    real_get_connection = ingest_cmd_module.get_connection

    def tracking_get_connection(db_path):
        conn = real_get_connection(db_path)
        opened.append(conn)
        return conn

    monkeypatch.setattr(ingest_cmd_module, "get_connection", tracking_get_connection)

    def boom(conn):
        raise RuntimeError("unexpected mid-command failure")

    monkeypatch.setattr(ingest_cmd_module, "run_ingest", boom)

    result = CliRunner().invoke(
        main,
        ["--db-path", str(tmp_path / "test.db"), "ingest", "--sources-path", str(sources_yaml)],
    )

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "unexpected mid-command failure" in result.output
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):  # closed connections raise on use
        opened[0].execute("SELECT 1")
