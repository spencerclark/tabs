import sqlite3

import pytest
from click.testing import CliRunner

import tabs.commands.sources_cmd as sources_cmd_module
from tabs.cli import main
from tabs.db import get_connection, init_db
from tabs.models import Source
from tabs.sources import sync_sources


def test_sources_command_lists_synced_sources(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    init_db(conn)
    sync_sources(conn, [Source("Krebs on Security", "https://krebsonsecurity.com/feed/", "AppSec", 2)])
    conn.close()

    runner = CliRunner()
    result = runner.invoke(main, ["--db-path", str(db_path), "sources"])

    assert result.exit_code == 0
    assert "Krebs on Security" in result.output
    assert "tier=2" in result.output
    assert "last_fetch=never" in result.output


def test_sources_command_reports_errors_cleanly_and_closes_the_connection(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    opened = []
    real_get_connection = sources_cmd_module.get_connection

    def tracking_get_connection(path):
        conn = real_get_connection(path)
        opened.append(conn)
        return conn

    monkeypatch.setattr(sources_cmd_module, "get_connection", tracking_get_connection)
    monkeypatch.setattr(
        sources_cmd_module,
        "init_db",
        lambda conn: (_ for _ in ()).throw(sqlite3.OperationalError("disk is full")),
    )

    result = CliRunner().invoke(main, ["--db-path", str(db_path), "sources"])

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "disk is full" in result.output
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):  # closed connections raise on use
        opened[0].execute("SELECT 1")


def _sync_then_set_earned_tier(db_path, institutional_tier, earned_tier):
    conn = get_connection(db_path)
    init_db(conn)
    sync_sources(
        conn,
        [Source("Krebs on Security", "https://krebsonsecurity.com/feed/", "AppSec", institutional_tier)],
    )
    conn.execute("UPDATE sources SET earned_tier = ?", (earned_tier,))
    conn.commit()
    conn.close()


def test_sources_command_shows_earned_tier_when_it_promotes_above_institutional(tmp_path):
    db_path = tmp_path / "test.db"
    _sync_then_set_earned_tier(db_path, institutional_tier=1, earned_tier=3)

    result = CliRunner().invoke(main, ["--db-path", str(db_path), "sources"])

    assert result.exit_code == 0
    # a source that has earned its way up is shown at the higher earned tier
    assert "tier=3" in result.output
    assert "tier=1" not in result.output


def test_sources_command_keeps_institutional_tier_as_a_floor(tmp_path):
    db_path = tmp_path / "test.db"
    _sync_then_set_earned_tier(db_path, institutional_tier=3, earned_tier=1)

    result = CliRunner().invoke(main, ["--db-path", str(db_path), "sources"])

    assert result.exit_code == 0
    # institutional tier is a floor: a lower earned tier must not drag it down
    assert "tier=3" in result.output
    assert "tier=1" not in result.output
