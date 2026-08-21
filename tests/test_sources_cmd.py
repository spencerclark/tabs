from click.testing import CliRunner

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
