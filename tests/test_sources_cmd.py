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
