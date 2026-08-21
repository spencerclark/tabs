from pathlib import Path

from tabs.db import get_connection, init_db
from tabs.sources import load_sources_yaml, sync_sources

SOURCES_YAML = """
- name: Krebs on Security
  feed_url: https://krebsonsecurity.com/feed/
  category: AppSec
  institutional_tier: 2
"""


def _write_yaml(tmp_path: Path) -> Path:
    yaml_path = tmp_path / "sources.yaml"
    yaml_path.write_text(SOURCES_YAML)
    return yaml_path


def test_load_sources_yaml_parses_entries(tmp_path):
    sources = load_sources_yaml(_write_yaml(tmp_path))

    assert len(sources) == 1
    assert sources[0].name == "Krebs on Security"
    assert sources[0].institutional_tier == 2


def test_sync_sources_inserts_new_source_with_earned_tier_seeded(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    sources = load_sources_yaml(_write_yaml(tmp_path))

    sync_sources(conn, sources)

    row = conn.execute(
        "SELECT * FROM sources WHERE feed_url = ?", (sources[0].feed_url,)
    ).fetchone()
    assert row["earned_tier"] == 2
    conn.close()


def test_sync_sources_preserves_earned_tier_on_resync(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    sources = load_sources_yaml(_write_yaml(tmp_path))
    sync_sources(conn, sources)
    conn.execute(
        "UPDATE sources SET earned_tier = 5 WHERE feed_url = ?", (sources[0].feed_url,)
    )
    conn.commit()

    sync_sources(conn, sources)  # re-sync the same source list

    row = conn.execute(
        "SELECT earned_tier FROM sources WHERE feed_url = ?", (sources[0].feed_url,)
    ).fetchone()
    assert row["earned_tier"] == 5
    conn.close()
