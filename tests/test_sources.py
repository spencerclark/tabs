from pathlib import Path

import pytest

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


def test_load_sources_yaml_accepts_an_empty_file(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text("")

    assert load_sources_yaml(path) == []


def test_load_sources_yaml_rejects_a_top_level_mapping(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text("name: Krebs\nfeed_url: https://k.example/feed\n")

    with pytest.raises(ValueError, match="list of source entries"):
        load_sources_yaml(path)


def test_load_sources_yaml_rejects_a_non_mapping_entry(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text("- just a string\n")

    with pytest.raises(ValueError, match="entry 1"):
        load_sources_yaml(path)


def test_load_sources_yaml_rejects_a_missing_field(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(
        "- name: Krebs on Security\n"
        "  feed_url: https://krebsonsecurity.com/feed/\n"
        "  institutional_tier: 2\n"
    )

    with pytest.raises(ValueError) as excinfo:
        load_sources_yaml(path)

    message = str(excinfo.value)
    assert "category" in message
    assert "Krebs on Security" in message  # names the offending entry


def test_load_sources_yaml_rejects_a_non_integer_tier(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(
        "- name: Krebs on Security\n"
        "  feed_url: https://krebsonsecurity.com/feed/\n"
        "  category: AppSec\n"
        "  institutional_tier: high\n"
    )

    with pytest.raises(ValueError, match="institutional_tier"):
        load_sources_yaml(path)


def test_load_sources_yaml_rejects_a_blank_required_field(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(
        "- name: ''\n"
        "  feed_url: https://krebsonsecurity.com/feed/\n"
        "  category: AppSec\n"
        "  institutional_tier: 2\n"
    )

    with pytest.raises(ValueError, match="name"):
        load_sources_yaml(path)


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
