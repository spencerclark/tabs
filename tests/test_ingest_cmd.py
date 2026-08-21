from click.testing import CliRunner

import tabs.commands.ingest_cmd as ingest_cmd_module
from tabs.cli import main


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
