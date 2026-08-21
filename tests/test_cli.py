from click.testing import CliRunner

from tabs.cli import main


def test_version_flag_prints_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output
