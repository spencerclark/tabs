from pathlib import Path

import click

from tabs import __version__
from tabs.commands.ingest_cmd import ingest_cmd
from tabs.commands.sources_cmd import sources_cmd

DEFAULT_DB_PATH = Path("data/tabs.db")


@click.group()
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_DB_PATH,
    help="Path to the SQLite database file.",
)
@click.version_option(__version__, prog_name="tabs")
@click.pass_context
def main(ctx: click.Context, db_path: Path) -> None:
    """tabs — AppSec & AI Security knowledge base."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path


main.add_command(ingest_cmd)
main.add_command(sources_cmd)


if __name__ == "__main__":
    main()
