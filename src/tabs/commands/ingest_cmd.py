from pathlib import Path

import click

from tabs.db import get_connection, init_db
from tabs.ingest.orchestrator import run_ingest
from tabs.sources import load_sources_yaml, sync_sources

DEFAULT_SOURCES_PATH = Path("sources.yaml")


@click.command(name="ingest")
@click.option(
    "--sources-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_SOURCES_PATH,
    help="Path to the sources.yaml allowlist.",
)
@click.pass_context
def ingest_cmd(ctx: click.Context, sources_path: Path) -> None:
    """Sync the source allowlist, then fetch and store new articles from every source."""
    conn = get_connection(ctx.obj["db_path"])
    init_db(conn)
    sync_sources(conn, load_sources_yaml(sources_path))
    summary = run_ingest(conn)
    click.echo(
        f"sources_ok={summary['sources_ok']} "
        f"sources_failed={summary['sources_failed']} "
        f"articles_stored={summary['articles_stored']}"
    )
    conn.close()
