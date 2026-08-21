from pathlib import Path

import anthropic
import click

from tabs.db import get_connection, init_db
from tabs.ingest.orchestrator import run_ingest
from tabs.sources import load_sources_yaml, sync_sources

DEFAULT_SOURCES_PATH = Path("sources.yaml")


@click.command(name="ingest")
@click.option(
    "--sources-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_SOURCES_PATH,
    help="Path to the sources.yaml allowlist.",
)
@click.pass_context
def ingest_cmd(ctx: click.Context, sources_path: Path) -> None:
    """Sync the source allowlist, then fetch, store, and curate new articles from every source."""
    conn = get_connection(ctx.obj["db_path"])
    try:
        init_db(conn)
        sync_sources(conn, load_sources_yaml(sources_path))
        client = anthropic.Anthropic()
        summary = run_ingest(conn, client)
        click.echo(
            f"sources_ok={summary['sources_ok']} "
            f"sources_failed={summary['sources_failed']} "
            f"articles_stored={summary['articles_stored']} "
            f"articles_out_of_scope={summary['articles_out_of_scope']} "
            f"articles_uncurated={summary['articles_uncurated']} "
            f"claims_extracted={summary['claims_extracted']} "
            f"perspectives_extracted={summary['perspectives_extracted']}"
        )
    except (click.ClickException, click.Abort, click.exceptions.Exit):
        raise  # Click's own control flow, already renders cleanly
    except Exception as exc:  # noqa: BLE001 — a clean one-line error beats a traceback
        raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()
