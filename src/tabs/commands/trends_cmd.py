import re

import click

from tabs.db import get_connection, init_db
from tabs.trends.notable import notable_stories
from tabs.trends.spikes import detect_spikes

DEFAULT_SINCE = "7d"
DEFAULT_LIMIT = 10

_SINCE_PATTERN = re.compile(r"^(\d+)d$")


def parse_since(value: str) -> int:
    """Parse a --since value like '7d' or '30d' into a day count.

    Deliberately minimal (digits + a literal 'd' suffix only, no weeks/months) — SPEC §9's
    CLI table only ever shows day-suffixed examples ('--since 30d'), and nothing in this
    phase needs a richer duration grammar yet.
    """
    match = _SINCE_PATTERN.match(value)
    if not match:
        raise click.BadParameter(f"expected a number of days like '7d', got {value!r}")
    return int(match.group(1))


@click.command(name="trends")
@click.option(
    "--since", default=DEFAULT_SINCE,
    help="Window to report on, e.g. '7d' or '30d' (default: 7d).",
)
@click.pass_context
def trends_cmd(ctx: click.Context, since: str) -> None:
    """Show notable stories and category/sub-tag volume spikes for the window."""
    conn = get_connection(ctx.obj["db_path"])
    try:
        init_db(conn)
        since_days = parse_since(since)

        click.echo("Notable Stories")
        stories = notable_stories(conn, since_days, limit=DEFAULT_LIMIT)
        if not stories:
            click.echo("  (none)")
        for story in stories:
            click.echo(
                f"  [{story.category}] corroborated by {story.corroboration_count} source(s), "
                f"last seen {story.most_recent_retrieved_at} — {story.sample_claim_text}"
            )

        click.echo("")
        click.echo("Trending Topics")
        spikes = detect_spikes(conn, since_days)
        if not spikes:
            click.echo("  (none)")
        for spike in spikes:
            tag = spike.sub_tag or "(overall)"
            change = "new" if spike.pct_change is None else f"+{spike.pct_change:.0%}"
            click.echo(
                f"  [{spike.category}] {tag}: {spike.current_volume} "
                f"(baseline avg {spike.baseline_avg:.1f}, {change})"
            )
    except (click.ClickException, click.Abort, click.exceptions.Exit):
        raise  # Click's own control flow, already renders cleanly
    except Exception as exc:  # noqa: BLE001 — a clean one-line error beats a traceback
        raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()
