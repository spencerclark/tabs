import click

from tabs.db import get_connection, init_db


@click.command(name="sources")
@click.pass_context
def sources_cmd(ctx: click.Context) -> None:
    """List allowlisted sources with effective tier and health."""
    conn = get_connection(ctx.obj["db_path"])
    init_db(conn)
    rows = conn.execute(
        "SELECT name, category, institutional_tier, earned_tier, "
        "consecutive_failures, last_successful_fetch_at FROM sources ORDER BY name"
    ).fetchall()
    for row in rows:
        effective_tier = max(row["institutional_tier"], row["earned_tier"])
        last_fetch = row["last_successful_fetch_at"] or "never"
        click.echo(
            f"{row['name']:30} {row['category']:20} "
            f"tier={effective_tier} failures={row['consecutive_failures']} "
            f"last_fetch={last_fetch}"
        )
    conn.close()
