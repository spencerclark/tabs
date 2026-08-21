# tabs
Keeping up with the current understanding and practices of AI and application security. 

## Running

Install: `pip install -e ".[dev]"`

The source allowlist lives in `sources.yaml` at the repo root — edit it to add,
remove, or retier sources (see the comments in the file for what each
`institutional_tier` value means). `tabs` never adds a source on its own.

- `tabs sources` — list allowlisted sources with their effective tier and health
  (consecutive failures, last successful fetch).
- `tabs ingest [--sources-path PATH]` — sync `sources.yaml` into the database,
  then fetch and store new articles from every source. `--sources-path`
  defaults to `sources.yaml` in the current directory.
- Both commands accept a global `--db-path PATH` before the subcommand
  (default: `data/tabs.db`), e.g. `tabs --db-path data/tabs.db sources`.

To run ingestion daily via cron, add a line like this to your crontab
(`crontab -e`), adjusting the path and using an absolute path to the `tabs`
executable from your virtualenv (find it with `which tabs` after activating
the env). `data/` is gitignored and won't exist on a fresh clone, so create
it before redirecting logs into it:

    0 6 * * * cd /Users/spencer/projects/tabs && mkdir -p data && /path/to/venv/bin/tabs ingest >> data/ingest.log 2>&1
