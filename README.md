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

## Curation (requires an Anthropic API key, and costs money per run)

`tabs ingest` also curates content: a cheap triage pass (Claude Haiku 4.5)
filters every genuinely new feed entry for relevance before the article body
is even fetched, and an extraction pass (Claude Sonnet 5) pulls structured
claims and perspectives out of any article that's newly stored — either a
brand-new in-scope article, or a previously-ingested one whose content
actually changed on a re-check. Set `ANTHROPIC_API_KEY` in your environment
before running `tabs ingest` (see Anthropic's documentation for how to obtain
a key). Claims land in an `unverified` state — confidence scoring, conflict
detection, and story clustering are a later phase.

If every Anthropic API call in a run fails (e.g. a missing or invalid API
key), `tabs ingest` exits non-zero instead of silently reporting success —
worth alerting on if you're running this under cron.

Every extracted claim is then embedded (Voyage AI `voyage-4-lite`) and compared against
recent same-category claims to detect corroboration and conflicts, producing a composite
confidence score that gates each claim to `verified`, `unverified`, or `misinformation`
(SPEC.md §6.3-6.4). Set `VOYAGE_API_KEY` in your environment alongside `ANTHROPIC_API_KEY`.
Unlike a fully-broken Anthropic key, a fully-broken Voyage key does not currently fail the
run — every claim is scored on its own tier/certainty/type merits without corroboration,
logged per-claim in `run_log`, visible via the `claims_scored`/`claims_unscored` summary
counts rather than a non-zero exit.
