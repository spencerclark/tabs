# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`tabs` is an LLM-curated, searchable knowledge base of Application Security and AI Security news, opinions, and topics. Full design and requirements: `SPEC.md` at the repo root — read it before making design-level decisions; it covers scope phasing, the data model (claims/perspectives/conflicts/story clusters), source trust tiers, the curation pipeline, and non-negotiable constraints (allowlist-only sourcing, confidence gating, mandatory attribution, conflict surfacing).

Implementation proceeds in phases, each with its own plan under `docs/superpowers/plans/`. Phase 1 ("Foundation & Ingestion") is implemented: package scaffolding, SQLite storage, source allowlist sync, feed fetching, article storage/versioning, and the `tabs sources`/`tabs ingest` CLI commands. Later phases (LLM curation, scoring/conflict detection, search, digest generation) are not yet built — do not assume they exist.

## Commands

- Install (editable, with dev deps): `pip install -e ".[dev]"`
- Run the full test suite: `pytest -v`
- Run a single test file: `pytest tests/test_orchestrator.py -v`
- Run a single test: `pytest tests/test_orchestrator.py::test_run_ingest_stores_articles_and_records_success -v`
- Run the CLI: `tabs sources` / `tabs ingest` (see `README.md` for the full command reference, cron setup, and `sources.yaml` editing)

There is no separate lint/typecheck command configured yet.

## Architecture

- **`src/tabs/db.py`** — the SQLite schema (`sources`, `articles`, `claims`, `perspectives`, `conflicts`, `story_clusters`, `anomaly_flags`, `run_log`) and `get_connection()`/`init_db()`. Only `sources`/`articles`/`run_log` are populated by code that exists today; the rest are provisioned ahead of later phases so no migration is needed when curation lands. `init_db()` is idempotent (`CREATE TABLE IF NOT EXISTS`) and safe to call on every command invocation.
- **`src/tabs/models.py`** / **`src/tabs/sources.py`** — the `Source` dataclass and allowlist loading/sync. `sources.yaml` at the repo root is the user-edited source of truth; `sync_sources()` upserts by `feed_url`, seeding `earned_tier = institutional_tier` on first insert but never overwriting `earned_tier` or health fields on re-sync (earned tier drifts independently over time — see SPEC.md §4.4). A source's effective tier for scoring is `max(institutional_tier, earned_tier)`.
- **`src/tabs/ingest/fetch.py`** — `fetch_feed()` wraps `feedparser` with retry/exponential backoff and a per-feed rate-limit delay; raises `FeedFetchError` after exhausting retries.
- **`src/tabs/ingest/storage.py`** — `fetch_article_text()` fetches and extracts an article's body text (HTML tags/scripts/styles stripped, whitespace normalized — this extraction is what gets hashed and stored, not raw HTML; see the module docstring/comments for why raw-HTML hashing was rejected). `store_article()` is idempotent by content hash: unchanged content returns the existing row's id with `created=False`; changed or new content inserts a new row (versioned via `previous_version_id` for a URL seen before) with `created=True`.
- **`src/tabs/ingest/orchestrator.py`** — `run_ingest(conn)` is the integration point: iterates every allowlisted source, fetches its feed, fetches+stores each entry's article (skipping URLs already ingested outside the 14-day re-check window), rate-limits per-article fetches, and tracks per-source health (`consecutive_failures`, `last_successful_fetch_at`) plus a `run_log` audit trail (including a row for every completed run, not just errors). A failure fetching or storing one article/source is logged and skipped — it never aborts the rest of the run; this resilience property is load-bearing and covered by tests that simulate mid-run failures.
- **`src/tabs/commands/`** — CLI subcommands (`sources_cmd.py`, `ingest_cmd.py`), registered on the `click.Group` in `src/tabs/cli.py`. Both commands validate `sources.yaml` and convert failures into `click.ClickException` rather than letting raw tracebacks reach the user; both close their DB connection in a `finally` block.
- **`tests/test_integration.py`** — the one test that drives the real CLI end-to-end (`sync_sources` → `run_ingest` → `fetch_feed` → `fetch_article_text` → `store_article`), stubbing only the true external boundary (`feedparser.parse`, `requests.get`). Every other test file unit-tests one module, monkeypatching its immediate dependencies.

## Known residual risks (see `SPEC.md` §6.5 and the Phase 1 review history for context)

- `fetch_article_text()` validates URL scheme and caps response size, but does not resolve hostnames to block private/link-local IP ranges — full SSRF hardening is deferred. `feed_url` in `sources.yaml` has no scheme validation at all (operator-controlled, lower risk, but inconsistent with the article-fetch gate).
- Text extraction in `storage.py` is a minimal stdlib-only pass (not a proper HTML-to-text library) and can silently produce an empty or truncated string on malformed markup (e.g. an unclosed `<script>` tag). This is a known gap to revisit before Phase 2's curation pipeline starts consuming `full_text` at scale.
