# AppSec & AI Security KB — Phase 1: Foundation & Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `tabs` Python CLI with a SQLite knowledge base, a user-maintained source allowlist, and a daily ingestion pipeline that fetches allowlisted feeds, caches full article text, versions changed articles, and tracks per-source health — with no LLM calls yet.

**Architecture:** A `click`-based CLI (`tabs`) backed by a single SQLite file. `sources.yaml` at the repo root is the user-edited source of truth for the allowlist; `tabs ingest` syncs it into the DB, fetches each feed via `feedparser` with retry/backoff, fetches full article text via `requests`, and stores/versions articles by content hash. This phase implements SPEC.md §3(constraint 1), §4.4, §4.5, §4.6, §5, and the `sources`/`ingest` rows of §9. Curation (§6), scoring/conflicts (§4.1–4.3, §6.4), search (§8), trends/review (§7, §9), and digest (§9, §12) are separate later-phase plans.

**Tech Stack:** Python 3.11+, `click` (CLI), `feedparser` (RSS/Atom), `requests` (article fetch), `pyyaml` (source config), `pytest` (tests). Package layout uses a `src/` layout with `pyproject.toml`.

## Global Constraints

(Copied from `SPEC.md`; apply to every task below.)

- Allowlist-only sourcing — the system never adds a source on its own; `sources.yaml` is the only way a source enters the system (SPEC §3.1, §5.1).
- Every stored claim/perspective record carries mandatory attribution: `source_id`, `article_url`, author (nullable), `published_at`, `retrieved_at` (SPEC §4.6) — this applies to claims/perspectives, not to the `articles` table itself, which has its own fields (`source_id`, `url`, `published_at`, `retrieved_at`; no `author` column) per SPEC §4.5. Claims/perspectives reference articles via `article_id` (normalized FK) rather than duplicating `article_url` as a column — the URL is retrievable via a join to `articles.url`.
- Transient fetch failures (network errors, 5xx, rate limits) retry with exponential backoff, then are skipped and logged — one failing source must never abort the rest of the run (SPEC §5.4).
- Per-source consecutive failure streaks are tracked separately from the per-run transient log, so a persistently broken feed is visible as "needs attention" (SPEC §5.4).
- Articles ingested within the last 14 days are re-fetched each run to detect edits/retractions; older articles are not re-fetched (SPEC §5.3).
- Full article text is cached locally for internal re-analysis only. It must never be reproduced in any exported/shareable output in later phases — later plans must not violate this (SPEC §4.5).
- Language is Python, matching the repo's existing `.gitignore` (SPEC §14).

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/tabs/__init__.py`
- Create: `src/tabs/cli.py`
- Test: `tests/test_cli.py`
- Modify: `.gitignore` (append `data/`)

**Interfaces:**
- Produces: `tabs.__version__` (str); `tabs.cli.main` — a `click.Group` entry point registered as the `tabs` console script.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from click.testing import CliRunner

from tabs.cli import main


def test_version_flag_prints_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tabs'`

- [ ] **Step 3: Write the package files**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "tabs"
version = "0.1.0"
description = "AppSec & AI Security knowledge base"
requires-python = ">=3.11"
dependencies = [
    "click>=8.1",
    "feedparser>=6.0",
    "pyyaml>=6.0",
    "requests>=2.31",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
tabs = "tabs.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

```python
# src/tabs/__init__.py
__version__ = "0.1.0"
```

```python
# src/tabs/cli.py
from pathlib import Path

import click

from tabs import __version__

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


if __name__ == "__main__":
    main()
```

Append to `.gitignore`:
```
# tabs local data
data/
```

- [ ] **Step 4: Install and run test to verify it passes**

Run: `pip install -e ".[dev]" && pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/tabs/__init__.py src/tabs/cli.py tests/test_cli.py .gitignore
git commit -m "feat: scaffold tabs CLI package"
```

---

### Task 2: SQLite schema

**Files:**
- Create: `src/tabs/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: nothing (first module in the dependency chain).
- Produces: `get_connection(db_path: Path) -> sqlite3.Connection` (row_factory set to `sqlite3.Row`, foreign keys on); `init_db(conn: sqlite3.Connection) -> None` (idempotent — safe to call every run). Tables created: `sources`, `articles`, `claims`, `perspectives`, `conflicts`, `story_clusters`, `anomaly_flags`, `run_log`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
from tabs.db import get_connection, init_db

EXPECTED_TABLES = {
    "sources", "articles", "claims", "perspectives",
    "conflicts", "story_clusters", "anomaly_flags", "run_log",
}


def test_init_db_creates_all_tables(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {row["name"] for row in rows}

    assert EXPECTED_TABLES.issubset(table_names)
    conn.close()


def test_init_db_is_idempotent(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    init_db(conn)  # must not raise

    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tabs.db'`

- [ ] **Step 3: Write the schema module**

```python
# src/tabs/db.py
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    feed_url TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    institutional_tier INTEGER NOT NULL,
    earned_tier INTEGER NOT NULL,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_successful_fetch_at TEXT
);

CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES sources(id),
    url TEXT NOT NULL,
    title TEXT,
    full_text TEXT,
    content_hash TEXT NOT NULL,
    published_at TEXT,
    retrieved_at TEXT NOT NULL,
    previous_version_id INTEGER REFERENCES articles(id)
);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    source_id INTEGER NOT NULL REFERENCES sources(id),
    claim_text TEXT NOT NULL,
    supporting_excerpt TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    category TEXT NOT NULL,
    sub_tags TEXT,
    status TEXT NOT NULL DEFAULT 'unverified',
    confidence_score REAL,
    llm_certainty REAL,
    corroboration_count INTEGER NOT NULL DEFAULT 0,
    story_cluster_id INTEGER REFERENCES story_clusters(id),
    author TEXT,
    published_at TEXT,
    retrieved_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS perspectives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    source_id INTEGER NOT NULL REFERENCES sources(id),
    perspective_text TEXT NOT NULL,
    supporting_excerpt TEXT NOT NULL,
    category TEXT NOT NULL,
    sub_tags TEXT,
    author TEXT,
    published_at TEXT,
    retrieved_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_a_id INTEGER NOT NULL REFERENCES claims(id),
    claim_b_id INTEGER NOT NULL REFERENCES claims(id),
    resolution TEXT NOT NULL,
    winning_claim_id INTEGER REFERENCES claims(id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS story_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    summary TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS anomaly_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reviewed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_started_at TEXT NOT NULL,
    run_finished_at TEXT,
    source_id INTEGER REFERENCES sources(id),
    status TEXT NOT NULL,
    message TEXT
);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_db.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tabs/db.py tests/test_db.py
git commit -m "feat: add SQLite schema for sources, articles, and downstream tables"
```

---

### Task 3: Source config loading and sync

**Files:**
- Create: `src/tabs/models.py`
- Create: `src/tabs/sources.py`
- Create: `sources.yaml`
- Test: `tests/test_sources.py`

**Interfaces:**
- Consumes: `tabs.db.get_connection`, `tabs.db.init_db` (Task 2).
- Produces: `Source` dataclass (`name: str`, `feed_url: str`, `category: str`, `institutional_tier: int`); `load_sources_yaml(path: Path) -> list[Source]`; `sync_sources(conn: sqlite3.Connection, sources: list[Source]) -> None` — upserts by `feed_url`, preserving `earned_tier` and health fields on existing rows, seeding `earned_tier = institutional_tier` on insert.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sources.py
from pathlib import Path

from tabs.db import get_connection, init_db
from tabs.sources import load_sources_yaml, sync_sources

SOURCES_YAML = """
- name: Krebs on Security
  feed_url: https://krebsonsecurity.com/feed/
  category: AppSec
  institutional_tier: 2
"""


def _write_yaml(tmp_path: Path) -> Path:
    yaml_path = tmp_path / "sources.yaml"
    yaml_path.write_text(SOURCES_YAML)
    return yaml_path


def test_load_sources_yaml_parses_entries(tmp_path):
    sources = load_sources_yaml(_write_yaml(tmp_path))

    assert len(sources) == 1
    assert sources[0].name == "Krebs on Security"
    assert sources[0].institutional_tier == 2


def test_sync_sources_inserts_new_source_with_earned_tier_seeded(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    sources = load_sources_yaml(_write_yaml(tmp_path))

    sync_sources(conn, sources)

    row = conn.execute(
        "SELECT * FROM sources WHERE feed_url = ?", (sources[0].feed_url,)
    ).fetchone()
    assert row["earned_tier"] == 2
    conn.close()


def test_sync_sources_preserves_earned_tier_on_resync(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    sources = load_sources_yaml(_write_yaml(tmp_path))
    sync_sources(conn, sources)
    conn.execute(
        "UPDATE sources SET earned_tier = 5 WHERE feed_url = ?", (sources[0].feed_url,)
    )
    conn.commit()

    sync_sources(conn, sources)  # re-sync the same source list

    row = conn.execute(
        "SELECT earned_tier FROM sources WHERE feed_url = ?", (sources[0].feed_url,)
    ).fetchone()
    assert row["earned_tier"] == 5
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tabs.sources'`

- [ ] **Step 3: Write the implementation**

```python
# src/tabs/models.py
from dataclasses import dataclass


@dataclass
class Source:
    name: str
    feed_url: str
    category: str
    institutional_tier: int
```

```python
# src/tabs/sources.py
import sqlite3
from pathlib import Path

import yaml

from tabs.models import Source


def load_sources_yaml(path: Path) -> list[Source]:
    with open(path) as f:
        raw = yaml.safe_load(f) or []
    return [
        Source(
            name=entry["name"],
            feed_url=entry["feed_url"],
            category=entry["category"],
            institutional_tier=entry["institutional_tier"],
        )
        for entry in raw
    ]


def sync_sources(conn: sqlite3.Connection, sources: list[Source]) -> None:
    for source in sources:
        existing = conn.execute(
            "SELECT id FROM sources WHERE feed_url = ?", (source.feed_url,)
        ).fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO sources
                    (name, feed_url, category, institutional_tier, earned_tier)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    source.name,
                    source.feed_url,
                    source.category,
                    source.institutional_tier,
                    source.institutional_tier,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE sources
                SET name = ?, category = ?, institutional_tier = ?
                WHERE feed_url = ?
                """,
                (source.name, source.category, source.institutional_tier, source.feed_url),
            )
    conn.commit()
```

Create `sources.yaml` at the repo root with a small starter set of feeds I have high confidence are currently valid (from SPEC.md §11's secondary tier — institutional-tier sources often lack a stable single RSS feed and need hand-verification, so they're left for you to add):

```yaml
# tabs source allowlist.
# institutional_tier: 3 = institutional (standards bodies, CERTs, vendor PSIRTs)
#                      2 = established secondary (independent researchers, security news)
#                      1 = tertiary
# Add more from SPEC.md §11 as you verify their feed URLs — this starter set
# covers only sources whose feed URL is well-known and stable.

- name: Krebs on Security
  feed_url: https://krebsonsecurity.com/feed/
  category: AppSec
  institutional_tier: 2

- name: Schneier on Security
  feed_url: https://www.schneier.com/feed/atom/
  category: AppSec
  institutional_tier: 2

- name: BleepingComputer
  feed_url: https://www.bleepingcomputer.com/feed/
  category: AppSec
  institutional_tier: 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sources.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tabs/models.py src/tabs/sources.py sources.yaml tests/test_sources.py
git commit -m "feat: load and sync the source allowlist from sources.yaml"
```

---

### Task 4: `tabs sources` CLI command

**Files:**
- Create: `src/tabs/commands/__init__.py`
- Create: `src/tabs/commands/sources_cmd.py`
- Modify: `src/tabs/cli.py`
- Test: `tests/test_sources_cmd.py`

**Interfaces:**
- Consumes: `tabs.db.get_connection`, `tabs.db.init_db` (Task 2); `ctx.obj["db_path"]` (Task 1).
- Produces: `sources_cmd` — a `click.Command` named `sources`, registered on `main`. Prints one line per source: name, category, effective tier (`max(institutional_tier, earned_tier)`), consecutive failures, last successful fetch (or `never`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sources_cmd.py
from click.testing import CliRunner

from tabs.cli import main
from tabs.db import get_connection, init_db
from tabs.models import Source
from tabs.sources import sync_sources


def test_sources_command_lists_synced_sources(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    init_db(conn)
    sync_sources(conn, [Source("Krebs on Security", "https://krebsonsecurity.com/feed/", "AppSec", 2)])
    conn.close()

    runner = CliRunner()
    result = runner.invoke(main, ["--db-path", str(db_path), "sources"])

    assert result.exit_code == 0
    assert "Krebs on Security" in result.output
    assert "tier=2" in result.output
    assert "last_fetch=never" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources_cmd.py -v`
Expected: FAIL with `Error: No such command 'sources'`

- [ ] **Step 3: Write the implementation**

```python
# src/tabs/commands/__init__.py
```

```python
# src/tabs/commands/sources_cmd.py
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
```

Modify `src/tabs/cli.py` — add the import and registration:

```python
from tabs.commands.sources_cmd import sources_cmd
```

(add this import near the top, after `from tabs import __version__`)

```python
main.add_command(sources_cmd)
```

(add this line right after the `main` function definition, before the `if __name__ == "__main__":` block)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sources_cmd.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tabs/commands/__init__.py src/tabs/commands/sources_cmd.py src/tabs/cli.py tests/test_sources_cmd.py
git commit -m "feat: add tabs sources command"
```

---

### Task 5: Feed fetching with retry/backoff

**Files:**
- Create: `src/tabs/ingest/__init__.py`
- Create: `src/tabs/ingest/fetch.py`
- Test: `tests/test_fetch.py`

**Interfaces:**
- Consumes: nothing new (uses `feedparser` directly).
- Produces: `FetchedEntry` dataclass (`url: str`, `title: str`, `published_at: str | None`, `summary: str`); `FeedFetchError(Exception)`; `fetch_feed(feed_url: str, sleep=time.sleep) -> list[FetchedEntry]` — retries up to 3 times with exponential backoff (2s, 4s, 8s) on a bozo/empty parse result, raises `FeedFetchError` after exhausting retries. `sleep` is injectable for fast tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetch.py
import feedparser
import pytest

from tabs.ingest.fetch import FeedFetchError, fetch_feed


def _make_parsed(bozo: bool, entries: list[dict]) -> feedparser.FeedParserDict:
    parsed = feedparser.FeedParserDict()
    parsed["bozo"] = bozo
    parsed["entries"] = entries
    return parsed


def test_fetch_feed_returns_entries_on_success(monkeypatch):
    entry = {
        "link": "https://example.com/a",
        "title": "A",
        "published": "2026-08-01",
        "summary": "summary text",
    }
    monkeypatch.setattr(feedparser, "parse", lambda url: _make_parsed(False, [entry]))

    entries = fetch_feed("https://example.com/feed.xml", sleep=lambda s: None)

    assert len(entries) == 1
    assert entries[0].url == "https://example.com/a"
    assert entries[0].title == "A"


def test_fetch_feed_retries_then_succeeds(monkeypatch):
    calls = {"count": 0}

    def fake_parse(url):
        calls["count"] += 1
        if calls["count"] < 2:
            return _make_parsed(True, [])
        return _make_parsed(
            False, [{"link": "u", "title": "t", "published": None, "summary": "s"}]
        )

    monkeypatch.setattr(feedparser, "parse", fake_parse)

    entries = fetch_feed("https://example.com/feed.xml", sleep=lambda s: None)

    assert calls["count"] == 2
    assert len(entries) == 1


def test_fetch_feed_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(feedparser, "parse", lambda url: _make_parsed(True, []))

    with pytest.raises(FeedFetchError):
        fetch_feed("https://example.com/feed.xml", sleep=lambda s: None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tabs.ingest'`

- [ ] **Step 3: Write the implementation**

```python
# src/tabs/ingest/__init__.py
```

```python
# src/tabs/ingest/fetch.py
import time
from dataclasses import dataclass

import feedparser

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2
REQUEST_DELAY_SECONDS = 1


class FeedFetchError(Exception):
    """Raised when a feed cannot be fetched after all retries are exhausted."""


@dataclass
class FetchedEntry:
    url: str
    title: str
    published_at: str | None
    summary: str


def fetch_feed(feed_url: str, sleep=time.sleep) -> list[FetchedEntry]:
    """Fetch and parse a feed, retrying transient failures with backoff."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        parsed = feedparser.parse(feed_url)
        if parsed.get("bozo") and not parsed.entries:
            last_error = parsed.get("bozo_exception")
            sleep(BACKOFF_BASE_SECONDS * (2**attempt))
            continue
        sleep(REQUEST_DELAY_SECONDS)
        return [
            FetchedEntry(
                url=entry.get("link", ""),
                title=entry.get("title", ""),
                published_at=entry.get("published"),
                summary=entry.get("summary", ""),
            )
            for entry in parsed.entries
        ]
    raise FeedFetchError(
        f"Failed to fetch {feed_url} after {MAX_RETRIES} attempts: {last_error}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fetch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tabs/ingest/__init__.py src/tabs/ingest/fetch.py tests/test_fetch.py
git commit -m "feat: fetch and parse RSS/Atom feeds with retry and backoff"
```

---

### Task 6: Article storage with content-hash versioning

**Files:**
- Create: `src/tabs/ingest/storage.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: nothing new (uses `requests` directly; writes to the `articles` table from Task 2).
- Produces: `fetch_article_text(url: str, http_get=requests.get) -> str`; `store_article(conn, source_id: int, url: str, title: str, published_at: str | None, full_text: str) -> int` — returns the existing article's id unchanged if content hash matches the latest stored version for that URL; otherwise inserts a new row, linking `previous_version_id` to the prior version if one existed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage.py
from tabs.db import get_connection, init_db
from tabs.ingest.storage import store_article


def test_store_article_inserts_new_article(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES ('S', 'https://s.example/feed', 'AppSec', 2, 2)"
    )
    conn.commit()

    article_id = store_article(
        conn, source_id=1, url="https://s.example/a", title="A",
        published_at="2026-08-01", full_text="original text",
    )

    row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
    assert row["url"] == "https://s.example/a"
    assert row["previous_version_id"] is None
    conn.close()


def test_store_article_returns_same_id_when_content_unchanged(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES ('S', 'https://s.example/feed', 'AppSec', 2, 2)"
    )
    conn.commit()
    first_id = store_article(
        conn, source_id=1, url="https://s.example/a", title="A",
        published_at="2026-08-01", full_text="original text",
    )

    second_id = store_article(
        conn, source_id=1, url="https://s.example/a", title="A",
        published_at="2026-08-01", full_text="original text",
    )

    assert first_id == second_id
    count = conn.execute("SELECT COUNT(*) AS n FROM articles").fetchone()["n"]
    assert count == 1
    conn.close()


def test_store_article_creates_new_version_when_content_changes(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES ('S', 'https://s.example/feed', 'AppSec', 2, 2)"
    )
    conn.commit()
    first_id = store_article(
        conn, source_id=1, url="https://s.example/a", title="A",
        published_at="2026-08-01", full_text="original text",
    )

    second_id = store_article(
        conn, source_id=1, url="https://s.example/a", title="A",
        published_at="2026-08-01", full_text="updated text",
    )

    assert second_id != first_id
    row = conn.execute("SELECT previous_version_id FROM articles WHERE id = ?", (second_id,)).fetchone()
    assert row["previous_version_id"] == first_id
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tabs.ingest.storage'`

- [ ] **Step 3: Write the implementation**

```python
# src/tabs/ingest/storage.py
import hashlib
import sqlite3
from datetime import datetime, timezone

import requests


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_article_text(url: str, http_get=requests.get) -> str:
    response = http_get(url, timeout=10)
    response.raise_for_status()
    return response.text


def store_article(
    conn: sqlite3.Connection,
    source_id: int,
    url: str,
    title: str,
    published_at: str | None,
    full_text: str,
) -> int:
    """Insert a new article, or a new version if content changed since the last fetch."""
    content_hash = _hash_content(full_text)
    retrieved_at = datetime.now(timezone.utc).isoformat()

    existing = conn.execute(
        "SELECT id, content_hash FROM articles WHERE url = ? ORDER BY id DESC LIMIT 1",
        (url,),
    ).fetchone()

    if existing is not None and existing["content_hash"] == content_hash:
        return existing["id"]

    previous_version_id = existing["id"] if existing is not None else None

    cursor = conn.execute(
        """
        INSERT INTO articles
            (source_id, url, title, full_text, content_hash,
             published_at, retrieved_at, previous_version_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_id, url, title, full_text, content_hash,
            published_at, retrieved_at, previous_version_id,
        ),
    )
    conn.commit()
    return cursor.lastrowid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_storage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tabs/ingest/storage.py tests/test_storage.py
git commit -m "feat: store and version articles by content hash"
```

---

### Task 7: Ingestion orchestrator

**Files:**
- Create: `src/tabs/ingest/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `fetch_feed`, `FeedFetchError` (Task 5); `fetch_article_text`, `store_article` (Task 6).
- Produces: `run_ingest(conn: sqlite3.Connection) -> dict` — iterates every row in `sources`, fetches its feed, fetches+stores each entry's article text (skipping URLs already ingested and outside the 14-day re-check window), updates `consecutive_failures`/`last_successful_fetch_at` per source, writes `run_log` rows, and returns `{"sources_ok": int, "sources_failed": int, "articles_stored": int}`. A failure fetching one source or one article is logged and skipped — it never stops the run.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orchestrator.py
import tabs.ingest.orchestrator as orchestrator_module
from tabs.db import get_connection, init_db
from tabs.ingest.fetch import FeedFetchError, FetchedEntry
from tabs.ingest.orchestrator import run_ingest


def _insert_source(conn, name, feed_url):
    conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES (?, ?, 'AppSec', 2, 2)",
        (name, feed_url),
    )
    conn.commit()


def test_run_ingest_stores_articles_and_records_success(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Good Source", "https://good.example/feed")

    entry = FetchedEntry(url="https://good.example/a", title="A", published_at=None, summary="s")
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [entry])
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    summary = run_ingest(conn)

    assert summary == {"sources_ok": 1, "sources_failed": 0, "articles_stored": 1}
    source_row = conn.execute("SELECT consecutive_failures, last_successful_fetch_at FROM sources").fetchone()
    assert source_row["consecutive_failures"] == 0
    assert source_row["last_successful_fetch_at"] is not None
    article_row = conn.execute("SELECT COUNT(*) AS n FROM articles").fetchone()
    assert article_row["n"] == 1
    conn.close()


def test_run_ingest_skips_failing_source_and_continues(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Bad Source", "https://bad.example/feed")
    _insert_source(conn, "Good Source", "https://good.example/feed")

    good_entry = FetchedEntry(url="https://good.example/a", title="A", published_at=None, summary="s")

    def fake_fetch_feed(feed_url):
        if feed_url == "https://bad.example/feed":
            raise FeedFetchError("boom")
        return [good_entry]

    monkeypatch.setattr(orchestrator_module, "fetch_feed", fake_fetch_feed)
    monkeypatch.setattr(orchestrator_module, "fetch_article_text", lambda url: "full text")

    summary = run_ingest(conn)

    assert summary == {"sources_ok": 1, "sources_failed": 1, "articles_stored": 1}
    bad_row = conn.execute(
        "SELECT consecutive_failures FROM sources WHERE name = 'Bad Source'"
    ).fetchone()
    assert bad_row["consecutive_failures"] == 1
    run_log_rows = conn.execute("SELECT status FROM run_log WHERE status = 'error'").fetchall()
    assert len(run_log_rows) >= 1
    conn.close()


def test_run_ingest_skips_previously_ingested_urls_outside_recheck_window(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    _insert_source(conn, "Source", "https://s.example/feed")
    conn.execute(
        "INSERT INTO articles (source_id, url, title, full_text, content_hash, "
        "published_at, retrieved_at, previous_version_id) "
        "VALUES (1, 'https://s.example/old', 'Old', 'text', 'hash', NULL, '2000-01-01T00:00:00+00:00', NULL)"
    )
    conn.commit()

    old_entry = FetchedEntry(url="https://s.example/old", title="Old", published_at=None, summary="s")
    monkeypatch.setattr(orchestrator_module, "fetch_feed", lambda feed_url: [old_entry])

    fetch_calls = []
    monkeypatch.setattr(
        orchestrator_module,
        "fetch_article_text",
        lambda url: fetch_calls.append(url) or "text",
    )

    summary = run_ingest(conn)

    assert fetch_calls == []  # old article, outside the 14-day re-check window: not re-fetched
    assert summary["articles_stored"] == 0
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tabs.ingest.orchestrator'`

- [ ] **Step 3: Write the implementation**

```python
# src/tabs/ingest/orchestrator.py
import sqlite3
from datetime import datetime, timedelta, timezone

from tabs.ingest.fetch import FeedFetchError, fetch_feed
from tabs.ingest.storage import fetch_article_text, store_article

RECENT_ARTICLE_WINDOW_DAYS = 14


def run_ingest(conn: sqlite3.Connection) -> dict:
    """Run one ingestion pass over every allowlisted source. Returns a summary dict."""
    summary = {"sources_ok": 0, "sources_failed": 0, "articles_stored": 0}

    sources = conn.execute("SELECT id, feed_url FROM sources").fetchall()
    for source in sources:
        try:
            entries = fetch_feed(source["feed_url"])
        except FeedFetchError as exc:
            _record_failure(conn, source["id"], str(exc))
            summary["sources_failed"] += 1
            continue

        recheck_urls = _urls_in_recheck_window(conn, source["id"])
        for entry in entries:
            already_seen = _already_ingested(conn, entry.url)
            if already_seen and entry.url not in recheck_urls:
                continue  # older article, already ingested, outside the re-check window

            try:
                full_text = fetch_article_text(entry.url)
            except Exception as exc:  # noqa: BLE001 — one bad article must not kill the run
                _log_run(conn, source["id"], "error", f"article fetch failed: {entry.url}: {exc}")
                continue

            store_article(conn, source["id"], entry.url, entry.title, entry.published_at, full_text)
            summary["articles_stored"] += 1

        _record_success(conn, source["id"])
        summary["sources_ok"] += 1

    return summary


def _urls_in_recheck_window(conn: sqlite3.Connection, source_id: int) -> set[str]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RECENT_ARTICLE_WINDOW_DAYS)).isoformat()
    rows = conn.execute(
        "SELECT DISTINCT url FROM articles WHERE source_id = ? AND retrieved_at >= ?",
        (source_id, cutoff),
    ).fetchall()
    return {row["url"] for row in rows}


def _already_ingested(conn: sqlite3.Connection, url: str) -> bool:
    row = conn.execute("SELECT 1 FROM articles WHERE url = ? LIMIT 1", (url,)).fetchone()
    return row is not None


def _record_failure(conn: sqlite3.Connection, source_id: int, message: str) -> None:
    conn.execute(
        "UPDATE sources SET consecutive_failures = consecutive_failures + 1 WHERE id = ?",
        (source_id,),
    )
    conn.commit()
    _log_run(conn, source_id, "error", message)


def _record_success(conn: sqlite3.Connection, source_id: int) -> None:
    conn.execute(
        "UPDATE sources SET consecutive_failures = 0, last_successful_fetch_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), source_id),
    )
    conn.commit()


def _log_run(conn: sqlite3.Connection, source_id: int, status: str, message: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO run_log (run_started_at, run_finished_at, source_id, status, message) "
        "VALUES (?, ?, ?, ?, ?)",
        (now, now, source_id, status, message),
    )
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tabs/ingest/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: orchestrate per-source ingestion with health tracking and run log"
```

---

### Task 8: `tabs ingest` CLI command and cron setup

**Files:**
- Create: `src/tabs/commands/ingest_cmd.py`
- Modify: `src/tabs/cli.py`
- Test: `tests/test_ingest_cmd.py`

**Interfaces:**
- Consumes: `run_ingest` (Task 7); `sync_sources`, `load_sources_yaml` (Task 3); `get_connection`, `init_db` (Task 2).
- Produces: `ingest_cmd` — a `click.Command` named `ingest`, registered on `main`. Syncs `sources.yaml` into the DB, runs `run_ingest`, and prints a one-line summary.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_cmd.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest_cmd.py -v`
Expected: FAIL with `Error: No such command 'ingest'`

- [ ] **Step 3: Write the implementation**

```python
# src/tabs/commands/ingest_cmd.py
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
```

Modify `src/tabs/cli.py` — add the import and registration alongside the existing `sources_cmd` wiring:

```python
from tabs.commands.ingest_cmd import ingest_cmd
```

```python
main.add_command(ingest_cmd)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest_cmd.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: All tests PASS (Tasks 1–8)

- [ ] **Step 6: Document the cron entry**

Add a `## Running` section to a new `README.md` at the repo root (create it if it doesn't already cover this) with:

```markdown
## Running

Install: `pip install -e ".[dev]"`

Run ingestion manually: `tabs ingest`

To run daily via cron, add a line like this to your crontab (`crontab -e`),
adjusting the path and using an absolute path to the `tabs` executable from
your virtualenv (find it with `which tabs` after activating the env):

    0 6 * * * cd /Users/spencer/projects/tabs && /path/to/venv/bin/tabs ingest >> data/ingest.log 2>&1
```

- [ ] **Step 7: Commit**

```bash
git add src/tabs/commands/ingest_cmd.py src/tabs/cli.py tests/test_ingest_cmd.py README.md
git commit -m "feat: add tabs ingest command and cron setup instructions"
```

---

## Self-Review Notes

- **Spec coverage:** §3 constraint 1 (allowlist-only) → Tasks 3, 8. §4.4 (sources table/tiers) → Task 2, 3, 4. §4.5 (articles, versioning, full-text caching) → Task 2, 6. §4.6 (attribution fields) → Task 2 schema, Task 6 (`published_at`/`retrieved_at` always set). §5.1 (allowlist mgmt) → Task 3. §5.2 (cadence/rate-limit) → Task 5 (`REQUEST_DELAY_SECONDS`), Task 8 (cron doc). §5.3 (re-fetch/versioning) → Task 6, 7. §5.4 (error handling/health) → Task 7. §9 `tabs sources`/`tabs ingest` rows → Task 4, 8. §14 (Python) → Task 1.
- Remaining SPEC sections (§4.1–4.3 claims/perspectives/conflicts content, §6 curation, §7 trends, §8 search, §9 remaining commands, §12 digest scheduling, §13 golden-set testing) are out of scope for this phase by design — see later phase plans.
- **Placeholder scan:** no TBD/TODO markers; every step has complete, runnable code.
- **Type consistency:** `Source` dataclass fields match `load_sources_yaml`/`sync_sources` usage across Tasks 3 and 8. `FetchedEntry` fields match `orchestrator.py` usage across Tasks 5 and 7. `ctx.obj["db_path"]` set in Task 1 is read identically in Tasks 4 and 8.
