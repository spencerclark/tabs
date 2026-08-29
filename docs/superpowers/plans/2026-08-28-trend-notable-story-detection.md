# AppSec & AI Security KB — Phase 3: Trend & Notable Story Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tabs trends [--since <window>]`: on-demand notable-story ranking (story clusters by corroboration count and recency) and category/sub-tag volume-spike detection (current window vs. a trailing baseline average), computed directly from the existing `claims`/`perspectives`/`story_clusters` tables — no new LLM calls, no precomputed trend table.

**Architecture:** A new `src/tabs/trends/` package: `volume.py` (`category_volume`/`sub_tag_volume`, plain SQL `GROUP BY` aggregation over `claims` + `perspectives` in a `[start, end)` window, expanding the `sub_tags` JSON array via SQLite's `json_each`), `spikes.py` (`detect_spikes`, comparing a current window's volume against a trailing baseline average built from `volume.py`), and `notable.py` (`notable_stories`, ranking `story_clusters` by the `corroboration_count` already maintained by Phase 2b's `score/storage.py`). `src/tabs/commands/trends_cmd.py` wires both into a new `tabs trends` CLI command, registered in `cli.py` alongside `sources`/`ingest`. This phase implements SPEC.md §7 (the `tabs trends` half only — digest wiring is out of scope, see Deferred Scope).

**Tech Stack:** No new dependencies. Uses SQLite's built-in JSON1 extension (`json_each`, bundled with Python's stdlib `sqlite3` — no new dependency) to expand the `sub_tags` JSON-array column that `curate/storage.py` already writes.

**Spec:** `SPEC.md` §7 (Trend & Notable Story Detection), §9 (CLI) — see also the brainstorming decisions folded into §7 by commit `d67b105`.

## Deferred Scope (explicitly out of this plan)

- **`tabs digest` and digest wiring.** SPEC §7: "Surfaced two ways: automatically as a section in the scheduled digest, and on demand via `tabs trends`." This plan builds the on-demand path only; wiring trend/notable-story output into a scheduled digest is deferred until digest generation itself is built (confirmed during brainstorming — SPEC §7 already records this).
- **Story cluster summaries.** `story_clusters.summary` stays `NULL`, unchanged from Phase 2b (see that phase's Deferred Scope). `notable_stories` returns the cluster's most recent non-misinformation claim's text as a display stand-in, not a generated summary.
- **Cluster merging / a single blended global ranking.** This phase reports notable stories and spikes per category (and per category/sub-tag), matching SPEC's language exactly. SPEC does not ask for a single cross-category "top trend of all" ranking, so none is built.
- **Caching or precomputing trend numbers.** Explicitly decided against during brainstorming (SPEC §7: "computed on demand from the existing tables at query time — no separate precomputed trend-tracking table"). Every `tabs trends` invocation re-aggregates from scratch.

## Global Constraints

(Copied/scoped from `SPEC.md` §7, and from explicit decisions made while brainstorming this plan; apply to every task below.)

- Trend volume counts **both claims and perspectives**. Perspectives are never truth-gated (SPEC §4.1 — "recorded as 'who said what,' not fact-checked") and have no `status` column at all, so every perspective in a window counts unconditionally.
- Claims with `status = 'misinformation'` are **excluded** from both trend volume and notable-story ranking (SPEC §7, confirmed during brainstorming) — a debunked claim must not inflate a category's volume or a cluster's prominence. `unverified`/`verified` claims both count; this exclusion is specifically and only about `misinformation`.
- Volume/spike numbers are computed **on demand** via SQL aggregation at query time. No new table, no cached/precomputed trend data (SPEC §7).
- The baseline window generalizes SPEC §7's "current week vs. trailing 4-week average" example to any `--since` value: the baseline is always the `since_days * BASELINE_MULTIPLIER`-day period immediately preceding the current window, non-overlapping (confirmed during brainstorming — `BASELINE_MULTIPLIER` defaults to `4`).
- Spike significance is a tunable parameter, not a value fixed by SPEC (§7: "a percentage-change floor with a minimum volume guard to avoid noise from low-volume tags"). Implemented as named, documented constants in `trends/spikes.py` (`MIN_VOLUME_GUARD`, `SPIKE_THRESHOLD_PCT`), matching the precedent set by `score/scoring.py`'s tunable constants in Phase 2b.
- All window/recency comparisons use `retrieved_at`, matching the established convention from Phase 2b's `score/matching.py` and `score/conflicts.py` (`published_at` is unreliable raw feed text; `retrieved_at` is always ISO 8601 and reliably parseable/comparable as a string).
- Notable-story corroboration counts are read from the **existing** `claims.corroboration_count` column (maintained by Phase 2b's `score/storage.py._join_story_cluster`) rather than recomputed independently — one existing definition of "how corroborated is this cluster," not two that could drift apart.
- This phase's CLI surface is `tabs trends [--since <window>]` only, per SPEC §9's table and the brainstormed scope decision. No `--category`/`--tier` filters are added — SPEC's CLI table doesn't list any for `tabs trends`, unlike `tabs search`.
- Language is Python, matching the existing codebase (SPEC §14).

---

### Task 1: Volume aggregation (`category_volume`, `sub_tag_volume`) + a supporting index

**Files:**
- Modify: `src/tabs/db.py`
- Modify: `tests/test_db.py`
- Create: `src/tabs/trends/__init__.py`
- Create: `src/tabs/trends/volume.py`
- Test: `tests/test_trends_volume.py`

**Interfaces:**
- Consumes: the existing `claims`/`perspectives` tables (`category`, `sub_tags`, `status`, `retrieved_at` columns).
- Produces: `idx_perspectives_category_retrieved_at` index; `category_volume(conn, start: str, end: str) -> dict[str, int]`; `sub_tag_volume(conn, start: str, end: str) -> dict[tuple[str, str], int]` (keyed by `(category, sub_tag)`). Both take ISO-8601 `start`/`end` strings and query `[start, end)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_db.py` (after the existing tests):

```python
def test_init_db_creates_the_perspectives_category_retrieved_at_index(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)

    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()

    assert "idx_perspectives_category_retrieved_at" in {row["name"] for row in rows}
    conn.close()
```

```python
# tests/test_trends_volume.py
import json
from datetime import datetime, timedelta, timezone

from tabs.db import get_connection, init_db
from tabs.trends.volume import category_volume, sub_tag_volume


def _insert_source(conn, name="source"):
    cursor = conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES (?, ?, 'AppSec', 2, 2)",
        (name, f"https://{name}.example/feed"),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_article(conn, source_id, url):
    cursor = conn.execute(
        "INSERT INTO articles (source_id, url, title, full_text, content_hash, "
        "published_at, retrieved_at, previous_version_id) "
        "VALUES (?, ?, 'T', 'text', 'hash', NULL, ?, NULL)",
        (source_id, url, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_claim(
    conn, article_id, source_id, category="AppSec", sub_tags=None, status="verified",
    retrieved_at=None,
):
    retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO claims (article_id, source_id, claim_text, supporting_excerpt, "
        "claim_type, category, sub_tags, status, llm_certainty, retrieved_at, created_at) "
        "VALUES (?, ?, 'claim', 'excerpt', 'factual', ?, ?, ?, 0.5, ?, ?)",
        (
            article_id, source_id, category, json.dumps(sub_tags or []), status,
            retrieved_at, retrieved_at,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_perspective(
    conn, article_id, source_id, category="AppSec", sub_tags=None, retrieved_at=None,
):
    retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO perspectives (article_id, source_id, perspective_text, "
        "supporting_excerpt, category, sub_tags, retrieved_at, created_at) "
        "VALUES (?, ?, 'perspective', 'excerpt', ?, ?, ?, ?)",
        (article_id, source_id, category, json.dumps(sub_tags or []), retrieved_at, retrieved_at),
    )
    conn.commit()
    return cursor.lastrowid


def _window():
    now = datetime.now(timezone.utc)
    return (now - timedelta(days=1)).isoformat(), (now + timedelta(days=1)).isoformat()


def test_category_volume_counts_claims_and_perspectives_together(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    _insert_claim(conn, article_id, source_id, category="AppSec")
    _insert_perspective(conn, article_id, source_id, category="AppSec")
    start, end = _window()

    counts = category_volume(conn, start, end)

    assert counts == {"AppSec": 2}
    conn.close()


def test_category_volume_excludes_misinformation_claims(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    _insert_claim(conn, article_id, source_id, category="AppSec", status="verified")
    _insert_claim(conn, article_id, source_id, category="AppSec", status="misinformation")
    start, end = _window()

    counts = category_volume(conn, start, end)

    assert counts == {"AppSec": 1}
    conn.close()


def test_category_volume_excludes_items_outside_the_window(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    _insert_claim(conn, article_id, source_id, category="AppSec", retrieved_at=old)
    start, end = _window()

    counts = category_volume(conn, start, end)

    assert counts == {}
    conn.close()


def test_sub_tag_volume_counts_each_tag_on_a_multi_tagged_item(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    _insert_claim(
        conn, article_id, source_id, category="AppSec",
        sub_tags=["Supply Chain", "AuthN/AuthZ"],
    )
    start, end = _window()

    counts = sub_tag_volume(conn, start, end)

    assert counts == {("AppSec", "Supply Chain"): 1, ("AppSec", "AuthN/AuthZ"): 1}
    conn.close()


def test_sub_tag_volume_ignores_items_with_no_sub_tags(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    _insert_claim(conn, article_id, source_id, category="AppSec", sub_tags=[])
    start, end = _window()

    counts = sub_tag_volume(conn, start, end)

    assert counts == {}
    conn.close()


def test_sub_tag_volume_excludes_misinformation_claims(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    _insert_claim(
        conn, article_id, source_id, category="AppSec", sub_tags=["Supply Chain"],
        status="misinformation",
    )
    start, end = _window()

    counts = sub_tag_volume(conn, start, end)

    assert counts == {}
    conn.close()


def test_sub_tag_volume_counts_claims_and_perspectives_together(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    _insert_claim(conn, article_id, source_id, category="AppSec", sub_tags=["Prompt Injection"])
    _insert_perspective(
        conn, article_id, source_id, category="AppSec", sub_tags=["Prompt Injection"],
    )
    start, end = _window()

    counts = sub_tag_volume(conn, start, end)

    assert counts == {("AppSec", "Prompt Injection"): 2}
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trends_volume.py tests/test_db.py -v`
Expected: `test_trends_volume.py` FAILS with `ModuleNotFoundError: No module named 'tabs.trends'`; the new `test_db.py` test FAILS (no such index yet) while pre-existing `test_db.py` tests still pass.

- [ ] **Step 3: Add the index and the volume module**

Modify `src/tabs/db.py`:
- Change `SCHEMA_VERSION = 2` to `SCHEMA_VERSION = 3`.
- Immediately after the `perspectives` table's `CREATE TABLE` statement (right before the existing `idx_claims_category_retrieved_at` comment/index), add:

```sql
-- trends.volume's category/sub-tag aggregation filters perspectives by category and a
-- retrieved_at window, mirroring claims' existing idx_claims_category_retrieved_at.
CREATE INDEX IF NOT EXISTS idx_perspectives_category_retrieved_at
    ON perspectives(category, retrieved_at);
```

So the two indexes now sit together, right after the `perspectives` table and before the `conflicts` table:

```sql
-- trends.volume's category/sub-tag aggregation filters perspectives by category and a
-- retrieved_at window, mirroring claims' existing idx_claims_category_retrieved_at.
CREATE INDEX IF NOT EXISTS idx_perspectives_category_retrieved_at
    ON perspectives(category, retrieved_at);

-- score.matching's candidate retrieval filters claims by category and a retrieved_at
-- recency cutoff, once per newly-extracted claim.
CREATE INDEX IF NOT EXISTS idx_claims_category_retrieved_at ON claims(category, retrieved_at);
```

Create `src/tabs/trends/__init__.py` (empty).

Create `src/tabs/trends/volume.py`:

```python
import sqlite3


def category_volume(conn: sqlite3.Connection, start: str, end: str) -> dict[str, int]:
    """Count of claims + perspectives retrieved in [start, end), grouped by top-level
    category.

    Claims with status='misinformation' are excluded (SPEC §7 — a debunked claim must
    not inflate a category's volume). Perspectives are never truth-gated (SPEC §4.1) and
    have no status column, so every perspective in the window counts.
    """
    counts: dict[str, int] = {}
    _accumulate(counts, conn.execute(
        "SELECT category, COUNT(*) AS n FROM claims "
        "WHERE status != 'misinformation' AND retrieved_at >= ? AND retrieved_at < ? "
        "GROUP BY category",
        (start, end),
    ).fetchall())
    _accumulate(counts, conn.execute(
        "SELECT category, COUNT(*) AS n FROM perspectives "
        "WHERE retrieved_at >= ? AND retrieved_at < ? GROUP BY category",
        (start, end),
    ).fetchall())
    return counts


def sub_tag_volume(conn: sqlite3.Connection, start: str, end: str) -> dict[tuple[str, str], int]:
    """Count of claims + perspectives retrieved in [start, end), grouped by
    (category, sub_tag).

    sub_tags is a JSON array column; json_each expands it so an item tagged with several
    sub_tags contributes to each one's count. COALESCE guards a NULL sub_tags value (the
    column is nullable) — json_each(NULL) would otherwise error. Same misinformation
    exclusion as category_volume.
    """
    counts: dict[tuple[str, str], int] = {}
    _accumulate_tags(counts, conn.execute(
        "SELECT category, je.value AS sub_tag, COUNT(*) AS n "
        "FROM claims, json_each(COALESCE(claims.sub_tags, '[]')) AS je "
        "WHERE status != 'misinformation' AND retrieved_at >= ? AND retrieved_at < ? "
        "GROUP BY category, je.value",
        (start, end),
    ).fetchall())
    _accumulate_tags(counts, conn.execute(
        "SELECT category, je.value AS sub_tag, COUNT(*) AS n "
        "FROM perspectives, json_each(COALESCE(perspectives.sub_tags, '[]')) AS je "
        "WHERE retrieved_at >= ? AND retrieved_at < ? GROUP BY category, je.value",
        (start, end),
    ).fetchall())
    return counts


def _accumulate(counts: dict[str, int], rows) -> None:
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + row["n"]


def _accumulate_tags(counts: dict[tuple[str, str], int], rows) -> None:
    for row in rows:
        key = (row["category"], row["sub_tag"])
        counts[key] = counts.get(key, 0) + row["n"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trends_volume.py tests/test_db.py -v`
Expected: PASS (7 new `test_trends_volume.py` tests + all `test_db.py` tests, including the 1 new one)

- [ ] **Step 5: Commit**

```bash
git add src/tabs/db.py tests/test_db.py src/tabs/trends/__init__.py src/tabs/trends/volume.py tests/test_trends_volume.py
git commit -m "feat: add category/sub-tag volume aggregation for trend detection"
```

---

### Task 2: Spike detection

**Files:**
- Create: `src/tabs/trends/spikes.py`
- Test: `tests/test_trends_spikes.py`

**Interfaces:**
- Consumes: `category_volume`, `sub_tag_volume` (Task 1).
- Produces: `BASELINE_MULTIPLIER`, `MIN_VOLUME_GUARD`, `SPIKE_THRESHOLD_PCT` (tunable constants); `Spike` (dataclass: `category: str`, `sub_tag: Optional[str]`, `current_volume: int`, `baseline_avg: float`, `pct_change: Optional[float]` — `sub_tag=None` marks a category-level spike; `pct_change=None` marks a brand-new tag with no baseline history); `detect_spikes(conn, since_days: int) -> list[Spike]`, sorted with brand-new tags first, then by `pct_change` descending.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trends_spikes.py
import json
from datetime import datetime, timedelta, timezone

from tabs.db import get_connection, init_db
from tabs.trends.spikes import (
    BASELINE_MULTIPLIER,
    MIN_VOLUME_GUARD,
    SPIKE_THRESHOLD_PCT,
    detect_spikes,
)


def _insert_source(conn, name="source"):
    cursor = conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES (?, ?, 'AppSec', 2, 2)",
        (name, f"https://{name}.example/feed"),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_article(conn, source_id, url):
    cursor = conn.execute(
        "INSERT INTO articles (source_id, url, title, full_text, content_hash, "
        "published_at, retrieved_at, previous_version_id) "
        "VALUES (?, ?, 'T', 'text', 'hash', NULL, ?, NULL)",
        (source_id, url, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_claim_days_ago(conn, article_id, source_id, category, days_ago, sub_tags=None):
    retrieved_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    conn.execute(
        "INSERT INTO claims (article_id, source_id, claim_text, supporting_excerpt, "
        "claim_type, category, sub_tags, status, llm_certainty, retrieved_at, created_at) "
        "VALUES (?, ?, 'claim', 'excerpt', 'factual', ?, ?, 'verified', 0.5, ?, ?)",
        (article_id, source_id, category, json.dumps(sub_tags or []), retrieved_at, retrieved_at),
    )
    conn.commit()


def _setup(conn):
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    return source_id, article_id


def _fill_baseline(conn, article_id, source_id, category, count_per_window, since_days=7):
    for window in range(BASELINE_MULTIPLIER):
        days_ago = since_days + window * since_days + 1
        for _ in range(count_per_window):
            _insert_claim_days_ago(conn, article_id, source_id, category, days_ago=days_ago)


def test_detect_spikes_flags_a_category_with_no_baseline_history_as_new(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id, article_id = _setup(conn)
    for _ in range(MIN_VOLUME_GUARD):
        _insert_claim_days_ago(conn, article_id, source_id, "AppSec", days_ago=1)

    spikes = detect_spikes(conn, since_days=7)

    matching = [s for s in spikes if s.category == "AppSec" and s.sub_tag is None]
    assert len(matching) == 1
    assert matching[0].pct_change is None
    assert matching[0].current_volume == MIN_VOLUME_GUARD
    conn.close()


def test_detect_spikes_ignores_a_category_below_the_minimum_volume_guard(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id, article_id = _setup(conn)
    for _ in range(MIN_VOLUME_GUARD - 1):
        _insert_claim_days_ago(conn, article_id, source_id, "AppSec", days_ago=1)

    spikes = detect_spikes(conn, since_days=7)

    assert spikes == []
    conn.close()


def test_detect_spikes_ignores_a_category_whose_change_is_below_the_threshold(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id, article_id = _setup(conn)
    _fill_baseline(conn, article_id, source_id, "AppSec", count_per_window=10)
    for _ in range(10):  # matches the baseline average exactly: 0% change
        _insert_claim_days_ago(conn, article_id, source_id, "AppSec", days_ago=1)

    spikes = detect_spikes(conn, since_days=7)

    assert spikes == []
    conn.close()


def test_detect_spikes_flags_a_category_whose_volume_exceeds_the_threshold(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id, article_id = _setup(conn)
    _fill_baseline(conn, article_id, source_id, "AppSec", count_per_window=4)
    # baseline_avg = (4*4)/4 = 4; comfortably above the 4 * (1 + SPIKE_THRESHOLD_PCT) floor
    current_volume = int(4 * (1 + SPIKE_THRESHOLD_PCT)) + 3
    for _ in range(current_volume):
        _insert_claim_days_ago(conn, article_id, source_id, "AppSec", days_ago=1)

    spikes = detect_spikes(conn, since_days=7)

    matching = [s for s in spikes if s.category == "AppSec" and s.sub_tag is None]
    assert len(matching) == 1
    assert matching[0].current_volume == current_volume
    assert matching[0].baseline_avg == 4.0
    assert matching[0].pct_change >= SPIKE_THRESHOLD_PCT
    conn.close()


def test_detect_spikes_flags_a_sub_tag_spike_independently_of_its_category(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id, article_id = _setup(conn)
    for _ in range(MIN_VOLUME_GUARD):
        _insert_claim_days_ago(
            conn, article_id, source_id, "AppSec", days_ago=1, sub_tags=["Supply Chain"],
        )

    spikes = detect_spikes(conn, since_days=7)

    assert any(s.sub_tag == "Supply Chain" for s in spikes)
    conn.close()


def test_detect_spikes_sorts_new_tags_before_tags_ranked_by_pct_change_descending(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id, article_id = _setup(conn)
    _fill_baseline(conn, article_id, source_id, "AI Security", count_per_window=4)
    for _ in range(20):
        _insert_claim_days_ago(conn, article_id, source_id, "AI Security", days_ago=1)
    for _ in range(MIN_VOLUME_GUARD):
        _insert_claim_days_ago(conn, article_id, source_id, "AppSec", days_ago=1)

    spikes = detect_spikes(conn, since_days=7)

    assert spikes[0].category == "AppSec"
    assert spikes[0].pct_change is None
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trends_spikes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tabs.trends.spikes'`

- [ ] **Step 3: Write the implementation**

```python
# src/tabs/trends/spikes.py
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from tabs.trends.volume import category_volume, sub_tag_volume

# SPEC §7: "current week vs. trailing 4-week average" is an illustrative example, not a
# fixed rule. BASELINE_MULTIPLIER generalizes it to any --since window: the baseline is
# always the (since_days * BASELINE_MULTIPLIER)-day period immediately preceding the
# current window, non-overlapping (confirmed during brainstorming).
BASELINE_MULTIPLIER = 4
# Tunable per SPEC §7 ("a percentage-change floor with a minimum volume guard to avoid
# noise from low-volume tags... not a value fixed by this spec").
MIN_VOLUME_GUARD = 3
SPIKE_THRESHOLD_PCT = 0.5


@dataclass
class Spike:
    category: str
    sub_tag: Optional[str]  # None => a category-level spike, not a sub-tag one
    current_volume: int
    baseline_avg: float
    pct_change: Optional[float]  # None when baseline_avg == 0 (a brand-new tag)


def detect_spikes(conn: sqlite3.Connection, since_days: int) -> list[Spike]:
    """Flag category/sub-tag volume spikes: the current `since_days`-day window vs. the
    average of the BASELINE_MULTIPLIER windows immediately preceding it.

    Sorted with brand-new tags (no baseline history at all) first, then by pct_change
    descending — a tag with no history and a tag with a huge jump are both worth a
    reader's attention before a milder, well-established increase.
    """
    now = datetime.now(timezone.utc)
    current_start = now - timedelta(days=since_days)
    baseline_start = current_start - timedelta(days=since_days * BASELINE_MULTIPLIER)
    now_iso = now.isoformat()
    current_start_iso = current_start.isoformat()
    baseline_start_iso = baseline_start.isoformat()

    current_categories = category_volume(conn, current_start_iso, now_iso)
    baseline_categories = category_volume(conn, baseline_start_iso, current_start_iso)
    current_sub_tags = sub_tag_volume(conn, current_start_iso, now_iso)
    baseline_sub_tags = sub_tag_volume(conn, baseline_start_iso, current_start_iso)

    spikes = []
    for category, current_volume in current_categories.items():
        spike = _evaluate(category, None, current_volume, baseline_categories.get(category, 0))
        if spike is not None:
            spikes.append(spike)
    for (category, sub_tag), current_volume in current_sub_tags.items():
        baseline_total = baseline_sub_tags.get((category, sub_tag), 0)
        spike = _evaluate(category, sub_tag, current_volume, baseline_total)
        if spike is not None:
            spikes.append(spike)

    spikes.sort(key=lambda s: (s.pct_change is not None, -(s.pct_change or 0)))
    return spikes


def _evaluate(
    category: str, sub_tag: Optional[str], current_volume: int, baseline_total: int,
) -> Optional[Spike]:
    if current_volume < MIN_VOLUME_GUARD:
        return None
    baseline_avg = baseline_total / BASELINE_MULTIPLIER
    if baseline_avg == 0:
        return Spike(category, sub_tag, current_volume, baseline_avg, None)
    pct_change = (current_volume - baseline_avg) / baseline_avg
    if pct_change < SPIKE_THRESHOLD_PCT:
        return None
    return Spike(category, sub_tag, current_volume, baseline_avg, pct_change)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trends_spikes.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tabs/trends/spikes.py tests/test_trends_spikes.py
git commit -m "feat: add category/sub-tag volume-spike detection"
```

---

### Task 3: Notable stories

**Files:**
- Create: `src/tabs/trends/notable.py`
- Test: `tests/test_trends_notable.py`

**Interfaces:**
- Consumes: the existing `story_clusters`/`claims` tables (`corroboration_count`, `story_cluster_id`, `status`, `retrieved_at`, `claim_text`, `category` columns).
- Produces: `NotableStory` (dataclass: `story_cluster_id: int`, `category: str`, `corroboration_count: int`, `most_recent_retrieved_at: str`, `sample_claim_text: str`); `notable_stories(conn, since_days: int, limit: int = 10) -> list[NotableStory]`, ranked by `corroboration_count` descending then `most_recent_retrieved_at` descending.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trends_notable.py
from datetime import datetime, timedelta, timezone

from tabs.db import get_connection, init_db
from tabs.trends.notable import notable_stories


def _insert_source(conn, name="source"):
    cursor = conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES (?, ?, 'AppSec', 2, 2)",
        (name, f"https://{name}.example/feed"),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_article(conn, source_id, url):
    cursor = conn.execute(
        "INSERT INTO articles (source_id, url, title, full_text, content_hash, "
        "published_at, retrieved_at, previous_version_id) "
        "VALUES (?, ?, 'T', 'text', 'hash', NULL, ?, NULL)",
        (source_id, url, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_cluster(conn, category="AppSec"):
    cursor = conn.execute(
        "INSERT INTO story_clusters (category, summary, created_at) VALUES (?, NULL, ?)",
        (category, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_claim(
    conn, article_id, source_id, category, claim_text, story_cluster_id,
    corroboration_count=0, status="verified", days_ago=0,
):
    retrieved_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    cursor = conn.execute(
        "INSERT INTO claims (article_id, source_id, claim_text, supporting_excerpt, "
        "claim_type, category, sub_tags, status, llm_certainty, corroboration_count, "
        "story_cluster_id, retrieved_at, created_at) "
        "VALUES (?, ?, ?, 'excerpt', 'factual', ?, '[]', ?, 0.5, ?, ?, ?, ?)",
        (
            article_id, source_id, claim_text, category, status, corroboration_count,
            story_cluster_id, retrieved_at, retrieved_at,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def test_notable_stories_ranks_by_corroboration_count_descending(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    low = _insert_cluster(conn)
    high = _insert_cluster(conn)
    _insert_claim(
        conn, article_id, source_id, "AppSec", "Low corroboration", low, corroboration_count=1,
    )
    _insert_claim(
        conn, article_id, source_id, "AppSec", "High corroboration", high, corroboration_count=5,
    )

    stories = notable_stories(conn, since_days=7)

    assert [s.story_cluster_id for s in stories] == [high, low]
    conn.close()


def test_notable_stories_excludes_clusters_with_no_activity_in_the_window(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    cluster_id = _insert_cluster(conn)
    _insert_claim(
        conn, article_id, source_id, "AppSec", "Stale claim", cluster_id, days_ago=30,
    )

    stories = notable_stories(conn, since_days=7)

    assert stories == []
    conn.close()


def test_notable_stories_excludes_misinformation_claims(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    cluster_id = _insert_cluster(conn)
    _insert_claim(
        conn, article_id, source_id, "AppSec", "Debunked", cluster_id, status="misinformation",
    )

    stories = notable_stories(conn, since_days=7)

    assert stories == []
    conn.close()


def test_notable_stories_includes_the_most_recent_claim_text_as_a_sample(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    cluster_id = _insert_cluster(conn)
    _insert_claim(
        conn, article_id, source_id, "AppSec", "Older claim text", cluster_id, days_ago=2,
    )
    _insert_claim(
        conn, article_id, source_id, "AppSec", "Newest claim text", cluster_id, days_ago=0,
    )

    stories = notable_stories(conn, since_days=7)

    assert stories[0].sample_claim_text == "Newest claim text"
    conn.close()


def test_notable_stories_respects_the_limit(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    for i in range(3):
        cluster_id = _insert_cluster(conn)
        _insert_claim(
            conn, article_id, source_id, "AppSec", f"Claim {i}", cluster_id,
            corroboration_count=i,
        )

    stories = notable_stories(conn, since_days=7, limit=2)

    assert len(stories) == 2
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trends_notable.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tabs.trends.notable'`

- [ ] **Step 3: Write the implementation**

```python
# src/tabs/trends/notable.py
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class NotableStory:
    story_cluster_id: int
    category: str
    corroboration_count: int
    most_recent_retrieved_at: str
    sample_claim_text: str


def notable_stories(
    conn: sqlite3.Connection, since_days: int, limit: int = 10,
) -> list[NotableStory]:
    """Story clusters with at least one non-misinformation claim retrieved within the
    last `since_days` days, ranked by corroboration count then recency (SPEC §7).

    corroboration_count is read directly from claims.corroboration_count — maintained by
    Phase 2b's score/storage.py._join_story_cluster — rather than recomputed here, so
    there stays one existing definition of "how corroborated is this cluster," not two
    that could drift apart. Every non-misinformation member of a cluster shares the same
    corroboration_count value, so MAX() just reads it out of the grouped rows.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
    rows = conn.execute(
        """
        SELECT sc.id AS story_cluster_id, sc.category,
               MAX(c.corroboration_count) AS corroboration_count,
               MAX(c.retrieved_at) AS most_recent_retrieved_at
        FROM story_clusters sc
        JOIN claims c ON c.story_cluster_id = sc.id
        WHERE c.status != 'misinformation' AND c.retrieved_at >= ?
        GROUP BY sc.id
        ORDER BY corroboration_count DESC, most_recent_retrieved_at DESC
        LIMIT ?
        """,
        (cutoff, limit),
    ).fetchall()

    stories = []
    for row in rows:
        sample = conn.execute(
            "SELECT claim_text FROM claims "
            "WHERE story_cluster_id = ? AND status != 'misinformation' "
            "ORDER BY retrieved_at DESC LIMIT 1",
            (row["story_cluster_id"],),
        ).fetchone()
        stories.append(NotableStory(
            story_cluster_id=row["story_cluster_id"],
            category=row["category"],
            corroboration_count=row["corroboration_count"],
            most_recent_retrieved_at=row["most_recent_retrieved_at"],
            sample_claim_text=sample["claim_text"],
        ))
    return stories
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trends_notable.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tabs/trends/notable.py tests/test_trends_notable.py
git commit -m "feat: add notable-story ranking by corroboration count and recency"
```

---

### Task 4: `tabs trends` CLI command

**Files:**
- Create: `src/tabs/commands/trends_cmd.py`
- Test: `tests/test_trends_cmd.py`
- Modify: `src/tabs/cli.py`
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `notable_stories`/`NotableStory` (Task 3); `detect_spikes`/`Spike` (Task 2).
- Produces: `parse_since(value: str) -> int` (raises `click.BadParameter` on a malformed value); the `trends` Click command, registered on `main` in `cli.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trends_cmd.py
from datetime import datetime, timezone

import click
import pytest
from click.testing import CliRunner

from tabs.cli import main
from tabs.commands.trends_cmd import parse_since
from tabs.db import get_connection, init_db


def _insert_source(conn, name="source"):
    cursor = conn.execute(
        "INSERT INTO sources (name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES (?, ?, 'AppSec', 2, 2)",
        (name, f"https://{name}.example/feed"),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_article(conn, source_id, url):
    cursor = conn.execute(
        "INSERT INTO articles (source_id, url, title, full_text, content_hash, "
        "published_at, retrieved_at, previous_version_id) "
        "VALUES (?, ?, 'T', 'text', 'hash', NULL, ?, NULL)",
        (source_id, url, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_cluster(conn, category="AppSec"):
    cursor = conn.execute(
        "INSERT INTO story_clusters (category, summary, created_at) VALUES (?, NULL, ?)",
        (category, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_claim(conn, article_id, source_id, category, claim_text, story_cluster_id, corroboration_count):
    retrieved_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO claims (article_id, source_id, claim_text, supporting_excerpt, "
        "claim_type, category, sub_tags, status, llm_certainty, corroboration_count, "
        "story_cluster_id, retrieved_at, created_at) "
        "VALUES (?, ?, ?, 'excerpt', 'factual', ?, '[]', 'verified', 0.5, ?, ?, ?, ?)",
        (
            article_id, source_id, claim_text, category, corroboration_count,
            story_cluster_id, retrieved_at, retrieved_at,
        ),
    )
    conn.commit()


def test_parse_since_parses_a_day_count():
    assert parse_since("30d") == 30


def test_parse_since_rejects_a_malformed_value():
    with pytest.raises(click.BadParameter):
        parse_since("banana")


def test_trends_command_reports_notable_stories_and_spikes(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    init_db(conn)
    source_id = _insert_source(conn)
    article_id = _insert_article(conn, source_id, "https://source.example/a")
    cluster_id = _insert_cluster(conn)
    _insert_claim(
        conn, article_id, source_id, "AppSec", "A notable claim", cluster_id,
        corroboration_count=3,
    )
    conn.close()

    result = CliRunner().invoke(main, ["--db-path", str(db_path), "trends"])

    assert result.exit_code == 0, result.output
    assert "Notable Stories" in result.output
    assert "A notable claim" in result.output
    assert "Trending Topics" in result.output


def test_trends_command_reports_no_activity_cleanly(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    init_db(conn)
    conn.close()

    result = CliRunner().invoke(main, ["--db-path", str(db_path), "trends"])

    assert result.exit_code == 0, result.output
    assert "(none)" in result.output


def test_trends_command_accepts_a_since_option(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    init_db(conn)
    conn.close()

    result = CliRunner().invoke(main, ["--db-path", str(db_path), "trends", "--since", "30d"])

    assert result.exit_code == 0, result.output


def test_trends_command_rejects_a_malformed_since_value_cleanly(tmp_path):
    result = CliRunner().invoke(
        main, ["--db-path", str(tmp_path / "test.db"), "trends", "--since", "banana"],
    )

    assert result.exit_code != 0
    assert "Traceback" not in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trends_cmd.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tabs.commands.trends_cmd'`

- [ ] **Step 3: Write the implementation**

Create `src/tabs/commands/trends_cmd.py`:

```python
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
```

Modify `src/tabs/cli.py` — add the import and registration:

```python
from tabs.commands.ingest_cmd import ingest_cmd
from tabs.commands.sources_cmd import sources_cmd
from tabs.commands.trends_cmd import trends_cmd
```

```python
main.add_command(ingest_cmd)
main.add_command(sources_cmd)
main.add_command(trends_cmd)
```

Modify `README.md` — add a `tabs trends` bullet to the existing command list (after the `tabs ingest` bullet, before the "Both commands accept..." line):

```markdown
- `tabs sources` — list allowlisted sources with their effective tier and health
  (consecutive failures, last successful fetch).
- `tabs ingest [--sources-path PATH]` — sync `sources.yaml` into the database,
  then fetch and store new articles from every source. `--sources-path`
  defaults to `sources.yaml` in the current directory.
- `tabs trends [--since 7d]` — show notable stories (story clusters ranked by
  corroboration count and recency) and category/sub-tag volume spikes for the
  window. Computed on demand from existing data — no extra API calls or cost.
- Both commands accept a global `--db-path PATH` before the subcommand
```

(Change "Both commands" to "All three commands" in that same line, since there are now three.)

Modify `CLAUDE.md` — in the `## Architecture` bullet list, add a new bullet after the `src/tabs/score/` bullet and before `src/tabs/commands/`:

```markdown
- **`src/tabs/trends/`** — the Phase 3 trend/notable-story layer, pure SQL aggregation over existing data (no LLM calls, no new tables). **`volume.py`** — `category_volume()`/`sub_tag_volume()`, `GROUP BY` counts of claims (excluding `misinformation`) + perspectives over a `[start, end)` window, expanding the `sub_tags` JSON array via SQLite's `json_each`. **`spikes.py`** — `detect_spikes()`, comparing a current window's volume against the average of the `BASELINE_MULTIPLIER` windows immediately preceding it (generalizing SPEC §7's "current week vs. trailing 4-week average" example to any `--since` value). **`notable.py`** — `notable_stories()`, ranking `story_clusters` by the `corroboration_count` Phase 2b's `score/storage.py` already maintains, not a value recomputed here.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trends_cmd.py -v`
Expected: PASS (6 tests)

Then run the full suite to confirm nothing else broke:

Run: `pytest -v`
Expected: PASS (every prior test plus this plan's new ones)

- [ ] **Step 5: Commit**

```bash
git add src/tabs/commands/trends_cmd.py tests/test_trends_cmd.py src/tabs/cli.py README.md CLAUDE.md
git commit -m "feat: add the tabs trends CLI command"
```
