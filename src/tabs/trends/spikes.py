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
