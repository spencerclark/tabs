import json
import sqlite3
from datetime import datetime, timezone

from tabs.score.conflicts import ConflictCandidate, resolve_conflict
from tabs.score.embeddings import embed_text
from tabs.score.judging import judge_candidate_matches
from tabs.score.matching import find_candidate_claims
from tabs.score.scoring import compute_confidence_score, gate_status


class ScoringResult:
    """What happened when scoring/corroborating one claim, for the caller to log."""

    def __init__(self, status: str, embedding_failed: bool, judgment_failed: bool):
        self.status = status
        self.embedding_failed = embedding_failed
        self.judgment_failed = judgment_failed


def score_and_corroborate_claim(
    conn: sqlite3.Connection, client, voyage_client, claim_id: int,
) -> ScoringResult:
    """Embed, corroborate/conflict-match, score, and gate one newly-extracted claim.

    Every step degrades gracefully rather than raising: corroboration/conflict matching
    is a reinforcing signal on top of a claim's own tier/certainty/type (SPEC §6.4), not
    a precondition for the claim being usable at all, so a Voyage or judgment failure
    just means this claim is scored without that signal this round — reflected in the
    returned ScoringResult flags for the caller to log, not as an exception. Any other
    failure (a genuine bug, a DB error) is allowed to propagate — the caller's own
    per-claim guard is the safety net for those, not this function's job to swallow.
    """
    claim = conn.execute(
        "SELECT article_id, category, claim_type, claim_text, llm_certainty, "
        "source_id, retrieved_at, corroboration_count "
        "FROM claims WHERE id = ?",
        (claim_id,),
    ).fetchone()

    embedding_failed = False
    embedding = None
    try:
        embedding = embed_text(voyage_client, claim["claim_text"])
    except Exception:  # noqa: BLE001 — one bad claim's embedding must not kill the run
        embedding_failed = True

    if embedding is not None:
        conn.execute(
            "UPDATE claims SET embedding = ? WHERE id = ?", (json.dumps(embedding), claim_id),
        )
        conn.commit()

    judgment_failed = False
    if embedding is not None:
        candidates = find_candidate_claims(
            conn, claim_id=claim_id, article_id=claim["article_id"],
            source_id=claim["source_id"], category=claim["category"], embedding=embedding,
        )
        if candidates:
            source_row = conn.execute(
                "SELECT institutional_tier, earned_tier FROM sources WHERE id = ?",
                (claim["source_id"],),
            ).fetchone()
            own_tier = max(source_row["institutional_tier"], source_row["earned_tier"])

            judgments = None
            try:
                judgments = judge_candidate_matches(client, claim["claim_text"], candidates)
            except Exception:  # noqa: BLE001 — one bad claim's judgment must not kill the run
                judgment_failed = True

            if judgments is None:
                judgment_failed = True
            else:
                by_id = {c.claim_id: c for c in candidates}
                corroborating = []
                for judgment in judgments.judgments:
                    candidate = by_id.get(judgment.candidate_claim_id)
                    if candidate is None:
                        continue  # model echoed an id we never offered — ignore, don't guess
                    if judgment.relationship == "corroborating":
                        corroborating.append(candidate)
                    elif judgment.relationship == "conflicting":
                        _record_conflict(
                            conn, claim_id, own_tier, claim["retrieved_at"],
                            claim["corroboration_count"], candidate,
                        )

                if corroborating:
                    _join_story_cluster(conn, claim_id, claim["category"], corroborating)

    status = _rescore_claim(conn, claim_id)
    return ScoringResult(status=status, embedding_failed=embedding_failed, judgment_failed=judgment_failed)


def _record_conflict(conn, new_claim_id, new_claim_tier, new_retrieved_at, new_corroboration_count, candidate):
    """Create a conflicts record between the new claim and a candidate judged conflicting.

    If the cascade resolves via a tier difference, immediately marks the losing claim as
    misinformation (sticky — SPEC §6.4: "regardless of the contradicted claim's own
    score"). A recency- or corroboration-count-based resolution still records
    winning_claim_id on the conflicts row, but does not touch either claim's status.
    """
    a = ConflictCandidate(new_claim_id, new_claim_tier, new_retrieved_at, new_corroboration_count)
    b = ConflictCandidate(
        candidate.claim_id, candidate.effective_tier, candidate.retrieved_at,
        candidate.corroboration_count,
    )
    resolution, winning_claim_id, misinformation_claim_id = resolve_conflict(a, b)

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO conflicts (claim_a_id, claim_b_id, resolution, winning_claim_id, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (new_claim_id, candidate.claim_id, resolution, winning_claim_id, now),
    )
    if misinformation_claim_id is not None:
        conn.execute(
            "UPDATE claims SET status = 'misinformation' WHERE id = ?", (misinformation_claim_id,),
        )
    conn.commit()


def _join_story_cluster(conn, claim_id, category, corroborating_candidates):
    """Join claim_id to the story cluster of its best corroborating match, creating a new
    cluster if none of the matches already has one, then recompute and re-gate every
    member's corroboration_count/status so the cluster stays internally consistent.

    Every corroborating candidate that doesn't already belong to a cluster joins this one
    too (not just the single best match) — the model judged all of them as describing the
    same event, so all of them should count toward the corroboration signal.

    If multiple corroborating candidates exist in DIFFERENT existing clusters, this joins
    only the one belonging to the best (highest-similarity) match's cluster — merging two
    separate existing clusters together is not attempted; a known simplification for this
    phase (see the plan's Deferred Scope).
    """
    corroborating_candidates = sorted(
        corroborating_candidates, key=lambda c: c.similarity, reverse=True,
    )
    existing_cluster_id = next(
        (c.story_cluster_id for c in corroborating_candidates if c.story_cluster_id is not None),
        None,
    )
    if existing_cluster_id is None:
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            "INSERT INTO story_clusters (category, summary, created_at) VALUES (?, NULL, ?)",
            (category, now),
        )
        cluster_id = cursor.lastrowid
    else:
        cluster_id = existing_cluster_id

    unclustered_ids = [
        c.claim_id for c in corroborating_candidates if c.story_cluster_id is None
    ]
    if unclustered_ids:
        placeholders = ",".join("?" for _ in unclustered_ids)
        conn.execute(
            f"UPDATE claims SET story_cluster_id = ? WHERE id IN ({placeholders})",
            (cluster_id, *unclustered_ids),
        )

    conn.execute("UPDATE claims SET story_cluster_id = ? WHERE id = ?", (cluster_id, claim_id))
    conn.commit()

    member_ids = [
        row["id"] for row in
        conn.execute("SELECT id FROM claims WHERE story_cluster_id = ?", (cluster_id,)).fetchall()
    ]
    corroboration_count = len(member_ids) - 1
    conn.execute(
        "UPDATE claims SET corroboration_count = ? WHERE story_cluster_id = ?",
        (corroboration_count, cluster_id),
    )
    conn.commit()

    for member_id in member_ids:
        if member_id != claim_id:  # claim_id is rescored once, by the caller, after this returns
            _rescore_claim(conn, member_id)


def _rescore_claim(conn, claim_id):
    """(Re)compute confidence_score and status for one claim from its current DB state.

    A claim already marked misinformation is left untouched — that status is a sticky,
    tier-based override (SPEC §6.4: "regardless of the contradicted claim's own score"),
    not something corroboration should ever revise back to verified/unverified.
    """
    row = conn.execute(
        "SELECT c.claim_type, c.llm_certainty, c.corroboration_count, c.status, "
        "s.institutional_tier, s.earned_tier "
        "FROM claims c JOIN sources s ON s.id = c.source_id WHERE c.id = ?",
        (claim_id,),
    ).fetchone()
    if row["status"] == "misinformation":
        return "misinformation"

    score = compute_confidence_score(
        effective_tier=max(row["institutional_tier"], row["earned_tier"]),
        corroboration_count=row["corroboration_count"],
        llm_certainty=row["llm_certainty"],
        claim_type=row["claim_type"],
    )
    status = gate_status(score, contradicted_by_higher_tier=False)
    conn.execute(
        "UPDATE claims SET confidence_score = ?, status = ? WHERE id = ?",
        (score, status, claim_id),
    )
    conn.commit()
    return status
