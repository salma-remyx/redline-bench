#!/usr/bin/env python3
"""Reliability-gated consensus for the RedlineBench judge panel.

Adapted (Mode 2) from *Project Kaleidoscope: Contextual, Human-Aligned
Evaluation for Real-World AI Applications* (arXiv:2607.14673v1).
Kaleidoscope's core mechanism is *reliability-gated automated scoring*: an
automated judge's score is trusted only when its agreement with reviewable
(human) labels meets a configured threshold, and cases below the threshold
are flagged for human review instead of being auto-scored.

RedlineBench has no human labels at panel-aggregation time, so the
human-label signal is substituted by a parameter-free proxy that
``panel.py`` already has the inputs for — each judge's mean pairwise
agreement with the rest of the panel (a self-referential stand-in for
"agreement with labels"; an optional ``--reference`` judge plays the
reviewable-label role upstream). The gating + flag-for-review *mechanism*
itself is kept at full fidelity:

  * Judges whose reliability falls below ``threshold`` are dropped from
    the consensus vote (only reliable judges contribute).
  * Rubrics left with no reliable judge are flagged for human review and
    fall back to strict majority, so the gated score stays defined.

The result is reported ALONGSIDE the official majority-vote score in
``panel_summary.json`` (key ``reliability_gated``) — the published panel
number is unchanged. Only the reliability-gating mechanism is ported;
Kaleidoscope's persona-based test generation, contextualized-rubric
authoring, and human-review UI live upstream of panel aggregation and are
intentionally out of scope here.
"""

from __future__ import annotations


def reliability_gated_consensus(
    rubric_sets_per_judge: list[dict[str, tuple[str, int, str | None]]],
    judge_labels: list[str],
    reliabilities: dict[str, float],
    threshold: float,
) -> tuple[dict[str, str], dict[str, int], int]:
    """Reduce N judges' per-rubric verdicts by reliability-weighted vote.

    Kaleidoscope's "only reliable judges contribute": for each rubric,
    only judges with ``reliabilities[label] >= threshold`` take part; their
    votes are weighted by reliability, and the side (PASS/FAIL) with the
    larger total reliability wins (ties resolve to FAIL, matching
    ``panel.majority_vote_per_rubric``). Rubrics with no reliable judge are
    *flagged for human review*: they fall back to strict majority among
    ALL voters so the gated score stays defined, and are counted in the
    returned ``n_flagged``.

    Args mirror ``panel.majority_vote_per_rubric``'s first argument, plus
    the per-judge reliability map and the gate threshold. Returns
    ``(panel_verdicts, weights, n_flagged)`` where ``weights`` carries the
    (judge-invariant) rubric weight, identical to the majority-vote path.
    """
    all_rids: set[str] = set().union(
        *[set(rs.keys()) for rs in rubric_sets_per_judge]
    ) if rubric_sets_per_judge else set()
    panel_verdicts: dict[str, str] = {}
    weights: dict[str, int] = {}
    n_flagged = 0
    for rid in all_rids:
        voters = [
            (judge_labels[i], rs[rid])
            for i, rs in enumerate(rubric_sets_per_judge)
            if rid in rs
        ]
        weights[rid] = voters[0][1][1] if voters else 0
        reliable = [
            (lbl, row[0]) for lbl, row in voters
            if reliabilities.get(lbl, 0.0) >= threshold
        ]
        if not reliable:
            # No judge reliable enough -> flag for human review; fall back
            # to strict majority so the gated score stays defined.
            n_flagged += 1
            votes = [row[0] for _, row in voters]
            n_pass = sum(1 for v in votes if v == "PASS")
            panel_verdicts[rid] = "PASS" if n_pass * 2 > len(votes) else "FAIL"
            continue
        pass_w = sum(reliabilities[lbl] for lbl, v in reliable if v == "PASS")
        fail_w = sum(reliabilities[lbl] for lbl, v in reliable if v == "FAIL")
        panel_verdicts[rid] = "PASS" if pass_w > fail_w else "FAIL"
    return panel_verdicts, weights, n_flagged


def reliability_report(
    judge_labels: list[str],
    reliabilities: dict[str, float],
    threshold: float,
    total_flagged: int,
    n_rubric_sets: int,
) -> dict:
    """Build the reliability report written alongside the official score.

    ``total_flagged`` / ``n_rubric_sets`` aggregate the per-task flag
    counts across the whole panel run (rubrics with no reliable judge ->
    flagged for human review, per Kaleidoscope).
    """
    per_judge = dict(sorted(reliabilities.items(), key=lambda kv: -kv[1]))
    gated = [
        lbl for lbl in judge_labels
        if reliabilities.get(lbl, 0.0) < threshold
    ]
    return {
        "threshold": threshold,
        "per_judge_reliability": per_judge,
        "gated_judges": gated,
        "n_rubrics_flagged_for_review": total_flagged,
        "flag_rate": round(total_flagged / n_rubric_sets, 4) if n_rubric_sets else 0.0,
    }
