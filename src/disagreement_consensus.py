"""Disagreement-aware consensus diagnostics for the judge panel.

Adapted from *VERDICT: Training-Free Step-Wise Verification of
Multimodal Reasoning via Disagreement-Aware Consensus*
(arXiv:2608.10665). VERDICT treats the scores of several frozen
verifiers as a coupled scoring problem — a coordination game with a
closed-form equilibrium in which agreement signals validity while
disagreement reveals instability. That insight ports cleanly onto
RedlineBench's multi-judge panel: the per-rubric verdict maps that
`panel.majority_vote_per_rubric()` already consumes are exactly the
frozen-verifier inputs, and the panel's strict-majority vote is a
simple aggregation that throws away the disagreement structure.

Here the rubric plays the role of VERDICT's reasoning step, and each
judge's PASS/FAIL verdict is a binary verifier score in [0, 1]. The
consensus score is the mean verifier score s = n_pass / n_voters; the
game's unique equilibrium threshold is delta* = 0.5 (a step is
consensus-valid iff s > delta*, i.e. exactly the panel's strict
majority — so the vote itself is untouched); and the per-step
dispersion |s − delta*| measures how far the step sits from the
equilibrium, i.e. how unstable the majority verdict is: small
dispersion means the verdict flipped on one judge's opinion.

This is a diagnostic signal only — it never overrides the canonical
majority vote. Substituted for the paper's machinery: the multimodal
step-wise scoring / cross-modal verifiers become the repo's per-rubric
judge verdicts, and the learned / threshold-swept filtering becomes a
parameter-free dispersion readout that lands in `panel_summary.json`.
"""

from __future__ import annotations

from statistics import mean

# Equilibrium threshold of the coupled-scoring game: a step is
# consensus-valid iff its mean verifier score strictly exceeds 0.5,
# matching the panel's strict-majority vote.
DELTA_STAR = 0.5


def consensus_stats(
    rubric_sets_per_judge: list[dict[str, tuple[str, int, str | None]]],
) -> dict:
    """Disagreement-aware consensus statistics for one (model, task).

    Input: the same per-judge maps that `panel.majority_vote_per_rubric`
    consumes — each judge's rubric_id → (verdict, weight, category).

    Returns:
      ``verdict_pass_fraction``: fraction of graded rubrics whose
        consensus score clears delta* (i.e. the majority vote says PASS).
      ``mean_dispersion``: mean |s - delta*| over graded rubrics —
        VERDICT's instability signal. Low mean dispersion means the
        panel's verdicts sit near the equilibrium on average, so the
        majority vote is fragile: small judge perturbations flip it.
      ``unstable_rubrics``: rubric_ids whose |s - delta*| is minimal
        (verdict decided by a single vote), sorted for determinism.
      ``n_rubrics``: number of rubrics graded by at least one judge.
    """
    all_rids: set[str] = (
        set().union(*(set(rs) for rs in rubric_sets_per_judge))
        if rubric_sets_per_judge
        else set()
    )
    dispersions: list[float] = []
    unstable: list[str] = []
    n_pass_verdicts = 0
    for rid in sorted(all_rids):
        votes = [rs[rid][0] for rs in rubric_sets_per_judge if rid in rs]
        if not votes:
            continue
        s = sum(1 for v in votes if v == "PASS") / len(votes)
        if s > DELTA_STAR:
            n_pass_verdicts += 1
        dispersion = abs(s - DELTA_STAR)
        dispersions.append(dispersion)
        if dispersion <= 1.0 / (2 * len(votes)):
            unstable.append(rid)
    return {
        "verdict_pass_fraction": (
            round(n_pass_verdicts / len(dispersions), 4) if dispersions else None
        ),
        "mean_dispersion": (
            round(mean(dispersions), 4) if dispersions else None
        ),
        "unstable_rubrics": unstable,
        "n_rubrics": len(dispersions),
    }


def summarize_panel_consensus(
    grades_per_judge: dict[str, dict],
    rubric_rows,
) -> dict:
    """Aggregate per-task consensus stats over a judge panel.

    ``grades_per_judge``: judge label → (model, task) → grade dict —
    exactly `panel.load_judge`'s output. ``rubric_rows``: the
    grade-dict → per-rubric map extractor (`panel._rubric_rows`),
    injected so this module stays free of grade-format knowledge.

    Returns a dict keyed by model, each mapping to
    ``{"mean_dispersion": …, "verdict_pass_fraction": …}`` averaged over
    that model's tasks — VERDICT's stability-conscious read on where
    the panel leaderboard is trustworthy versus fragile.
    """
    if not grades_per_judge:
        return {}
    common = set.intersection(*(set(j) for j in grades_per_judge.values()))
    by_model: dict[str, list[dict]] = {}
    for model, _task in common:
        rubric_sets = [
            rubric_rows(g[(model, _task)]) for g in grades_per_judge.values()
        ]
        by_model.setdefault(model, []).append(consensus_stats(rubric_sets))
    out = {}
    for model, stats in by_model.items():
        dis = [s["mean_dispersion"] for s in stats if s["mean_dispersion"] is not None]
        pfs = [
            s["verdict_pass_fraction"]
            for s in stats
            if s["verdict_pass_fraction"] is not None
        ]
        out[model] = {
            "mean_dispersion": round(mean(dis), 4) if dis else None,
            "verdict_pass_fraction": round(mean(pfs), 4) if pfs else None,
        }
    return dict(sorted(out.items(), key=lambda kv: -(kv[1]["mean_dispersion"] or 0.0)))
