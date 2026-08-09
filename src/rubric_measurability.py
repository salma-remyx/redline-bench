#!/usr/bin/env python3
"""Bayesian rubric-measurability diagnostic for the judge panel.

Adapted from *CalibratedRubric: Task-Adaptive Rubric Banks for
Open-Ended LLM Evaluation* (arXiv:2607.29252) — the paper's
Beta-Bernoulli agreement posterior, which scores how *measurable* each
rubric is, i.e. how reliably the judge panel reaches a stable verdict
on it. This is a more principled, uncertainty-aware successor to
``panel``'s pairwise ``judge_agreement`` /
``ranking_stable_across_judges`` heuristic: low-measurability rubrics
are precisely where panel consensus fails, so flagging them tells the
team which attorney-authored rubrics to revisit — directly serving the
stated goal of refining multi-judge consensus for nuanced legal
reasoning.

What is ported (Mode 2 — adapted):
  * The Beta-Bernoulli agreement posterior and the per-rubric
    measurability score, applied to the panel's pooled per-rubric
    verdict matrix.

What is intentionally scoped out:
  * IRT-based bank assembly — RedlineBench's rubrics are
    attorney-authored / fixed per task, not selected from a candidate
    pool, so there is no bank to greedily assemble.
  * Task-label perturbation study and the separate JudgmentBench-style
    eval framework — evaluation belongs downstream.

The posterior. Each judge verdict is a Bernoulli draw on the rubric's
latent pass-rate ``p``. With a uniform ``Beta(1, 1)`` prior and
``n_pass`` PASS / ``n_fail`` FAIL observations the posterior is
``Beta(1 + n_pass, 1 + n_fail)``. Measurability is the posterior
expected pairwise agreement

    m = E[p^2 + (1 - p)^2] = (a(a+1) + b(b+1)) / ((a+b)(a+b+1))

which is ~1 when the posterior mass sits at 0 or 1 (judges concur) and
dips toward 0.5 when it sits near a coin flip. It rises with judge
count: a rubric confirmed PASS by 30 pooled votes scores higher than
one graded PASS by 3, because the posterior tightens — exactly the
uncertainty awareness a pointwise agreement rate lacks. Each rubric is
pooled across every judge, model and task it appears in (matching the
paper's pooling across response blocks), so the score discriminates
even though RedlineBench runs only three judges per grade.
"""

from __future__ import annotations

# Uniform, non-informative prior. The panel's odd judge count means a
# single split vote already carries signal; the prior just keeps the
# posterior finite for rubrics seen only once.
_PRIOR_ALPHA = 1.0
_PRIOR_BETA = 1.0

# A rubric is "measurable" when the panel reproduces its majority more
# reliably than a coin flip. 0.5 is chance for an agreement
# probability, so the default sits just above it; tunable via the
# panel CLI's --measurability-threshold.
DEFAULT_MEASURABILITY_THRESHOLD = 0.6


def measurability_of(n_pass: int, n_fail: int) -> dict:
    """Beta-Bernoulli agreement posterior for a single rubric.

    Returns the posterior parameters plus a scalar ``measurability``
    score (posterior expected pairwise agreement, in ``[0.5, 1]``) and
    the majority verdict among the observed judges.
    """
    n_pass = max(0, int(n_pass))
    n_fail = max(0, int(n_fail))
    a = _PRIOR_ALPHA + n_pass
    b = _PRIOR_BETA + n_fail
    n = a + b
    score = (a * (a + 1) + b * (b + 1)) / (n * (n + 1))
    if n_pass > n_fail:
        majority = "PASS"
    elif n_fail > n_pass:
        majority = "FAIL"
    else:
        majority = "TIE"
    return {
        "alpha": a,
        "beta": b,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_judges": n_pass + n_fail,
        "majority": majority,
        "posterior_mean": round(a / n, 4),
        "measurability": round(score, 4),
    }


def pooled_measurability(
    votes_by_rubric: dict,
    threshold: float = DEFAULT_MEASURABILITY_THRESHOLD,
) -> tuple[dict, dict]:
    """Per-rubric measurability plus an aggregate summary over the panel.

    ``votes_by_rubric``: ``{rubric_id: (n_pass, n_fail)}`` — typically
    each rubric pooled across every judge / model / task it appears in
    (the shape ``panel.main`` builds while running the majority vote).

    Returns ``(per_rubric, summary)``:

      * ``per_rubric`` maps ``rubric_id`` to the posterior dict from
        :func:`measurability_of`, with a ``measurable`` bool added.
      * ``summary`` carries the rubric counts, mean measurability, the
        threshold used, and the sorted list of low-measurability
        rubrics the paper says to filter — the actionable surface that
        the old pairwise agreement rate could not produce.
    """
    per_rubric: dict[str, dict] = {}
    for rid, votes in votes_by_rubric.items():
        n_pass, n_fail = votes
        info = measurability_of(n_pass, n_fail)
        info["measurable"] = info["measurability"] >= threshold
        per_rubric[rid] = info

    n_total = len(per_rubric)
    low = sorted(rid for rid, i in per_rubric.items() if not i["measurable"])
    summary = {
        "n_rubrics": n_total,
        "n_measurable": n_total - len(low),
        "n_low_measurability": len(low),
        "mean_measurability": (
            round(sum(i["measurability"] for i in per_rubric.values()) / n_total, 4)
            if n_total
            else None
        ),
        "measurability_threshold": threshold,
        "low_measurability_rubrics": low,
    }
    return per_rubric, summary
