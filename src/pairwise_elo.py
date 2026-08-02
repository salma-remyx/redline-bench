#!/usr/bin/env python3
"""Pairwise-comparison Elo aggregation for the judge panel.

Alongside the pointwise panel score (each judge grades every model
independently, then the panel majority-votes per rubric), this module
offers the *pairwise* aggregation path from "(Towards) Scalable Reliable
Automated Evaluation with Large Language Models" (arXiv:2607.28282):

  1. Multi-judge *pairwise* comparisons — for each task, each judge casts
     a head-to-head vote between two model outputs, which damps any one
     judge's bias (the panel already fields several judges).
  2. An *adjustable agreement threshold* (unanimity -> majority) decides
     whether a comparison is confident enough to count; below it the
     comparison abstains, trading coverage for confidence — exactly the
     knob the paper exposes.
  3. An *Elo rating* over the counted comparisons yields a stable,
     interpretable ranking.

ADAPTATION (Mode 2). The paper collects each judge's pairwise preference
by asking the LLM "which output is better?" directly. This repo has no
pairwise-judge call path, so we *derive* each judge's pairwise preference
from that judge's existing per-task pointwise weighted score
(``grade["score"]["weighted"]``), which ``panel`` already computes. Same
signal — a judge's head-to-head preference between two outputs on a task
— zero extra LLM calls. The paper's separate competency-profile benchmark
suite is cut; this module feeds the panel's own summary artifact.
"""

from __future__ import annotations

DEFAULT_K = 32.0
DEFAULT_BASE = 1500.0
DEFAULT_SCALE = 400.0
DEFAULT_THRESHOLD = 0.5  # strict majority; 1.0 = full unanimity


def expected_score(rating_a: float, rating_b: float, *, scale: float = DEFAULT_SCALE) -> float:
    """Standard Elo expected score for A vs B (in [0, 1])."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def judge_pairwise_vote(score_a: float, score_b: float, *, eps: float = 1e-9) -> str:
    """One judge's head-to-head preference from two pointwise scores.

    Returns ``"A"`` / ``"B"`` / ``"TIE"`` (scores within ``eps`` tie).
    """
    if score_a > score_b + eps:
        return "A"
    if score_b > score_a + eps:
        return "B"
    return "TIE"


def aggregate_votes(votes: list[str], agreement_threshold: float) -> str:
    """Reduce one (task, model-pair)'s judge votes to a consensus.

    ``votes``: per-judge ``"A"`` / ``"B"`` / ``"TIE"``.
    ``agreement_threshold`` in (0, 1]: the fraction of judges that must
    agree on the SAME non-tie winner. ``1.0`` = unanimity, ``0.5`` =
    strict majority. A unanimously-all-tie panel returns ``"TIE"``;
    otherwise the plurality non-tie winner is counted only if its share
    clears the threshold, else the comparison ``"ABSTAIN"``s (no Elo
    update).
    """
    n = len(votes)
    if n == 0:
        return "ABSTAIN"
    na = sum(1 for v in votes if v == "A")
    nb = sum(1 for v in votes if v == "B")
    if na == 0 and nb == 0:
        return "TIE"
    winner, wcount = ("A", na) if na >= nb else ("B", nb)
    return winner if wcount / n >= agreement_threshold else "ABSTAIN"


def compute_elo(
    comparisons: list[tuple[str, str, str]],
    models: list[str],
    *,
    k: float = DEFAULT_K,
    base: float = DEFAULT_BASE,
    scale: float = DEFAULT_SCALE,
) -> dict[str, float]:
    """Elo ratings over ``comparisons`` = ``[(a, b, outcome), ...]`` with
    outcome ``"A"`` (a wins) / ``"B"`` (b wins) / ``"TIE"``.

    Each model starts at ``base``; a single deterministic pass — the
    caller sorts comparisons upstream, so output is reproducible.
    """
    ratings = {m: base for m in models}
    points = {"A": 1.0, "B": 0.0, "TIE": 0.5}
    for a, b, outcome in comparisons:
        if a not in ratings or b not in ratings:
            continue
        ea = expected_score(ratings[a], ratings[b], scale=scale)
        sa = points[outcome]
        ratings[a] += k * (sa - ea)
        ratings[b] += k * ((1.0 - sa) - (1.0 - ea))
    return ratings


def pairwise_comparisons(
    judges: dict[str, dict[tuple[str, str], dict]],
    common: set[tuple[str, str]],
    *,
    agreement_threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[tuple[str, str, str]], dict]:
    """Build consensus (task, model-pair) comparisons from the panel's
    per-judge per-task weighted scores.

    ``judges`` is ``panel.main()``'s ``judges`` dict (label ->
    (model, task) -> grade); ``common`` is the (model, task) keys every
    judge graded. Returns ``(comparisons, stats)``.
    """
    labels = sorted(judges)
    models_per_task: dict[str, list[str]] = {}
    for (_, task) in common:
        models_per_task.setdefault(task, set())
    for (model, task) in common:
        models_per_task[task].add(model)

    comparisons: list[tuple[str, str, str]] = []
    considered = abstained = unanimous = 0
    for task in sorted(models_per_task):
        models = sorted(models_per_task[task])
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                votes: list[str] = []
                for label in labels:
                    ga = judges[label].get((a, task))
                    gb = judges[label].get((b, task))
                    if ga is None or gb is None:
                        continue
                    wa = ga.get("score", {}).get("weighted", 0.0)
                    wb = gb.get("score", {}).get("weighted", 0.0)
                    votes.append(judge_pairwise_vote(wa, wb))
                outcome = aggregate_votes(votes, agreement_threshold)
                if outcome == "ABSTAIN":
                    abstained += 1
                    continue
                considered += 1
                # Full-judge consensus on a (non-tie) winner — the
                # paper's unanimity-confidence signal.
                if outcome != "TIE" and votes.count(outcome) == len(votes):
                    unanimous += 1
                comparisons.append((a, b, outcome))

    total = considered + abstained
    stats = {
        "n_judges": len(labels),
        "n_considered": considered,
        "n_abstained": abstained,
        "n_unanimous": unanimous,
        "coverage": round(considered / total, 4) if total else 0.0,
    }
    return comparisons, stats


def panel_pairwise_elo(
    judges: dict[str, dict[tuple[str, str], dict]],
    common: set[tuple[str, str]],
    *,
    agreement_threshold: float = DEFAULT_THRESHOLD,
    k: float = DEFAULT_K,
) -> dict:
    """End-to-end pairwise-Elo aggregation for ``panel.main()``.

    Returns a JSON-friendly dict written to ``panel_summary.json`` under
    ``"pairwise_elo"``: ``ratings`` (model -> Elo, in ranked order),
    ``ranking`` (models by Elo desc), the ``agreement_threshold`` used,
    coverage / abstention counts, and ``n_unanimous`` (comparisons where
    every judge agreed on a winner — the paper's unanimity signal).
    """
    comparisons, stats = pairwise_comparisons(
        judges, common, agreement_threshold=agreement_threshold
    )
    models = sorted({m for (m, _) in common})
    ratings = compute_elo(comparisons, models, k=k)
    ranking = sorted(ratings, key=lambda m: (-ratings[m], m))
    return {
        "ratings": {m: round(ratings[m], 1) for m in ranking},
        "ranking": ranking,
        "agreement_threshold": agreement_threshold,
        "n_judges": stats["n_judges"],
        "n_considered": stats["n_considered"],
        "n_abstained": stats["n_abstained"],
        "n_unanimous": stats["n_unanimous"],
        "coverage": stats["coverage"],
    }
