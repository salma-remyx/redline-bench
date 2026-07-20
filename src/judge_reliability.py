#!/usr/bin/env python3
"""Measurement-validity audit for the judge panel.

``panel.py`` already reports per-judge sensitivity (does the model ranking
depend on the judge?) and pairwise rubric-level agreement. This module adds
the *measurement-validity* probes called for by "When the Judge Changes, So
Does the Measurement: Auditing LLM-as-Judge Reliability" (arXiv:2607.08535),
whose central point is that an LLM-as-judge report is incomplete unless it
says how much a score moves when the evaluator is swapped, where the
judge-sensitive rubric slices live, and whether the panel's judges fail
independently or redundantly.

All three probes run over the per-judge grade trees that ``panel.main()``
already loads (each judge graded the SAME fixed candidate outputs), so they
need no extra LLM calls:

  1. **Score drift** -- for each judge pair, the distribution of
     per-(model, task) weighted-score differences. This is the paper's
     "the score can move even when the candidate stays fixed" signal,
     quantified: a candidate's standing should not hinge on which judge
     happened to grade it.
  2. **Rubric flips by category** -- the rubric-level disagreement rate
     between each judge pair, stratified by rubric category. These are the
     "dataset slices" the paper asks reports to carry: a high per-category
     flip rate marks a slice of the rubric set whose verdicts depend on the
     judge rather than the candidate.
  3. **Error dependence** -- pairwise correlation of each judge's
     per-rubric disagreements, computed leave-one-out against the rest of
     the panel. High correlation means the judges stumble on the same
     candidates, so adding more judges to the jury buys little independent
     signal -- the paper's "repeated-sample juries add little when errors
     are correlated" finding, as a parameter-free proxy.

Adapted port (Mode 2). Two of the paper's bias probes are intentionally out
of scope here: *position* bias needs pairwise comparison with the candidate
order swapped (RedlineBench grades each rubric PASS/FAIL, not A-vs-B), and
*verbosity* bias needs the candidate output length, which the panel
substrate -- stored verdicts and weights -- does not carry. The paper's
cross-model scaling study (Qwen3 1.7B->32B, MiniMax M2->M2.7) is a
benchmark-suite comparison, not a method, and belongs to a downstream eval;
what is kept at full fidelity is the measurement-validity audit itself,
computed over whichever judges the panel is configured with. The paper's
error-dependence estimator is replaced by the leave-one-out correlation
proxy above; the repo's existing ``majority_vote_per_rubric`` is reused as
the single source of truth for the vote semantics.
"""

from __future__ import annotations

from itertools import combinations
from statistics import mean

from panel import majority_vote_per_rubric


def _rubric_rows(grade: dict) -> dict:
    """rubric_id -> (verdict, category) for a single grade.

    Mirrors the row shape ``panel._rubric_rows`` produces, dropping the
    weight (which the audit does not need). Kept local rather than importing
    panel's private ``_rubric_rows`` so the audit does not reach into
    panel's private surface.
    """
    return {
        p["rubric_id"]: (p["verdict"], p.get("category"))
        for p in grade.get("score", {}).get("per_rubric", [])
    }


def _weighted_of(grade: dict) -> float:
    """A judge's stored standalone weighted score for one (model, task)."""
    return float(grade.get("score", {}).get("weighted", 0.0) or 0.0)


def _score_drift(judges: dict, common: set) -> dict:
    """Per judge-pair distribution of per-(model, task) weighted-score moves.

    Both judges graded the same fixed output, so any score difference is
    pure evaluator-replacement ambiguity -- exactly the movement the paper
    warns can be misread as a change in candidate quality.
    """
    out: dict = {}
    for a, b in combinations(judges, 2):
        diffs = [
            _weighted_of(judges[b][(m, t)]) - _weighted_of(judges[a][(m, t)])
            for (m, t) in common
        ]
        if not diffs:
            continue
        abs_diffs = [abs(d) for d in diffs]
        out[f"{a} vs {b}"] = {
            # Mean absolute movement of a candidate's score when the
            # evaluator is swapped (the headline drift figure).
            "mean_abs": round(mean(abs_diffs), 4),
            # Mean signed movement (b - a): >0 means b grades more generously.
            "mean_signed": round(mean(diffs), 4),
            "max_abs": round(max(abs_diffs), 4),
            "n": len(diffs),
        }
    return out


def _rubric_flips(judges: dict, common: set) -> dict:
    """Per judge-pair rubric disagreement rate, stratified by category.

    The category breakdown gives the "dataset slices" the paper asks for:
    which slices of the rubric set are judge-sensitive rather than
    candidate-sensitive.
    """
    out: dict = {}
    for a, b in combinations(judges, 2):
        cat_total: dict = {}
        cat_flip: dict = {}
        for (m, t) in common:
            ra = _rubric_rows(judges[a][(m, t)])
            rb = _rubric_rows(judges[b][(m, t)])
            for rid in set(ra) & set(rb):
                cat = ra[rid][1]
                cat_total[cat] = cat_total.get(cat, 0) + 1
                if ra[rid][0] != rb[rid][0]:
                    cat_flip[cat] = cat_flip.get(cat, 0) + 1
        total = sum(cat_total.values())
        if total == 0:
            continue
        out[f"{a} vs {b}"] = {
            "overall": round(sum(cat_flip.values()) / total, 4),
            "by_category": {
                str(c): round(cat_flip.get(c, 0) / n, 4)
                for c, n in sorted(cat_total.items(), key=lambda kv: str(kv[0]))
            },
            "n": total,
        }
    return out


def _pearson(xs: list, ys: list) -> float | None:
    """Pearson correlation over two equal-length 0/1 vectors.

    Returns ``None`` when undefined (empty input or a constant vector, the
    latter meaning one judge agreed with the leave-one-out reference on
    every rubric and so carries no error-variance to correlate).
    """
    n = len(xs)
    if n == 0:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / ((vx ** 0.5) * (vy ** 0.5))


def _error_dependence(judges: dict, common: set) -> dict:
    """Pairwise correlation of judges' leave-one-out disagreements.

    For each judge we form a 0/1 "error" indicator over the
    rubric x (model, task) grid: 1 where the judge's verdict disagrees with
    the strict majority of the *other* judges (leave-one-out, so that being
    in the panel majority does not exempt a judge from the probe). Pairwise
    correlation of these vectors estimates whether the judges fail on the
    same candidates -- high correlation means repeated-sample juries add
    little independent signal, the paper's "errors are correlated" finding.
    """
    labels = list(judges)
    errors: dict = {lbl: {} for lbl in labels}
    for (m, t) in common:
        rows_by_label = {lbl: _rubric_rows(judges[lbl][(m, t)]) for lbl in labels}
        for lbl in labels:
            # Leave-one-out reference: the panel majority excluding this judge,
            # computed with the repo's shared vote so the semantics match
            # `panel.main()` exactly.
            others = [rows_by_label[j2] for j2 in labels if j2 != lbl]
            ref, _weights = majority_vote_per_rubric(others)
            for rid, (v, _cat) in rows_by_label[lbl].items():
                if rid in ref:
                    errors[lbl][(m, t, rid)] = 1 if v != ref[rid] else 0

    matrix: dict = {}
    corrs: list = []
    for a, b in combinations(labels, 2):
        keys = set(errors[a]) & set(errors[b])
        if not keys:
            continue
        r = _pearson(
            [errors[a][k] for k in keys],
            [errors[b][k] for k in keys],
        )
        if r is None:
            continue
        matrix[f"{a} vs {b}"] = round(r, 4)
        corrs.append(r)
    return {
        "matrix": matrix,
        "mean": round(mean(corrs), 4) if corrs else None,
    }


def audit_panel(judges: dict, common: set) -> dict:
    """Compute the measurement-validity audit over a loaded judge panel.

    ``judges`` is ``{label: {(model, task): grade}}`` -- exactly the
    structure ``panel.main()`` builds via ``panel.load_judge``. The audit
    never raises on a degenerate panel (fewer than two judges or no shared
    rubrics): it returns the probes it can and leaves the rest empty, so
    wiring it into ``panel.main()`` cannot break a run.
    """
    n_judges = len(judges)
    if n_judges >= 2:
        score_drift = _score_drift(judges, common)
        rubric_flips = _rubric_flips(judges, common)
        error_dependence = (
            _error_dependence(judges, common) if common else {"matrix": {}, "mean": None}
        )
    else:
        score_drift, rubric_flips, error_dependence = {}, {}, {"matrix": {}, "mean": None}
    return {
        "n_judges": n_judges,
        "n_pairs": len(common),
        "score_drift": score_drift,
        "rubric_flips": rubric_flips,
        "error_dependence": error_dependence,
    }


def format_audit(audit: dict) -> str:
    """One-line human summary of :func:`audit_panel`, for the run log."""
    drift = audit.get("score_drift", {}) or {}
    if drift:
        mean_abs = round(mean(v["mean_abs"] for v in drift.values()), 4)
        drift_txt = f"mean |score drift| {mean_abs}"
    else:
        drift_txt = "no judge pairs"
    ed = audit.get("error_dependence", {}) or {}
    ed_mean = ed.get("mean")
    ed_txt = f"error-dep corr {ed_mean}" if ed_mean is not None else "error-dep n/a"
    return f"judge reliability: {drift_txt}, {ed_txt}"
