"""Diagnosis-oriented error taxonomy for RedlineBench rubric verdicts.

The headline metrics summarize a model's contract-redlining quality as a
single turn-weighted score plus per-dimension weighted pass-rates. This
module adds the *diagnosis-oriented* layer: it takes the per-rubric FAIL
verdicts already on disk (verdict + benchmark evaluation-dimension +
criteria text) and decomposes them into a two-level error profile per
model — (evaluation dimension → error type) — plus document-level
coverage. The point is to show *where* a model is weak (which kind of
rule it violates, and how often), not just how weak it is overall.

  - **dimension** (top level) — the benchmark's own five evaluation
    dimensions (Commercial context, Legal correctness, Negotiation
    quality, Deal-closing orientation, Counterparty-acceptance
    prediction), read from each rubric's ``category`` field.
  - **error_type** (leaf) — a parameter-free keyword classifier over the
    rubric's ``criteria`` text (the rule that was checked) mapping to a
    legal-concept failure mode. Penalty rubrics the model *triggered*
    (PASS on a negative-weight rubric — an edit the attorney flagged as
    undesirable) are diagnosed directly as the ``over_aggression`` error
    type, since the penalty itself is the diagnosis.

This is the diagnosis-oriented *evaluation* half of GB/T-Bench's
hierarchical review-error taxonomy + evaluation protocol, adapted to
legal redlining. Adapted from *Benchmarking and Enhancing LLMs for
Rule-Intensive Review of National Standard Documents*
(arXiv:2608.06312). The paper's GB/T-Reviewer multi-agent framework,
its counterexample-generation pipeline, and its national-standard-
specific 25-type schema are intentionally out of scope: RedlineBench
already has its multi-judge panel, rubric set, and on-disk verdicts, so
this consumes those verdicts rather than re-deriving them.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

# The benchmark's five canonical evaluation dimensions (see README's
# "Metrics" table). Rubric `category` values match one of these; anything
# else is kept verbatim rather than forced into a wrong bucket.
_DIMENSIONS: tuple[str, ...] = (
    "Commercial context",
    "Legal correctness",
    "Negotiation quality",
    "Deal-closing orientation",
    "Counterparty-acceptance prediction",
)

# Leaf error types — a parameter-free keyword classifier over the rubric's
# `criteria` text (the rule that was checked). First match wins (entries are
# ordered most-specific first). Each entry maps a legal-concept failure mode
# to the substrings that signal it. Rubrics whose criteria matches none of
# these bucket as `other`; that residual is itself informative (rubrics the
# taxonomy doesn't yet cover). This keyword proxy stands in for the paper's
# specialized error-type assignment, which is GB/T-standard-specific.
_ERROR_TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("definition", ("definition", "defined term")),
    ("cross_reference", ("cross-reference", "cross reference", "section reference")),
    ("liability_indemnity", ("liabilit", "indemnif", "damages", "limitation of liab")),
    ("termination", ("terminat", "expir", "renewal", "non-renew")),
    ("payment_pricing", ("payment", "pricing", "invoice", "price", "late fee")),
    ("confidentiality", ("confidential", "non-disclosure")),
    ("intellectual_property", ("intellectual property", "ownership", "work product", "license grant")),
    ("governing_law_dispute", ("governing law", "jurisdiction", "venue", "arbitration", "dispute resol")),
    ("warranty_rep", ("warranty", "warranties", "representation")),
    ("data_privacy", ("data protection", "personal data", "privacy", "gdpr")),
    ("scope_change", ("scope of", "change of control", "assignment", "subcontract")),
    ("risk_exposure", ("unlimited liabilit", "exposure", "tail risk", "downside")),
    ("wording_precision", ("ambiguous", "vague", "imprecise", "wording", "clarity")),
    ("structure_format", ("numbering", "heading", "section number", "bullet", "formatting")),
    ("consistency_conflict", ("consistent", "contradict", "conflict", "align with")),
)

_SOURCE = (
    "Diagnosis-oriented error taxonomy: dimension = rubric category "
    "(benchmark evaluation dimension); error_type = criteria keyword "
    "classifier (triggered penalties -> over_aggression). Adapted from "
    "GB/T-Bench, arXiv:2608.06312."
)


def _normalize_dimension(raw: str | None) -> str:
    """Map a rubric ``category`` to a canonical dimension label.

    Returns one of the five benchmark dimensions when ``raw`` matches
    (case-insensitive equality or containment); otherwise returns the
    raw value verbatim (or ``Uncategorized`` when blank) so unknown
    categories are not silently mis-bucketed.
    """
    s = (raw or "").strip()
    if not s:
        return "Uncategorized"
    low = s.lower()
    for canon in _DIMENSIONS:
        if low == canon.lower() or canon.lower() in low:
            return canon
    return s


def _classify_error_type(criteria: str | None) -> str:
    """First-matching legal-concept error type for a rubric's criteria
    text, or ``other`` if none match."""
    text = (criteria or "").lower()
    if not text:
        return "other"
    for error_type, keys in _ERROR_TYPES:
        for k in keys:
            if k in text:
                return error_type
    return "other"


def _is_rubric_error(p: dict) -> bool:
    """Whether a per-rubric verdict counts as a diagnosable error.

    Reward rubrics (positive weight) err on FAIL (the model fell short of
    the rule); penalty rubrics (negative weight) err when *triggered* —
    PASS means the model made the undesirable edit the penalty guards
    against.
    """
    verdict = p.get("verdict")
    if bool(p.get("is_penalty")):
        return verdict == "PASS"
    return verdict == "FAIL"


def classify_rubric(per_rubric: dict) -> tuple[str, str]:
    """Assign one rubric verdict to its ``(dimension, error_type)``.

    Penalty-triggered rubrics map to the ``over_aggression`` error type
    regardless of criteria (the penalty itself is the diagnosis);
    reward-rubric failures are classified by their criteria text.
    """
    dimension = _normalize_dimension(per_rubric.get("category"))
    if per_rubric.get("is_penalty"):
        return dimension, "over_aggression"
    return dimension, _classify_error_type(per_rubric.get("criteria"))


def _dominant(counts: Counter) -> str | None:
    """Highest-count key, breaking ties alphabetically for determinism."""
    if not counts:
        return None
    return sorted(counts, key=lambda k: (-counts[k], k))[0]


def _breakdown(counts: Counter, weights: Counter, total: int) -> dict:
    """Per-key ``{n_errors, weight, share}`` sorted by descending count."""
    return {
        k: {
            "n_errors": counts[k],
            "weight": weights.get(k, 0),
            "share": round(counts[k] / total, 4) if total else 0.0,
        }
        for k in sorted(counts, key=lambda kk: (-counts[kk], kk))
    }


def diagnose_error_profile(
    by_model: dict[str, Iterable[dict]],
) -> dict:
    """Build a per-model hierarchical error profile from trial rows.

    ``by_model`` is the ``rows_by_model`` grouping the metrics pipeline
    already produces: ``{model: [trial_row, …]}``, where each row carries
    a ``_per_rubric`` list of ``{rubric_id, verdict, weight, is_penalty,
    category, criteria}`` entries (the same rows
    ``aggregate.summarize_model`` consumes).

    Returns a diagnosis-oriented profile: per model, the count + weighted
    severity of diagnosed errors split by dimension and by error type,
    the leaf ``"dimension::error_type"`` hierarchy, the dominant
    dimension/type, and ``task_coverage_clean`` (share of the model's
    tasks with zero diagnosed errors — the document-level coverage
    angle). A pooled ``overall`` view shows where the whole field is
    weak.
    """
    by_model_out: dict[str, dict] = {}
    overall_dim: Counter = Counter()
    overall_type: Counter = Counter()

    for model in sorted(by_model):
        rows = list(by_model[model])
        dim_counts: Counter = Counter()
        dim_weight: Counter = Counter()
        type_counts: Counter = Counter()
        type_weight: Counter = Counter()
        leaf: Counter = Counter()
        n_reward_fail = n_penalty_triggered = error_weight = 0
        n_tasks = n_clean = 0

        for r in rows:
            per = r.get("_per_rubric") or []
            task_errors = 0
            for p in per:
                if not _is_rubric_error(p):
                    continue
                task_errors += 1
                is_penalty = bool(p.get("is_penalty"))
                weight = abs(int(p.get("weight") or 0))
                if is_penalty:
                    n_penalty_triggered += 1
                else:
                    n_reward_fail += 1
                error_weight += weight
                dim, etype = classify_rubric(p)
                dim_counts[dim] += 1
                dim_weight[dim] += weight
                type_counts[etype] += 1
                type_weight[etype] += weight
                leaf[f"{dim}::{etype}"] += 1
                overall_dim[dim] += 1
                overall_type[etype] += 1
            n_tasks += 1
            if task_errors == 0:
                n_clean += 1

        n_errors = n_reward_fail + n_penalty_triggered
        by_model_out[model] = {
            "n_errors": n_errors,
            "n_reward_fail": n_reward_fail,
            "n_penalty_triggered": n_penalty_triggered,
            "error_weight": error_weight,
            "task_coverage_clean": (
                round(n_clean / n_tasks, 4) if n_tasks else 1.0
            ),
            "by_dimension": _breakdown(dim_counts, dim_weight, n_errors),
            "by_error_type": _breakdown(type_counts, type_weight, n_errors),
            "by_dimension_error_type": {
                k: leaf[k] for k in sorted(leaf, key=lambda kk: (-leaf[kk], kk))
            },
            "dominant_dimension": _dominant(dim_counts),
            "dominant_error_type": _dominant(type_counts),
        }

    return {
        "source": _SOURCE,
        "n_error_types": len(overall_type),
        "by_model": by_model_out,
        "overall": {
            "by_dimension": {
                k: overall_dim[k]
                for k in sorted(overall_dim, key=lambda kk: (-overall_dim[kk], kk))
            },
            "by_error_type": {
                k: overall_type[k]
                for k in sorted(overall_type, key=lambda kk: (-overall_type[kk], kk))
            },
        },
    }
