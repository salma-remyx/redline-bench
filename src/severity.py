#!/usr/bin/env python3
"""Action-graded severity scale for RedlineBench rubric failures.

Adapted (Mode 2) from "Beyond Attack-Success Rate: Action-Graded Severity
Scale for Tool-Using AI Agents" (arXiv:2607.07474). That paper argues a
binary attack-success rate throws away *how harmful* each compromise was,
and grades an agent's tool-call trajectory on an ordinal L0–L6 scale
(reversible? cross-scope? privilege-expanding?) using a deterministic
oracle plus a 3-judge LLM panel.

RedlineBench hits the same problem one layer down: its rubric pipeline
collapses every criterion to a single PASS/FAIL bit, so a model that
omits a critical indemnity clause and a model that fumbles a formatting
detail both register as one identical "FAIL". This module keeps the
paper's core instrument — a fixed ordinal scale graded by a
deterministic oracle over typed records — and re-points it at the
per-rubric records ``judging.aggregate()`` already emits:

  paper concept                 -> RedlineBench analog
  ------------------------------ ------------------------------------------
  action trajectory             -> one per-rubric verdict record
  reversible / cross-scope /    -> legal-impact gates read off the rubric's
  privilege gates                  importance ``weight``, penalty flag, and
                                  category (clause criticality as a
                                  parameter-free proxy for harm magnitude)
  L0–L6 scale                   -> preliminary L0–L4 scale (suggested experiment)

What is intentionally NOT ported (Mode 2 substitutions / scoping):

  - The 3-frontier-model judge panel. The repo already runs a 3-judge
    panel for the binary verdict; the deterministic oracle is the novel
    instrument and the clean, dependency-free integration. Wiring an LLM
    severity judge through ``judging.call_judge`` to reproduce the
    paper's oracle↔panel agreement (Krippendorff α = 0.91) is a natural
    follow-up, out of scope here.
  - The AgentDojo trajectory/benchmark suite — evaluation already lives
    in this repo's own ``runs/<run-id>/`` tree.

Composes into ``metrics_summary.run()``'s output dict alongside (not in
place of) the existing binary score.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean

# ─── the ordinal scale ───────────────────────────────────────────────
#
# A preliminary five-level scale (L0–L4). Each level is a fixed gate
# over a per-rubric record's metadata; the oracle is deterministic and
# parameter-free, mirroring the paper's deterministic oracle.

SEVERITY_SCALE: list[dict] = [
    {
        "level": 0, "label": "L0", "name": "no-harm",
        "desc": (
            "Rubric satisfied (PASS), or a penalty rubric correctly NOT "
            "triggered. No failure to remediate."
        ),
    },
    {
        "level": 1, "label": "L1", "name": "cosmetic",
        "desc": (
            "Low-importance miss (|weight| <= 2): formatting, structure, "
            "or other non-substantive redline error. Reversible / cosmetic."
        ),
    },
    {
        "level": 2, "label": "L2", "name": "minor-ambiguity",
        "desc": (
            "Moderate-importance miss (|weight| 3-5): minor legal "
            "ambiguity or partial edit leaving the clause weaker."
        ),
    },
    {
        "level": 3, "label": "L3", "name": "substantive",
        "desc": (
            "High-importance miss (|weight| 6-7), or an undesirable edit "
            "the model made on a low/moderate penalty rubric. Substantive "
            "legal misinterpretation creating real exposure."
        ),
    },
    {
        "level": 4, "label": "L4", "name": "critical",
        "desc": (
            "Critical-clause omission (|weight| >= 8) or a serious "
            "undesirable edit (penalty |weight| >= 6). The legal analog "
            "of an irreversible, cross-scope action."
        ),
    },
]

_LABELS = [s["label"] for s in SEVERITY_SCALE]


# ─── the deterministic oracle ────────────────────────────────────────


def grade_severity(per_rubric: dict) -> int:
    """Grade one per-rubric record on the L0–L4 ordinal scale.

    ``per_rubric`` is exactly the unit ``judging.aggregate()`` emits in
    its ``per_rubric`` list (and that ``runs_reader`` / ``panel_reader``
    carry on each trial row under ``_per_rubric``): a dict with
    ``verdict`` (``"PASS"``/``"FAIL"``), integer ``weight``, and
    ``is_penalty``. Returns the integer level (0–4).
    """
    verdict = (per_rubric.get("verdict") or "FAIL").upper()
    if verdict not in ("PASS", "FAIL"):
        verdict = "FAIL"
    weight = abs(int(per_rubric.get("weight") or 0))
    is_penalty = bool(per_rubric.get("is_penalty"))

    # A penalty rubric describes an edit the attorney flagged as
    # undesirable, so the *desired* verdict is FAIL (don't make the edit).
    if is_penalty:
        if verdict == "PASS":
            # Model made the undesirable edit -> active harm, graded by
            # how seriously the attorney weighted the forbidden change.
            return 4 if weight >= 6 else 3
        return 0  # correctly avoided -> no failure to remediate

    if verdict == "PASS":
        return 0  # satisfied the criterion -> no harm

    # Plain FAIL on a positive-weight rubric: the model missed a required
    # edit. Severity scales with the clause's importance weight.
    if weight >= 8:
        return 4
    if weight >= 6:
        return 3
    if weight >= 3:
        return 2
    return 1


def grade_label(per_rubric: dict) -> str:
    """The L0–L4 *label* (e.g. ``"L4"``) for one record."""
    return SEVERITY_SCALE[grade_severity(per_rubric)]["label"]


# ─── benchmark-level rollup ──────────────────────────────────────────


def _rollup(records: list[dict]) -> dict:
    """Severity distribution over a set of per-rubric records."""
    levels = [grade_severity(rec) for rec in records]
    if not levels:
        return _empty_rollup()
    counts = Counter(levels)
    n = len(levels)
    failures = [lv for lv in levels if lv > 0]
    nf = len(failures)
    high = sum(1 for lv in failures if lv >= 3)
    return {
        "levels": {
            _LABELS[lv]: counts.get(lv, 0) for lv in range(len(_LABELS))
        },
        # Share is over ALL records (incl. L0): L0's share is effectively
        # the pass rate, and L1–L4 sum to the fail rate.
        "share": {
            _LABELS[lv]: round(counts.get(lv, 0) / n, 4)
            for lv in range(len(_LABELS))
        },
        "mean_severity": round(mean(levels), 4),
        "mean_fail_severity": round(mean(failures), 4) if failures else 0.0,
        "max_severity": max(levels),
        "n_records": n,
        "n_failures": nf,
        # The paper's headline: severity exposes cases the binary metric
        # flattens. Here that is the share of FAILs graded L3+ —
        # indistinguishable from cosmetic FAILs under a raw pass-rate.
        "high_severity_failures": {
            "count": high,
            "share_of_failures": round(high / nf, 4) if nf else 0.0,
        },
    }


def _empty_rollup() -> dict:
    return {
        "levels": {lbl: 0 for lbl in _LABELS},
        "share": {lbl: 0.0 for lbl in _LABELS},
        "mean_severity": 0.0,
        "mean_fail_severity": 0.0,
        "max_severity": 0,
        "n_records": 0,
        "n_failures": 0,
        "high_severity_failures": {"count": 0, "share_of_failures": 0.0},
    }


def summarize_severity(trials: list[dict]) -> dict:
    """Grade every per-rubric record across ``trials`` and roll up into a
    benchmark-level severity summary for ``metrics_summary.run()``.

    ``trials`` are the rows ``runs_reader`` / ``panel_reader`` emit — each
    carries a ``_per_rubric`` list (the per-rubric output of
    ``judging.aggregate()``). This adds nothing to the rubric pipeline;
    it reads what that pipeline already produced.
    """
    pairs: list[tuple[str, dict]] = []
    for trial in trials:
        model = trial.get("model") or "unknown"
        for rec in trial.get("_per_rubric") or []:
            pairs.append((model, rec))

    by_model: dict[str, dict] = {}
    for model in sorted({m for m, _ in pairs}):
        by_model[model] = _rollup([rec for m, rec in pairs if m == model])

    return {
        "scale": SEVERITY_SCALE,
        **_rollup([rec for _, rec in pairs]),
        "by_model": by_model,
    }
