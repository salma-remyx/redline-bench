#!/usr/bin/env python3
"""No-reference judge calibration + over-crediting sensitivity for RedlineBench.

RedlineBench's public metrics rest on a 3-judge panel that grades each rubric
PASS/FAIL with NO reference answer in the prompt — exactly the reference-free
LLM-judge setting studied in "LLM Judges Can Be Too Generous When There Is No
Reference Answer" (arXiv:2607.12885). That paper shows reference-free judges
systematically OVER-CREDIT incorrect answers, and that consulting a reference
answer flips a large fraction of verdicts (up to 85% in their settings). Its
recommendation is to CALIBRATE a judge against a reference-aware sample before
trusting its reference-free verdicts.

This module is a sibling diagnostic to `panel.py`'s `judge_agreement` and
`sensitivity_per_judge`. It reads the SAME per-judge verdict trees `panel.py`
does (via `panel.load_judge`), reuses `panel.majority_vote_per_rubric` for the
panel verdict, and measures — against a reference judge (gold / human /
rollout-grade standard supplied via `--reference`) — the paper's two signals:

  * over-credit rate  — among rubrics the reference FAILs, the fraction the
    reference-free judge/panel PASSes (the paper's headline "too generous"
    signal; higher = more incorrect answers credited as correct).
  * flip rate         — the fraction of rubrics whose verdict changes when the
    reference is consulted (the paper's "reference-driven decision flip").
  * under-credit rate — the symmetric direction (reference PASS, judge FAIL),
    reported alongside so the net generosity bias is visible.

Adapted port (Mode 2): the paper re-prompts the SAME judge with and without a
reference answer injected into the prompt and measures the within-judge flip.
RedlineBench does not store reference-conditioned re-prompts, so that auxiliary
is substituted by its target-native equivalent already on disk — comparing each
reference-free panel judge (and the panel majority) against the separate
reference-judge verdicts that `panel.py --reference` already consumes. The
over-crediting / decision-flip measurements themselves are the paper's core
mechanism, implemented at full fidelity; only the reference-injection auxiliary
is substituted. The weighted-score and panel-majority contracts are untouched.

Usage (peer to `python -m panel`):

    python -m judge_calibration \\
        --judge "gpt-5.4-mini=results/judge/gpt-5.4-mini" \\
        --judge "claude-haiku=results/judge/claude-haiku" \\
        --judge "gemini-3.1-flash-lite=results/judge/gemini-3.1-flash-lite" \\
        --reference "gpt-5.5=results/judge/gpt-5.5" \\
        --out results/calibration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from panel import _rubric_rows, load_judge, majority_vote_per_rubric


def verdict_map(grade: dict) -> dict[str, str]:
    """rubric_id -> verdict for a single grade, via panel._rubric_rows."""
    return {rid: row[0] for rid, row in _rubric_rows(grade).items()}


def judge_verdict_trees(
    judge_grades: dict[tuple[str, str], dict],
) -> dict[tuple[str, str], dict[str, str]]:
    """(model, task) -> {rubric_id: verdict} for one judge's loaded grade tree."""
    return {mt: verdict_map(grade) for mt, grade in judge_grades.items()}


def generosity_metrics(
    candidate: dict, reference: dict
) -> dict:
    """Over-credit / under-credit / flip rates of `candidate` vs `reference`,
    computed over the rubric keys BOTH graded (intersection).

    `candidate`: key -> "PASS"/"FAIL" from the reference-free judge/panel.
    `reference`: key -> "PASS"/"FAIL" from the gold/reference standard.

    Keys are opaque (callers pass `(model, task, rubric_id)` tuples) — each key
    is one independent judgment call. Rates are None where the denominator is
    empty so callers can tell "0 over-credits" from "nothing to measure".
    """
    keys = set(candidate) & set(reference)
    n = len(keys)
    over = sum(1 for k in keys if candidate[k] == "PASS" and reference[k] == "FAIL")
    under = sum(1 for k in keys if candidate[k] == "FAIL" and reference[k] == "PASS")
    n_ref_fail = sum(1 for k in keys if reference[k] == "FAIL")
    n_ref_pass = sum(1 for k in keys if reference[k] == "PASS")
    n_cand_pass = sum(1 for k in keys if candidate[k] == "PASS")
    return {
        "n_rubrics": n,
        "n_over_credit": over,
        "n_under_credit": under,
        "n_reference_fail": n_ref_fail,
        "n_reference_pass": n_ref_pass,
        "over_credit_rate": round(over / n_ref_fail, 4) if n_ref_fail else None,
        "under_credit_rate": round(under / n_ref_pass, 4) if n_ref_pass else None,
        "flip_rate": round((over + under) / n, 4) if n else None,
        "candidate_pass_rate": round(n_cand_pass / n, 4) if n else None,
        "reference_pass_rate": round(n_ref_pass / n, 4) if n else None,
    }


def _with_bias(metrics: dict) -> dict:
    """Attach net generosity bias (over-credit − under-credit). Positive means
    the reference-free judge over-credits; None if either rate is unmeasurable."""
    over = metrics["over_credit_rate"]
    under = metrics["under_credit_rate"]
    metrics["net_generosity_bias"] = (
        round(over - under, 4) if over is not None and under is not None else None
    )
    return metrics


def _flatten(tree, keys) -> dict:
    """Flatten {(model,task): {rid: verdict}} over the given (model,task) keys
    into {(model, task, rid): verdict} so each judgment call is its own row."""
    return {(mt, rid): v for mt in keys for rid, v in tree[mt].items()}


def calibrate(
    judges: dict[str, dict],
    reference_grades: dict,
    *,
    include_panel: bool = True,
) -> dict:
    """Calibration + over-crediting diagnostics for each reference-free judge
    (and optionally the panel majority) against `reference_grades`.

    `judges`: label -> loaded grade tree (the shape `panel.load_judge` returns).
    `reference_grades`: a single judge's loaded grade tree used as the gold
        standard (e.g. GPT-5.5 rollout grades, or a human-annotated set).

    The panel majority is rebuilt with `panel.majority_vote_per_rubric` over the
    rubric sets `panel._rubric_rows` produces — bit-identical to the panel
    verdict that feeds the public metrics — so its over-credit rate is the
    generosity of the score the benchmark actually reports.

    Measurements are restricted to the (model, task) pairs ALL judges AND the
    reference graded (intersection set, matching `panel.py`'s `common`).
    """
    candidate_trees = {lbl: judge_verdict_trees(g) for lbl, g in judges.items()}
    ref_tree = judge_verdict_trees(reference_grades)

    common = set(ref_tree)
    for tree in candidate_trees.values():
        common &= set(tree)

    ref_flat = _flatten(ref_tree, common)

    per_judge: dict[str, dict] = {}
    for lbl, tree in candidate_trees.items():
        cand_flat = _flatten(tree, common)
        per_judge[lbl] = _with_bias(generosity_metrics(cand_flat, ref_flat))

    panel_metrics = None
    if include_panel and judges:
        cand_flat: dict = {}
        for mt in common:
            rubric_sets = [_rubric_rows(judges[lbl][mt]) for lbl in judges]
            panel_verdicts, _weights = majority_vote_per_rubric(rubric_sets)
            for rid, v in panel_verdicts.items():
                cand_flat[(mt, rid)] = v
        panel_metrics = _with_bias(generosity_metrics(cand_flat, ref_flat))

    summary: dict = {
        "n_judges": len(judges),
        "n_common_model_task": len(common),
        "reference_free_judges": list(judges),
        "per_judge": per_judge,
    }
    if panel_metrics is not None:
        summary["panel_majority"] = panel_metrics
        over = panel_metrics["over_credit_rate"]
        summary["panel_over_credits"] = over is not None and over > 0
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", action="append", required=True,
                    help="label=path/to/reference-free/judge/output/tree (repeatable)")
    ap.add_argument("--reference", required=True,
                    help="label=path for the reference/gold judge tree "
                         "(e.g. gpt-5.5 rollout grades)")
    ap.add_argument("--out", default="results/calibration")
    args = ap.parse_args()

    judges = {}
    for spec in args.judge:
        label, path = spec.split("=", 1)
        judges[label] = load_judge(Path(path))
    rlabel, rpath = args.reference.split("=", 1)
    reference = load_judge(Path(rpath))

    summary = calibrate(judges, reference)
    summary["reference_label"] = rlabel

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"{len(judges)} reference-free judges vs reference '{rlabel}', "
          f"{summary['n_common_model_task']} common (model,task) pairs",
          flush=True)
    for lbl, m in summary["per_judge"].items():
        print(f"  {lbl}: over-credit={m['over_credit_rate']} "
              f"flip={m['flip_rate']} net_bias={m['net_generosity_bias']}")
    panel = summary.get("panel_majority")
    if panel:
        print(f"  panel majority: over-credit={panel['over_credit_rate']} "
              f"flip={panel['flip_rate']} net_bias={panel['net_generosity_bias']}")
        print(f"panel over-credits incorrect answers: "
              f"{summary['panel_over_credits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
