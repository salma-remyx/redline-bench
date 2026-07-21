#!/usr/bin/env python3
"""Directional confusion metrics for judge calibration.

RedlineBench's panel reports each judge family as a single scalar
leaderboard (``panel.sensitivity_per_judge``). That scalar collapses two
very different failure modes into one number: a judge that *over-credits*
(rubber-stamps PASS) and a judge that *over-rejects* (FAIL-happy) can
land at the same score. The direction of the error is exactly what a
downstream consensus vote — or, as in the source paper, a reinforcement
loop that treats the judge as its reward signal — would amplify, so
hiding it behind a single F1 is unsafe.

This module breaks the scalar apart into the confusion-matrix view:
false-positive rate, false-negative rate, pass-rate drift, pass-class
precision/recall/F1, and Cohen's kappa against a gold reference. The
caller (``panel.main()``) supplies gold as either the panel's designated
``--reference`` judge or, absent one, a leave-one-out majority of the
*other* panel judges — consensus-as-gold, the target-native stand-in for
the paper's human gold labels.

Adapted from arXiv:2607.08700, "Do You Need a Frontier Model as a
Citation Verifier? Benchmarking Rubric LLMs for Deep-Research Source
Attribution" — which audits LLM judges with FPR / FNR / pass-rate-drift
rather than scalar F1 alone and finds that, at comparable F1, cheaper
judges stay competitive once their directional bias is exposed. The
audit there scored citation-rubric decisions against human gold; here
the per-rubric PASS/FAIL verdict stands in for the rubric decision and
panel consensus replaces the human adjudication. The directional-metric
mechanism (FPR/FNR/drift/kappa, the paper's contribution) is kept at
full fidelity; only the gold source is substituted.
"""

from __future__ import annotations

PASS = "PASS"
FAIL = "FAIL"


def confusion_counts(judge_labels, gold_labels):
    """Tally a 2x2 confusion matrix (PASS = positive) from two parallel
    label iterables. Returns ``(tp, fp, fn, tn)``; pairs whose labels are
    not exactly ``PASS``/``FAIL`` are skipped (partial / ungraded rubrics).
    """
    tp = fp = fn = tn = 0
    for j, g in zip(judge_labels, gold_labels):
        if j not in (PASS, FAIL) or g not in (PASS, FAIL):
            continue
        if j == PASS and g == PASS:
            tp += 1
        elif j == PASS and g == FAIL:
            fp += 1
        elif j == FAIL and g == PASS:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def confusion_metrics(judge_labels, gold_labels):
    """Full directional-bias report for one judge vs gold.

    ``judge_labels`` / ``gold_labels`` are parallel iterables of
    ``"PASS"`` / ``"FAIL"`` — the per-rubric verdict strings the panel
    already carries. Returns a dict with the confusion counts and the
    paper's directional metrics:

      * ``false_positive_rate`` — over-crediting (PASS when gold FAIL)
      * ``false_negative_rate`` — over-rejecting (FAIL when gold PASS)
      * ``pass_rate_drift``     — judge pass-rate minus gold pass-rate
      * ``precision`` / ``recall`` / ``f1`` — pass-class; the scalar the
        paper shows obscures error direction
      * ``kappa``               — Cohen's kappa, chance-corrected agreement
      * ``dominant_error``      — ``over-crediting`` / ``over-rejecting`` /
        ``balanced``; the prescriptive label for judge selection
    """
    tp, fp, fn, tn = confusion_counts(judge_labels, gold_labels)
    n = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    judge_pass = tp + fp
    gold_pass = tp + fn
    judge_pass_rate = judge_pass / n if n else 0.0
    gold_pass_rate = gold_pass / n if n else 0.0

    # Cohen's kappa — chance-corrected agreement (paper reports kappa).
    po = (tp + tn) / n if n else 0.0
    pe = (((judge_pass * gold_pass) + ((n - judge_pass) * (n - gold_pass)))
          / (n * n)) if n else 0.0
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0

    if fpr > fnr:
        dominant = "over-crediting"
    elif fnr > fpr:
        dominant = "over-rejecting"
    else:
        dominant = "balanced"

    return {
        "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "pass_rate_drift": round(judge_pass_rate - gold_pass_rate, 4),
        "judge_pass_rate": round(judge_pass_rate, 4),
        "gold_pass_rate": round(gold_pass_rate, 4),
        "kappa": round(kappa, 4),
        "dominant_error": dominant,
    }


def format_confusion(label, metrics):
    """One-line human summary of a judge's directional bias."""
    if not metrics or metrics.get("n", 0) == 0:
        return f"{label}: (no graded rubrics)"
    return (
        f"{label}: FPR={metrics['false_positive_rate']} "
        f"FNR={metrics['false_negative_rate']} "
        f"drift={metrics['pass_rate_drift']} F1={metrics['f1']} "
        f"kappa={metrics['kappa']} [{metrics['dominant_error']}]"
    )
