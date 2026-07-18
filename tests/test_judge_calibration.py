"""Tests for no-reference judge calibration / over-crediting diagnostics.

Covers src/judge_calibration.py through its integration with the EXISTING
`panel` module: synthetic per-judge grade trees are built in the exact shape
`panel.load_judge` produces, then run through `panel.majority_vote_per_rubric`
(the real vote the public metrics use) AND `judge_calibration.calibrate`, and
the over-credit / flip diagnostics are checked against hand-computed values —
the direction the paper predicts (reference-free judges over-credit incorrect
answers; consulting a reference flips those verdicts).
"""

import importlib

panel = importlib.import_module("panel")
judge_calibration = importlib.import_module("judge_calibration")


def _row(rid, verdict, weight=4, category="legal"):
    return {"rubric_id": rid, "verdict": verdict, "weight": weight,
            "category": category}


def _grade(per_rubric):
    return {"score": {"weighted": 0.0, "per_rubric": per_rubric}}


def _tree(verdicts_by_rid, model="m1", task="redline-s1-t1-g01a"):
    """One judge's grade tree: {(model, task): grade}, the load_judge shape."""
    return {(model, task): _grade([_row(r, v) for r, v in verdicts_by_rid.items()])}


def test_generous_judge_over_credits_every_reference_fail():
    # Reference (gold) FAILs r2/r3; the reference-free judge PASSes everything.
    reference = _tree({"r1": "PASS", "r2": "FAIL", "r3": "FAIL"})
    generous = _tree({"r1": "PASS", "r2": "PASS", "r3": "PASS"})

    summary = judge_calibration.calibrate({"gen": generous}, reference)
    m = summary["per_judge"]["gen"]

    assert m["n_over_credit"] == 2          # r2, r3 PASS'd despite reference FAIL
    assert m["over_credit_rate"] == 1.0     # both reference-FAILs over-credited
    assert m["under_credit_rate"] == 0.0    # nothing wrongly FAIL'd
    assert m["net_generosity_bias"] == 1.0  # strongly over-credits
    # 2 of 3 rubrics flip when the reference is consulted.
    assert m["flip_rate"] == round(2 / 3, 4)


def test_judge_matching_reference_has_zero_over_credit_and_zero_flips():
    reference = _tree({"r1": "PASS", "r2": "FAIL", "r3": "PASS"})
    clone = _tree({"r1": "PASS", "r2": "FAIL", "r3": "PASS"})

    m = judge_calibration.calibrate({"clone": clone}, reference)["per_judge"]["clone"]

    assert m["over_credit_rate"] == 0.0
    assert m["under_credit_rate"] == 0.0
    assert m["flip_rate"] == 0.0
    assert m["net_generosity_bias"] == 0.0


def test_under_crediting_is_captured_too():
    # Harsh judge FAILs a rubric the reference PASSes -> under-credit, while
    # agreeing with the reference's one FAIL -> 0 over-credit.
    reference = _tree({"r1": "PASS", "r2": "PASS", "r3": "FAIL"})
    harsh = _tree({"r1": "FAIL", "r2": "PASS", "r3": "FAIL"})

    m = judge_calibration.calibrate({"harsh": harsh}, reference)["per_judge"]["harsh"]

    assert m["n_under_credit"] == 1
    assert m["under_credit_rate"] == 0.5     # 1 of 2 reference-PASSes wrongly FAIL'd
    assert m["n_over_credit"] == 0
    assert m["over_credit_rate"] == 0.0      # 0 of 1 reference-FAILs over-credited
    assert m["net_generosity_bias"] == -0.5  # net under-crediting


def test_empty_intersection_yields_none_rates():
    reference = _tree({"r1": "PASS"})
    # Same (model, task) but a disjoint rubric id -> nothing in common.
    other = _tree({"rZZZ": "PASS"})

    m = judge_calibration.calibrate({"o": other}, reference)["per_judge"]["o"]

    assert m["n_rubrics"] == 0
    assert m["over_credit_rate"] is None
    assert m["flip_rate"] is None


def test_panel_majority_uses_panels_real_vote():
    """The panel-majority diagnostic must reuse panel.majority_vote_per_rubric,
    not a reimplementation — confirming real integration with panel.py."""
    # Three judges, odd count. Two say PASS on r2, one says FAIL -> panel PASS.
    # Reference says r2 is FAIL -> the panel over-credits r2.
    reference = _tree({"r1": "PASS", "r2": "FAIL"})
    judges = {
        "j1": _tree({"r1": "PASS", "r2": "PASS"}),
        "j2": _tree({"r1": "PASS", "r2": "PASS"}),
        "j3": _tree({"r1": "PASS", "r2": "FAIL"}),
    }

    summary = judge_calibration.calibrate(judges, reference)
    panel_metrics = summary["panel_majority"]

    # Cross-check against panel.py's own majority vote on the same rubric.
    rubric_sets = [panel._rubric_rows(g[("m1", "redline-s1-t1-g01a")])
                   for g in judges.values()]
    panel_verdicts, _ = panel.majority_vote_per_rubric(rubric_sets)
    assert panel_verdicts["r2"] == "PASS"  # 2-of-3 majority

    # So the panel PASSes r2 while the reference FAILs it -> over-credit.
    assert panel_metrics["n_over_credit"] == 1
    assert panel_metrics["over_credit_rate"] == 1.0
    assert summary["panel_over_credits"] is True


def test_calibrate_restricts_to_pairs_all_judges_and_reference_graded():
    # Judge A graded an extra (model, task) the reference didn't — it must be
    # excluded from the measurement (panel.py's intersection semantics).
    reference = {("m1", "redline-s1-t1-g01a"): _grade([_row("r1", "PASS")])}
    judge_a = {
        ("m1", "redline-s1-t1-g01a"): _grade([_row("r1", "PASS")]),
        ("m1", "redline-s1-t1-g01b"): _grade([_row("r1", "PASS")]),
    }

    summary = judge_calibration.calibrate({"a": judge_a}, reference)

    assert summary["n_common_model_task"] == 1
    assert summary["per_judge"]["a"]["n_rubrics"] == 1
