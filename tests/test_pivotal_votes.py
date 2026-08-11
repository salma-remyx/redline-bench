"""Integration tests for the pivotal-vote (affected-set) panel analysis.

Exercises the wiring through NON-NEW modules: builds judge grade dicts in
the exact shape ``panel._rubric_rows`` consumes, runs the shared
``panel.majority_vote_per_rubric`` vote, and feeds it to ``pivotal_votes``
— then drives ``panel.main()`` end-to-end on a synthetic on-disk judge
tree to confirm the pivotal summary actually lands in panel_summary.json.

The pivotal detection itself is the core result of *Blind to the Pivotal
Vote* (arXiv:2608.06940v1): only rubrics decided by a one-vote margin
can flip under single-ballot substitution.
"""

import importlib
import json

panel = importlib.import_module("panel")
pivotal_votes = importlib.import_module("pivotal_votes")


def _grade(per_rubric):
    """Grade dict in the shape rejudge / the verifier writes."""
    return {"score": {"per_rubric": per_rubric, "weighted": 0.0}}


def _rubric(rid, verdict, weight):
    return {"rubric_id": rid, "verdict": verdict,
            "weight": weight, "category": None}


def test_pivotal_detection_through_panel_vote():
    # 3 judges: r1 unanimous PASS, r2 = 2-1 PASS (pivotal), r3 = 2-1 FAIL
    # (pivotal). Differing weights (3 / 1 / 1) keep the count fraction
    # and the weight fraction distinct so both are exercised.
    rubric_sets = [
        {"r1": ("PASS", 3, None), "r2": ("PASS", 1, None), "r3": ("FAIL", 1, None)},
        {"r1": ("PASS", 3, None), "r2": ("PASS", 1, None), "r3": ("FAIL", 1, None)},
        {"r1": ("PASS", 3, None), "r2": ("FAIL", 1, None), "r3": ("PASS", 1, None)},
    ]
    panel_verdicts, weights = panel.majority_vote_per_rubric(rubric_sets)
    assert panel_verdicts == {"r1": "PASS", "r2": "PASS", "r3": "FAIL"}

    stats = pivotal_votes.pivotal_task_stats(rubric_sets, weights)
    # r1 is unanimous (margin 3) → not pivotal; r2, r3 are 2-1 splits.
    assert stats["pivotal_rubric_ids"] == ["r2", "r3"]
    assert stats["n_pivotal"] == 2 and stats["n_rubrics"] == 3
    # Count fraction 2/3; the call-reduction rule mirrors it exactly.
    assert stats["fraction_pivotal"] == round(2 / 3, 4)
    assert stats["verification_invocation_rate"] == stats["fraction_pivotal"]
    # Weight share: pivotal positive weight 2 over total 5 = 0.4 (and
    # deliberately != the 0.667 count fraction).
    assert stats["pivotal_weight_share"] == 0.4


def test_unanimous_panel_has_no_pivotal_rubrics():
    rubric_sets = [
        {"r1": ("PASS", 2, None), "r2": ("FAIL", 2, None)},
    ] * 3  # all three judges identical → margins of 3, nothing pivotal
    _, weights = panel.majority_vote_per_rubric(rubric_sets)
    stats = pivotal_votes.pivotal_task_stats(rubric_sets, weights)
    assert stats["n_pivotal"] == 0
    assert stats["fraction_pivotal"] == 0.0
    assert stats["pivotal_weight_share"] == 0.0
    assert stats["pivotal_rubric_ids"] == []


def test_score_swing_bounded_by_pivotal_weight(monkeypatch):
    # When every pivotal rubric flips against the panel, the score swing
    # can move at most the pivotal positive weight (here 1) over the
    # total positive weight (here 3): |Δ| <= 1/3.
    rubric_sets = [
        {"r1": ("PASS", 2, None), "r2": ("PASS", 1, None)},
        {"r1": ("PASS", 2, None), "r2": ("PASS", 1, None)},
        {"r1": ("PASS", 2, None), "r2": ("FAIL", 1, None)},
    ]
    panel_verdicts, weights = panel.majority_vote_per_rubric(rubric_sets)
    swing = pivotal_votes.pivotal_score_swing(
        panel_verdicts, weights, rubric_sets,
    )
    assert 0.0 < swing <= round(1 / 3, 4)


def test_main_writes_pivotal_summary(tmp_path, monkeypatch):
    # End-to-end through panel.main(): a 3-judge on-disk tree where r2 is
    # a 2-1 split (pivotal) and r1 is unanimous. Proves the wiring edit
    # surfaces pivotal_votes in the written summary + per-model rollup.
    task = "redline-s1-t1-g01a"
    model = "gpt55"
    judges = {
        "jA": {"r1": ("PASS", 2), "r2": ("PASS", 1)},
        "jB": {"r1": ("PASS", 2), "r2": ("PASS", 1)},
        "jC": {"r1": ("PASS", 2), "r2": ("FAIL", 1)},
    }
    judge_root = tmp_path / "judges"
    for label, rubrics in judges.items():
        d = judge_root / label / model
        d.mkdir(parents=True)
        per_rubric = [_rubric(rid, v, w) for rid, (v, w) in rubrics.items()]
        (d / f"{task}.json").write_text(json.dumps(_grade(per_rubric)))

    out = tmp_path / "panel_out"
    monkeypatch.setattr("sys.argv", [
        "panel",
        f"--judge=jA={judge_root / 'jA'}",
        f"--judge=jB={judge_root / 'jB'}",
        f"--judge=jC={judge_root / 'jC'}",
        f"--out={out}",
    ])
    assert panel.main() == 0

    summary = json.loads((out / "panel_summary.json").read_text())
    pv = summary["pivotal_votes"]
    assert pv["overall"]["n_rubrics"] == 2
    assert pv["overall"]["n_pivotal"] == 1            # only r2
    assert pv["overall"]["fraction_pivotal"] == 0.5
    assert pv["overall"]["pivotal_weight_share"] == round(1 / 3, 4)
    assert pv["per_model"]["gpt55"]["n_pivotal"] == 1
