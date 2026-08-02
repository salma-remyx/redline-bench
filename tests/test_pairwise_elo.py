"""Integration tests for the pairwise-Elo panel aggregation.

Exercises the wiring through the NON-NEW ``panel`` module: builds a tiny
synthetic judge-grade tree, runs ``panel.main()`` (which now calls
``pairwise_elo.panel_pairwise_elo``), and asserts the Elo ranking lands in
``panel_summary.json`` alongside the pointwise panel leaderboard. The
second test drives ``pairwise_elo`` directly to pin the paper's core knob
— the agreement threshold trading coverage for confidence.
"""

import json
import sys

import panel
import pairwise_elo

# Task names must match panel._NAME_RE (redline-s\d+-t\d+-g\d+[a-z]).
_TASKS = ["redline-s1-t1-g1a", "redline-s2-t2-g2a"]
_MODELS = ["modelA", "modelB"]


def _grade(verdict: str, weighted: float) -> dict:
    """A minimal grade dict: one rubric + the canonical weighted score."""
    return {
        "score": {
            "weighted": weighted,
            "per_rubric": [
                {"rubric_id": "r1", "verdict": verdict, "weight": 1, "category": "core"}
            ],
        }
    }


def _build_tree(root, judges_grades) -> None:
    """Write <root>/<model>/<task>.json grade files for one judge."""
    for (model, task), grade in judges_grades.items():
        d = root / model
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{task}.json").write_text(json.dumps(grade))


def test_pairwise_elo_wired_into_panel(monkeypatch, tmp_path):
    # 3 judges, 2 models, 2 tasks. modelA passes everything (weighted 1.0),
    # modelB fails everything (weighted 0.0) — so every judge prefers A.
    trees = {}
    for label in ("j1", "j2", "j3"):
        root = tmp_path / label
        grades = {
            ("modelA", t): _grade("PASS", 1.0) for t in _TASKS
        }
        grades.update({("modelB", t): _grade("FAIL", 0.0) for t in _TASKS})
        _build_tree(root, grades)
        trees[label] = root

    out = tmp_path / "panel"
    argv = ["panel"]
    for label, root in trees.items():
        argv += ["--judge", f"{label}={root}"]
    argv += ["--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)

    assert panel.main() == 0

    summary = json.loads((out / "panel_summary.json").read_text())
    assert "pairwise_elo" in summary
    pelo = summary["pairwise_elo"]

    # Elo ranking agrees with the pointwise panel ranking: modelA first.
    assert pelo["ranking"][0] == "modelA"
    assert pelo["ranking"] == summary["panel_ranking"]
    assert pelo["ratings"]["modelA"] > 1500.0
    assert pelo["ratings"]["modelB"] < 1500.0
    # Unanimous preferences -> nothing abstains, full coverage.
    assert pelo["n_considered"] == 2  # one model-pair per task, 2 tasks
    assert pelo["n_abstained"] == 0
    assert pelo["n_unanimous"] == 2
    assert pelo["coverage"] == 1.0


def test_agreement_threshold_trades_coverage_for_confidence():
    # 3 judges on one task: two prefer A, one dissenter prefers B.
    grade_a = {"score": {"weighted": 1.0}}
    grade_b = {"score": {"weighted": 0.0}}
    task = "t1"
    judges = {
        "j1": {("modelA", task): grade_a, ("modelB", task): grade_b},
        "j2": {("modelA", task): grade_a, ("modelB", task): grade_b},
        "j3": {("modelA", task): grade_b, ("modelB", task): grade_a},  # dissenter
    }
    common = {("modelA", task), ("modelB", task)}

    maj = pairwise_elo.panel_pairwise_elo(judges, common, agreement_threshold=0.5)
    una = pairwise_elo.panel_pairwise_elo(judges, common, agreement_threshold=1.0)

    # Majority (2/3 >= 0.5) counts A as the winner; full coverage.
    assert maj["n_considered"] == 1 and maj["n_abstained"] == 0
    assert maj["coverage"] == 1.0
    assert maj["ranking"][0] == "modelA"
    # Unanimity (2/3 < 1.0) abstains — higher confidence, zero coverage.
    assert una["n_considered"] == 0 and una["n_abstained"] == 1
    assert una["coverage"] == 0.0
