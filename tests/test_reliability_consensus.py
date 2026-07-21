"""Integration tests for the reliability-gated panel consensus.

Drives the REAL call site — ``panel.main()`` (a NON-NEW module) — over
on-disk judge trees, then asserts ``panel_summary.json`` carries the new
``reliability_gated`` view ALONGSIDE the unchanged official majority-vote
score, and that the reliability report correctly gates the noisy judge.
A second test exercises ``reliability_consensus`` directly to show gating
can flip a verdict that blind majority vote gets wrong.

This is the reliability-gating mechanism adapted from *Project
Kaleidoscope* (arXiv:2607.14673v1) — see ``reliability_consensus.py``.
"""

import json
import sys

import panel
import reliability_consensus

_MODEL = "agentco"
_TASKS = ["redline-s1-t1-g01a", "redline-s1-t1-g01b"]
# rubric_id -> (weight,) — same weight set every judge grades.
_RUBRICS = {"r1": 2, "r2": 3}


def _grade(verdicts: dict[str, str]) -> dict:
    per = [
        {"rubric_id": rid, "verdict": verdicts[rid], "weight": w, "category": "legal"}
        for rid, w in _RUBRICS.items()
    ]
    pos = sum(w for w in _RUBRICS.values() if w > 0)
    earned = sum(w for rid, w in _RUBRICS.items() if verdicts[rid] == "PASS" and w > 0)
    return {"score": {"weighted": round(earned / pos, 4) if pos else 0.0, "per_rubric": per}}


def _write_judge_tree(judge_root, verdicts: dict[str, str]) -> None:
    """<judge_root>/<model>/<task>.json — the layout ``panel.load_judge`` expects."""
    model_dir = judge_root / _MODEL
    model_dir.mkdir(parents=True, exist_ok=True)
    for task in _TASKS:
        (model_dir / f"{task}.json").write_text(json.dumps(_grade(verdicts)))


def test_panel_main_reports_reliability_gated_alongside_official(monkeypatch, tmp_path):
    # good-a / good-b agree with each other; noisy dissents on every rubric.
    roots = {
        "good-a": {"r1": "PASS", "r2": "PASS"},
        "good-b": {"r1": "PASS", "r2": "PASS"},
        "noisy": {"r1": "FAIL", "r2": "FAIL"},
    }
    paths = {}
    for label, verdicts in roots.items():
        root = tmp_path / label
        _write_judge_tree(root, verdicts)
        paths[label] = str(root)

    out = tmp_path / "panel"
    argv = ["panel", f"--out={out}"]
    for label, p in paths.items():
        argv.append(f"--judge={label}={p}")
    monkeypatch.setattr(sys, "argv", argv)
    assert panel.main() == 0

    summary = json.loads((out / "panel_summary.json").read_text())

    # Official majority-vote score is PRESERVED.
    assert summary["panel_leaderboard"][_MODEL] == 1.0

    # The reliability-gated view is reported alongside it.
    assert "reliability_gated" in summary
    rg = summary["reliability_gated"]
    assert rg["leaderboard"][_MODEL] == 1.0
    assert rg["matches_official_ranking"] is True

    # Reliability proxy: good-* agree with the panel (0.5 each), noisy is
    # isolated (0.0) -> gated at the default 0.5 threshold.
    report = rg["report"]
    assert report["threshold"] == 0.5
    assert report["per_judge_reliability"]["noisy"] == 0.0
    assert report["per_judge_reliability"]["good-a"] == 0.5
    assert report["gated_judges"] == ["noisy"]
    assert report["n_rubrics_flagged_for_review"] == 0

    # Pairwise agreement still reported (existing behavior unchanged).
    assert summary["judge_agreement"]["good-a vs good-b"] == 1.0
    assert summary["judge_agreement"]["good-a vs noisy"] == 0.0

    # CSV carries the gated column next to the official one.
    header = (out / "panel_leaderboard.csv").read_text().splitlines()[0]
    assert "panel_majority" in header and "panel_reliability_gated" in header


def test_gating_flips_verdict_that_majority_gets_wrong():
    # judge A is reliable + heavy and dissents; B is barely-reliable; C is
    # gated. Blind majority (B+C) says PASS; reliability-gated says FAIL.
    rubric_sets = [
        {"r1": ("FAIL", 2, None)},   # A
        {"r1": ("PASS", 2, None)},   # B
        {"r1": ("PASS", 2, None)},   # C
    ]
    reliab = {"A": 0.9, "B": 0.5, "C": 0.1}
    verdicts, _weights, flagged = reliability_consensus.reliability_gated_consensus(
        rubric_sets, ["A", "B", "C"], reliab, threshold=0.4
    )
    assert verdicts["r1"] == "FAIL"   # fail_w 0.9 > pass_w 0.5
    assert flagged == 0

    maj_verdicts, _ = panel.majority_vote_per_rubric(rubric_sets)
    assert maj_verdicts["r1"] == "PASS"  # 2 of 3 -> PASS; gating flipped it


def test_rubric_with_no_reliable_judge_is_flagged_for_review():
    # Every judge below threshold -> rubric flagged, falls back to majority.
    rubric_sets = [{"r1": ("PASS", 2, None)}, {"r1": ("PASS", 2, None)}]
    reliab = {"A": 0.1, "B": 0.2}
    verdicts, _weights, flagged = reliability_consensus.reliability_gated_consensus(
        rubric_sets, ["A", "B"], reliab, threshold=0.5
    )
    assert verdicts["r1"] == "PASS"  # majority fallback
    assert flagged == 1              # flagged for human review
