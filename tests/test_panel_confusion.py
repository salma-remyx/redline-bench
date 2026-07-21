"""Integration test for the per-judge directional-confusion layer.

Exercises the wiring at ``panel.main()`` (a NON-NEW module): builds three
judge output trees plus a reference ("gold") tree on disk, runs the panel
CLI through its public entry point, and asserts the new
``sensitivity_confusion`` summary breaks each judge's scalar score apart
into the paper's directional metrics (FPR / FNR / pass-rate drift /
kappa). Two judges engineered to tie on raw verdict count must be told
apart by *direction*: one over-credits, one over-rejects.

The directional-confusion mechanism is adapted from arXiv:2607.08700
("Do You Need a Frontier Model as a Citation Verifier? Benchmarking
Rubric LLMs for Deep-Research Source Attribution"); gold here is the
designated --reference judge (or leave-one-out panel consensus), the
target-native stand-in for the paper's human gold labels.
"""

import importlib
import json
import sys

import pytest

panel = importlib.import_module("panel")

_TASK = "redline-s1-t1-g1a"
_MODELS = ["modelA", "modelB"]
# Reference (gold) verdicts per rubric. Judges below are engineered around it.
_GOLD = {"r1": "PASS", "r2": "PASS", "r3": "FAIL", "r4": "FAIL"}


def _grade(verdicts):
    """Build a grade JSON shaped like ``rejudge`` output (what load_judge reads)."""
    passed = sum(1 for v in verdicts.values() if v == "PASS")
    return {
        "score": {
            "weighted": round(passed / len(verdicts), 4),
            "per_rubric": [
                {"rubric_id": rid, "verdict": v, "weight": 1, "category": "core"}
                for rid, v in verdicts.items()
            ],
        }
    }


def _write_tree(root, per_model_verdicts):
    root.mkdir(parents=True, exist_ok=True)
    for model in _MODELS:
        (root / model).mkdir(parents=True, exist_ok=True)
        (root / model / f"{_TASK}.json").write_text(json.dumps(_grade(per_model_verdicts)))


def _run_panel(monkeypatch, tmp_path, judges, reference=None):
    argv = ["panel", "--out", str(tmp_path / "out")]
    for label, root in judges.items():
        argv += ["--judge", f"{label}={root}"]
    if reference:
        argv += ["--reference", f"{reference[0]}={reference[1]}"]
    monkeypatch.setattr(sys, "argv", argv)
    rc = panel.main()
    assert rc == 0
    return json.loads((tmp_path / "out" / "panel_summary.json").read_text())


def test_confusion_separates_direction_against_reference(monkeypatch, tmp_path):
    """Two judges with identical 2-PASS / 2-FAIL verdicts must be
    distinguished by error direction: `cheap` over-credits (all PASS),
    `strict` over-rejects (all FAIL), `balanced` matches gold exactly."""
    _write_tree(tmp_path / "cheap", {r: "PASS" for r in _GOLD})        # 4 PASS
    _write_tree(tmp_path / "strict", {r: "FAIL" for r in _GOLD})       # 4 FAIL
    _write_tree(tmp_path / "balanced", dict(_GOLD))                    # matches gold
    _write_tree(tmp_path / "gold", dict(_GOLD))

    summary = _run_panel(
        monkeypatch, tmp_path,
        judges={"cheap": tmp_path / "cheap",
                "strict": tmp_path / "strict",
                "balanced": tmp_path / "balanced"},
        reference=("gold", tmp_path / "gold"),
    )

    conf = summary["sensitivity_confusion"]
    assert set(conf) == {"cheap", "strict", "balanced"}

    # cheap: PASS on every rubric -> 2 TP (r1,r2) + 2 FP (r3,r4), no FN.
    assert conf["cheap"]["false_positive_rate"] == 1.0
    assert conf["cheap"]["false_negative_rate"] == 0.0
    assert conf["cheap"]["pass_rate_drift"] == pytest.approx(0.5)
    assert conf["cheap"]["dominant_error"] == "over-crediting"

    # strict: FAIL on every rubric -> 2 FN (r1,r2) + 2 TN (r3,r4), no FP.
    assert conf["strict"]["false_positive_rate"] == 0.0
    assert conf["strict"]["false_negative_rate"] == 1.0
    assert conf["strict"]["pass_rate_drift"] == pytest.approx(-0.5)
    assert conf["strict"]["dominant_error"] == "over-rejecting"

    # balanced: identical to gold -> perfect agreement.
    assert conf["balanced"]["false_positive_rate"] == 0.0
    assert conf["balanced"]["false_negative_rate"] == 0.0
    assert conf["balanced"]["kappa"] == 1.0
    assert conf["balanced"]["dominant_error"] == "balanced"

    # Existing scalar sensitivity layer is untouched.
    assert set(summary["sensitivity_per_judge"]) == {"cheap", "strict", "balanced"}


def test_confusion_falls_back_to_leave_one_out_consensus(monkeypatch, tmp_path):
    """With no --reference, gold is the leave-one-out majority of the other
    judges; the confusion layer still populates with non-empty counts."""
    _write_tree(tmp_path / "cheap", {r: "PASS" for r in _GOLD})
    _write_tree(tmp_path / "strict", {r: "FAIL" for r in _GOLD})
    _write_tree(tmp_path / "balanced", dict(_GOLD))

    summary = _run_panel(
        monkeypatch, tmp_path,
        judges={"cheap": tmp_path / "cheap",
                "strict": tmp_path / "strict",
                "balanced": tmp_path / "balanced"},
    )

    conf = summary["sensitivity_confusion"]
    assert set(conf) == {"cheap", "strict", "balanced"}
    for label, metrics in conf.items():
        assert metrics["n"] > 0
        assert metrics["dominant_error"] in {
            "over-crediting", "over-rejecting", "balanced"}
    # `cheap` rubber-stamps PASS; against any consensus it can only be
    # over-crediting (it never produces a false negative).
    assert conf["cheap"]["false_negative_rate"] == 0.0
