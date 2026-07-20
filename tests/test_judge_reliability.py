"""Integration test for the judge reliability audit wired into panel.main.

Exercises the wiring at ``panel.main`` (the NON-NEW call site): with three
judge grade trees planted on disk, running the panel CLI must emit a
``judge_reliability_audit`` block in ``panel_summary.json`` whose score-drift,
rubric-flip, and error-dependence probes reflect the planted judge
disagreements. The audit runs over the stored grade trees the panel already
loads -- no LLM call is made.
"""

import importlib
import json
import sys
from pathlib import Path

panel = importlib.import_module("panel")

_TASK = "redline-s1-t1-g01a"

# rubric_id -> category; two categories so flips are sliceable.
_RUBRICS = [
    ("r1", "substance"),
    ("r2", "form"),
    ("r3", "substance"),
    ("r4", "form"),
]

# judge label -> per-rubric verdicts + stored weighted score. B and C grade
# identically (and more generously than A) so their error patterns coincide.
_JUDGES = {
    "A": {"r1": "PASS", "r2": "PASS", "r3": "FAIL", "r4": "PASS", "w": 0.4},
    "B": {"r1": "PASS", "r2": "FAIL", "r3": "PASS", "r4": "PASS", "w": 0.7},
    "C": {"r1": "PASS", "r2": "FAIL", "r3": "PASS", "r4": "PASS", "w": 0.5},
}


def _write_tree(root: Path, label: str, model: str) -> Path:
    d = root / label / model
    d.mkdir(parents=True)
    v = _JUDGES[label]
    grade = {
        "score": {
            "weighted": v["w"],
            "per_rubric": [
                {"rubric_id": rid, "verdict": v[rid], "weight": 2, "category": cat}
                for rid, cat in _RUBRICS
            ],
        }
    }
    (d / f"{_TASK}.json").write_text(json.dumps(grade))
    return root / label


def test_panel_main_emits_reliability_audit(tmp_path, monkeypatch):
    roots = [_write_tree(tmp_path, lbl, "modelA") for lbl in _JUDGES]
    out = tmp_path / "panel"
    monkeypatch.setattr(sys, "argv", [
        "panel",
        f"--judge=A={roots[0]}",
        f"--judge=B={roots[1]}",
        f"--judge=C={roots[2]}",
        f"--out={out}",
    ])

    rc = panel.main()
    assert rc == 0

    summary = json.loads((out / "panel_summary.json").read_text())
    audit = summary["judge_reliability_audit"]
    assert audit["n_judges"] == 3
    assert audit["n_pairs"] == 1

    # Score drift: B's stored weighted score (0.7) minus A's (0.4) == 0.3.
    drift = audit["score_drift"]
    assert drift["A vs B"]["mean_abs"] == 0.3
    assert drift["A vs B"]["n"] == 1

    # Rubric flips: A and B disagree on r2 (form) and r3 (substance) => 2/4.
    flips = audit["rubric_flips"]
    assert flips["A vs B"]["overall"] == 0.5
    assert set(flips["A vs B"]["by_category"]) == {"substance", "form"}

    # Error dependence: B and C grade identically, so their leave-one-out
    # disagreement patterns correlate perfectly; overall mean is positive.
    ed = audit["error_dependence"]
    assert ed["matrix"]["B vs C"] == 1.0
    assert ed["mean"] is not None and ed["mean"] > 0


def test_audit_degenerate_panel_is_empty(tmp_path, monkeypatch):
    """A single judge cannot drift or correlate -- the audit must not crash."""
    root = _write_tree(tmp_path, "A", "modelA")
    out = tmp_path / "panel"
    monkeypatch.setattr(sys, "argv", [
        "panel",
        f"--judge=A={root}",
        f"--out={out}",
    ])

    rc = panel.main()
    assert rc == 0
    audit = json.loads((out / "panel_summary.json").read_text())["judge_reliability_audit"]
    assert audit["score_drift"] == {}
    assert audit["error_dependence"]["mean"] is None
