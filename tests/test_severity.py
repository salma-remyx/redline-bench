"""Tests for the action-graded severity scale (``src/severity.py``).

Two integration angles, both reaching NON-NEW modules:

  1. ``severity.grade_severity`` grades the REAL output of
     ``judging.aggregate`` — the module the spec names as the data
     source. A high-weight omission, a triggered penalty, and a plain
     pass must land on distinct levels.
  2. ``metrics_summary.run`` (the call site) emits a ``severity`` block
     computed from a synthetic runs tree, proving the wiring edit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Both light: judging only imports litellm lazily inside call_judge, and
# severity is pure stdlib.
judging = __import__("judging")
severity = __import__("severity")


def _rubrics() -> list[dict]:
    return [
        {"id": "r_pass", "weight": 8, "category": "Legal correctness",
         "criteria": "Inserts the 30-day cure right."},
        {"id": "r_minor", "weight": 2, "category": "Deal-closing orientation",
         "criteria": "Tightens governing-law phrasing."},
        {"id": "r_mid", "weight": 5, "category": "Negotiation quality",
         "criteria": "Narrows the warranty disclaimer."},
        {"id": "r_high", "weight": 6, "category": "Legal correctness",
         "criteria": "Caps the indemnity."},
        {"id": "r_critical", "weight": 9, "category": "Legal correctness",
         "criteria": "Deletes the uncapped indemnity."},
        {"id": "r_penalty_minor", "weight": -3,
         "category": "Negotiation quality",
         "criteria": "Strikes a minor definition (undesirable)."},
        {"id": "r_penalty_critical", "weight": -7,
         "category": "Legal correctness",
         "criteria": "Strikes the entire limitation-of-liability section."},
    ]


def test_grade_severity_maps_aggregate_output():
    """grade_severity consumes judging.aggregate()'s real per_rubric."""
    verdicts = [
        {"rubric_id": "r_pass", "verdict": "PASS", "justification": "ok"},
        {"rubric_id": "r_minor", "verdict": "FAIL", "justification": "missed"},
        {"rubric_id": "r_mid", "verdict": "FAIL", "justification": "missed"},
        {"rubric_id": "r_high", "verdict": "FAIL", "justification": "missed"},
        {"rubric_id": "r_critical", "verdict": "FAIL", "justification": "missed"},
        # Model made the undesirable edit on both penalty rubrics:
        {"rubric_id": "r_penalty_minor", "verdict": "PASS",
         "justification": "did the bad edit"},
        {"rubric_id": "r_penalty_critical", "verdict": "PASS",
         "justification": "did the bad edit"},
    ]
    agg = judging.aggregate(verdicts, _rubrics())
    by_id = {p["rubric_id"]: p for p in agg["per_rubric"]}

    # aggregate() must have flagged the negative-weight rubrics as penalties.
    assert by_id["r_penalty_critical"]["is_penalty"] is True

    assert severity.grade_severity(by_id["r_pass"]) == 0            # satisfied
    assert severity.grade_severity(by_id["r_minor"]) == 1           # |w|=2
    assert severity.grade_severity(by_id["r_mid"]) == 2            # |w|=5
    assert severity.grade_severity(by_id["r_high"]) == 3           # |w|=6
    assert severity.grade_severity(by_id["r_critical"]) == 4       # |w|=9
    # Penalty PASS = model made the undesirable edit: |w|=3 -> L3, |w|=7 -> L4.
    assert severity.grade_severity(by_id["r_penalty_minor"]) == 3
    assert severity.grade_severity(by_id["r_penalty_critical"]) == 4
    # A penalty correctly avoided (FAIL) is no harm.
    avoided_agg = judging.aggregate(
        [{"rubric_id": "r_penalty_critical", "verdict": "FAIL",
          "justification": "avoided"}], _rubrics(),
    )
    avoided = {p["rubric_id"]: p for p in avoided_agg["per_rubric"]}["r_penalty_critical"]
    assert severity.grade_severity(avoided) == 0


def _write_grade(path: Path, per_rubric: list[dict], *, weighted: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "side": "A", "task_id": "t1",
        "score": {"weighted": weighted, "per_rubric": per_rubric},
    }))


def test_metrics_summary_emits_severity_block(tmp_path):
    """The call site (metrics_summary.run) wires severity into its output."""
    try:
        metrics_summary = __import__("metrics_summary")
    except ImportError as exc:  # optional heavy deps (lxml, huggingface_hub)
        pytest.skip(f"metrics_summary deps unavailable: {exc}")

    runs = tmp_path / "runs"
    _write_grade(
        runs / "trajectories" / "modelA" / "redline-s1-t1-g01a" / "grade.json",
        [
            {"rubric_id": "r1", "verdict": "PASS", "weight": 8,
             "is_penalty": False, "category": "Legal correctness",
             "criteria": "x"},
            {"rubric_id": "r2", "verdict": "FAIL", "weight": 9,
             "is_penalty": False, "category": "Legal correctness",
             "criteria": "y"},
            {"rubric_id": "r3", "verdict": "FAIL", "weight": 2,
             "is_penalty": False, "category": "Deal-closing orientation",
             "criteria": "z"},
        ],
    )
    out = tmp_path / "metrics_summary.json"
    rc = metrics_summary.run(
        runs=runs, out=out, benchmark_dir=tmp_path, judge_method="single",
    )
    assert rc == 0

    data = json.loads(out.read_text())
    sev = data["severity"]
    # 1 PASS -> L0, one |w|=9 FAIL -> L4, one |w|=2 FAIL -> L1.
    assert sev["levels"]["L0"] == 1
    assert sev["levels"]["L1"] == 1
    assert sev["levels"]["L4"] == 1
    assert sev["n_failures"] == 2
    # The binary metric flattens the L4 and L1 failures into one number;
    # the severity block keeps them apart.
    assert sev["high_severity_failures"]["count"] == 1
    assert sev["high_severity_failures"]["share_of_failures"] == 0.5
    assert "modelA" in sev["by_model"]
    assert sev["by_model"]["modelA"]["levels"]["L4"] == 1
