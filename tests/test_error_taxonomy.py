"""Integration + unit tests for the diagnosis-oriented error profile.

The integration test drives the real call site — ``metrics_summary.run``
(a NON-NEW module) — on a minimal single-judge runs dir and asserts the
wired-in ``error_profile`` block is computed and written, exercising the
``diagnose_error_profile`` hook the metrics pipeline now calls. The unit
tests cover the classifier and the coverage math directly.
"""

import importlib
import json

import pytest

error_taxonomy = importlib.import_module("error_taxonomy")


def _grade(per_rubric, *, side="A", weighted=0.5):
    return {
        "side": side,
        "score": {"weighted": weighted, "per_rubric": per_rubric},
    }


# A reward FAIL (liability), a reward PASS, and a triggered penalty
# (PASS on a negative-weight rubric). Together they exercise the two
# error classes the taxonomy diagnoses.
_PER_RUBRIC = [
    {
        "rubric_id": "r1", "verdict": "FAIL", "weight": 8,
        "is_penalty": False, "category": "Legal correctness",
        "criteria": "Did the model add a liability cap?",
    },
    {
        "rubric_id": "r2", "verdict": "PASS", "weight": 5,
        "is_penalty": False, "category": "Commercial context",
        "criteria": "Keeps the payment terms intact.",
    },
    {
        "rubric_id": "r3", "verdict": "PASS", "weight": -6,
        "is_penalty": True, "category": "Negotiation quality",
        "criteria": "Penalty: over-redlines immaterial issues.",
    },
]


def test_run_emits_error_profile(tmp_path):
    """The wiring edit: run() computes + writes the error_profile block."""
    # metrics_summary pulls in docx_metrics (lxml) — skip if heavy deps
    # are absent, mirroring test_smoke_imports.py's HEAVY convention.
    metrics_summary = pytest.importorskip("metrics_summary")
    runs = tmp_path / "runs"
    grade_dir = runs / "trajectories" / "mymodel" / "redline-s1-t1-g01a"
    grade_dir.mkdir(parents=True)
    (grade_dir / "grade.json").write_text(json.dumps(_grade(_PER_RUBRIC)))

    out = tmp_path / "metrics_summary.json"
    rc = metrics_summary.run(
        runs=runs, out=out, benchmark_dir=tmp_path, judge_method="single",
    )
    assert rc == 0

    data = json.loads(out.read_text())
    assert "error_profile" in data

    prof = data["error_profile"]["by_model"]["mymodel"]
    # One reward FAIL + one triggered penalty = two diagnosed errors.
    assert prof["n_reward_fail"] == 1
    assert prof["n_penalty_triggered"] == 1
    assert prof["n_errors"] == 2
    assert prof["error_weight"] == 14  # |8| + |-6|

    # Leaf hierarchy: liability FAIL under Legal correctness, triggered
    # penalty under Negotiation quality.
    leaf = prof["by_dimension_error_type"]
    assert leaf["Legal correctness::liability_indemnity"] == 1
    assert leaf["Negotiation quality::over_aggression"] == 1

    # Task had errors, so coverage is 0; field-wide pool reflects both.
    assert prof["task_coverage_clean"] == 0.0
    assert data["error_profile"]["overall"]["by_dimension"]["Legal correctness"] == 1


def test_classify_penalty_always_over_aggression():
    dim, etype = error_taxonomy.classify_rubric(
        {"is_penalty": True, "category": "Deal-closing orientation",
         "criteria": "anything"}
    )
    assert dim == "Deal-closing orientation"
    assert etype == "over_aggression"


def test_unknown_category_kept_verbatim():
    dim, _ = error_taxonomy.classify_rubric(
        {"is_penalty": False, "category": "Exotic new dimension",
         "criteria": "did the model do X"}
    )
    assert dim == "Exotic new dimension"


def test_coverage_clean_when_no_errors():
    rows = [{
        "model": "m", "_per_rubric": [
            {"verdict": "PASS", "weight": 5, "is_penalty": False,
             "category": "Legal correctness", "criteria": "fine"},
        ],
    }]
    prof = error_taxonomy.diagnose_error_profile({"m": rows})["by_model"]["m"]
    assert prof["n_errors"] == 0
    assert prof["task_coverage_clean"] == 1.0
    assert prof["dominant_error_type"] is None
