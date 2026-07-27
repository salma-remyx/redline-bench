"""Tests for the LLM severity-judge panel (``src/severity_panel.py``).

Integration angles, each reaching NON-NEW modules:

  1. ``grade_record_with_panel`` routes through the real
     ``judging.call_judge`` chokepoint (litellm faked via
     ``sys.modules``), reduces the judges' ordinal levels by high
     median, and its calls land in the ``judge_audit`` SQLite trail
     readable by ``audit_reader.summarize``.
  2. ``metrics_summary.run(..., severity_panel=True)`` emits a
     ``severity_panel`` block whose Krippendorff alpha matches a
     hand-computed value, while the binary pipeline and the
     deterministic ``severity`` block are untouched.
  3. ``krippendorff_alpha`` unit checks against hand-computed values
     (ordinal and interval metrics).
"""

from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

# Light at import time: judging imports litellm lazily inside
# call_judge; severity / severity_panel are pure stdlib.
judging = importlib.import_module("judging")
judge_audit = importlib.import_module("judge_audit")
audit_reader = importlib.import_module("audit_reader")
severity = importlib.import_module("severity")
severity_panel = importlib.import_module("severity_panel")


# ─── helpers ─────────────────────────────────────────────────────────


def _fake_response(payload: str):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=payload))]
    )


def _install_litellm(monkeypatch, payloads: dict[str, str]):
    """Fake `litellm` so completion() returns a per-model payload."""
    fake = types.ModuleType("litellm")
    fake.completion = lambda **kw: _fake_response(payloads[kw["model"]])
    monkeypatch.setitem(sys.modules, "litellm", fake)


def _payload(level: str) -> str:
    return (
        '{"verdicts": [{"rubric_id": "r1", "verdict": "' + level + '", '
        '"justification": "graded."}]}'
    )


def _record() -> dict:
    return {
        "rubric_id": "r1", "verdict": "FAIL", "weight": 7,
        "is_penalty": False, "category": "Legal correctness",
        "criteria": "Caps the indemnity at the fees paid.",
        "justification": "Identifies the clause but inserts no cap.",
    }


# ─── tag-free serialization ──────────────────────────────────────────


def test_tag_free_account_withholds_oracle_inputs():
    rec = _record()
    account = severity_panel.tag_free_account(rec)
    # The judge sees the criterion, dimension, and outcome...
    assert rec["criteria"] in account
    assert rec["category"] in account
    assert rec["justification"] in account
    assert "does NOT contain" in account
    # ...but nothing the oracle grades from.
    assert str(rec["weight"]) not in account
    assert "weight" not in account.lower()
    assert "penalty" not in account.lower()
    assert "L3" not in account and "L4" not in account


def test_tag_free_account_states_penalty_pass_neutrally():
    """A penalty rubric the model triggered (verdict PASS) must read as
    'edit present', not as a failure — the harm judgment is the
    panel's, not the metadata's."""
    rec = {**_record(), "verdict": "PASS",
           "criteria": "Strikes the entire limitation-of-liability section."}
    account = severity_panel.tag_free_account(rec)
    assert "CONTAINS the edit described" in account
    assert "does NOT contain" not in account


# ─── panel grading through the real chokepoint ───────────────────────


def test_panel_grading_routes_through_call_judge_and_audits(
    monkeypatch, tmp_path
):
    _install_litellm(monkeypatch, {
        "j1": _payload("L2"), "j2": _payload("L4"), "j3": _payload("L3"),
    })
    db = tmp_path / "audit.sqlite3"
    monkeypatch.setenv(judge_audit.AUDIT_DB_ENV, str(db))

    out = severity_panel.grade_record_with_panel(
        _record(), judges=("j1", "j2", "j3"),
    )

    # High median of {2, 4, 3} is 3.
    assert out["judge_levels"] == {"j1": 2, "j2": 4, "j3": 3}
    assert out["panel_level"] == 3
    assert out["panel_label"] == "L3"

    # The three severity-judge calls landed in the shared audit trail,
    # tagged with the severity system prompt.
    summary = audit_reader.summarize(str(db))
    assert summary["n_calls"] == 3
    assert summary["n_ok"] == 3
    conn = sqlite3.connect(db)
    prompts = conn.execute(
        "SELECT DISTINCT system_prompt FROM judge_calls"
    ).fetchall()
    conn.close()
    assert prompts == [(severity_panel.SEVERITY_JUDGE_SYSTEM_PROMPT,)]


def test_panel_grading_rejects_unparseable_level(monkeypatch):
    _install_litellm(monkeypatch, {
        "j1": '{"verdicts": [{"rubric_id": "r1", "verdict": "HIGH", '
              '"justification": "not a level."}]}',
    })
    with pytest.raises(ValueError, match="no L-label verdict"):
        severity_panel.grade_record_with_panel(_record(), judges=("j1",))


# ─── Krippendorff's alpha ────────────────────────────────────────────


def test_alpha_perfect_agreement():
    assert severity_panel.krippendorff_alpha(
        [[0, 0], [2, 2], [4, 4]], metric="ordinal",
    ) == 1.0


def test_alpha_hand_computed_max_disagreement():
    # Units [[1,2],[2,1]]: Do = 1 (interval) with De = 2/3 -> α = -0.5.
    # Ordinal coincides here (n_1 = n_2 = 2 -> δ² = 4, Do = 4, De = 8/3).
    assert severity_panel.krippendorff_alpha(
        [[1, 2], [2, 1]], metric="interval",
    ) == pytest.approx(-0.5)
    assert severity_panel.krippendorff_alpha(
        [[1, 2], [2, 1]], metric="ordinal",
    ) == pytest.approx(-0.5)


def test_alpha_missing_and_undefined():
    # Units with <2 ratings don't contribute; nothing pairable -> None.
    assert severity_panel.krippendorff_alpha([[1, None], [None, 2]]) is None
    # Constant data with perfect agreement -> 1.0.
    assert severity_panel.krippendorff_alpha([[3, 3], [3, 3]]) == 1.0


# ─── metrics_summary integration (the call site) ─────────────────────


def _write_grade(path: Path, per_rubric: list[dict], *, weighted: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "side": "A", "task_id": "t1",
        "score": {"weighted": weighted, "per_rubric": per_rubric},
    }))


def _synthetic_runs(tmp_path: Path) -> Path:
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
    return runs


def test_metrics_summary_emits_severity_panel_block(monkeypatch, tmp_path):
    """The call site (metrics_summary.run) wires the panel into its output."""
    try:
        metrics_summary = __import__("metrics_summary")
    except ImportError as exc:  # optional heavy deps (lxml, huggingface_hub)
        pytest.skip(f"metrics_summary deps unavailable: {exc}")

    payload = _payload("L4")
    _install_litellm(monkeypatch, {m: payload for m in ("j1", "j2", "j3")})
    out = tmp_path / "metrics_summary.json"
    rc = metrics_summary.run(
        runs=_synthetic_runs(tmp_path), out=out, benchmark_dir=tmp_path,
        judge_method="single", severity_panel=True,
        severity_judges=["j1", "j2", "j3"],
    )
    assert rc == 0

    data = json.loads(out.read_text())
    sp = data["severity_panel"]
    assert sp["judges"] == ["j1", "j2", "j3"]
    # Two oracle-flagged failures (L4 and L1); the PASS stays unjudged.
    assert sp["n_failures_judged"] == 2
    assert [r["oracle_level"] for r in sp["records"]] == [4, 1]
    assert [r["panel_level"] for r in sp["records"]] == [4, 4]
    # Hand-computed: units [[4,4],[1,4]], ordinal metric (n_1=1, n_4=3,
    # δ²(1,4) = 4) -> Do = 2, De = 2 -> α = 0.0.
    assert sp["krippendorff_alpha_ordinal"] == pytest.approx(0.0)
    assert sp["exact_agreement"] == 0.5
    assert sp["within_one_level"] == 0.5

    # The binary pipeline and deterministic oracle block are untouched.
    assert data["severity"]["n_failures"] == 2
    assert data["severity"]["levels"]["L4"] == 1
    assert data["leaderboard"][0]["model"] == "modelA"


def test_metrics_summary_severity_panel_off_by_default(tmp_path):
    """Without --severity-panel, no LLM calls and a null block."""
    try:
        metrics_summary = __import__("metrics_summary")
    except ImportError as exc:
        pytest.skip(f"metrics_summary deps unavailable: {exc}")

    out = tmp_path / "metrics_summary.json"
    rc = metrics_summary.run(
        runs=_synthetic_runs(tmp_path), out=out, benchmark_dir=tmp_path,
        judge_method="single",
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["severity_panel"] is None
    # Deterministic oracle still computed.
    assert data["severity"]["n_failures"] == 2
