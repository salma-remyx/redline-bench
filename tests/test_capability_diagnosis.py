"""Tests for the CRAFT-style capability diagnosis.

The first test drives the real call site — ``panel.main()`` (a NON-NEW
module) — end to end over fabricated judge grade trees and asserts the
diagnosis is wired into ``panel_summary.json``. The rest exercise the
clustering / selection logic of ``capability_diagnosis`` directly.
"""

import importlib
import json
import sys

panel = importlib.import_module("panel")
capability_diagnosis = importlib.import_module("capability_diagnosis")


def _rub(rid, verdict, weight, category, criteria, justification=""):
    return {
        "rubric_id": rid,
        "verdict": verdict,
        "weight": weight,
        "is_penalty": weight < 0,
        "category": category,
        "criteria": criteria,
        "justification": justification,
    }


def _grade(per_rubric):
    return {"score": {"weighted": 0.0, "per_rubric": per_rubric}}


def test_panel_main_emits_capability_diagnosis(tmp_path, monkeypatch):
    """Integration: panel.main() must populate panel_summary.json with a
    capability diagnosis that flags the weak Liability capability."""
    judge_dir = tmp_path / "judge"
    model_dir = judge_dir / "weakmodel"
    model_dir.mkdir(parents=True)

    g01a = _grade([
        _rub("r1", "FAIL", 5, "Liability", "Inserts a liability cap",
             "no cap inserted at the indemnification section."),
        _rub("r2", "FAIL", 5, "Liability", "Deletes the indemnity clause",
             "indemnity clause left intact."),
        _rub("r3", "PASS", 5, "Data Protection", "Inserts a DPA reference",
             "DPA reference added."),
    ])
    g01b = _grade([
        _rub("r4", "FAIL", 5, "Liability", "Rejects the liability insertion",
             "opposing insertion not rejected."),
        _rub("r5", "PASS", 5, "Data Protection", "Preserves the data clause",
             "data clause left intact."),
        _rub("r6", "PASS", 5, "Termination", "Inserts a termination notice",
             "notice inserted."),
    ])
    (model_dir / "redline-s1-t1-g01a.json").write_text(json.dumps(g01a))
    (model_dir / "redline-s1-t1-g01b.json").write_text(json.dumps(g01b))

    out = tmp_path / "panel"
    monkeypatch.setattr(
        sys, "argv",
        ["panel", "--judge", f"weakmodel={judge_dir}", "--out", str(out)],
    )
    assert panel.main() == 0

    summary = json.loads((out / "panel_summary.json").read_text())
    diag = summary["capability_diagnosis"]["weakmodel"]
    assert diag["n_rubrics"] == 6

    weak = diag["weak_capabilities"]
    assert weak, "expected at least one weak capability"
    liab = [w for w in weak if w["capability"] == "Liability"]
    assert liab, f"expected Liability as a weak capability; got {[w['capability'] for w in weak]}"
    assert liab[0]["pass_rate"] == 0.0
    assert liab[0]["n_rubrics"] == 3
    assert liab[0]["example_failures"], "seed failure justifications must be populated"


def test_penalty_rubric_inverts_capability_success():
    """A penalty rubric (negative weight) succeeds on FAIL, not PASS."""
    s = capability_diagnosis._capability_success
    assert s("PASS", 5) is True
    assert s("FAIL", 5) is False
    assert s("PASS", -4) is False   # made the undesirable edit -> failure
    assert s("FAIL", -4) is True    # correctly avoided it -> success


def test_diagnose_picks_category_when_subclusters_too_small():
    """When keyword sub-clusters fall below min_support, the failure is
    reported at the category level (CRAFT: clearest supported granularity)."""
    rows = [
        _rub("r1", "FAIL", 5, "Liability", "Inserts a liability cap"),
        _rub("r2", "FAIL", 5, "Liability", "Deletes the indemnity clause"),
        _rub("r3", "FAIL", 5, "Liability", "Rejects the liability insertion"),
        _rub("r4", "PASS", 5, "Data Protection", "Inserts a DPA reference"),
        _rub("r5", "PASS", 5, "Data Protection", "Preserves the data clause"),
        _rub("r6", "PASS", 5, "Data Protection", "Adds a processing term"),
    ]
    out = capability_diagnosis.diagnose_weak_capabilities({"m": rows})["m"]
    weak = out["weak_capabilities"]
    assert [w["capability"] for w in weak] == ["Liability"]
    assert weak[0]["pass_rate"] == 0.0
    assert weak[0]["n_rubrics"] == 3
    assert not any(w["capability"] == "Data Protection" for w in weak)
    assert any(s["capability"] == "Data Protection" for s in out["strongest_capabilities"])


def test_diagnose_picks_keyword_subcluster_when_supported():
    """When a keyword sub-cluster has enough support, the failure is reported
    there rather than at the (not-weak) parent category."""
    rows = [
        _rub(f"r{i}", "FAIL", 5, "Liability", "Inserts a liability cap", f"fail {i}")
        for i in range(4)
    ] + [
        _rub("p1", "PASS", 5, "Liability", "Deletes the insurance endorsement"),
        _rub("p2", "PASS", 5, "Liability", "Rejects the warranty waiver"),
        _rub("p3", "PASS", 5, "Liability", "Preserves the indemnity clause"),
        _rub("p4", "PASS", 5, "Liability", "Accepts the audit right"),
        _rub("p5", "PASS", 5, "Liability", "Maintains the insurance minimum"),
        _rub("p6", "PASS", 5, "Liability", "Retains the subrogation waiver"),
    ]
    out = capability_diagnosis.diagnose_weak_capabilities({"m": rows})["m"]
    weak = out["weak_capabilities"]
    # The 4 failing "liability cap" probes cluster at the keyword level.
    assert any(w["level"] == "keyword" and "liability" in w["capability"] for w in weak), weak
    # The parent Liability category is mostly passing (6/10) -> not reported.
    assert not any(w["level"] == "category" for w in weak), weak
