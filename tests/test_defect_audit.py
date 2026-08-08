"""Integration tests for the content-level benchmark-defect audit.

Populates the SQLite trail through the real chokepoint
(``judging.call_judge`` — a NON-NEW module) with ``litellm`` faked via
``sys.modules``, then asserts ``defect_audit.detect_defects`` flags both
defect classes adapted from SciCode-Verified (arXiv:2608.04975): the
mechanical parse-mismatch (an ok-recorded call whose raw response does not
parse) and the content-level contradictory-spec class (a rubric the panel
splits on, whose minority PASS is a wrongly-rejected flip candidate).
"""

import importlib
import sys
import types

judging = importlib.import_module("judging")
judge_audit = importlib.import_module("judge_audit")
defect_audit = importlib.import_module("defect_audit")

_PASS = (
    '{"verdicts": [{"rubric_id": "r1", "verdict": "PASS", '
    '"justification": "edits the right clause."}]}'
)
_FAIL = (
    '{"verdicts": [{"rubric_id": "r1", "verdict": "FAIL", '
    '"justification": "misses the right clause."}]}'
)


def _install_litellm(monkeypatch, payload):
    fake = types.ModuleType("litellm")
    fake.completion = lambda **kw: types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=payload))]
    )
    monkeypatch.setitem(sys.modules, "litellm", fake)


def test_flags_parse_mismatch_and_contradictory_spec(monkeypatch, tmp_path):
    db = tmp_path / "audit.sqlite3"
    monkeypatch.setenv(judge_audit.AUDIT_DB_ENV, str(db))

    user = "USER"  # same prompt => same task-turn => same panel
    # Two panel judges that DISAGREE on r1 -> contradictory_spec + a flip.
    _install_litellm(monkeypatch, _PASS)
    assert judging.call_judge("judge-a", "SYS", user)["verdicts"][0]["verdict"] == "PASS"
    _install_litellm(monkeypatch, _FAIL)
    assert judging.call_judge("judge-b", "SYS", user)["verdicts"][0]["verdict"] == "FAIL"

    # A third call the trail recorded as ok but whose raw response is garbage
    # -> the mechanical parse-mismatch defect class.
    judge_audit.log_judge_call(
        model="judge-c", system="SYS", user=user,
        raw_response="not json at all", attempts=1, latency_ms=5.0, ok=True,
    )

    report = defect_audit.detect_defects(str(db))
    by_class = report["by_class"]
    assert by_class[defect_audit.PARSE_MISMATCH] == 1
    assert by_class[defect_audit.CONTRADICTORY_SPEC] == 1
    # Majority FAIL with a dissenting PASS -> wrongly-rejected flip candidate.
    assert report["regrade"]["wrongly_rejected"] == 1
    assert report["n_flip_candidates"] == 1
    assert any(
        f["direction"] == "fail_to_pass" and f["rubric_id"] == "r1"
        for f in report["flips"]
    )

    line = defect_audit.format_defects(report)
    assert "1 contradictory-spec" in line
    assert "1 wrongly-rejected" in line


def test_no_defects_when_panel_agrees(monkeypatch, tmp_path):
    db = tmp_path / "audit.sqlite3"
    monkeypatch.setenv(judge_audit.AUDIT_DB_ENV, str(db))

    user = "USER"
    _install_litellm(monkeypatch, _PASS)
    judging.call_judge("judge-a", "SYS", user)
    judging.call_judge("judge-b", "SYS", user)

    report = defect_audit.detect_defects(str(db))
    assert report["n_defects"] == 0
    assert report["n_flip_candidates"] == 0
    assert report["regrade"]["wrongly_rejected"] == 0


def test_empty_db_audits_cleanly(tmp_path):
    # A configured-but-empty DB file must audit cleanly, not raise.
    db = tmp_path / "empty.sqlite3"
    db.touch()
    report = defect_audit.detect_defects(str(db))
    assert report["n_calls"] == 0
    assert report["n_defects"] == 0
