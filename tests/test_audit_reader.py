"""Integration tests for the judge-call audit-trail reader.

Populates the SQLite trail through the real chokepoint
(``judging.call_judge`` — a NON-NEW module) with ``litellm`` faked via
``sys.modules``, then asserts ``audit_reader.summarize`` reconstructs the
call counts and — crucially — flags an ``ok``-recorded call whose raw
response does not parse into a valid ``verdicts`` object (the
verdict-format-mismatch class the audit trail exists to expose).
"""

import importlib
import sys
import types

judging = importlib.import_module("judging")
judge_audit = importlib.import_module("judge_audit")
audit_reader = importlib.import_module("audit_reader")

_GOOD = (
    '{"verdicts": [{"rubric_id": "r1", "verdict": "PASS", '
    '"justification": "edits the right clause."}]}'
)


def _install_litellm(monkeypatch, payload):
    fake = types.ModuleType("litellm")
    fake.completion = lambda **kw: types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=payload))]
    )
    monkeypatch.setitem(sys.modules, "litellm", fake)


def test_summarize_counts_and_flags_unparseable(monkeypatch, tmp_path):
    db = tmp_path / "audit.sqlite3"
    monkeypatch.setenv(judge_audit.AUDIT_DB_ENV, str(db))

    # One real, well-formed judge call through the shared chokepoint.
    _install_litellm(monkeypatch, _GOOD)
    assert judging.call_judge("m", "SYS", "USER")["verdicts"][0]["verdict"] == "PASS"

    # A call the trail recorded as ok but whose raw response is malformed —
    # exactly the phantom-ok verdict-format mismatch the reader must surface.
    judge_audit.log_judge_call(
        model="m", system="SYS", user="USER",
        raw_response="not json at all", attempts=1, latency_ms=5.0, ok=True,
    )

    summary = audit_reader.summarize(str(db))
    assert summary["n_calls"] == 2
    assert summary["n_ok"] == 2
    assert summary["n_failed"] == 0
    assert summary["n_unparseable_ok"] == 1
    assert summary["max_attempts"] == 1

    line = audit_reader.format_summary(summary)
    assert "1 ok-but-unparseable" in line


def test_summarize_records_failed_and_retried_calls(monkeypatch, tmp_path):
    db = tmp_path / "audit.sqlite3"
    monkeypatch.setenv(judge_audit.AUDIT_DB_ENV, str(db))

    def _always_fail(**kw):
        raise RuntimeError("boom")

    fake = types.ModuleType("litellm")
    fake.completion = _always_fail
    monkeypatch.setitem(sys.modules, "litellm", fake)
    monkeypatch.setattr(judging, "MAX_RETRIES", 3)
    monkeypatch.setattr(judging.time, "sleep", lambda *_a, **_k: None)

    import pytest
    with pytest.raises(RuntimeError):
        judging.call_judge("m", "SYS", "USER")

    summary = audit_reader.summarize(str(db))
    assert summary["n_calls"] == 1
    assert summary["n_ok"] == 0
    assert summary["n_failed"] == 1
    # A terminal failure is logged with attempts == MAX_RETRIES (> 1).
    assert summary["n_retried"] == 1
    assert summary["max_attempts"] == 3


def test_read_rows_empty_when_never_written(tmp_path):
    # A configured-but-empty DB file must summarize cleanly, not raise.
    db = tmp_path / "empty.sqlite3"
    db.touch()
    assert audit_reader.read_rows(str(db)) == []
    assert audit_reader.summarize(str(db))["n_calls"] == 0
