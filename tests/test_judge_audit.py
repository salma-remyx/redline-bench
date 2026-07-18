"""Integration tests for the judge-call SQLite audit trail.

Exercises the wiring at ``judging.call_judge`` (the shared judge chokepoint
imported by re-judging / panel / reproduce): with the audit env var set, a
judge call must (a) still return its parsed verdict dict unchanged and
(b) append a row to the SQLite audit log capturing the raw response, attempt
count, and outcome. With the env var unset, no DB is created. ``litellm`` is
faked via ``sys.modules`` so no network call is made.
"""

import importlib
import sqlite3
import sys
import types

import pytest

judging = importlib.import_module("judging")
judge_audit = importlib.import_module("judge_audit")

_PAYLAOD = (
    '{"verdicts": [{"rubric_id": "r1", "verdict": "PASS", '
    '"justification": "edits the right clause."}]}'
)
_PARSED = {
    "verdicts": [
        {"rubric_id": "r1", "verdict": "PASS",
         "justification": "edits the right clause."}
    ]
}


def _fake_response(payload: str):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=payload))]
    )


def _install_litellm(monkeypatch, completion):
    """Install a fake `litellm` module (call_judge imports it locally)."""
    fake = types.ModuleType("litellm")
    fake.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake)


def test_call_judge_returns_verdict_and_audits(monkeypatch, tmp_path):
    _install_litellm(monkeypatch, lambda **kw: _fake_response(_PAYLAOD))
    db = tmp_path / "audit.sqlite3"
    monkeypatch.setenv(judge_audit.AUDIT_DB_ENV, str(db))

    result = judging.call_judge("fake-model", "SYS", "USER")

    assert result == _PARSED
    assert db.exists()
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT model, system_prompt, raw_response, attempts, ok, error "
        "FROM judge_calls"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    model, system, raw, attempts, ok, error = rows[0]
    assert model == "fake-model"
    assert system == "SYS"
    assert raw == _PAYLAOD
    assert attempts == 1
    assert ok == 1
    assert error is None


def test_audit_disabled_by_default(monkeypatch, tmp_path):
    _install_litellm(monkeypatch, lambda **kw: _fake_response(_PAYLAOD))
    monkeypatch.delenv(judge_audit.AUDIT_DB_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    assert judge_audit.audit_path() is None
    result = judging.call_judge("fake-model", "SYS", "USER")

    # Default judging path unchanged: no audit DB materializes anywhere.
    assert result == _PARSED
    assert not list(tmp_path.glob("*.sqlite*"))


def test_failed_call_is_audited_as_not_ok(monkeypatch, tmp_path):
    def _always_fail(**kw):
        raise RuntimeError("transient boom")

    _install_litellm(monkeypatch, _always_fail)
    monkeypatch.setattr(judging, "MAX_RETRIES", 2)
    monkeypatch.setattr(judging.time, "sleep", lambda *_a, **_k: None)
    db = tmp_path / "audit.sqlite3"
    monkeypatch.setenv(judge_audit.AUDIT_DB_ENV, str(db))

    with pytest.raises(RuntimeError):
        judging.call_judge("fake-model", "SYS", "USER")

    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT ok, attempts, error FROM judge_calls").fetchall()
    conn.close()
    assert len(rows) == 1
    ok, attempts, error = rows[0]
    assert ok == 0
    assert attempts == 2
    assert error is not None and "transient boom" in error
