"""Integration tests for the inference-backend provenance disclosure.

Exercises the wiring at the two NON-NEW call sites the disclosure is
stamped onto:

  * ``judge_audit.log_judge_call`` — the audit-trail chokepoint reached
    from ``judging.call_judge`` — must stamp the LiteLLM backend + version
    onto each recorded row (the per-call disclosure arXiv:2608.04714 asks
    for).
  * ``metrics_summary.run`` — the benchmark output builder — must embed a
    top-level ``inference_provenance`` block (judge backend + generation
    config + deterministic flag, plus the agent harness when run via
    ``reproduce``).

The metrics-summary path isolates the disclosure wiring by stubbing the
unrelated grading collectors (same isolation pattern ``test_judge_audit``
uses with its faked ``litellm``).
"""

import importlib
import json
import sqlite3

import pytest

inference_provenance = importlib.import_module("inference_provenance")
judge_audit = importlib.import_module("judge_audit")
metrics_summary = importlib.import_module("metrics_summary")


# ─── pure-function coverage of the disclosure logic ───────────────────


def test_judge_disclosure_shape():
    d = inference_provenance.judge_disclosure()
    assert d["backend"] == "litellm"
    assert "generation_config" in d
    # temperature is deliberately unset → not deterministic (the paper's
    # point: an unset param inherits the backend's default).
    assert d["generation_config"]["temperature"] is None
    assert d["deterministic"] is False
    assert "arXiv:2608.04714" in d["note"]


@pytest.mark.parametrize(
    "cfg, expected",
    [
        ({"temperature": 0}, True),              # greedy → deterministic
        ({"temperature": None}, False),          # unset → backend default
        ({"temperature": 0.7}, False),           # sampled
        ({}, False),                             # nothing pinned
    ],
)
def test_is_deterministic(cfg, expected):
    assert inference_provenance.is_deterministic(cfg) is expected


def test_provider_of_parses_litellm_model_string():
    assert inference_provenance.provider_of("anthropic/claude-opus-4-8") == "anthropic"
    assert inference_provenance.provider_of("claude-opus-4-8") is None
    assert inference_provenance.provider_of(None) is None


def test_backend_version_is_str_or_none():
    # Best-effort: the installed distribution version when litellm is
    # present (e.g. "1.96.2"), None when it isn't. Never crashes.
    v = inference_provenance.backend_version()
    assert v is None or isinstance(v, str)


def test_agent_harness_disclosure_absent_without_context():
    assert inference_provenance.agent_harness_disclosure() is None


def test_benchmark_disclosure_carries_agent_harness():
    d = inference_provenance.benchmark_disclosure(
        judge_method="panel",
        agent="claude-code",
        agent_model="anthropic/claude-opus-4-8",
        harbor_env="modal",
    )
    assert d["judge_method"] == "panel"
    assert d["agent_harness"]["agent"] == "claude-code"
    assert d["agent_harness"]["provider"] == "anthropic"
    assert d["agent_harness"]["env"] == "modal"


# ─── wiring: judge audit trail stamps the backend per call ────────────


def test_log_judge_call_stamps_backend(monkeypatch, tmp_path):
    """log_judge_call (non-new module) must record the LiteLLM backend +
    version on every audited row, and must migrate an old-schema DB."""
    db = tmp_path / "audit.sqlite3"
    monkeypatch.setenv(judge_audit.AUDIT_DB_ENV, str(db))

    judge_audit.log_judge_call(
        model="anthropic/claude-haiku", system="SYS", user="USER",
        raw_response='{"verdicts": []}', attempts=1, latency_ms=12.0, ok=True,
    )

    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT backend, backend_version FROM judge_calls"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    backend, version = rows[0]
    assert backend == "litellm"
    # version is best-effort (str when litellm is installed, None otherwise)
    # but always stamped, and always equal to what the module reports.
    assert version == inference_provenance.backend_version()


def test_log_judge_call_migrates_legacy_schema(monkeypatch, tmp_path):
    """A DB created with the original (pre-backend) schema must be migrated
    in place rather than rejected."""
    db = tmp_path / "audit.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE judge_calls ("
        "ts REAL NOT NULL, model TEXT, system_prompt TEXT, user_prompt TEXT, "
        "raw_response TEXT, attempts INTEGER, latency_ms REAL, "
        "ok INTEGER NOT NULL, error TEXT)"
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv(judge_audit.AUDIT_DB_ENV, str(db))
    judge_audit.log_judge_call(
        model="m", system="S", user="U", raw_response="r",
        attempts=1, latency_ms=1.0, ok=True,
    )

    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT backend, backend_version FROM judge_calls"
    ).fetchone()
    conn.close()
    assert row[0] == "litellm"
    assert row[1] == inference_provenance.backend_version()


# ─── wiring: metrics summary embeds the disclosure block ──────────────


def _stub_leaderboard():
    return [{
        "model": "claude-opus-4-8",
        "overall_turn_weighted": 0.5,
        "best_at_k_turn_weighted": 0.5,
        "ci": [0.0, 1.0],
    }]


def test_metrics_summary_embeds_backend_disclosure(monkeypatch, tmp_path):
    """run() (non-new module) must write an inference_provenance block,
    populated from the agent-harness args, onto the summary JSON."""
    monkeypatch.setattr(
        metrics_summary, "collect_panel_rows",
        lambda *a, **k: [{"stub": True}],
    )
    monkeypatch.setattr(
        metrics_summary, "rows_by_model", lambda trials: {"m": trials},
    )
    monkeypatch.setattr(
        metrics_summary, "build_leaderboard", lambda by_model: _stub_leaderboard(),
    )
    monkeypatch.setattr(
        metrics_summary, "_build_docx_metrics", lambda *a, **k: ({}, {}),
    )

    runs = tmp_path / "runs"
    runs.mkdir()
    out = tmp_path / "out.json"
    rc = metrics_summary.run(
        runs=runs, out=out, benchmark_dir=tmp_path, judge_method="panel",
        agent="claude-code",
        agent_model="anthropic/claude-opus-4-8",
        harbor_env="modal",
    )

    assert rc == 0
    data = json.loads(out.read_text())
    prov = data["inference_provenance"]
    assert prov["judge"]["backend"] == "litellm"
    assert "generation_config" in prov["judge"]
    assert prov["judge"]["deterministic"] is False
    assert prov["agent_harness"]["agent"] == "claude-code"
    assert prov["agent_harness"]["provider"] == "anthropic"
    assert prov["agent_harness"]["env"] == "modal"


def test_metrics_summary_disclosure_without_agent_context(monkeypatch, tmp_path):
    """Standalone summary (no agent info) still discloses the judge backend
    and records the agent harness as absent rather than fabricating it."""
    monkeypatch.setattr(
        metrics_summary, "collect_panel_rows",
        lambda *a, **k: [{"stub": True}],
    )
    monkeypatch.setattr(
        metrics_summary, "rows_by_model", lambda trials: {"m": trials},
    )
    monkeypatch.setattr(
        metrics_summary, "build_leaderboard", lambda by_model: _stub_leaderboard(),
    )
    monkeypatch.setattr(
        metrics_summary, "_build_docx_metrics", lambda *a, **k: ({}, {}),
    )

    runs = tmp_path / "runs"
    runs.mkdir()
    out = tmp_path / "out.json"
    rc = metrics_summary.run(
        runs=runs, out=out, benchmark_dir=tmp_path, judge_method="panel",
    )

    assert rc == 0
    data = json.loads(out.read_text())
    prov = data["inference_provenance"]
    assert prov["judge"]["backend"] == "litellm"
    assert prov["agent_harness"] is None
