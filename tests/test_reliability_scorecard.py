"""Integration + unit tests for the CM-LRS reliability scorecard.

Exercises the wiring at two NON-NEW modules:

  * ``judging.call_judge_raw`` — the shared judge chokepoint that
    ``reliability_scorecard.score_reliability`` funnels through (so the
    opt-in audit trail covers it like any other judge call).
  * ``rejudge.regrade_one`` — the re-judge call site that, with
    ``reliability=True``, produces a per-trial scorecard alongside the
    existing verdict score.

``litellm`` is faked via ``sys.modules`` so no network call is made.
"""

import importlib
import json
import sys
import types

reliability_scorecard = importlib.import_module("reliability_scorecard")
rejudge = importlib.import_module("rejudge")


def _fake_response(payload: str):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=payload))]
    )


def _install_litellm(monkeypatch, completion):
    """Install a fake `litellm` module (call_judge_raw imports it locally)."""
    fake = types.ModuleType("litellm")
    fake.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake)


_RATINGS_PAYLOAD = json.dumps({
    "ratings": [
        {"dimension_id": "D1", "score": 5, "justification": "edits match source text at p-012."},
        {"dimension_id": "D2", "score": 4, "justification": "anchors cite [p-012] and §16.1."},
        {"dimension_id": "D3", "score": 5, "justification": "30-day cure period unchanged."},
        {"dimension_id": "D4", "score": 3, "justification": "indemnity edit present, cap missing."},
        {"dimension_id": "D5", "score": 5, "justification": "no invented provisions."},
        {"dimension_id": "D6", "score": 4, "justification": "appropriately aggressive for turn 1."},
        {"dimension_id": "D7", "score": 4, "justification": "rationale comment on each edit."},
    ]
})

_TASK = {
    "scenario_id": 1, "side": "A", "level": 1,
    "rubrics": [{"id": "r1", "criteria": "Inserts a 30-day cure right.", "weight": 5}],
}


# ─── unit: aggregation ──────────────────────────────────────────────


def test_aggregate_equal_weighted_mean():
    sc = reliability_scorecard.aggregate_reliability(json.loads(_RATINGS_PAYLOAD)["ratings"])
    # Equal weights → plain mean of [5,4,5,3,5,4,4] = 30/7.
    assert sc["scale"] == 5
    assert sc["aggregate"] == round(30 / 7, 4)
    assert len(sc["dimensions"]) == 7
    by_id = {d["dimension_id"]: d for d in sc["dimensions"]}
    assert by_id["D1"]["score"] == 5.0
    assert by_id["D4"]["score"] == 3.0
    assert by_id["D7"]["weight"] == 1.0


def test_aggregate_tunable_weights_and_clamp():
    ratings = json.loads(_RATINGS_PAYLOAD)["ratings"]
    # Up-weight D4 (the weak dimension, score 3) so the aggregate drops.
    sc = reliability_scorecard.aggregate_reliability(
        ratings, weights={"D4": 10.0},
    )
    flat = reliability_scorecard.aggregate_reliability(ratings)
    assert sc["aggregate"] < flat["aggregate"]
    by_id = {d["dimension_id"]: d for d in sc["dimensions"]}
    assert by_id["D4"]["weight"] == 10.0
    assert by_id["D1"]["weight"] == 1.0  # unspecified dims default to 1.0

    # Out-of-range and non-numeric scores are clamped/treated as 0, never NaN.
    bad = reliability_scorecard.aggregate_reliability([
        {"dimension_id": "D1", "score": 99},
        {"dimension_id": "D2", "score": "garbage"},
    ])
    d1, d2 = bad["dimensions"][0], bad["dimensions"][1]
    assert d1["score"] == 5.0  # clamped to scale
    assert d2["score"] == 0.0  # non-numeric → 0
    assert 0.0 <= bad["aggregate"] <= 5.0


# ─── integration: score_reliability through the judging chokepoint ──


def test_score_reliability_uses_call_judge_raw(monkeypatch):
    _install_litellm(monkeypatch, lambda **kw: _fake_response(_RATINGS_PAYLOAD))

    sc = reliability_scorecard.score_reliability("fake-judge", _TASK, "annotated body")

    # Goes through judging (NON-NEW module) → same aggregate as the unit test.
    assert sc["aggregate"] == round(30 / 7, 4)
    assert len(sc["dimensions"]) == 7
    assert sc["dimensions"][0]["name"] == "factual_accuracy"


# ─── integration: regrade_one writes the scorecard when asked ───────


def _verdict_payload():
    return json.dumps({"verdicts": [
        {"rubric_id": "r1", "verdict": "PASS", "justification": "inserts the cure right."}
    ]})


def _branching_litellm():
    """Return verdicts for the verdict judge, ratings for the reliability judge,
    branching on the system prompt (the reliability prompt says 'RELIABILITY')."""
    def _completion(**kw):
        system = kw["messages"][0]["content"]
        payload = _RATINGS_PAYLOAD if "RELIABILITY" in system else _verdict_payload()
        return _fake_response(payload)
    return _completion


def _make_trial(tmp_path):
    trial = tmp_path / "redline-s1-t1-g01a__run0"
    (trial / "verifier").mkdir(parents=True)
    (trial / "verifier" / "annotated_view.md").write_text("# annotated redline\n++30-day cure++ at [p-012]")
    grade = {
        "task_id": "redline-s1-t1-g01a", "scenario_id": 1, "side": "A", "level": 1,
        "gate": {"passed": True},
        "score": {"per_rubric": [
            {"rubric_id": "r1", "criteria": "Inserts a 30-day cure right.",
             "weight": 5, "category": "Legal correctness"}
        ]},
    }
    (trial / "verifier" / "grade.json").write_text(json.dumps(grade))
    return trial


def test_regrade_one_with_reliability_writes_scorecard(monkeypatch, tmp_path):
    _install_litellm(monkeypatch, _branching_litellm())
    trial = _make_trial(tmp_path)
    out_dir = tmp_path / "out"

    status = rejudge.regrade_one(
        trial, "fake-judge", out_dir, "test-model", reliability=True,
    )

    assert status == "graded"
    written = json.loads((out_dir / "test-model" / "redline-s1-t1-g01a.json").read_text())
    assert written["score"]["weighted"] == 1.0  # verdict judge still ran
    assert "reliability" in written
    rel = written["reliability"]
    assert rel["scale"] == 5
    assert 0.0 <= rel["aggregate"] <= 5.0
    assert len(rel["dimensions"]) == 7


def test_regrade_one_default_omits_reliability(monkeypatch, tmp_path):
    """Default regrade (no --reliability) is byte-for-byte unchanged: no
    extra judge call, no 'reliability' key."""
    calls = {"n": 0}

    def _completion(**kw):
        calls["n"] += 1
        return _fake_response(_verdict_payload())

    _install_litellm(monkeypatch, _completion)
    trial = _make_trial(tmp_path)
    out_dir = tmp_path / "out"

    rejudge.regrade_one(trial, "fake-judge", out_dir, "test-model")

    written = json.loads((out_dir / "test-model" / "redline-s1-t1-g01a.json").read_text())
    assert "reliability" not in written
    assert calls["n"] == 1  # only the verdict judge ran
