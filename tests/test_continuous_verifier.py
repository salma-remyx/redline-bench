"""Tests for the continuous LLM-as-a-Verifier scoring path.

Exercises the wiring end-to-end through the NON-new modules:
  - ``judging.grade_continuous`` (call-site hook in the existing judge module)
    → ``continuous_verifier`` (the new capability module)
  - ``panel.weighted_score`` consuming the thresholded discrete verdict the
    continuous grade also emits (panel backward-compat).

The LLM call is monkeypatched so no network is involved; the pure logit-math
is tested directly without any monkeypatching.
"""

import importlib

import pytest

continuous_verifier = importlib.import_module("continuous_verifier")
judging = importlib.import_module("judging")
panel = importlib.import_module("panel")


# --- pure logit-math (no LLM, no monkeypatch) ---


def test_expected_score_renormalizes_over_score_tokens_only():
    # "9" dominates; "the" is a non-score token and must be dropped; the
    # renormalized expectation should land near 9/9 == ~1.0.
    s = continuous_verifier.expected_score(
        {"9": -0.05, "8": -2.5, "7": -5.0, "the": -0.01}, scale=10
    )
    assert 0.0 <= s <= 1.0
    assert s > 0.9  # heavy mass on "9"


def test_expected_score_nan_when_no_score_token():
    s = continuous_verifier.expected_score({"yes": -0.1, "no": -0.2}, scale=10)
    assert s != s  # NaN sentinel → caller falls back


def test_expected_score_midpoint_for_balanced_distribution():
    # equal mass on "0" and "9" → 4.5/9 == 0.5
    s = continuous_verifier.expected_score({"0": -0.5, "9": -0.5}, scale=10)
    assert abs(s - 0.5) < 1e-9


# --- criteria decomposition (axis 3) ---


def test_decompose_splits_compound_criterion():
    crit = "Deletes the indemnity cap; inserts a 30-day cure right while preserving the governing-law clause"
    parts = continuous_verifier.decompose_criterion(crit)
    assert len(parts) == 3
    assert "indemnity cap" in parts[0]
    assert "cure right" in parts[1]


def test_decompose_returns_original_when_atomic():
    crit = "Inserts a single sentence"
    assert continuous_verifier.decompose_criterion(crit) == [crit]


# --- continuous aggregation: the weighted-math generalization ---


def _rubrics():
    return [
        {"id": "r1", "weight": 4, "category": "legal", "criteria": "deletes X"},
        {"id": "r2", "weight": 6, "category": "comm", "criteria": "inserts Y"},
        {"id": "r3", "weight": -2, "category": "pen", "criteria": "undesirable edit"},
    ]


def test_continuous_aggregate_matches_weighted_score_generalization():
    grade = continuous_verifier.continuous_aggregate(
        {"r1": 1.0, "r2": 0.5, "r3": 1.0}, _rubrics()
    )
    # numerator = 4*1.0 + 6*0.5 - 2*1.0 = 5.0; denom = 4+6 = 10 → 0.5
    assert abs(grade["continuous_weighted"] - 0.5) < 1e-9
    assert grade["weighted"] == grade["continuous_weighted"]
    by_id = {p["rubric_id"]: p for p in grade["per_rubric"]}
    assert by_id["r1"]["verdict"] == "PASS"   # 1.0 >= 0.5
    assert by_id["r2"]["verdict"] == "PASS"   # 0.5 >= 0.5 threshold
    assert by_id["r3"]["is_penalty"] is True


def test_continuous_aggregate_threshold_verdicts_flow_into_panel_weighted_score():
    # The discrete verdicts the continuous grade emits must be consumable by
    # the EXISTING panel.weighted_score unchanged (backward-compat bridge).
    grade = continuous_verifier.continuous_aggregate(
        {"r1": 1.0, "r2": 0.0, "r3": 0.0}, _rubrics()
    )
    verdicts = {p["rubric_id"]: p["verdict"] for p in grade["per_rubric"]}
    weights = {p["rubric_id"]: p["weight"] for p in grade["per_rubric"]}
    # panel.weighted_score on those thresholded verdicts == r1 only passes → 4/10
    assert abs(panel.weighted_score(verdicts, weights) - 0.4) < 1e-9


# --- integration: judging.grade_continuous drives continuous_verifier ---


class _FakeResp:
    """Minimal stand-in for a litellm completion response with logprobs."""

    def __init__(self, top_logprobs, text):
        entry = type("E", (), {"top_logprobs": [{"token": t, "logprob": lp}
                                                for t, lp in top_logprobs]})()
        msg = type("M", (), {"content": text})()
        choice = type("C", (), {"message": msg,
                                "logprobs": type("L", (), {"content": [entry]})()})()
        self.choices = [choice]


def _fake_completion_factory():
    """Returns a fake litellm.completion that heavily favors the "9" digit."""
    calls = {"n": 0}

    def _completion(*args, **kwargs):
        calls["n"] += 1
        return _FakeResp([("9", -0.02), ("8", -3.0), ("7", -6.0)], "9")

    return _completion, calls


def test_grade_continuous_invokes_verifier_and_returns_compatible_grade(monkeypatch):
    litellm = pytest.importorskip("litellm")
    fake, calls = _fake_completion_factory()
    monkeypatch.setattr(litellm, "completion", fake)

    task = {
        "scenario_id": "s1", "side": "A", "level": 1,
        "rubrics": _rubrics(),
    }
    grade = judging.grade_continuous(
        "fake/judge", task, "annotated doc body", scale=10, repeats=1, decompose=False,
    )

    # Every rubric was scored via one verifier call each.
    assert calls["n"] == len(task["rubrics"])
    # Continuous scores near 1.0 (the fake favors "9"); aggregate therefore ~1.0
    # minus nothing (penalty r3 also scores high → subtracts, capping below 1).
    assert grade["continuous_weighted"] >= 0.7
    # Output shape is compatible with the existing aggregate()/panel reader.
    assert set(grade) >= {"weighted", "continuous_weighted", "per_rubric", "n_pass", "n_total"}
    assert all("continuous" in p and "verdict" in p for p in grade["per_rubric"])
    assert all(p["verdict"] == "PASS" for p in grade["per_rubric"])


def test_grade_continuous_decompose_makes_more_calls(monkeypatch):
    litellm = pytest.importorskip("litellm")
    fake, calls = _fake_completion_factory()
    monkeypatch.setattr(litellm, "completion", fake)

    task = {
        "scenario_id": "s1", "side": "A", "level": 1,
        "rubrics": [
            {"id": "r1", "weight": 5, "category": "legal",
             "criteria": "Deletes clause A; inserts clause B while preserving C"},
        ],
    }
    judging.grade_continuous("fake/judge", task, "doc", decompose=True)
    # decomposition → 3 sub-clauses → 3 verifier calls for that one rubric
    assert calls["n"] == 3
