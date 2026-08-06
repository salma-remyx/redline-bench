"""Integration test for the opt-in judge prompt-design sensitivity
diagnostic wired into ``metrics_summary``.

Re-judges a fixture trial through the shared ``judging.call_judge``
chokepoint (a NON-NEW module) with ``litellm`` faked via ``sys.modules``,
then asserts the per-axis adherence-delta diagnostics that
``metrics_summary.run`` writes into the summary JSON reflect those faked
verdicts. The diagnostic is adapted from VeyraBench (arXiv:2607.19257).
"""

import importlib
import json
import re
import sys
import types

judging = importlib.import_module("judging")
metrics_summary = importlib.import_module("metrics_summary")

_GRADE = {
    "task_id": 1, "model": "modelA", "scenario_id": 1, "side": "A", "level": 1,
    "gate": {"passed": True},
    "score": {
        "weighted": 1.0, "n_pass": 2, "n_total": 2,
        "per_rubric": [
            {"rubric_id": "r1", "verdict": "PASS", "weight": 5,
             "category": "edits", "criteria": "Inserts clause X.",
             "justification": ""},
            {"rubric_id": "r2", "verdict": "PASS", "weight": 5,
             "category": "edits", "criteria": "Deletes clause Y.",
             "justification": ""},
        ],
    },
}

_VIEW = (
    "[p-001] This contract contains clause X and clause Y.\n"
    "++X++ inserted here.\n"
    "~~Y~~ deleted here.\n"
)


def _install_litellm(monkeypatch):
    """Fake judge: the canonical system prompt with full context => all
    PASS; any prompt-design perturbation (reformatted / instruction-thinned
    / context-truncated) => all FAIL. That deterministic split is exactly
    the adherence delta the diagnostic measures."""
    def completion(**kw):
        msgs = {m["role"]: m["content"] for m in kw["messages"]}
        canonical = (
            msgs["system"] == judging.JUDGE_SYSTEM_PROMPT
            and "[context-truncated]" not in msgs["user"]
        )
        verdict = "PASS" if canonical else "FAIL"
        ids = re.findall(r"- id: `([^`]+)`", msgs["user"])
        content = json.dumps({"verdicts": [
            {"rubric_id": i, "verdict": verdict, "justification": "fake"}
            for i in ids
        ]})
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=content))])

    fake = types.ModuleType("litellm")
    fake.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake)


def _make_runs(tmp_path):
    runs = tmp_path / "runs"
    trial = runs / "trajectories" / "modelA" / "redline-s1-t1-g1a"
    (trial / "verifier").mkdir(parents=True)
    (trial / "grade.json").write_text(json.dumps(_GRADE))
    (trial / "verifier" / "annotated_view.md").write_text(_VIEW)
    return runs


def test_prompt_sensitivity_wired_into_run(monkeypatch, tmp_path):
    _install_litellm(monkeypatch)
    runs = _make_runs(tmp_path)
    out = tmp_path / "summary.json"

    rc = metrics_summary.run(
        runs=runs, out=out, benchmark_dir=tmp_path,
        judge_method="single", prompt_sensitivity=4,
        prompt_sensitivity_judge="fake/judge",
    )

    assert rc == 0
    data = json.loads(out.read_text())
    ps = data["prompt_sensitivity"]

    # Canonical prompt => all PASS => baseline score 1.0.
    assert ps["mean_baseline_score"] == 1.0
    assert ps["analyzed_trials"] == 1
    assert ps["judge_model"] == "fake/judge"
    # Every perturbation flips both rubrics => max fragility on each axis.
    for axis in ("format", "instruction_count", "context_length"):
        assert ps["axes"][axis]["mean_flip_rate"] == 1.0, axis
        assert ps["axes"][axis]["mean_abs_score_delta"] == 1.0, axis
    # The re-judging actually went through judging.call_judge (litellm).
    assert "plain_text" in ps["axes"]["format"]["by_variant"]
    assert "50pct" in ps["axes"]["context_length"]["by_variant"]


def test_prompt_sensitivity_off_by_default(tmp_path):
    runs = _make_runs(tmp_path)
    out = tmp_path / "summary.json"

    rc = metrics_summary.run(
        runs=runs, out=out, benchmark_dir=tmp_path, judge_method="single",
    )

    assert rc == 0
    assert "prompt_sensitivity" not in json.loads(out.read_text())
