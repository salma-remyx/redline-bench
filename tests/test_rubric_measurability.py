"""Integration test for the Bayesian rubric-measurability diagnostic.

Writes a tiny 3-judge / 2-model grade tree to disk in the layout
``panel.load_judge`` reads, drives the real ``panel.main()`` CLI
(``panel`` — a NON-NEW module) over it via ``sys.argv``, and asserts
the resulting ``panel_summary.json`` carries the CalibratedRubric-style
measurability diagnostic — a rubric the panel agrees on is flagged
measurable while one it splits on is flagged low-measurability.
"""

import importlib
import json
import sys

panel = importlib.import_module("panel")
rubric_measurability = importlib.import_module("rubric_measurability")

_TASKS = ["redline-s1-t1-g1a", "redline-s1-t1-g2a"]
_MODELS = ["mA", "mB"]
_JUDGES = ["j0", "j1", "j2"]


def _grade(per_rubric):
    return {"score": {"weighted": 0.0, "per_rubric": per_rubric}}


def _write_tree(root, verdict_fn):
    """Write root/<judge>/<model>/<task>.json — the layout load_judge rglobs."""
    for j in _JUDGES:
        for model in _MODELS:
            mdir = root / j / model
            mdir.mkdir(parents=True, exist_ok=True)
            for task in _TASKS:
                verdicts = verdict_fn(j, model, task)
                per_rubric = [
                    {"rubric_id": rid, "verdict": v, "weight": 1}
                    for rid, v in verdicts.items()
                ]
                (mdir / f"{task}.json").write_text(json.dumps(_grade(per_rubric)))


def test_measurability_flags_split_rubric(tmp_path, monkeypatch):
    judge_root = tmp_path / "judges"
    # r_agree: every judge PASSes everywhere -> pooled (12, 0), highly
    # measurable. r_split: j0/j1 PASS, j2 FAIL -> pooled (8, 4), the
    # panel never stabilises -> low-measurability.
    def verdicts(judge, model, task):
        return {"r_agree": "PASS", "r_split": "PASS" if judge != "j2" else "FAIL"}

    _write_tree(judge_root, verdicts)

    out = tmp_path / "panel"
    argv = ["panel", "--out", str(out)]
    for j in _JUDGES:
        argv += ["--judge", f"{j}={judge_root / j}"]
    monkeypatch.setattr(sys, "argv", argv)
    assert panel.main() == 0

    summary = json.loads((out / "panel_summary.json").read_text())
    meas = summary["rubric_measurability"]
    assert meas["n_rubrics"] == 2
    assert meas["low_measurability_rubrics"] == ["r_split"]

    detail = summary["rubric_measurability_per_rubric"]
    assert detail["r_agree"]["measurable"] is True
    assert detail["r_agree"]["majority"] == "PASS"
    assert detail["r_split"]["measurable"] is False


def test_measurability_threshold_is_configurable(tmp_path, monkeypatch):
    judge_root = tmp_path / "judges"

    def verdicts(judge, model, task):
        return {"r_agree": "PASS"}

    _write_tree(judge_root, verdicts)
    out = tmp_path / "panel"
    argv = ["panel", "--out", str(out), "--measurability-threshold", "0.99"]
    for j in _JUDGES:
        argv += ["--judge", f"{j}={judge_root / j}"]
    monkeypatch.setattr(sys, "argv", argv)
    assert panel.main() == 0

    # Even a unanimous rubric is below an absurd 0.99 bar -> flagged low.
    summary = json.loads((out / "panel_summary.json").read_text())
    assert summary["rubric_measurability"]["low_measurability_rubrics"] == ["r_agree"]


def test_measurability_of_extremes():
    assert rubric_measurability.measurability_of(12, 0)["measurability"] > 0.8
    balanced = rubric_measurability.measurability_of(6, 6)
    assert 0.5 <= balanced["measurability"] < 0.6
    assert balanced["majority"] == "TIE"
