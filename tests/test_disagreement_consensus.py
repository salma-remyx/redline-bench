"""Integration tests for the disagreement-aware consensus diagnostics.

Exercises the wiring in the NON-NEW module ``panel``: two fake judge
trees (3 rubrics, 3 judges) are written to disk, run through the real
``panel.main()`` CLI, and the test asserts the new
``disagreement_consensus`` block lands in ``panel_summary.json`` with
the dispersion math VERDICT predicts — and that the canonical
majority-vote leaderboard is unchanged by the addition.
"""

import json
import subprocess
import sys
from pathlib import Path


def _grade(per_rubric):
    return {"score": {"weighted": 0.5, "per_rubric": per_rubric}}


def _rub(rid, verdict, weight=5):
    return {"rubric_id": rid, "verdict": verdict, "weight": weight}


def _write_judge(root: Path, label: str, grades):
    d = root / label / "modelA"
    d.mkdir(parents=True)
    for task, grade in grades.items():
        (d / f"{task}.json").write_text(json.dumps(grade))
    return d


def test_panel_main_emits_disagreement_consensus(tmp_path):
    # 3 judges x 3 rubrics on one task. Judge 3 dissents everywhere, so
    # every rubric is a 2-1 split: s = 2/3 for PASS-majority rubrics,
    # s = 1/3 for the FAIL-majority one — |s - 0.5| = 1/6 either way,
    # and all three rubrics are one-vote unstable.
    task = "redline-s1-t1-g1a"
    rub = [
        [_rub("r1", "PASS"), _rub("r2", "PASS"), _rub("r3", "FAIL")],
        [_rub("r1", "PASS"), _rub("r2", "FAIL"), _rub("r3", "FAIL")],
        [_rub("r1", "FAIL"), _rub("r2", "PASS"), _rub("r3", "PASS")],
    ]
    judges = tmp_path / "judges"
    specs = []
    for i, per_rubric in enumerate(rub):
        label = f"j{i}"
        _write_judge(judges, label, {task: _grade(per_rubric)})
        specs.append(f"{label}={judges / label}")

    out = tmp_path / "panel"
    argv = [sys.executable, "-m", "panel"]
    for spec in specs:
        argv += ["--judge", spec]
    res = subprocess.run(
        argv + ["--out", str(out)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr

    summary = json.loads((out / "panel_summary.json").read_text())
    assert "disagreement_consensus" in summary
    stats = summary["disagreement_consensus"]["modelA"]
    assert stats["mean_dispersion"] == round(1 / 6, 4)
    # r1 and r2 are PASS-majority (s = 2/3), r3 is FAIL-majority (s = 1/3).
    assert stats["verdict_pass_fraction"] == round(2 / 3, 4)

    # Majority vote stays canonical: r1/r2 PASS, r3 FAIL -> full score
    # on positive weights, unchanged by the diagnostics block.
    assert summary["panel_leaderboard"]["modelA"] == round(2 / 3, 4)


def test_unanimous_panel_has_max_dispersion(tmp_path):
    # Unanimous judges: s ∈ {0, 1}, |s - 0.5| = 0.5 — maximal distance
    # from the equilibrium, i.e. a maximally stable panel verdict, and
    # no unstable rubrics.
    task = "redline-s1-t1-g1a"
    judges = tmp_path / "judges"
    specs = []
    for i in range(3):
        label = f"j{i}"
        _write_judge(judges, label, {
            task: _grade([_rub("r1", "PASS"), _rub("r2", "FAIL")])
        })
        specs.append(f"{label}={judges / label}")

    out = tmp_path / "panel"
    argv = [sys.executable, "-m", "panel"]
    for spec in specs:
        argv += ["--judge", spec]
    subprocess.run(
        argv + ["--out", str(out)],
        capture_output=True, text=True, check=True,
    )
    stats = json.loads((out / "panel_summary.json").read_text())[
        "disagreement_consensus"
    ]["modelA"]
    assert stats["mean_dispersion"] == 0.5
    assert stats["verdict_pass_fraction"] == 0.5
