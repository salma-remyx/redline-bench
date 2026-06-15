"""Walk a RedlineBench `runs/<run-id>/` directory and emit one row per
graded `(model, task)` trial in the same shape `aggregate.summarize_model`
already consumes.

Two layouts live under `runs/<run-id>/`:

  - `trajectories/<dir>/<task>/grade.json` — per-model traces for any
    model that ran end-to-end in the live sandbox (currently opus48,
    gpt55, gemini35). Each trace dir is one trial; the docx beside the
    grade is `redline.docx`.

  - `archival-fable5/<task>/grade.json` — Claude Fable 5's graded
    rubric verdicts (from the same Harbor run as the other models;
    comparable on the leaderboard). The Harbor run did not preserve
    Fable 5's `.docx` outputs, so the behavioral metrics use docx
    files borrowed from an earlier experiment on the same benchmark
    and stored at `<task>/old_experiment_run/redline.docx`.

The default reader walks only `trajectories/*/` — Fable 5 is opt-in
via `include_fable_5=True` (the CLI exposes this as `--add-fable-5`).
Auto-discovery means: drop a new model's traces into
`trajectories/<some_dir>/` and the reader will pick them up on the
next run; no code changes needed.

This module is the seam between the on-disk layout and the in-memory
trial rows the rest of the report pipeline operates on. If the trace
layout changes, only this file needs to change.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from aggregate import DIAG_KEYS

# `redline-s{S}-t{T}-g{NN}{variant}` — same regex aggregate.py uses, but
# without the `__` job-name suffix since we walk traces directly, not
# Harbor's per-trial directory naming.
_TASK_NAME_RE = re.compile(r"^redline-s(\d+)-t(\d+)-g(\d+)([a-z])$")


# Hard-coded directory-name override for Claude Fable 5. The reference
# trace layout uses a top-level directory name (`archival-fable5`)
# that's intentionally distinct from the actively-run models'
# directories under `trajectories/` — the layout flags Fable 5 as
# a reference run with a different on-disk shape. But that special
# directory name is poor as a model identifier in the report
# ("archival-fable5" doesn't read as a model). Map it here.
_ARCHIVAL_DIR_TO_MODEL: dict[str, str] = {
    "archival-fable5": "claude-fable-5",
}


def _model_name_for(trace_root_dir: str) -> str:
    """Resolve a trace's containing directory to a model identifier.
    For models under `trajectories/<dir>/`, the directory name IS the
    model id (auto-discovered — drop a new model under a new dir and
    it shows up). For `archival-fable5/`, hard-coded to
    `claude-fable-5`."""
    return _ARCHIVAL_DIR_TO_MODEL.get(trace_root_dir, trace_root_dir)


def _row_from_grade(
    grade_path: Path,
    *,
    model_dir: str,
    trial: int = 1,
) -> dict | None:
    """Read one `grade.json` and return the standardized row dict, or
    `None` if the file is malformed or the task name doesn't match the
    benchmark's naming convention.

    Model identity is derived from `model_dir` — `grade.json` doesn't
    carry a model field (see schemas/grade.schema.json: the canonical
    `model` field lives in `prediction.schema.json` instead). The dir
    name is the model id, except for the Fable 5 special case (see
    `_ARCHIVAL_DIR_TO_MODEL`).
    """
    try:
        grade = json.loads(grade_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    task_name = grade_path.parent.name
    m = _TASK_NAME_RE.match(task_name)
    if not m:
        return None
    sid, turn, group, variant = m.groups()

    score = grade.get("score", {}) or {}
    weighted = score.get("weighted", 0.0)
    # Reward = weighted score, clamped to [0, 1] to mirror what the
    # Harbor verifier writes to reward.json (which `aggregate.collect_trials`
    # consumes when reading from the Harbor job tree directly).
    try:
        reward = max(0.0, min(1.0, float(weighted)))
    except (TypeError, ValueError):
        reward = 0.0

    # Diagnostics live directly under `score` in grade.json (see
    # schemas/grade.schema.json). The keys mirror what `aggregate.collect_trials`
    # pulls from reward.json so downstream code sees an identical shape
    # regardless of which reader produced the row.
    diag_vals = {k: score.get(k) for k in DIAG_KEYS}

    return {
        "model": _model_name_for(model_dir),
        "task": task_name,
        "task_id": grade.get("task_id"),
        "scenario": int(sid),
        "turn": int(turn),
        "side": grade.get("side"),
        "input_group": f"s{sid}-t{turn}-g{group}",
        "variant": variant,
        "reward": reward,
        "gate_passed": bool(grade.get("gate", {}).get("passed", True)),
        "n_pass": score.get("n_pass"),
        "n_total": score.get("n_total"),
        "n_penalties_triggered": score.get("n_penalties_triggered", 0),
        **diag_vals,
        # `model_dir` (e.g. "opus48", "archival-fable5") is what
        # downstream code uses to locate the per-trace `redline.docx`
        # for docx-driven metrics. We keep it on the row instead of
        # forcing every consumer to derive it.
        "model_dir": model_dir,
        "trial": trial,
        "_per_rubric": score.get("per_rubric", []),
    }


def collect_from_runs_dir(
    runs_dir: Path,
    *,
    include_fable_5: bool = False,
) -> list[dict]:
    """Walk `runs_dir` (e.g. `runs/ref1-trial1`) and return one row per
    graded `(model, task, trial)`.

    `runs_dir/trajectories/<dir>/<task>/grade.json` → main rows.
    `runs_dir/archival-fable5/<task>/grade.json` → only if
    `include_fable_5=True` (opt-in because Fable 5 lives in a different
    trace layout).

    Note: each `<dir>` under `trajectories/` is treated as one model's
    full set of traces — model identity comes from `grade.json::model`,
    not the directory name. The dir name is preserved on each row as
    `model_dir` so docx-driven metrics can find the right `redline.docx`.
    """
    rows: list[dict] = []

    trajectories_root = runs_dir / "trajectories"
    if trajectories_root.is_dir():
        for model_subdir in sorted(trajectories_root.iterdir()):
            if not model_subdir.is_dir():
                continue
            for trace_dir in sorted(model_subdir.iterdir()):
                if not trace_dir.is_dir():
                    continue
                grade = trace_dir / "grade.json"
                if not grade.exists():
                    continue
                row = _row_from_grade(grade, model_dir=model_subdir.name)
                if row is not None:
                    rows.append(row)

    if include_fable_5:
        archival_root = runs_dir / "archival-fable5"
        if archival_root.is_dir():
            for trace_dir in sorted(archival_root.iterdir()):
                if not trace_dir.is_dir():
                    continue
                grade = trace_dir / "grade.json"
                if not grade.exists():
                    continue
                row = _row_from_grade(grade, model_dir="archival-fable5")
                if row is not None:
                    rows.append(row)

    return rows


def rows_by_model(rows: list[dict]) -> dict[str, list[dict]]:
    """Group `collect_from_runs_dir` output by model. Convenience for
    the leaderboard build, which iterates per model."""
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["model"]].append(r)
    return dict(out)


def best_at_k_rows(rows: list[dict]) -> list[dict]:
    """Collapse multiple trials per `(model, task)` to the single
    highest-reward trial per pair — i.e. best@k.

    With one trial per task (the current state of `runs/ref1-trial1`),
    this is an identity transform: every `(model, task)` already has
    exactly one row, so the max-pick is that same row. The function is
    written generically so that future runs with multiple trials per
    task aggregate correctly.

    Returns one row per `(model, task)` — the one with the highest
    `reward`. Ties broken by trial number (lower first), then by the
    order rows were originally collected.
    """
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_pair[(r["model"], r["task"])].append(r)
    best: list[dict] = []
    for (_, _), candidates in by_pair.items():
        # Highest reward; break ties on lowest trial number, then
        # earliest in the input order.
        candidates_sorted = sorted(
            candidates,
            key=lambda r: (-r["reward"], r.get("trial", 1)),
        )
        best.append(candidates_sorted[0])
    return best
