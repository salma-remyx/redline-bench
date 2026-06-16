"""Judge-panel-driven rows for the metrics pipeline.

By default the metrics pipeline reads per-rubric verdicts from a single
GPT-5.5 judge living under `runs/<run-id>/trajectories/<model>/<task>/grade.json`.
That single-judge setup has a built-in conflict of interest — GPT-5.5
is one of the four models being benchmarked, so it grades its own
outputs. The 3-judge panel (`gpt-5.4-mini`, `claude-haiku`,
`gemini-3.1-flash-lite`) is purpose-built to fix that: every panel
judge is FROM a different family than any of the actively-graded
models, and a rubric-level majority vote bakes in cross-family
consensus before the score aggregator ever sees it.

This module produces rows in the same shape `runs_reader` does, but
with the per-rubric verdicts reconciled by panel-majority instead of
read straight from GPT-5.5. The downstream `aggregate.summarize_model`
pipeline doesn't know or care which source produced the verdicts —
that's the whole point of keeping the row schema stable.

Data layout (already on disk):

  runs/<run-id>/panel/judges/<judge>/<panel_model_name>/<task>.json

where `<judge>` is one of the 3 panel members, `<panel_model_name>` is
the full model identifier (e.g. `claude-opus-4-8`, `gpt-5.5`,
`gemini-3.5-flash`, `claude-fable-5`), and `<task>` matches the
`redline-s{S}-t{T}-g{NN}{variant}` naming convention.

Two on-disk verdict shapes are supported:

  * `score.per_rubric` — fully-aggregated rows (verdict + weight +
    category + is_penalty), written by the post-hoc `panel.py` /
    `rejudge.py`.
  * `verdicts` — raw judge output (`{rubric_id, verdict, justification}`,
    NO weights), written by the Harbor verifier's `judge.py` and copied
    verbatim by `reproduce.assemble_runs`. Weights / category /
    is_penalty are joined from the task's `rubrics.json` (requires
    `benchmark_dir`).

Per-rubric majority logic mirrors `panel.py::main`'s vote: PASS if
`n_pass * 2 > len(votes)` (strict majority with an odd judge count).
Diagnostics (`redlines`, `edit_operations`, …) and the gate verdict
are docx-derived — not judge-dependent — so they're borrowed verbatim
from the trajectory grade.json sitting beside the docx.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from aggregate import DIAG_KEYS
from panel import majority_vote_per_rubric, weighted_score

# Same regex `runs_reader._TASK_NAME_RE` uses — verdict filenames are
# `<task>.json` so the stem matches the task name directly.
_TASK_NAME_RE = re.compile(r"^redline-s(\d+)-t(\d+)-g(\d+)([a-z])$")

# Per-task rubric metadata, cached by rubrics.json path so repeated
# (model, task) lookups across models don't re-read the file.
_RUBRIC_CACHE: dict[Path, dict[str, dict]] = {}


def _load_task_rubrics(
    benchmark_dir: Path | None, task: str,
) -> dict[str, dict]:
    """Map `rubric_id → {weight, category, is_penalty, criteria}` for a
    task, read from `<benchmark>/tasks/<task>/tests/rubrics.json`.

    Returns an empty dict when `benchmark_dir` is None or the file is
    missing/unreadable — callers then fall back to the weights embedded
    in `score.per_rubric` (the fully-aggregated format). `is_penalty` is
    derived as `weight < 0`, matching the live verifier's `judge.py`.
    """
    if benchmark_dir is None:
        return {}
    path = Path(benchmark_dir) / "tasks" / task / "tests" / "rubrics.json"
    cached = _RUBRIC_CACHE.get(path)
    if cached is not None:
        return cached
    out: dict[str, dict] = {}
    try:
        data = json.loads(path.read_text())
        for r in data.get("rubrics", []):
            w = int(r["weight"])
            out[r["id"]] = {
                "weight": w,
                "category": r.get("category"),
                "is_penalty": w < 0,
                "criteria": r.get("criteria"),
            }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        out = {}
    _RUBRIC_CACHE[path] = out
    return out


# Panel model directory names (under `panel/judges/<judge>/`) carry the
# FULL model identifier from the original benchmark run (e.g.
# `claude-opus-4-8`). Trajectory directories under
# `trajectories/<dir>/` use SHORT names (`opus48`). Map between them
# here — single source of truth so the leaderboard's display labels
# stay consistent with the single-judge path.
#
# `model_id` is what `runs_reader` puts on each row's `model` field
# (auto-discovered from the trajectory dir name; we mirror it here).
# `model_dir` is what `docx_metrics.find_model_docx_paths` keys docx
# files by — for the active models it's the trajectory subdirectory
# name, for Fable 5 it's the archival path.
_PANEL_MODEL_TO_TRAJECTORY: dict[str, tuple[str, str]] = {
    # panel_model → (model_id, model_dir)
    "gpt-5.5":          ("gpt55",          "gpt55"),
    "claude-opus-4-8":  ("opus48",         "opus48"),
    "gemini-3.5-flash": ("gemini35",       "gemini35"),
    "claude-fable-5":   ("claude-fable-5", "archival-fable5"),
}


def collect_panel_rows(
    runs_dir: Path,
    *,
    include_fable_5: bool = False,
    benchmark_dir: Path | None = None,
) -> list[dict]:
    """Walk `runs_dir/panel/judges/*/` and return rows in the same shape
    as `runs_reader.collect_from_runs_dir`, with per-rubric verdicts
    reconciled by panel-majority vote.

    Per-judge verdict files at
    `runs_dir/panel/judges/<judge>/<panel_model>/<task>.json` are
    grouped by `(panel_model, task)`. For each tuple where ALL judges
    voted (intersection set — matches `panel.py`'s `common`), we:

      1. Collect each judge's per-rubric verdicts.
      2. For each rubric: PASS if a strict majority voted PASS,
         FAIL otherwise. Weight + is_penalty + category + criteria
         are copied from the first judge that scored the rubric (they
         agree across judges by construction; rejudge.py ensures it).
      3. Recompute the weighted score using the majority verdicts.
      4. Borrow the gate verdict and docx-derived diagnostics
         (`redlines`, `edit_operations`, …) from the trajectory
         grade.json beside the model's docx — these are not
         judge-dependent.

    Returns an empty list if no panel data is present (caller should
    fall back to the single-judge reader in that case).
    """
    panel_root = runs_dir / "panel" / "judges"
    if not panel_root.is_dir():
        return []

    judges = sorted(p.name for p in panel_root.iterdir() if p.is_dir())
    if not judges:
        return []

    # by_model_task[(panel_model, task)] = {judge_name: verdict_dict}
    by_model_task: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for judge in judges:
        for model_dir in (panel_root / judge).iterdir():
            if not model_dir.is_dir():
                continue
            panel_model = model_dir.name
            if panel_model == "claude-fable-5" and not include_fable_5:
                continue
            for vf in model_dir.glob("*.json"):
                task = vf.stem
                try:
                    by_model_task[(panel_model, task)][judge] = json.loads(vf.read_text())
                except (json.JSONDecodeError, OSError):
                    continue

    rows: list[dict] = []
    for (panel_model, task), votes in by_model_task.items():
        if len(votes) < len(judges):
            # Mirror panel.py's `common = set.intersection(...)`: only
            # consider (model, task) tuples graded by EVERY judge. A
            # missing verdict from one judge would make the majority
            # math degenerate.
            continue

        m = _TASK_NAME_RE.match(task)
        if not m:
            continue
        sid, turn, group, variant = m.groups()

        # Collect per-judge rubric data into the shape
        # `panel.majority_vote_per_rubric` expects: a list of N maps
        # (one per judge), each rubric_id -> (verdict, weight, category).
        # Also collect rubric-level metadata (is_penalty, criteria) on
        # the side, since those don't fit the (verdict, weight,
        # category) tuple but the metrics rows need them.
        task_rubrics = _load_task_rubrics(benchmark_dir, task)
        rubric_meta: dict[str, dict] = {}
        rubric_sets: list[dict[str, tuple[str, int, str | None]]] = []
        for judge_name in judges:
            vd = votes[judge_name]
            judge_map: dict[str, tuple[str, int, str | None]] = {}
            scored = vd.get("score", {}).get("per_rubric")
            if scored:
                # Fully-aggregated rows carry their own weight/category.
                for p in scored:
                    rid = p["rubric_id"]
                    judge_map[rid] = (
                        p["verdict"], int(p["weight"]), p.get("category"),
                    )
                    if rid not in rubric_meta:
                        rubric_meta[rid] = {
                            "is_penalty": bool(p.get("is_penalty", False)),
                            "category": p.get("category"),
                            "criteria": p.get("criteria"),
                        }
            else:
                # Raw verifier verdicts: join weights from rubrics.json.
                for p in vd.get("verdicts", []):
                    rid = p.get("rubric_id")
                    rdef = task_rubrics.get(rid)
                    if rdef is None:
                        continue
                    judge_map[rid] = (
                        p.get("verdict", "FAIL"), rdef["weight"], rdef["category"],
                    )
            rubric_sets.append(judge_map)

        # `rubrics.json` is the authoritative source of is_penalty /
        # criteria and covers rubrics every judge omitted (still scored,
        # as FAIL). Fold it in without clobbering metadata already taken
        # from the aggregated `score.per_rubric` rows.
        for rid, rdef in task_rubrics.items():
            rubric_meta.setdefault(rid, {
                "is_penalty": rdef["is_penalty"],
                "category": rdef["category"],
                "criteria": rdef["criteria"],
            })

        # Strict-majority vote per rubric — shared helper with
        # `panel.main()` so the math is bit-identical to what the
        # post-hoc panel CLI computes. Votes only among judges that
        # actually graded each rubric.
        panel_verdicts, vote_weights = majority_vote_per_rubric(rubric_sets)

        # Score over the FULL rubric set when rubrics.json is available: a
        # rubric no judge returned counts as FAIL and still contributes
        # its (positive) weight to the denominator — matching the live
        # verifier's `aggregate()`. With the aggregated `score.per_rubric`
        # format every rubric is already present, so this is a no-op; the
        # fallback preserves the prior voted-set behavior.
        if task_rubrics:
            weights = {rid: rdef["weight"] for rid, rdef in task_rubrics.items()}
            panel_verdicts = {
                rid: panel_verdicts.get(rid, "FAIL") for rid in task_rubrics
            }
        else:
            weights = vote_weights

        # Rebuild the per_rubric list the metrics row carries downstream,
        # attaching the metadata `majority_vote_per_rubric` doesn't
        # propagate (is_penalty, criteria text).
        majority_per_rubric: list[dict] = []
        for rid, verdict in panel_verdicts.items():
            meta = rubric_meta.get(rid, {})
            majority_per_rubric.append({
                "rubric_id": rid,
                "verdict": verdict,
                "weight": weights[rid],
                "is_penalty": meta.get("is_penalty", False),
                "category": meta.get("category"),
                "criteria": meta.get("criteria"),
            })

        reward = round(weighted_score(panel_verdicts, weights), 4)

        # Per-rubric counters for the row schema.
        n_total = len(majority_per_rubric)
        n_pass = sum(1 for r in majority_per_rubric if r["verdict"] == "PASS")
        n_penalties_triggered = sum(
            1 for r in majority_per_rubric
            if r["is_penalty"] and r["verdict"] == "PASS"
        )

        # Borrow gate, diagnostics, AND task metadata (side / task_id)
        # from the trajectory grade.json. The panel verdict files
        # sometimes ship with side=None / task_id=None for a subset of
        # tasks (a rejudge.py metadata-propagation bug — affects ~12
        # gemini-3.5-flash tasks in the current run). The trajectory
        # grade.json is the canonical source of those fields and
        # mirrors them reliably. Verdict-file metadata is only used as
        # a fallback when the trajectory grade is unavailable.
        sample = next(iter(votes.values()))
        side = sample.get("side")
        task_id = sample.get("task_id")
        gate_passed = True
        diag_vals: dict[str, object] = {k: None for k in DIAG_KEYS}

        model_id, model_dir = _PANEL_MODEL_TO_TRAJECTORY.get(
            panel_model, (panel_model, panel_model),
        )
        # Where the trajectory grade.json lives — active models under
        # `trajectories/<dir>/`, Fable 5 under `archival-fable5/`.
        if panel_model == "claude-fable-5":
            traj_grade = runs_dir / "archival-fable5" / task / "grade.json"
        else:
            traj_grade = runs_dir / "trajectories" / model_dir / task / "grade.json"
        if traj_grade.exists():
            try:
                tg = json.loads(traj_grade.read_text())
                gate_passed = bool(tg.get("gate", {}).get("passed", True))
                tscore = tg.get("score", {}) or {}
                for k in DIAG_KEYS:
                    diag_vals[k] = tscore.get(k)
                # Trajectory grade is authoritative for task metadata.
                # Only adopt non-None values — keep the verdict-file
                # fallback for fields the trajectory grade might also
                # lack.
                if tg.get("side") is not None:
                    side = tg.get("side")
                if tg.get("task_id") is not None:
                    task_id = tg.get("task_id")
            except (json.JSONDecodeError, OSError):
                pass

        rows.append({
            "model": model_id,
            "task": task,
            "task_id": task_id,
            "scenario": int(sid),
            "turn": int(turn),
            "side": side,
            "input_group": f"s{sid}-t{turn}-g{group}",
            "variant": variant,
            "reward": reward,
            "gate_passed": gate_passed,
            "n_pass": n_pass,
            "n_total": n_total,
            "n_penalties_triggered": n_penalties_triggered,
            **diag_vals,
            "model_dir": model_dir,
            "trial": 1,
            "_per_rubric": majority_per_rubric,
        })

    return rows
