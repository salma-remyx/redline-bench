#!/usr/bin/env python3
"""Aggregate RedlineBench Harbor job results into public metrics.

Walks one or more Harbor job directories, collects every graded trial, and
writes:

    <out>/per_task_scores.csv     one row per trial (model, task, scores, diagnostics)
    <out>/summary_metrics.json    per model: group-averaged overall score +
                                  per-turn / per-side / per-scenario breakdowns

Scoring follows the benchmark's grouping rule: tasks within one input group
share an identical model-facing input and differ only in rubric set, so
per-task rewards are averaged within each input group first, and the overall
score is the mean over groups.

Usage:
    python -m aggregate --jobs jobs/matrix-* --out results/run1
    python -m aggregate --jobs jobs/matrix-gemini --model gemini-3.5-flash
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

_NAME_RE = re.compile(r"(redline-s(\d+)-t(\d+)-g(\d+)([a-z]))__")

DIAG_KEYS = (
    "redlines", "edit_operations", "total_revisions", "touched_paragraphs",
    "comments_added", "excess_redlines", "median_insertion_chars",
)


def collect_trials(job_dirs: list[Path], model_override: str | None) -> list[dict]:
    rows = []
    for job in job_dirs:
        for trial in sorted(job.glob("*__*")):
            grade_p = trial / "verifier" / "grade.json"
            reward_p = trial / "verifier" / "reward.json"
            if not reward_p.exists():
                continue
            m = _NAME_RE.search(trial.name)
            if not m:
                continue
            task_name, sid, turn, group, variant = m.groups()
            reward = json.loads(reward_p.read_text())
            grade = json.loads(grade_p.read_text()) if grade_p.exists() else {}
            model = model_override
            result_p = trial / "result.json"
            if model is None and result_p.exists():
                try:
                    info = json.loads(result_p.read_text()).get("agent_info") or {}
                    model = (info.get("model_info") or {}).get("name")
                except Exception:  # noqa: BLE001
                    model = None
            score = grade.get("score", {})
            rows.append({
                "model": model or "unknown",
                "task": task_name,
                "task_id": grade.get("task_id"),
                "scenario": int(sid),
                "turn": int(turn),
                "side": grade.get("side"),
                "input_group": f"s{sid}-t{turn}-g{group}",
                "variant": variant,
                "reward": float(reward.get("reward", 0.0)),
                "gate_passed": bool(grade.get("gate", {}).get("passed", True)),
                "n_pass": score.get("n_pass"),
                "n_total": score.get("n_total"),
                "n_penalties_triggered": score.get("n_penalties_triggered", 0),
                **{k: reward.get(k) for k in DIAG_KEYS},
                "job": job.name,
                "trial": trial.name,
                "_per_rubric": score.get("per_rubric", []),
            })
    return rows


def rubric_rows(trials: list[dict]) -> list[dict]:
    """Flatten to one row per (trial, rubric) for rubric-level pass/fail export."""
    out = []
    for t in trials:
        for p in t["_per_rubric"]:
            out.append({
                "model": t["model"], "task": t["task"], "scenario": t["scenario"],
                "turn": t["turn"], "side": t["side"], "input_group": t["input_group"],
                "rubric_id": p["rubric_id"], "category": p.get("category"),
                "weight": p.get("weight"), "is_penalty": p.get("is_penalty", False),
                "verdict": p["verdict"],
            })
    return out


def category_scores(rows: list[dict]) -> dict:
    """Per rubric category: weighted pass rate (Σ weight of PASS / Σ |weight|),
    pooled across this model's trials. Penalty rubrics contribute |weight| to the
    denominator and are 'correct' when NOT triggered."""
    cat = defaultdict(lambda: {"earned": 0, "total": 0})
    for r in rows:
        for p in r["_per_rubric"]:
            w = int(p.get("weight", 0))
            c = p.get("category") or "(uncategorized)"
            aw = abs(w)
            cat[c]["total"] += aw
            passed = p["verdict"] == "PASS"
            if w >= 0 and passed:
                cat[c]["earned"] += aw
            elif w < 0 and not passed:
                cat[c]["earned"] += aw
    return {c: round(v["earned"] / v["total"], 4) for c, v in cat.items() if v["total"]}


def group_average(rows: list[dict]) -> dict[str, float]:
    """input_group -> mean reward across its variants (for one model)."""
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["input_group"]].append(r["reward"])
    return {g: mean(v) for g, v in by_group.items()}


def summarize_model(rows: list[dict]) -> dict:
    groups = group_average(rows)
    by_dim = lambda key: {  # noqa: E731
        str(k): round(mean(gv for g, gv in groups.items() if g in grp_set), 4)
        for k, grp_set in _groups_by(rows, key).items()
    }
    by_turn = by_dim("turn")

    # ── New (scenario × turn) cell aggregation ───────────────────────────
    # `by_scenario_turn`: 12 cells keyed "<scenario>-<turn>". Each cell is
    # the mean group-score for that single (scenario, turn) pair. Same for
    # `by_side_turn` (8 cells: side ∈ {A,B} × turn ∈ {1..4}).
    by_scenario_turn = _by_two_dim(rows, groups, "scenario", "turn")
    by_side_turn = _by_two_dim(rows, groups, "side", "turn")

    # `overall_score_turn_weighted`: mean over scenario-turn cells.
    # This gives each scenario/turn pair one vote regardless of how many
    # input groups that cell contains.
    overall_score_turn_weighted = (
        round(mean(by_scenario_turn.values()), 4) if by_scenario_turn else None
    )

    # Per-scenario turn-weighted: average over the 4 (scenario, turn)
    # cells per scenario. Each scenario has exactly 4 cells in the
    # 12-cell partition, and `mean(s1, s2, s3) == overall_score_turn_weighted`
    # by construction.
    by_scenario_turn_weighted = _turn_weighted_by_dim(by_scenario_turn)

    # Per-side: project the 12-cell `by_scenario_turn` partition onto
    # side rather than naively averaging the 8 (side, turn) cells.
    #
    # The Crosby benchmark alternates sides by turn — each (scenario,
    # turn) cell has exactly ONE side (no cell has both A and B input
    # groups). If we averaged the 8 (side, turn) buckets equally per
    # side, each "turn bucket" for a side mixes scenarios unevenly
    # (side-A turn-1 may cover s1+s3 while side-A turn-2 only covers
    # s2), so `mean(A, B)` doesn't reduce to the overall 12-cell mean.
    #
    # Projecting the 12 (scenario, turn) cells onto side keeps each
    # cell as a single unit — side X's score is the mean of cells where
    # side=X, and `mean(A, B) == overall_score_turn_weighted` exactly
    # whenever the cells split evenly between sides (6/6 in the
    # current corpus). This is the same notion the headline uses, just
    # filtered by side.
    by_side_turn_weighted = _by_side_from_scenario_turn(rows, by_scenario_turn)

    return {
        "n_trials": len(rows),
        "n_input_groups": len(groups),
        "n_gate_failures": sum(1 for r in rows if not r["gate_passed"]),
        "overall_score": round(mean(groups.values()), 4) if groups else None,
        "overall_score_turn_weighted": overall_score_turn_weighted,
        # `mean_per_task_reward` is the non-deduplicated flat mean over all
        # raw trial rewards. `None` for empty input — matches `overall_score`'s
        # convention so callers can rely on uniform handling.
        "mean_per_task_reward": (
            round(mean(r["reward"] for r in rows), 4) if rows else None
        ),
        "by_turn": by_turn,
        "by_scenario": by_dim("scenario"),
        "by_side": by_dim("side"),
        # Turn-weighted breakdowns used by the metrics summary for
        # side and scenario slices.
        "by_scenario_turn": by_scenario_turn,
        "by_side_turn": by_side_turn,
        "by_scenario_turn_weighted": by_scenario_turn_weighted,
        "by_side_turn_weighted": by_side_turn_weighted,
        "by_rubric_category": category_scores(rows),
        "diagnostics_mean": {
            k: round(mean(r[k] for r in rows if r.get(k) is not None), 1)
            for k in DIAG_KEYS
            if any(r.get(k) is not None for r in rows)
        },
    }


def _groups_by(rows: list[dict], key: str) -> dict:
    out = defaultdict(set)
    for r in rows:
        out[r[key]].add(r["input_group"])
    return out


def _by_two_dim(
    rows: list[dict],
    groups: dict[str, float],
    dim_a: str,
    dim_b: str,
) -> dict[str, float]:
    """Group-score means split by two row-level dimensions.

    Returns `{"<dim_a_value>-<dim_b_value>": mean_group_score, …}` — e.g.
    for (scenario, turn) this is the 12-cell grid the summary's headline
    metric averages over.
    """
    # Build {(dim_a_value, dim_b_value): set(input_group)} so we can
    # average the right group-scores per cell.
    cell_groups: dict[tuple, set[str]] = defaultdict(set)
    for r in rows:
        cell_groups[(r[dim_a], r[dim_b])].add(r["input_group"])
    out: dict[str, float] = {}
    for (a, b), grp_set in cell_groups.items():
        values = [groups[g] for g in grp_set if g in groups]
        if values:
            out[f"{a}-{b}"] = round(mean(values), 4)
    return out


def _by_side_from_scenario_turn(
    rows: list[dict],
    by_scenario_turn: dict[str, float],
) -> dict[str, float]:
    """Project the 12 (scenario, turn) cell means onto side.

    The benchmark alternates sides per turn — each (scenario, turn)
    cell has exactly one side. We look up that side per cell from the
    rows, group cells by side, then average. If the cells split 6/6
    between A and B (which they do in the current corpus), this gives
    `mean(by_side[A], by_side[B]) == mean(by_scenario_turn.values())`
    exactly, eliminating the divergence the naive (side, turn) bucket
    average produces.

    If a (scenario, turn) cell happens to contain groups from BOTH
    sides (not the case in the current dataset, but defensive), the
    cell is assigned to the first side encountered — which preserves
    the projection's invariant that each cell is counted once.
    """
    # Determine the side of each (scenario, turn) cell from the rows.
    cell_side: dict[tuple[int, int], str] = {}
    for r in rows:
        key = (r["scenario"], r["turn"])
        if key not in cell_side:
            cell_side[key] = r["side"]

    by_side_lists: dict[str, list[float]] = defaultdict(list)
    for cell_key, val in by_scenario_turn.items():
        sc_str, t_str = cell_key.split("-")
        side = cell_side.get((int(sc_str), int(t_str)))
        if side:
            by_side_lists[side].append(val)
    return {s: round(mean(v), 4) for s, v in by_side_lists.items() if v}


def _turn_weighted_by_dim(by_two_dim: dict[str, float]) -> dict[str, float]:
    """Collapse a `{"<dim>-<turn>": v}` map back to `{"<dim>": mean over
    turns}`. Used to derive `by_<dim>_turn_weighted` from `by_<dim>_turn`."""
    by_outer: dict[str, list[float]] = defaultdict(list)
    for key, value in by_two_dim.items():
        # Split only on the LAST hyphen — supports any dim value (e.g.
        # "A", "1", "B", "2") not just numeric ones.
        outer, _ = key.rsplit("-", 1)
        by_outer[outer].append(value)
    return {k: round(mean(v), 4) for k, v in by_outer.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", nargs="+", required=True,
                    help="Harbor job directories (globs ok via shell)")
    ap.add_argument("--model", default=None,
                    help="model label override (default: read from trial result.json)")
    ap.add_argument("--out", default="results/public_metrics")
    args = ap.parse_args()

    job_dirs = [Path(j) for j in args.jobs if Path(j).is_dir()]
    if not job_dirs:
        print("no job directories found", file=sys.stderr)
        return 1
    rows = collect_trials(job_dirs, args.model)
    if not rows:
        print("no graded trials found", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fields = [k for k in rows[0].keys() if not k.startswith("_")]
    with (out / "per_task_scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    rrows = rubric_rows(rows)
    with (out / "rubric_level_verdicts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rrows[0].keys()))
        w.writeheader()
        w.writerows(rrows)

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    summary = {m: summarize_model(rs) for m, rs in sorted(by_model.items())}
    (out / "summary_metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"wrote {len(rows)} trials -> {out}/per_task_scores.csv")
    for m, s in summary.items():
        print(f"  {m}: overall={s['overall_score']} "
              f"turn_weighted={s['overall_score_turn_weighted']} "
              f"({s['n_trials']} trials, {s['n_input_groups']} groups, "
              f"{s['n_gate_failures']} gate failures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
