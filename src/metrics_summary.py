#!/usr/bin/env python3
"""Compute aggregate RedlineBench metrics into one JSON summary.

This script reads graded trials directly from a `runs/<run-id>/`
directory (auto-discovering models under `trajectories/*/`) and emits
the benchmark-level metrics derived from those raw grade artifacts:

  1. Overall score (turn-weighted, 12-cell average)
  2. Score by side (turn-weighted)
  3. Score by scenario (turn-weighted)
  4. Score by evaluation dimension (pooled weighted pass-rate per
     rubric category)
  5. Score breakdown by turn (per model)
  6. best@k (max reward per (model, task) across trials, then
     turn-weighted aggregation)
  7. Verbosity trap (turn 1 only — paragraph-index alignment is
     reliable when the input is the clean template)
  8. Surgicalness (inline/block share per model + human baseline)

Models are auto-discovered: any directory under
`runs/<run-id>/trajectories/` is treated as one model's full trace set,
and the model's identity comes from `grade.json::model` (not the
directory name). Adding a new model = drop its traces and re-run.

Usage:
    python -m metrics_summary \\
        --runs runs/ref1-trial1 \\
        --out metrics_summary.json

    python -m metrics_summary \\
        --runs runs/ref1-trial1 \\
        --add-fable-5 \\
        --out metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from aggregate import summarize_model
from dataset import get_benchmark_dir
from docx_metrics import (
    compute_surgicalness,
    compute_verbosity_turn1,
    find_expert_docx_paths,
    find_model_docx_paths,
    turn_of,
)
from panel_reader import collect_panel_rows
from runs_reader import (
    best_at_k_rows,
    collect_from_runs_dir,
    rows_by_model,
)


# ─── confidence interval ────────────────────────────────────────────


def bootstrap_ci(
    values: list[float], n: int = 5000, seed: int = 0
) -> list[float]:
    """2.5th / 97.5th percentile of `n` bootstrap means. Trims to 4
    decimals to match the rest of the JSON's precision.

    `values` is typically the 12 (scenario × turn) cell means for one
    model — the same sample the turn-weighted overall averages over,
    so the CI bounds the same statistic.
    """
    if len(values) < 2:
        return [round(values[0], 4), round(values[0], 4)] if values else [0.0, 0.0]
    rng = random.Random(seed)
    means = []
    k = len(values)
    for _ in range(n):
        means.append(sum(values[rng.randrange(k)] for _ in range(k)) / k)
    means.sort()
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n)], 4)]


def _model_seed(name: str) -> int:
    """Deterministic per-model RNG seed so bootstrap CIs are
    reproducible across runs."""
    return sum(name.encode("utf-8")) % (2**32)


# ─── leaderboard ────────────────────────────────────────────────────


def build_leaderboard(by_model: dict[str, list[dict]]) -> list[dict]:
    """Per-model summary rows for the metrics summary.

    Returns a list sorted descending by `overall_turn_weighted`. Each
    row carries:

      - `overall_turn_weighted` — the 12-cell (scenario × turn) mean;
        this is the headline score
      - `best_at_k_turn_weighted` — same 12-cell-weighted aggregation
        but on the max-per-(model, task) reduction. Identical to
        `overall_turn_weighted` when there's one trial per task;
        diverges if multiple trials exist per task.
      - `ci` — bootstrap 95% CI over the 12-cell sample
      - `by_turn` — per-turn means (4 numbers)
      - `by_side_turn_weighted` — A/B, each averaged over 4 turn cells
      - `by_scenario_turn_weighted` — 1/2/3, each averaged over 4 turn cells
      - `by_category` — pooled weighted pass-rate per rubric dimension
      - `n_gate_failures` — count of trials that failed the gate
    """
    rows: list[dict] = []
    for model, trials in by_model.items():
        s = summarize_model(trials)

        # Best-at-k: for each (model, task), keep the max-reward trial,
        # then re-summarize. With one trial per task this is identical
        # to s["overall_score_turn_weighted"].
        best_rows = best_at_k_rows(trials)
        s_best = summarize_model(best_rows)

        rows.append({
            "model": model,
            "overall_turn_weighted": s["overall_score_turn_weighted"],
            "best_at_k_turn_weighted": s_best["overall_score_turn_weighted"],
            "ci": bootstrap_ci(
                list(s["by_scenario_turn"].values()),
                seed=_model_seed(model),
            ),
            "by_turn": s["by_turn"],
            "by_side_turn_weighted": s["by_side_turn_weighted"],
            "by_scenario_turn_weighted": s["by_scenario_turn_weighted"],
            "by_category": s["by_rubric_category"],
            "diagnostics": s["diagnostics_mean"],
            "n_gate_failures": s["n_gate_failures"],
            "n_trials": s["n_trials"],
            "n_input_groups": s["n_input_groups"],
        })

    # Sort by the turn-weighted headline. Models without a score
    # (shouldn't happen in practice, but defensive) sink to the bottom.
    rows.sort(
        key=lambda r: -(r["overall_turn_weighted"] or 0.0)
    )
    return rows


# ─── docx-driven sections (verbosity + surgicalness) ────────────────


def _build_docx_metrics(
    runs_dir: Path,
    benchmark_dir: Path,
    *,
    include_fable_5: bool,
    inline_block_threshold: float = 0.30,
) -> tuple[dict, dict]:
    """Walk the on-disk docx files and compute the two docx-driven
    sections: verbosity (turn-1) + surgicalness (all turns).

    `benchmark_dir` is the resolved benchmark root (containing `tasks/`);
    the expert attorney redlines are read from
    `benchmark_dir/tasks/<task>/tests/attorney_redlines.docx`.
    """
    model_docx = find_model_docx_paths(runs_dir, include_fable_5=include_fable_5)
    expert_docx = find_expert_docx_paths(benchmark_dir)

    # Surgicalness: pool across ALL 140 tasks per actor. Each model
    # contributes the paths to its per-task redline.docx; the expert
    # baseline pools every available attorney_redlines.docx.
    surg_input_by_model: dict[str, list[Path]] = {
        m: list(per_task.values()) for m, per_task in model_docx.items()
    }
    expert_paths_all = list(expert_docx.values())
    surgicalness = compute_surgicalness(
        surg_input_by_model,
        expert_paths_all,
        inline_block_threshold=inline_block_threshold,
    )

    # Verbosity (turn 1): per model, build a list of
    # (task_name, model_docx, expert_docx_or_None) tuples. Filter to
    # turn-1 tasks. Expert baseline = the expert docx files at turn 1.
    by_model_turn1: dict[str, list[tuple[str, Path, Path | None]]] = {}
    for model, per_task in model_docx.items():
        items: list[tuple[str, Path, Path | None]] = []
        for task_name, docx in per_task.items():
            if turn_of(task_name) != 1:
                continue
            items.append((task_name, docx, expert_docx.get(task_name)))
        by_model_turn1[model] = items
    expert_turn1 = {
        task: path
        for task, path in expert_docx.items()
        if turn_of(task) == 1
    }
    verbosity = compute_verbosity_turn1(by_model_turn1, expert_turn1)

    return verbosity, surgicalness


# ─── main ───────────────────────────────────────────────────────────


def run(
    runs: str | Path,
    out: str | Path = "metrics_summary.json",
    *,
    benchmark_dir: str | Path | None = None,
    add_fable_5: bool = False,
    judge_method: str = "panel",
    surgicalness_threshold: float = 0.30,
) -> int:
    """Build the metrics summary JSON from a runs/<run-id>/ directory.

    `benchmark_dir` is the resolved benchmark root (containing `tasks/`),
    used only for the docx-driven expert baseline. If None, it is
    resolved via `dataset.get_benchmark_dir()` (local ./benchmark,
    $REDLINEBENCH_BENCHMARK_DIR, or a HuggingFace download). Callable
    in-process (e.g. from `reproduce.py`) without spawning a subprocess.
    """
    runs_dir = Path(runs).resolve()
    if not runs_dir.is_dir():
        print(f"ERROR: runs dir not found: {runs_dir}")
        return 1

    if benchmark_dir is None:
        benchmark_dir = get_benchmark_dir()
    benchmark_dir = Path(benchmark_dir)

    # ── grades (rubric-driven metrics) ────────────────────────────
    # Two row sources for the rubric pipeline:
    #   panel  — 3-judge majority vote (default; avoids any single
    #            judge grading a model from its own family)
    #   single — single-judge diagnostic path
    # Both produce rows with the same schema; `summarize_model`
    # downstream is source-agnostic.
    if judge_method == "panel":
        trials = collect_panel_rows(
            runs_dir, include_fable_5=add_fable_5, benchmark_dir=benchmark_dir,
        )
        if not trials:
            print(
                f"ERROR: --judge-method=panel but no panel verdicts found at "
                f"{runs_dir}/panel/judges/. Re-run the panel CLI or pass "
                f"--judge-method=single."
            )
            return 1
    else:
        trials = collect_from_runs_dir(runs_dir, include_fable_5=add_fable_5)
    if not trials:
        print(f"ERROR: no trials found under {runs_dir}")
        return 1
    by_model = rows_by_model(trials)

    leaderboard = build_leaderboard(by_model)
    models = [r["model"] for r in leaderboard]

    # ── docx-driven metrics (verbosity + surgicalness) ────────────
    verbosity, surgicalness = _build_docx_metrics(
        runs_dir, benchmark_dir,
        include_fable_5=add_fable_5,
        inline_block_threshold=surgicalness_threshold,
    )

    # ── action-graded severity (adapted from arXiv:2607.07474) ──────
    # Grade the legal-impact severity (L0–L4) of every per-rubric FAIL
    # the rubric pipeline already produced, exposing *how wrong* each
    # failure is rather than just that it failed. Local import keeps
    # this optional alongside the binary score without coupling
    # metrics_summary's import surface to the severity module.
    from severity import summarize_severity
    severity_summary = summarize_severity(trials)

    data = {
        "n_trials": len(trials),
        "n_models": len(by_model),
        "models": models,
        "include_fable_5": bool(add_fable_5),
        "judge_method": judge_method,
        "surgicalness_threshold": surgicalness_threshold,
        "leaderboard": leaderboard,
        "verbosity_turn1": verbosity,
        "surgicalness": surgicalness,
        "severity": severity_summary,
    }

    out_path = Path(out)
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))

    # Console summary.
    print(f"wrote {out_path}")
    print(f"  models       : {', '.join(models)}")
    print(f"  trials       : {len(trials)}")
    print(f"  fable-5      : {'included' if add_fable_5 else 'excluded'}")
    print(f"  judge method : {judge_method}"
          f"{' (gpt-5.4-mini + claude-haiku + gemini-3.1-flash-lite, majority vote)' if judge_method == 'panel' else ' (gpt-5.5)'}")
    print()
    print(f"  {'model':<28} {'turn_wgt':>10} {'best@k':>10} {'CI':>22}")
    for r in leaderboard:
        ci = r["ci"]
        print(
            f"  {r['model']:<28} "
            f"{r['overall_turn_weighted']:>10.4f} "
            f"{r['best_at_k_turn_weighted']:>10.4f} "
            f"  [{ci[0]:.4f}, {ci[1]:.4f}]"
        )
    hs = severity_summary["high_severity_failures"]
    print()
    print(
        f"  severity     : {severity_summary['n_failures']} rubric failures "
        f"({hs['count']} graded L3+, {hs['share_of_failures']:.0%} of failures), "
        f"mean fail = L{severity_summary['mean_fail_severity']:.1f}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs", required=True,
        help=(
            "Path to a runs/<run-id>/ directory (e.g. one assembled by "
            "`redlinebench-reproduce` from a fresh Harbor run)."
        ),
    )
    ap.add_argument(
        "--benchmark-dir", default=None,
        help=(
            "Benchmark root containing tasks/ (for the expert-redline "
            "baseline). Defaults to the dataset resolver: local "
            "./benchmark, $REDLINEBENCH_BENCHMARK_DIR, or a HuggingFace "
            "download of crosbylegal/RedlineBench."
        ),
    )
    ap.add_argument(
        "--add-fable-5", action="store_true",
        help=(
            "Include Claude Fable 5 (reference model from an earlier "
            "benchmark run) from runs/<run-id>/archival-fable5/. Off by "
            "default because Fable 5 traces have a different layout "
            "from the active models."
        ),
    )
    ap.add_argument(
        "--out", default="metrics_summary.json",
        help="Output path for the metrics summary JSON.",
    )
    ap.add_argument(
        "--surgicalness-threshold", type=float, default=0.30,
        help=(
            "Inline-vs-block threshold for the surgicalness metric. An "
            "event of size `s` in a paragraph of unchanged-baseline "
            "length `L` is inline if s/L < threshold, else block. "
            "Default 0.30."
        ),
    )
    ap.add_argument(
        "--judge-method", default="panel", choices=("panel", "single"),
        help=(
            "Source of per-rubric verdicts. 'panel' (default): 3-judge "
            "majority vote (gpt-5.4-mini + claude-haiku + "
            "gemini-3.1-flash-lite) read from "
            "runs/<run>/panel/judges/. 'single': diagnostic single-judge "
            "path that reads from trajectories/*/grade.json."
        ),
    )
    args = ap.parse_args()
    return run(
        runs=args.runs,
        out=args.out,
        benchmark_dir=args.benchmark_dir,
        add_fable_5=args.add_fable_5,
        judge_method=args.judge_method,
        surgicalness_threshold=args.surgicalness_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
