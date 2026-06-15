# RedlineBench source modules

Flat layout — modules live directly under `src/` (no nested package). The report
pipeline reads graded runs under `runs/<run-id>/`, computes the metrics defined in
`docs/REPORT-METRICS.md`, and renders the HTML report. The benchmark `tasks/` tree
is resolved/downloaded by `dataset.py` (see the top-level README).

## Entry points (console scripts → CLIs)

| Script | Module | What it does |
|---|---|---|
| `redlinebench-reproduce` | `reproduce` | End-to-end: download tasks → Harbor run → assemble runs → score → report → diff vs. published. |
| `redlinebench-aggregate` | `aggregate` | Per-task CSV/summary export from a Harbor job tree (`per_task_scores.csv` + `summary_metrics.json`). |
| `redlinebench-rejudge` | `rejudge` | Re-grade trajectory outputs with a different judge model; populates the per-judge verdict trees `panel` consumes. |
| `redlinebench-panel` | `panel` | Post-hoc 3-judge panel vote over per-judge verdict trees → `panel_summary.json` + `panel_leaderboard.csv`. |

## Modules

| Module | What it does |
|---|---|
| `dataset.py` | Resolves the benchmark root (local `./benchmark`, `$REDLINEBENCH_BENCHMARK_DIR`, or HuggingFace `crosbylegal/RedlineBench`). |
| `reproduce.py` | The end-to-end driver; assembles a Harbor job tree into the `runs/<id>/` layout and calls the report pipeline. |
| `report_metrics.py` | Orchestrator. Reads per-rubric verdicts (panel majority by default; single-judge optional), computes overall + by-turn / by-side / by-scenario / by-category / best@k / verbosity / surgicalness. Exposes `run(...)` for in-process use and a `--runs` CLI. |
| `build_report_html.py` | Renders `report_data.json` → `index.html` (dark/light theme). |
| `aggregate.py` | Variant-dedup + the 12-cell turn-weighted aggregation. `summarize_model` is the single entry point downstream code calls. |
| `runs_reader.py` | Single-judge row reader (walks `trajectories/*/grade.json`). |
| `panel_reader.py` | Panel-majority row reader (walks `panel/judges/<judge>/<model>/<task>.json`, applies majority vote via shared helpers in `panel.py`). |
| `docx_metrics.py` | OOXML walker for the docx-driven metrics (surgicalness + verbosity); reads the golden `tests/attorney_redlines.docx`. |
| `judging.py` | Shared judge system prompt + user-prompt assembly + the `call_judge` litellm wrapper. Used by `rejudge.py` and mirrored inline in the Harbor verifier. |

## Quick start

```bash
pip install -e .

# Full end-to-end reproduction (downloads the dataset from HuggingFace)
redlinebench-reproduce --agent claude-code --model anthropic/claude-opus-4-8 --n-concurrent 8

# Or build the report directly from an existing runs/<id>/ tree
python -m report_metrics --runs <runs-dir> --out report_data.json
python -m build_report_html --data report_data.json --logo ../assets/redlinebench-logo.svg --out index.html
```

See `docs/REPORT-METRICS.md` for the metric formulas and JSON schema.
