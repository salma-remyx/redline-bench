# Reproducing the RedlineBench report

The published leaderboard (`docs/report/index.html`,
`docs/report/report_data.json`) is produced by running agents over the 140
tasks, grading each with a 3-judge panel, and aggregating. This guide walks the
pipeline and how `redlinebench-reproduce` automates it.

## Prerequisites

- `pip install -e .` (this package) and `uv tool install harbor`.
- Docker running locally, **or** a Daytona account for cloud parallelism.
- API keys in `.env` (see `.env.template`): Anthropic + OpenAI + Gemini are all
  required because the verifier's 3-judge panel spans all three families.

## The dataset resolves automatically

You do not clone the data. `src/dataset.py` resolves the benchmark root in order:

1. a local `./benchmark/` directory, if present;
2. `$REDLINEBENCH_BENCHMARK_DIR`, if set;
3. otherwise a HuggingFace download of `crosbylegal/RedlineBench` (cached by
   `huggingface_hub`).

The resolved root always contains `tasks/`.

## One command

```bash
redlinebench-reproduce \
    --agent claude-code \
    --model anthropic/claude-opus-4-8 \
    --n-concurrent 8
```

This:

1. **Resolves/downloads** the benchmark.
2. **Runs Harbor**: `harbor run -p <benchmark>/tasks -a <agent> -m <model> …`,
   writing trials to `reproduce_out/jobs/<job>/`.
3. **Assembles** the `runs/<id>/` layout the report pipeline expects, copying from
   each trial: `verifier/grade.json` → `trajectories/<model>/<task>/grade.json`,
   `artifacts/contract.docx` → `…/redline.docx`, and the Harbor verifier's
   per-judge files `verifier/judges/*.json` → `panel/judges/<judge>/<model>/<task>.json`.
   (The verifier already runs the 3-judge panel inline, so there is no separate
   re-judging step.)
4. **Scores + reports**: `report_metrics.run(...)` → `report_data.json`.
5. **Compares** against `docs/report/report_data.json` and prints a delta table.

Add `--html --logo assets/redlinebench-logo.svg` to also render `index.html`.

### Useful flags

| Flag | Effect |
|---|---|
| `--task redline-s1-t1-g01a` | Run a single task (smoke test) instead of all 140. |
| `--env daytona` | Run on Daytona cloud sandboxes instead of local Docker. |
| `--n-concurrent N` | Parallel trials. |
| `--workdir DIR` | Where `jobs/` and `runs/` are written (default `reproduce_out/`). |
| `--baseline PATH` | Report JSON to diff against (default the published one). |

## Why deltas are expected

A full re-run is **non-deterministic**: agents sample, and the LLM judges are not
perfectly stable. Absolute scores shift run-to-run and with judge choice, but the
benchmark's findings are robust: every judge family independently produces the
same model ranking, and no reference model exceeds ~0.49 overall. Treat the delta
table as a sanity check, not a bit-exact gate.

## Running pieces by hand

The individual stages are also exposed as CLIs (see `src/README.md`):
`redlinebench-aggregate` (Harbor jobs → CSV/summary), `redlinebench-rejudge`
(re-grade with a different judge), `redlinebench-panel` (post-hoc panel vote),
and `report_metrics`/`build_report_html` for the report itself.
