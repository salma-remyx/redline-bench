# Evaluation

RedlineBench scores the edited contract. The verifier checks
that the output is a usable Word document, renders the redline into a stable
judge-readable view, grades attorney-authored rubric criteria, and aggregates
the resulting task scores. The scoring pipeline is detailed below.

## Per-Task Scoring

Each task produces one edited `.docx`. The verifier applies three steps.

First, the output must pass a validity gate. It must load as a `.docx` and
contain at least one tracked change or comment attributed to the task author.
Comments can be sufficient on turns where the right legal move is to accept the
counterparty's outstanding edits and close the issue.

Second, the verifier renders the redline into an annotated text view. Insertions,
deletions, and comments are exposed in a form the judge can read while still
being tied back to the Word document.

Third, a judge grades each attorney-authored rubric criterion as pass or fail.
Rubrics can carry positive weights, and some can carry negative weights for
undesirable redlining moves.

The task reward is the weighted rubric result, clamped to the valid scoring
range:

```text
reward = clamp(earned positive weight - triggered penalty weight) / total positive weight
```

The shared `weighted_score()` helper is used by the panel code, metrics readers,
and verifier-side judging logic.

## Judge Panel

The standard path uses a judge panel. Each judge evaluates the same rendered
redline against the same rubric criteria, and the panel verdict is the strict
majority for each criterion.

Only model-task pairs with complete judge coverage are included in panel scoring.
A single-judge mode exists for diagnostics, but panel scoring is the intended
comparison path.

## Input Groups

Some tasks share the same model-facing input and differ only by attorney rubric
set. These tasks form an input group.

The metrics summary first averages task scores within each input group. This prevents a
single contract state from receiving extra influence just because it has more
than one rubric variant.

## Benchmark Aggregation

After input-group averaging, scores are aggregated across the scenario and turn
grid. The summary also breaks results out by negotiation turn, party side,
scenario, and rubric category to better interpret model performance in specific parts of the redlining workflow.

## Document Diagnostics

Some diagnostics are computed directly from the `.docx` files rather than from
judge verdicts. These fields are useful for understanding redlining style:

- How many redlines and comments the model produced.
- How much of the document the model touched.
- Whether edits are mostly word-level or block-level.
- How verbose the model is on first-markup tasks.

These diagnostics help distinguish a targeted redline from a noisy one. They are
included as context, not as a substitute for rubric scoring.

## Metrics Summary

`metrics_summary.py` writes `metrics_summary.json`. It is the public entry point
for turning raw per-task grades and judge files into benchmark-level metrics.

The summary includes:

- Run metadata and model names.
- Leaderboard records.
- Overall scores and confidence intervals.
- Breakdowns by turn, side, scenario, and rubric category.
- Gate failures and input-group counts.
- Document-level diagnostics.

`metrics_summary.py` coordinates the calculation using helper modules: `aggregate.py` handles group and turn-weighted aggregation, `panel_reader.py` reads panel verdicts, `runs_reader.py` reads grade files, and `docx_metrics.py` computes document-level diagnostics.

