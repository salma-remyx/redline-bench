# RedlineBench — metrics & scoring

How a redline becomes a score, and how per-task scores become the leaderboard.
The report pipeline (`src/report_metrics.py`) writes a `report_data.json` with
this shape; `src/build_report_html.py` renders it to HTML.

## 1. Per-task scoring

### Validity gate
The output must be a loadable `.docx` containing at least one tracked change *or*
comment attributed to the task's author string. Comments alone can be valid: on
late deal-closing turns, accepting the counterparty's outstanding edits by
leaving them untouched is legitimate play. **Gate failure → reward 0.**

### LLM-judge rubric grading
The redline is rendered to an inline-annotated view (`~~deletions~~`,
`++insertions++`, `{cmt-N}` with a comment appendix) and graded PASS/FAIL against
each attorney-authored rubric. Rubrics carry a weight 1–10; a small number carry
**negative weights** (penalty rubrics — edits the attorney flagged as
undesirable; triggering them subtracts).

### Reward
```
reward = clamp(Σ earned − Σ penalties, 0, Σ positive weights) / Σ positive weights   ∈ [0, 1]
```
where `earned` = weights of PASS positive rubrics and `penalties` = weights of
triggered penalty rubrics. This is `weighted_score()` — defined once in `panel.py`
and shared by `panel_reader.py` and the verifier's vendored `judge.py`.

## 2. The judge panel

A 3-judge panel votes on each rubric and the **strict majority** wins
(`n_pass * 2 > n_judges`). Panel members — `gpt-5.4-mini`, `claude-haiku-4-5`,
`gemini-3.1-flash-lite` — are intentionally outside the families of any
benchmarked model, ruling out judge self-preference. Only `(model, task)` tuples
graded by *every* judge are counted. A single-judge diagnostic path
(`--judge-method single`) is also available.

## 3. Input-group averaging

Multiple attorneys independently authored rubrics over the same negotiation
states, so tasks in one input group (`g01a`, `g01b`, …) share an identical
model-facing input. Per-task scores are **first averaged within each input group**
so a model isn't over-rewarded for repeating one input under several graders.

## 4. Benchmark aggregation (12-cell turn-weighted)

Group scores are aggregated over a (scenario × turn) grid. The headline
`overall_turn_weighted` is the mean over the 12 cells, giving each
scenario-turn equal weight regardless of how many groups it contains. The report
also breaks scores out **by turn, by side, by scenario, and by rubric category**
(Legal correctness, Negotiation Quality, Commercial Context, Counterparty
Acceptance Prediction, Deal-closing Orientation). A 95% CI is computed by
**seeded** bootstrap over the 12-cell sample (deterministic given the same runs).

## 5. Deterministic diagnostics (reported, not ranked)

Per model: redline count, edit operations, total revisions, touched paragraphs,
comments added, excess redlines, median insertion length — useful for
distinguishing surgical redlines from volume.

## 6. Docx-driven behavioral metrics

Read directly from the output `.docx` (and the golden `attorney_redlines.docx`
under each task's `tests/`) by `src/docx_metrics.py`:

- **Surgicalness** (all turns) — share of inline (small, word-level) vs. block
  edits. An event of size `s` in a paragraph of unchanged-baseline length `L` is
  *inline* if `s/L < threshold` (default 0.30), else *block*.
- **Verbosity (turn 1)** — redlines per task, edits per touched paragraph, average
  edit length, and overlap rate vs. the expert baseline.

The expert baseline (`Fable 5` note): one model's docx outputs were borrowed from
an earlier experiment; `--add-fable-5` controls its inclusion.

## 7. `report_data.json` shape

```jsonc
{
  "n_trials": …, "n_models": …, "models": [...],
  "include_fable_5": bool, "judge_method": "panel", "surgicalness_threshold": 0.30,
  "leaderboard": [ { "model", "overall_turn_weighted", "best_at_k_turn_weighted",
                     "ci": [lo, hi], "by_turn", "by_side_turn_weighted",
                     "by_scenario_turn_weighted", "by_category", "diagnostics",
                     "n_gate_failures", "n_trials", "n_input_groups" }, … ],
  "verbosity_turn1": { "expert": {...}, "<model>": {...} },
  "surgicalness":    { "expert": {...}, "<model>": {...} }
}
```

See `schemas/` for the task / prediction / grade JSON-schema contracts.
