# RedlineBench schemas

Three contracts define how RedlineBench is consumed and produced:

1. **Task schema** ([`task.schema.json`](task.schema.json)) — what a benchmark task contains.
2. **Prediction schema** ([`prediction.schema.json`](prediction.schema.json)) — what any harness must submit per task.
3. **Grade schema** ([`grade.schema.json`](grade.schema.json)) — what the verifier emits per prediction.

See [`../examples/`](../examples) for one concrete task + a sample prediction + its grade.

## Task

A task is a directory under `harbor/tasks/<task_name>/`:

```
redline-s1-t1-g01a/
├── task.toml                         # Harbor config + metadata
├── instruction.md                    # the attorney brief shown to the agent
├── environment/
│   ├── Dockerfile
│   ├── app/contract.docx             # the document to redline (the model-facing input)
│   ├── app/grounding/                # playbook + commercial context (docx/pdf + extracted .md)
│   └── skills/contract-redliner/     # the redlining skill
└── tests/
    ├── rubrics.json                  # task context + attorney rubrics (hidden from the agent)
    └── judge.py, ...                 # the verifier
```

`task_name` = `redline-s{scenario}-t{turn}-g{group}{variant}`. Tasks sharing
`s{scenario}-t{turn}-g{group}` have an **identical model-facing input** and differ
only by rubric set (`variant`); scoring averages within the group. The logical
task fields (mirrored in `tests/rubrics.json`) are described by `task.schema.json`.

## Prediction

A prediction is what a harness produces for one task: the **redlined `.docx`,
edited in place**, plus identifying metadata. In Harbor this is the trial's
collected artifact (`/app/contract.docx`) recorded against the run. The
`prediction.schema.json` describes the metadata envelope any external harness
should submit alongside the document so results are attributable and reproducible.

There is intentionally **no single "golden" redline**. Rubrics define the scoring
standard and admit a solution space; `examples/` ships a *reference* output with
its score, not a canonical answer.

## Grade

The verifier loads the predicted `.docx`, gates it (loadable + ≥1 tracked change
or comment by the declared author, else 0), renders an inline-annotated view,
judges each rubric PASS/FAIL, and emits a weighted score plus diagnostics.
`grade.schema.json` describes that record (`verifier/grade.json`).
