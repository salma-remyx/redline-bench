# Benchmark Design

RedlineBench is built around contract redlining as a negotiation task. The agent
does not answer questions about a contract in isolation; it receives a contract
state, a party position, grounding materials, and instructions for the current
turn of a negotiation. It then outputs the redlined version of the contract it received.

## Code And Data

The repository contains the runner, scoring code, schemas, metrics tooling, and
the canonical redlining skill. The runnable benchmark tasks live in the
Hugging Face dataset [`crosbylegal/RedlineBench`](https://huggingface.co/datasets/crosbylegal/RedlineBench). The task set can be resolved locally or downloaded at run time.

## Task Model

Each task is a self-contained Harbor task under the benchmark data directory:

```text
tasks/redline-s1-t1-g01a/
  task.toml
  instruction.md
  environment/
    Dockerfile
    app/
      contract.docx
      grounding/
    skills/
      contract-redliner/
  tests/
    rubrics.json
    judge.py
    attorney_redlines.docx
```

The model-facing files live under `environment/app/`. The verifier-side files
live under `tests/` and are not mounted into the agent environment.

The task name encodes the scenario, negotiation turn, input group, and rubric
variant:

```text
redline-s{scenario}-t{turn}-g{group}{variant}
```

Tasks in the same input group share the same model-facing contract and context.
They differ by rubric set, which lets the benchmark evaluate the same output
against multiple attorney-authored views of the same negotiation state.

## Negotiation Structure

The benchmark uses simulated SaaS MSA negotiations. Scenarios vary the starting
document, party posture, and deal context. Later turns include the redlines and
comments already present in the document, so the agent has to respond to the
negotiation record rather than start from a clean page every time.

This design tests more than issue spotting. A strong output must
decide whether to push, concede, narrow, preserve, or explain a position in the
context of the current deal.

## Redline Output

The expected artifact is a Word `.docx` edited with native tracked changes and
comments. The benchmark does not treat a plain-text rewrite as a valid redline.

Agents use the bundled [contract-redliner skill](../skills/contract-redliner/SKILL.md)
to inspect and edit the document. The skill provides scripts for reading the
document, applying tracked insertions and deletions, adding comments, and marking
reserved sections while preserving numbering.

The copy under `skills/contract-redliner/` is the source copy in this repository.
Each benchmark task vendors the skill into its Harbor environment so the agent
can use it inside the task container.

## Schemas

The JSON schemas in `schemas/` define the records exchanged by the benchmark
tooling:

- `task.schema.json`: logical task metadata and rubric fields.
- `prediction.schema.json`: the metadata envelope for a submitted redline.
- `grade.schema.json`: the verifier output for a prediction.

The schemas describe the records used by the harness and metrics pipeline.
They do not replace the Harbor task directory structure; they document the data
contracts around it.

## Pipeline

At a high level, a reproduction run follows this path:

```text
benchmark tasks
  -> Harbor agent run
  -> edited contract.docx
  -> verifier and judge records
  -> assembled runs directory
  -> metrics_summary.json
```

The verifier renders each redlined document into a judge-readable view, applies
the validity checks, asks the judge panel to score rubric criteria, and writes
the grade records consumed by the metrics pipeline.

## Synthetic Data

The benchmark uses synthetic contracts, parties, and commercial facts. Do not add
real client material, confidential information, or personal data to benchmark
tasks or examples.
