# RedlineBench

A benchmark for measuring how well AI agents **redline contracts the way attorneys actually work**: by producing a real Word `.docx` with native tracked changes and threaded margin comments — the document an in-house lawyer would open in Word's Review pane — and grading it against attorney-authored rubrics with an LLM judge.

Each task drops the agent into a live contract negotiation: *you are in-house counsel for one party, at a specific turn of the negotiation — here is the contract as it stands, your playbook, and your commercial context. Produce your redline.*

- **Code** (this repo, GitHub): the reproduction driver, scoring, judging, and report tooling.
- **Data** ([`crosbylegal/RedlineBench`](https://huggingface.co/datasets/crosbylegal/RedlineBench) on HuggingFace): the 140 runnable tasks. **Not committed here** — it is downloaded on demand (see [Reproducing](#reproducing)).

## What's being tested

- **Legal judgment** — finding the provisions that genuinely move your party's risk, and fixing them with legally meaningful edits
- **Negotiation craft** — responding to a counterparty's redlines: accepting, rejecting, refining, replying on comment threads; knowing when a deal is done
- **Document mechanics** — real OOXML tracked changes (`<w:ins>`/`<w:del>`) with correct attribution, surgical word-level edits, preserved numbering and cross-references
- **Communication** — every edit carries a rationale comment in a disciplined negotiation voice

## Dataset structure

**140 tasks across 3 scenarios**, each scenario a complete multi-turn negotiation between a vendor (AgentCo, an AI hiring-platform company) and an enterprise customer:

| Scenario | Deal | Tasks |
|---|---|---|
| 1 | Vendor-led SaaS MSA — AgentCo marks up LargeCo's template first | 50 |
| 2 | Customer-led SaaS MSA — LargeCo writes the first markup | 40 |
| 3 | Professional-services MSA — AgentCo on GiantCo's procurement-heavy template | 50 |

Tasks span **4 negotiation turns**. Turn 1 is a clean-template first markup; turns 2–4 are response turns where the document arrives carrying the negotiation so far (real tracked changes and comment threads from prior turns), and the agent must classify and respond to every existing edit.

Task names encode the negotiation tree: `redline-s{scenario}-t{turn}-g{group}{variant}`. Tasks within one input group (e.g. `g01a`, `g01b`, `g01c`) **share an identical model-facing input and differ only in which attorney's rubric set grades the output** — measuring the same performance under multiple independent expert graders. Scoring accounts for this (see [Metrics](#metrics)).

### Task anatomy

The dataset is a single flat `tasks/` tree on HuggingFace. Each task is a self-contained, runnable [Harbor](https://harborframework.com) task:

```
tasks/redline-s1-t1-g01a/
├── task.toml            # config + metadata (scenario, turn, side, party, input_group, …)
├── instruction.md       # the attorney brief: representation, mechanics, turn context
├── environment/
│   ├── Dockerfile
│   ├── app/
│   │   ├── contract.docx        # the document to redline (edited in place)
│   │   └── grounding/           # playbook + commercial context (originals + extracted text)
│   └── skills/contract-redliner # the redlining skill (see below)
└── tests/               # verifier: LLM judge + rubrics (not visible to the agent)
    ├── rubrics.json
    ├── judge.py
    └── attorney_redlines.docx   # golden expert redline (verifier-side; 138/140 tasks)
```

The golden `attorney_redlines.docx` lives under `tests/` (the verifier side, never mounted into the agent's environment). Two turn-4 acceptance-only tasks (`redline-s2-t4-g03a`, `redline-s3-t4-g01a`) have no golden by design — the correct move there is to accept the counterparty's outstanding edits and close the deal.

### The contract-redliner skill

Agents edit the document through a bundled [skill](skills/contract-redliner/SKILL.md) — four self-contained Python scripts:

| Script | Purpose |
|---|---|
| `read_document.py` | Render the contract as Markdown with stable paragraph IDs + an appendix of every existing comment and tracked change |
| `propose_edits.py` | Apply a batch of tracked changes (replace / delete / insert_after) anchored to verbatim text, each with a rationale comment |
| `add_comment.py` | Standalone comment threads and threaded replies |
| `mark_reserved.py` | Whole-section removal that preserves downstream numbering |

The canonical copy lives at [`skills/contract-redliner/`](skills/contract-redliner); it is also vendored into each task's `environment/skills/` so Harbor can build the task container.

## Setup

1. Install [Harbor](https://harborframework.com) and have Docker running:

   ```bash
   uv tool install harbor
   ```

2. Install this package (Python ≥3.10):

   ```bash
   pip install -e .
   ```

3. API keys — copy `.env.template` to `.env` and fill in your own:

   | Variable | Used for |
   |---|---|
   | `OPENAI_API_KEY` | OpenAI panel judge (gpt-5.4-mini) and codex agents |
   | `ANTHROPIC_API_KEY` | Anthropic panel judge (claude-haiku-4-5) and claude-code agents |
   | `GEMINI_API_KEY` / `GOOGLE_GENERATIVE_AI_API_KEY` | Gemini panel judge (gemini-3.1-flash-lite) + Gemini/opencode agents |
   | `DAYTONA_API_KEY` | optional; cloud-parallel Harbor runs (`--env daytona`) |

## Reproducing

The benchmark data is **not** in this repo — it is resolved automatically: a local `./benchmark/` dir if present, else `$REDLINEBENCH_BENCHMARK_DIR`, else downloaded from [`crosbylegal/RedlineBench`](https://huggingface.co/datasets/crosbylegal/RedlineBench).

One command runs the whole pipeline (download tasks → Harbor agent run → assemble the 3-judge panel verdicts → score → report) and writes a `report_data.json` (pass `--baseline <report.json>` to also print a delta table against a prior run):

```bash
# Full benchmark (all 140 tasks)
redlinebench-reproduce --agent claude-code --model anthropic/claude-opus-4-8 --n-concurrent 8

# One-task smoke test
redlinebench-reproduce --agent claude-code --model anthropic/claude-opus-4-8 --task redline-s1-t1-g01a
```

A full re-run is **non-deterministic** (agent sampling + LLM judges), so run-to-run deltas are expected and informational — the benchmark's core finding is task difficulty (no reference model exceeds ~0.49), not an exact score. See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for the step-by-step pipeline, cloud parallelism (Daytona), and the scoring details in [`docs/REPORT-METRICS.md`](docs/REPORT-METRICS.md).

Harbor supports many [agents](https://www.harborframework.com/docs/agents) — `codex`, `opencode`, or your own — any of which can drive RedlineBench.

## Metrics

**Per task** the verifier:

1. **Validity gate** — the output must be a loadable `.docx` containing at least one tracked change *or* comment attributed to the task's author string. Gate failure → reward 0.
2. **LLM judge** — the redline is rendered to an inline-annotated view (`~~deletions~~`, `++insertions++`, `{cmt-N}` with a comment appendix) and graded PASS/FAIL against each rubric. Rubrics are weighted 1–10; a small number carry **negative weights** (penalties — edits the attorney flagged as undesirable).
3. **Score** — `reward = clamp(Σ earned − Σ penalties, 0, Σ positive weights) / Σ positive weights` ∈ [0, 1].

**Benchmark level**: per-task scores are first **averaged within each input group**, then aggregated as the mean over groups — overall and broken out per turn, per side, and per scenario. **Judging** uses a 3-judge panel (`gpt-5.4-mini` + `claude-haiku-4-5` + `gemini-3.1-flash-lite`, intentionally outside the families of benchmarked models) with strict-majority vote per rubric. See [`docs/REPORT-METRICS.md`](docs/REPORT-METRICS.md) for the formulas.

The reference models run through this pipeline are GPT-5.5, Claude Opus 4.8, Gemini 3.5 Flash, and Claude Fable 5.

## Repo layout

```
src/             # flat Python modules: reproduce, report_metrics, aggregate, panel, rejudge,
                 #   build_report_html, runs_reader, panel_reader, docx_metrics, judging, dataset
schemas/         # task / prediction / grade JSON schemas
skills/          # the canonical contract-redliner skill
docs/            # REPORT-METRICS.md and REPRODUCING.md
benchmark/       # gitignored — the dataset, downloaded from HuggingFace at runtime
```

## License

Code: MIT (© 2026 Crosby Legal). Dataset: CC-BY-4.0. See [LICENSE](LICENSE).
