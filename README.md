# RedlineBench

RedlineBench evaluates whether AI agents can redline contracts in the format lawyers actually review: a Word `.docx` with native tracked changes and threaded comments.

Each task places an agent inside a contract negotiation. The agent receives the current contract, the party it represents, negotiation context, and grounding materials. It must return the same `.docx`, edited in place, with comments explaining the redline.

This repository contains the code for running, judging, scoring, and summarizing the benchmark. The benchmark data is not committed here: the runnable tasks are distributed on Hugging Face at [`crosbylegal/RedlineBench`](https://huggingface.co/datasets/crosbylegal/RedlineBench) and are downloaded on demand.

## What It Tests

Redlining is not only clause drafting. A useful redline has to identify what matters, choose an appropriate negotiation move, preserve the document, and communicate the reason for the change.

RedlineBench focuses on four parts of that workflow:

- Legal judgment: finding provisions that matter for the represented party.
- Negotiation judgment: accepting, rejecting, narrowing, or explaining counterparty positions.
- Document mechanics: producing real Word tracked changes and comments, not a rewritten text file.
- Professional communication: giving concise rationale comments in the voice of counsel.

## Documentation

- [Guide](docs/GUIDE.md): setup, API keys, smoke tests, full reproduction, and CLI reference.
- [Benchmark Design](docs/BENCHMARK-DESIGN.md): task format, dataset layout, schemas, and the redlining skill.
- [Evaluation](docs/EVALUATION.md): validity checks, rubric judging, aggregation, diagnostics, and metrics summaries.

## Repository Layout

```text
src/        Python modules and console-script entry points
schemas/    JSON schemas for task, prediction, and grade records
skills/     canonical contract-redliner skill used by the benchmark tasks
docs/       human-readable guide, benchmark design, and evaluation notes
```

`benchmark/` is intentionally not committed. If present locally, it is treated as the benchmark data directory; otherwise the data resolver can download the tasks from Hugging Face.

## License

Code: MIT. Dataset: CC-BY-4.0. See [LICENSE](LICENSE).
