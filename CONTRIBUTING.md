# Contributing to RedlineBench

Thanks for your interest in improving RedlineBench. This repo holds the
**code** (reproduction driver, scoring, judging, report); the **data** lives
on HuggingFace at [`crosbylegal/RedlineBench`](https://huggingface.co/datasets/crosbylegal/RedlineBench).

## Ground rules

- **Synthetic data only.** Every contract, party, and grounding document in the
  benchmark is fictional (AgentCo / LargeCo / GiantCo). Never contribute real
  client material, names, or PII.
- **Never commit secrets.** API keys go in `.env` (gitignored); use `.env.template`
  as the reference. The `benchmark/` directory is gitignored and must stay that way.
- Keep pull requests small and focused.

## Development setup

```bash
pip install -e .
pip install pytest
pytest tests/
```

The modules live flat under `src/` (no nested package). The console scripts in
`pyproject.toml` map to them: `redlinebench-reproduce`, `redlinebench-aggregate`,
`redlinebench-rejudge`, `redlinebench-panel`.

## The contract-redliner skill

The canonical skill is `skills/contract-redliner/`. It is **also vendored** into
every task's `environment/skills/` on the dataset (Harbor needs it inside each
task to build the container). If you change the skill, the vendored copies on
HuggingFace must be regenerated to match — the top-level copy here is the source
of truth.

## Scoring changes

The scoring math (`weighted_score`, the 12-cell turn-weighted aggregation, the
panel majority vote) is shared across `panel.py`, `panel_reader.py`, and the
verifier's vendored `judge.py`. Changing it in one place means changing it in all
of them — call this out explicitly in your PR. See `docs/REPORT-METRICS.md` for the
formulas.
