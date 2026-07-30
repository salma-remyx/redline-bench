#!/usr/bin/env python3
"""CM-LRS-style reliability scorecard for redline outputs.

Adapted (Mode 2) from *Capital Markets LLM Reliability Score (CM-LRS):
From Plausible to Bankable* (arXiv:2607.21340). The paper's core
contribution is a **workflow-output-layer reliability scorecard**: an LLM
judge rates an output across seven 0-5 reliability dimensions and the
dimensions aggregate to a tunable weighted mean. Its headline claim is
"plausibility is cheap; bankability is the bar" — a single PASS/FAIL or
fluency signal does not capture whether a redline is defensible before a
counterparty or a reviewing attorney.

This module ports that **core mechanism** to RedlineBench at full
fidelity: seven LLM-judged 0-5 dimensions + a tunable weighted-mean
aggregate, funneled through the existing ``judging.call_judge_raw``
chokepoint so the opt-in ``judge_audit`` trail covers it like any other
judge call (no new infrastructure, no duplicated retry/audit logic).

Auxiliary components are substituted for target-native equivalents:

  * The paper's capital-markets dimension anchors (DCM terms extraction,
    M&A comparable reasoning, issuer-profile synthesis, …) are replaced
    with **contract-redlining anchors** that reuse this repo's existing
    signals — paragraph-ID-cited justifications, rationale comments, and
    OOXML tracked-change structure. The seven-dimension *structure*
    (factual accuracy → reviewability) is preserved.
  * The paper's five-workflow benchmark suite over SEC EDGAR filings is
    **cut** — evaluation at scale belongs in a downstream PR. Here the
    scorecard is produced per redline via the re-judge path
    (``redlinebench-rejudge --reliability``), reusing the same annotated
    view and task context the verdict judge already consumes.

What is preserved at full fidelity is the paper's actual contribution:
the multi-dimensional 0-5 rubric + the tunable weighted-mean
reliability signal at the workflow-output layer.
"""

from __future__ import annotations

import json
import re

import judging

#: Top of the per-dimension rubric scale (0–5, as in CM-LRS).
DEFAULT_SCALE = 5

# Seven reliability dimensions, structurally identical to CM-LRS's
# capital-markets rubric, reframed for the redline output a practitioner
# actually defends. Each carries the 0-5 anchor the judge scores against.
# D2 (evidence traceability) and D7 (reviewability) directly reflect the
# README's emphasis on paragraph-ID-cited judge justifications and
# disciplined rationale comments.
RELIABILITY_DIMENSIONS = [
    {
        "id": "D1",
        "name": "factual_accuracy",
        "anchor": (
            "Does the redline accurately reflect the contract's actual text "
            "and the represented party's commercial/legal position? "
            "0 = material misstatements or edits that contradict the source; "
            "5 = every edit and claim is factually correct."
        ),
    },
    {
        "id": "D2",
        "name": "evidence_traceability",
        "anchor": (
            "Is every edit and its rationale anchored to a specific paragraph "
            "id, section, or clause? 0 = edits are untraceable or free-floating; "
            "5 = all edits cite a precise location (e.g. [p-115], §16.1)."
        ),
    },
    {
        "id": "D3",
        "name": "numerical_consistency",
        "anchor": (
            "Are numbers (amounts, dates, notice periods, thresholds) handled "
            "consistently and correctly? 0 = figures invented, contradicted, or "
            "misstated; 5 = all figures match the source and are internally coherent."
        ),
    },
    {
        "id": "D4",
        "name": "workflow_completeness",
        "anchor": (
            "Does the redline address all the edits this negotiation turn requires? "
            "0 = most required edits missing; 5 = complete coverage of the turn's "
            "required changes."
        ),
    },
    {
        "id": "D5",
        "name": "source_discipline",
        "anchor": (
            "Does the redline stay grounded in the source contract and playbook "
            "without inventing provisions? 0 = hallucinated or misattributed clauses; "
            "5 = fully grounded in source material."
        ),
    },
    {
        "id": "D6",
        "name": "decision_usefulness",
        "anchor": (
            "Is the redline actionable and appropriately calibrated to the party's "
            "leverage and the deal stage? 0 = counterproductive or inert; 5 = advances "
            "the party's position with well-judged aggressiveness."
        ),
    },
    {
        "id": "D7",
        "name": "reviewability",
        "anchor": (
            "Are the edits and margin comments clear, justified, and audit-ready for a "
            "reviewing attorney? 0 = no rationale or opaque edits; 5 = every change "
            "carries a disciplined, citable rationale comment."
        ),
    },
]

#: Per-dimension weights defaulting to 1.0 — the paper's equal-weighted mean.
#: Override via ``score_reliability(..., weights={...})`` to tune the
#: aggregate to a workflow (e.g. up-weight D2 traceability for a regulated deal).
DEFAULT_WEIGHTS = {d["id"]: 1.0 for d in RELIABILITY_DIMENSIONS}


RELIABILITY_SYSTEM_PROMPT = """\
You are a senior commercial-contracts attorney scoring an AI-generated contract redline for RELIABILITY — whether the redline is defensible before a counterparty and a reviewing attorney, not merely fluent. You are STRICT but fair. Plausibility is cheap; bankability is the bar.

# Your job

Score the redline on each of seven reliability dimensions, 0–5, against the anchor for that dimension:

- 5 = exemplary: the dimension is fully and unambiguously satisfied.
- 3 = adequate: the dimension is mostly satisfied with minor gaps.
- 0 = failing: the dimension is unmet or materially violated.

Use the full 0–5 range and do not default to 5. A fluent-looking redline that is not actually defensible must not score 5.

# What you are looking at

The redlined document is rendered in CriticMarkup-style inline format:

- `~~strikethrough~~`   — a tracked deletion
- `++insertion++`       — a tracked insertion
- `~~old~~++new++`      — a tracked replacement
- `{cmt-N}`             — a comment anchor; the full comment body is in the appendix keyed by ID
- `[p-NNN]`             — paragraph ids for locating clauses

Grade on the OOXML state of the redline (the inline markers) plus the rationale comments. Traceability and source discipline matter: edits and claims should be anchored to specific paragraph ids / sections.

# Rules

1. **Justify each score** in **ONE short sentence, no more than 25 words**. Cite a paragraph id or section number when it sharpens the point. No preamble, no hedging.
2. **Score each dimension independently** — do not let one strong dimension inflate the others.
3. Return exactly one score per dimension, using the dimension's `id` verbatim.

# Output format

Return ONLY a JSON object matching this exact schema, with no prose before or after:

```json
{
  "ratings": [
    {
      "dimension_id": "D1",
      "score": 4,
      "justification": "ONE short sentence, ≤25 words, citing a paragraph/section when it sharpens the point"
    }
  ]
}
```

There must be exactly one entry per dimension (D1–D7).
"""


def build_reliability_user_prompt(task: dict, annotated_doc: str) -> str:
    """Build the per-task user prompt for the reliability scorecard.

    ``task`` carries the same shape the verdict judge consumes
    (``scenario_id`` / ``side`` / ``level``); ``annotated_doc`` is the
    inline-annotated redline view.
    """
    side_word = "vendor (provider-side)" if task["side"] == "A" else "customer-side"
    header = (
        f"# Task context\n\n"
        f"- Scenario: {task['scenario_id']}\n"
        f"- Side being represented: {task['side']} ({side_word})\n"
        f"- Negotiation turn (level): {task['level']}\n\n"
    )
    dims = "# Reliability dimensions (0–5)\n\n"
    for d in RELIABILITY_DIMENSIONS:
        dims += f"## {d['id']} — {d['name']}\n{d['anchor']}\n\n"
    return header + dims + "# Annotated redlined document\n\n" + annotated_doc


def parse_reliability_json(raw: str) -> dict:
    """Parse the judge's reliability response. Mirrors ``judging.parse_judge_json``'s
    fence/brace tolerance but validates a ``ratings`` list instead of ``verdicts``."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        brace = text.find("{")
        if brace >= 0:
            text = text[brace:]
    data = json.loads(text)
    if "ratings" not in data or not isinstance(data["ratings"], list):
        raise ValueError("reliability response missing 'ratings' list")
    return data


def aggregate_reliability(
    ratings: list[dict], weights: dict[str, float] | None = None,
) -> dict:
    """Tunable weighted-mean reliability score on a 0–5 scale.

    ``weights`` maps dimension id → weight (default 1.0 each = CM-LRS's
    equal-weighted mean); pass a partial map to up/down-weight selected
    dimensions. Missing or non-numeric scores count as 0. Returns the
    weighted mean over [0, 5] plus a per-dimension breakdown.
    """
    by_id: dict[str, dict] = {}
    for r in ratings:
        did = r.get("dimension_id")
        if did and did not in by_id:
            by_id[did] = r
    weight_map = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
    dims_out: list[dict] = []
    weighted_sum = 0.0
    total_weight = 0.0
    for d in RELIABILITY_DIMENSIONS:
        w = float(weight_map.get(d["id"], 1.0))
        r = by_id.get(d["id"], {})
        try:
            score = float(r.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(float(DEFAULT_SCALE), score))
        weighted_sum += score * w
        total_weight += w
        dims_out.append({
            "dimension_id": d["id"], "name": d["name"], "score": round(score, 4),
            "weight": w,
            "justification": r.get("justification", "(judge did not score this dimension)"),
        })
    aggregate_score = weighted_sum / total_weight if total_weight else 0.0
    return {
        "aggregate": round(max(0.0, min(float(DEFAULT_SCALE), aggregate_score)), 4),
        "scale": DEFAULT_SCALE,
        "dimensions": dims_out,
    }


def score_reliability(
    judge_model: str,
    task: dict,
    annotated_doc: str,
    *,
    weights: dict[str, float] | None = None,
) -> dict:
    """Judge a redline across the seven reliability dimensions and aggregate.

    The LLM call funnels through ``judging.call_judge_raw`` so it inherits
    the retry policy and the opt-in ``judge_audit`` trail exactly like a
    verdict judge call — one shared chokepoint, no duplicated logic.
    Returns the weighted-mean scorecard (see :func:`aggregate_reliability`).
    """
    user = build_reliability_user_prompt(task, annotated_doc)
    raw = judging.call_judge_raw(judge_model, RELIABILITY_SYSTEM_PROMPT, user)
    data = parse_reliability_json(raw)
    return aggregate_reliability(data["ratings"], weights=weights)
