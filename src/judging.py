#!/usr/bin/env python3
"""Canonical judge prompt + scoring for RedlineBench.

This is the importable single source of truth for the judge system prompt,
the per-rubric user prompt, the LLM call, and the weighted/penalty-aware
score aggregation. The per-task verifier (`harbor/tasks/*/tests/judge.py`) is
a vendored, self-contained copy of the same logic so it can run inside the
Harbor container; this module is what the repo-level tools (re-judging, the
judge panel) import.

Re-judging works from the annotated view that every trial already saved to
`verifier/annotated_view.md`, so no .docx re-rendering is needed.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time

MAX_RETRIES = 10

JUDGE_SYSTEM_PROMPT = """\
You are a senior commercial-contracts attorney grading an AI-generated contract redline against a fixed set of rubric criteria. You are STRICT but fair.

# Your job

For each rubric criterion, decide PASS or FAIL.

- PASS means the redline **clearly and unambiguously** satisfies the criterion. The required edit (deletion, insertion, replacement, accept, reject, etc.) is present at the right location with reasoning that lines up with the rubric.
- FAIL means the redline does not satisfy the criterion, or it satisfies it only partially, ambiguously, or in the wrong place. Partial credit is FAIL.

When the criterion uses an active verb like "deletes", "replaces", "rejects", "inserts", "preserves": look for that exact kind of edit in the document.

# What you are looking at

The redlined document is rendered below in CriticMarkup-style inline format:

- `~~strikethrough text~~`   — a tracked deletion (text the redliner removed)
- `++inserted text++`        — a tracked insertion (text the redliner added)
- `~~old~~++new++`           — a tracked replacement (delete + adjacent insert)
- `{cmt-N}`                  — a comment anchor; the full comment body is in the appendix at the bottom of the document, keyed by ID
- Paragraph IDs `[p-NNN]`    — useful when the criterion references a section by number; you can locate the paragraph by reading its text content

Section references in the rubric (e.g. "Section 1.3", "Exhibit A, Section 2") map to sections of the contract. Sections are numbered in the contract's auto-numbered list structure; you may need to scan the document text to find the right paragraph(s).

# How to read each rubric verb

Each criterion uses an active verb that tells you what STRUCTURAL change the redline must contain. Grade primarily on the OOXML state of the redline (the inline markers), not on the tone of any comments. Comments are evidence of intent; they don't substitute for the structural change.

- **"Inserts X"** — PASS iff `++X++` (or a paraphrase that clearly contains X) appears at the right location. A comment proposing X without `++X++` is FAIL.
- **"Deletes X"** — PASS iff `~~X~~` appears at the right location. A comment saying "we should delete X" without `~~X~~` is FAIL.
- **"Replaces X with Y"** — PASS iff `~~X~~++Y++` (or adjacent del+ins covering the substantive swap) appears at the right location.
- **"Rejects [an opposing-side edit]"** — for a side responding to a prior turn: PASS iff the redline contains a tracked change that undoes / strikes through / modifies the opposing edit's content (e.g., `~~opposing-inserted-text~~`, or replacement of an opposing insertion with different language).
- **"Accepts [an opposing-side edit]" / "Preserves X" / "Maintains X" / "Retains X" / "Leaves X"** — for a side responding to a prior turn: PASS iff the opposing-side change is **left structurally intact** — no new tracked change strikes through it, modifies it, replaces it, or contradicts it. **Comments are not dispositive here.** A model may push back, ask to narrow, or request future-turn changes in comments and STILL pass an "Accepts" rubric, so long as the structural state of the targeted text is unchanged in this turn's output. Comments only fail an "Accepts" rubric if the model added a contradicting tracked change in the same turn that effectively undoes the acceptance (e.g., struck through the opposing insertion, replaced it with different language, or inserted a directly contradictory clause that nullifies it).

# Other rules

1. **Justify each verdict** in **ONE short sentence, no more than 25 words**. Cite a paragraph id or section number when it sharpens the point. **No multi-sentence explanations, no preamble, no hedging.** The goal is a glanceable record, not an essay. Examples of the target tone and length:
   - PASS: `"Inserts the 30-day cure right at p-115 (Exhibit A §9) as required."`
   - FAIL: `"Identifies the correct clause in section 13.1 but fails to redline the indemnity piece."`
   - PASS on Accepts: `"AgentCo's insertion at p-084 left intact; no contradicting tracked change."`
   - FAIL on wrong location: `"Edits liability cap in §17 instead of the §16.1 indemnification clause the rubric points at."`
2. **Don't penalize a model for additional edits** outside the rubric — only grade what the rubric asks. The rubric is the ground truth.
   - One exception-shaped case: a rubric may carry a **negative importance weight** (e.g. `-4/10`). That criterion describes an edit the attorney flagged as undesirable. Your job does not change: return PASS iff the document contains the described edit, FAIL otherwise. The scoring layer handles the sign — do not invert your verdict.
3. **Be strict on location**: "Rejects in Section 1.3 the inclusion of PCI-DDS Standards" requires the edit to be in the PCI-DDS provision of the definition section — not, say, an unrelated PCI-related edit elsewhere.
4. **Detect malformed redlines**: if the relevant tracked change exists but contains a contradiction (e.g., both "10 days" AND "30 days" inserted in the same place, or new language that directly conflicts with what the rubric asks to accept), that's a FAIL — the redline didn't cleanly accomplish the criterion.

# Output format

Return ONLY a JSON object matching this exact schema, with no prose before or after:

```json
{
  "verdicts": [
    {
      "rubric_id": "rubric_…",
      "verdict": "PASS" | "FAIL",
      "justification": "ONE short sentence, ≤25 words, citing a paragraph or section when it sharpens the point"
    }
  ]
}
```

There must be exactly one entry per rubric. Use the rubric's `id` field verbatim as `rubric_id`.
"""


def build_user_prompt(task: dict, annotated_doc: str) -> str:
    side_word = "vendor (provider-side)" if task["side"] == "A" else "customer-side"
    header = (
        f"# Task context\n\n"
        f"- Scenario: {task['scenario_id']}\n"
        f"- Side being represented: {task['side']} ({side_word})\n"
        f"- Negotiation turn (level): {task['level']}\n\n"
    )
    rubrics_block = "# Rubrics to grade\n\n"
    for i, r in enumerate(task["rubrics"], 1):
        cat = r.get("category") or "(uncategorized)"
        rubrics_block += (
            f"## Rubric {i}\n"
            f"- id: `{r['id']}`\n"
            f"- category: {cat}\n"
            f"- importance weight: {r['weight']}/10\n"
            f"- **criterion**: {r['criteria'].strip()}\n"
            f"- justification (context for you, not for grading): "
            f"{(r.get('justification') or '').strip()}\n\n"
        )
    return header + rubrics_block + "# Annotated redlined document\n\n" + annotated_doc


def parse_judge_json(raw: str) -> dict:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        brace = text.find("{")
        if brace >= 0:
            text = text[brace:]
    data = json.loads(text)
    if "verdicts" not in data or not isinstance(data["verdicts"], list):
        raise ValueError("judge response missing 'verdicts' list")
    return data


def call_judge(model: str, system: str, user: str) -> dict:
    """Call the judge with retries. No temperature pin (reasoning models reject
    it); request json_object output, degrade once if unsupported; fail fast on
    deterministic 4xx."""
    import litellm

    kwargs: dict = {"response_format": {"type": "json_object"}}
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                timeout=600,
                **kwargs,
            )
            return parse_judge_json(resp.choices[0].message.content or "")
        except Exception as exc:  # noqa: BLE001
            if "response_format" in kwargs and "response_format" in str(exc):
                kwargs.pop("response_format")
                continue
            status = getattr(exc, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise RuntimeError(f"judge request invalid (no retry): {exc!r}") from exc
            last_exc = exc
            time.sleep(min(2**attempt, 60) + random.uniform(0, 1))
    raise RuntimeError(f"judge failed after {MAX_RETRIES} attempts: {last_exc!r}")


def aggregate(verdicts: list[dict], rubrics: list[dict]) -> dict:
    """Weighted score with penalty-rubric support.

    Positive-weight rubrics: PASS earns the weight. Negative-weight (penalty)
    rubrics: PASS subtracts |weight|. Denominator = sum of positive weights;
    final score clamped to [0, 1]. Missing verdicts count as FAIL.
    """
    by_id = {}
    for v in verdicts:
        rid = v.get("rubric_id")
        if rid and rid not in by_id:
            by_id[rid] = v
    per_rubric, earned, penalty, total_positive = [], 0, 0, 0
    for r in rubrics:
        w = int(r["weight"])
        if w > 0:
            total_positive += w
        v = by_id.get(r["id"])
        verdict = (v or {}).get("verdict", "FAIL")
        if verdict not in ("PASS", "FAIL"):
            verdict = "FAIL"
        if verdict == "PASS":
            if w > 0:
                earned += w
            elif w < 0:
                penalty += -w
        per_rubric.append({
            "rubric_id": r["id"], "verdict": verdict, "weight": w,
            "is_penalty": w < 0, "category": r.get("category"),
            "criteria": r["criteria"],
            "justification": (v or {}).get("justification", "(judge did not address this rubric)"),
        })
    raw = (earned - penalty) / total_positive if total_positive else 0.0
    return {
        "weighted": max(0.0, min(1.0, raw)),
        "earned_weight": earned, "penalty_weight": penalty, "total_weight": total_positive,
        "n_pass": sum(1 for p in per_rubric if p["verdict"] == "PASS" and not p["is_penalty"]),
        "n_total": sum(1 for p in per_rubric if not p["is_penalty"]),
        "per_rubric": per_rubric,
    }
