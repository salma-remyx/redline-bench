#!/usr/bin/env python3
"""Judge prompt-design sensitivity diagnostic.

How much do the judge's verdicts move when the system prompt is reformatted,
its instructions are thinned, or the document context is trimmed? A fragile
judge -- one whose verdicts flip under those perturbations -- is exactly the
verdict-fragility regression the audit trail was built to make debuggable.

This is a Mode-2 adapted port of VeyraBench -- "Prompt Design at Scale: How
Format, Instruction Count, and Context Length Shape Instruction Adherence and
Hallucination in Large Language Models" (arXiv:2607.19257) -- which runs a
controlled protocol crossing three prompt-design axes (format x
instruction-count x context-length) and measures how instruction adherence
decays under each.

The paper's CORE MECHANISM is kept at full fidelity: the controlled sweep of
those three axes and the measurement of the adherence delta each one produces.
The AUXILIARY components are substituted with target-native equivalents:

  - VeyraBench's synthetic "Book of Veyra" corpus -> the harness's own cached
    ``annotated_view`` trials (real redlines, real rubrics).
  - the paper's 5-model evaluation panel            -> the configured judge
    model, called through the shared ``judging.call_judge`` chokepoint.
  - the generatable rule-count ladder N = 10..160   -> subsampling the judge
    prompt's fixed set of rubric-verb instructions.
  - the 2k..512k-token context ladder               -> fractional truncation
    of the annotated document the judge reads.
  - the separate VeyraBench harness / eval framework -> an opt-in diagnostic
    emitted into ``metrics_summary``.

Output: a per-axis adherence-delta report. For each axis we report the mean
absolute change in the judge's weighted score and the mean verdict-flip rate
across the sampled trials, so the team can see which prompt-design decisions
their judge is most fragile to.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import judging
from judging import JUDGE_SYSTEM_PROMPT, aggregate, build_user_prompt

# ─── format axis ─────────────────────────────────────────────────────

# The judge's output contract is a fenced JSON schema block. Format
# transforms must not corrupt it, so we lift it out and re-insert it.
_FENCE_RE = re.compile(r"```json\n.*?\n```", re.S)
_SCHEMA_PLACEHOLDER = "\x00SCHEMA\x00"


def _split_schema(prompt: str) -> tuple[str, str]:
    m = _FENCE_RE.search(prompt)
    if not m:
        return prompt, ""
    block = m.group(0)
    return prompt.replace(block, _SCHEMA_PLACEHOLDER), block


def _join_schema(rest: str, schema: str) -> str:
    return rest.replace(_SCHEMA_PLACEHOLDER, schema) if schema else rest


def markdown_to_plain(prompt: str) -> str:
    """Markdown -> plain text: drop heading hashes, bullet markers, and
    emphasis/inline-code ticks while preserving content order."""
    rest, schema = _split_schema(prompt)
    rest = re.sub(r"^#{1,6}\s*", "", rest, flags=re.M)     # headings
    rest = re.sub(r"^\s*[-*+]\s+", "", rest, flags=re.M)   # bullets
    rest = rest.replace("**", "").replace("`", "")         # emphasis / code
    rest = re.sub(r"\n{3,}", "\n\n", rest)
    return _join_schema(rest, schema)


def markdown_to_prose(prompt: str) -> str:
    """Markdown -> flowing prose: headings and bullets become sentences."""
    rest, schema = _split_schema(prompt)
    lines: list[str] = []
    for line in rest.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1]:
                lines.append("")
            continue
        stripped = re.sub(r"^#{1,6}\s*", "", stripped)
        stripped = re.sub(r"^\s*[-*+]\s+", "", stripped)
        stripped = stripped.replace("**", "").replace("`", "")
        if stripped and not stripped.endswith((".", ":", "!", "?")):
            stripped += "."
        lines.append(stripped)
    return _join_schema("\n".join(lines), schema)


def markdown_to_tabular(prompt: str) -> str:
    """Markdown -> tabular: render the rubric-verb rules as a table.

    Only the "How to read each rubric verb" bullets are table-ified -- the
    part of the system prompt where format most plausibly moves compliance.
    Falls back to the original prompt if the section can't be isolated.
    """
    block_re = re.compile(r"(?:^- \*\*\".*\".*$\n?)+", re.M)
    m = block_re.search(prompt)
    if not m:
        return prompt
    block = m.group(0)
    rows = re.findall(r"- \*\*\"(.*?)\"\*\*\s*(.*)", block)
    if not rows:
        return prompt
    table = ["| Verb | What PASS requires |", "| --- | --- |"]
    for verb, body in rows:
        table.append(f'| "{verb.strip()}" | {body.strip()} |')
    return prompt.replace(block, "\n".join(table) + "\n")


FORMAT_VARIANTS: dict[str, callable] = {
    "plain_text": markdown_to_plain,
    "prose": markdown_to_prose,
    "tabular": markdown_to_tabular,
}


# ─── instruction-count axis ──────────────────────────────────────────

# The quoted-verb rules under "How to read each rubric verb" are the
# discrete instructions whose count we vary.
_VERB_RULE_RE = re.compile(r"^- \*\*\".*$", re.M)


def instruction_count_variants(prompt: str, counts: list[int]) -> dict[int, str]:
    """Subsample the judge's verb-rule instructions to ``counts`` sizes.

    VeyraBench sweeps a generatable rule count N = 10..160 and watches
    instruction-following collapse; the judge prompt carries a *fixed*
    instruction set, so this axis is approximated by varying how many of
    the verb-rule instructions the system prompt actually presents. Each
    requested count < the full set produces a thinned prompt.
    """
    rules = _VERB_RULE_RE.findall(prompt)
    if not rules:
        return {}
    full_block = "\n".join(rules)
    out: dict[int, str] = {}
    for c in counts:
        if c >= len(rules):
            out[c] = prompt
        else:
            out[c] = prompt.replace(full_block, "\n".join(rules[:c]), 1)
    return out


# ─── context-length axis ─────────────────────────────────────────────

_DOC_MARKER = "# Annotated redlined document"
_TRUNC_SENTINEL = "[context-truncated]"


def truncate_context(user_prompt: str, fraction: float) -> str:
    """Keep the task/rubric header and trim the annotated-document body to
    ``fraction`` of its length -- the judge-relevant analog of VeyraBench's
    2k..512k-token context ladder (recall degrades near the ceiling)."""
    head, sep, doc = user_prompt.partition(_DOC_MARKER)
    if not sep or fraction >= 1.0:
        return user_prompt
    keep = max(1, int(len(doc) * fraction))
    return head + _DOC_MARKER + doc[:keep] + f"\n\n{_TRUNC_SENTINEL}\n"


# ─── adherence delta ─────────────────────────────────────────────────


def _weighted(verdicts: list[dict], rubrics: list[dict]) -> float:
    return aggregate(verdicts, rubrics)["weighted"]


def flip_rate(baseline: list[dict], variant: list[dict]) -> float:
    """Fraction of rubric ids whose verdict differs between baseline and
    variant. A rubric present in only one side counts as a flip."""
    bv = {v.get("rubric_id"): v.get("verdict") for v in baseline}
    vv = {v.get("rubric_id"): v.get("verdict") for v in variant}
    ids = set(bv) | set(vv)
    if not ids:
        return 0.0
    flips = sum(1 for i in ids if bv.get(i) != vv.get(i))
    return round(flips / len(ids), 4)


def analyze_trial(
    task_ctx: dict,
    annotated_view: str,
    judge_model: str,
    *,
    call_fn: callable,
    instruction_counts: tuple[int, ...] = (2, 4),
    context_fractions: tuple[float, ...] = (0.5, 0.25),
) -> dict:
    """Re-judge one trial under the canonical prompt and every perturbation,
    returning the per-variant adherence delta vs the canonical baseline."""
    rubrics = task_ctx["rubrics"]
    user = build_user_prompt(task_ctx, annotated_view)
    base = call_fn(judge_model, JUDGE_SYSTEM_PROMPT, user)["verdicts"]
    base_score = _weighted(base, rubrics)

    perturbations: list[tuple[str, str, str, str]] = []  # axis, label, system, user

    for label, fn in FORMAT_VARIANTS.items():
        perturbations.append(("format", label, fn(JUDGE_SYSTEM_PROMPT), user))
    for c, sys_v in instruction_count_variants(
        JUDGE_SYSTEM_PROMPT, list(instruction_counts)
    ).items():
        perturbations.append(("instruction_count", str(c), sys_v, user))
    for f in context_fractions:
        perturbations.append(
            ("context_length", f"{int(f * 100)}pct", JUDGE_SYSTEM_PROMPT,
             truncate_context(user, f))
        )

    rows = []
    for axis, label, sys_v, usr_v in perturbations:
        verdicts = call_fn(judge_model, sys_v, usr_v)["verdicts"]
        score = _weighted(verdicts, rubrics)
        rows.append({
            "axis": axis,
            "variant": label,
            "score": round(score, 4),
            "score_delta": round(score - base_score, 4),
            "abs_score_delta": round(abs(score - base_score), 4),
            "flip_rate": flip_rate(base, verdicts),
        })
    return {"baseline_score": round(base_score, 4), "variants": rows}


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def summarize(results: list[dict]) -> dict:
    """Aggregate per-trial variant rows into per-axis fragility stats."""
    by_axis: dict[str, list[dict]] = {}
    for r in results:
        for row in r["variants"]:
            by_axis.setdefault(row["axis"], []).append(row)

    axes: dict[str, dict] = {}
    for axis, rows in by_axis.items():
        by_variant: dict[str, list[dict]] = {}
        for row in rows:
            by_variant.setdefault(row["variant"], []).append(row)
        axes[axis] = {
            "n_variant_trials": len(rows),
            "mean_abs_score_delta": _mean([r["abs_score_delta"] for r in rows]),
            "mean_flip_rate": _mean([r["flip_rate"] for r in rows]),
            "by_variant": {
                v: {
                    "mean_abs_score_delta": _mean(
                        [r["abs_score_delta"] for r in vs]
                    ),
                    "mean_flip_rate": _mean([r["flip_rate"] for r in vs]),
                }
                for v, vs in sorted(by_variant.items())
            },
        }
    return {
        "n_trials": len(results),
        "mean_baseline_score": (
            _mean([r["baseline_score"] for r in results]) if results else 0.0
        ),
        "axes": axes,
        "note": (
            "Higher abs_score_delta / flip_rate => the judge's verdicts are "
            "more fragile to that prompt-design axis."
        ),
    }


# ─── trial discovery + entry point ───────────────────────────────────


def _find_annotated_view(trial: Path) -> Path | None:
    for cand in (trial / "verifier" / "annotated_view.md",
                 trial / "annotated_view.md"):
        if cand.exists():
            return cand
    hits = list(trial.rglob("annotated_view.md"))
    return hits[0] if hits else None


def discover_trials(
    runs_dir: str | Path,
) -> list[tuple[dict, str, dict]]:
    """Return ``(task_ctx, annotated_view, meta)`` for every trial under
    ``runs_dir`` that has both a ``grade.json`` and a findable
    ``annotated_view.md``. Reuses the same rubric shape ``rejudge`` builds."""
    out: list[tuple[dict, str, dict]] = []
    for grade_p in sorted(Path(runs_dir).rglob("grade.json")):
        trial = grade_p.parent
        view = _find_annotated_view(trial)
        if view is None:
            continue
        try:
            grade = json.loads(grade_p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        per = grade.get("score", {}).get("per_rubric") or []
        if not per:
            continue
        task_ctx = {
            "scenario_id": grade.get("scenario_id", 0),
            "side": grade.get("side", "A"),
            "level": grade.get("level", 1),
            "rubrics": [
                {
                    "id": p.get("rubric_id"),
                    "criteria": p.get("criteria", ""),
                    "weight": p.get("weight", 1),
                    "category": p.get("category"),
                    "justification": "",
                }
                for p in per
            ],
        }
        meta = {"trial": trial.name, "model": grade.get("model", "unknown")}
        out.append((task_ctx, view.read_text(), meta))
    return out


def analyze_runs(
    runs_dir: str | Path,
    judge_model: str,
    *,
    sample: int = 8,
    call_fn: callable | None = None,
    instruction_counts: tuple[int, ...] = (2, 4),
    context_fractions: tuple[float, ...] = (0.5, 0.25),
    seed: int = 0,
) -> dict:
    """Discover cached trials, re-judge each under the canonical prompt and
    every prompt-design perturbation, and return per-axis adherence-delta
    diagnostics. ``call_fn`` defaults to the shared ``judging.call_judge``
    chokepoint; tests inject a fake. ``sample`` caps the trial count (0 = all).
    """
    if call_fn is None:
        call_fn = judging.call_judge
    trials = discover_trials(runs_dir)
    discovered = len(trials)
    if sample and len(trials) > sample:
        trials = random.Random(seed).sample(trials, sample)

    results = [
        analyze_trial(
            ctx, view, judge_model,
            call_fn=call_fn,
            instruction_counts=instruction_counts,
            context_fractions=context_fractions,
        )
        for ctx, view, _ in trials
    ]
    summary = summarize(results)
    summary.update({
        "judge_model": judge_model,
        "sample": sample,
        "discovered_trials": discovered,
        "analyzed_trials": len(results),
    })
    return summary
