#!/usr/bin/env python3
"""Continuous rubric scoring via LLM-as-a-Verifier (logit expectation).

The canonical RedlineBench judge (`judging.py`) asks the model for a *discrete*
PASS/FAIL verdict per rubric. This module implements the core mechanism of
*LLM-as-a-Verifier*: instead of taking the argmax verdict, take the
**expectation over the distribution of scoring-token logits** to get a
continuous score in [0, 1] per rubric.

Adapted from "LLM-as-a-Verifier: A General-Purpose Verification Framework"
(arXiv:2607.05391). The paper's verification framework scales along three
axes, all supported here:

  1. **Score granularity** — score each criterion on a fine-grained 0…(scale-1)
     scale instead of binary {0,1}, and read the expected value of the
     score-token distribution rather than its argmax. Finer scales separate
     borderline cases the discrete verdict collapses.
  2. **Repeated evaluation** — sample the score `repeats` times at nonzero
     temperature and average the expectations; averaging independent draws
     shrinks the variance of the estimate.
  3. **Criteria decomposition** — split a compound criterion into its
     sub-clauses, score each, and average; simpler per-call judgments reduce
     complexity.

What is intentionally NOT ported (out of scope for this repo):
  - The paper's bespoke ranking algorithm over candidate solutions
    (RedlineBench grades one redline per task, not a candidate set).
  - The Terminal-Bench / SWE-Bench / RL-feedback evaluation suites — those
    are downstream-of-evaluation concerns; this module only adds a scoring
    signal the existing panel consumes.

Auxiliary components substituted with target-native equivalents (Mode 2):
  - The paper's separate verifier inference path → the repo's existing
    ``litellm.completion`` judge call, with ``logprobs`` enabled.
"""

from __future__ import annotations

import math
import re

# Default scoring scale: 10 levels (digits 0–9). Each level is a single
# tokenizer token for every mainstream judge family, so its logit shows up in
# ``top_logprobs`` at the score position. Larger scales are supported but may
# span multi-token values (e.g. "10"), which the parser drops rather than
# misreads.
DEFAULT_SCALE = 10

# How many of the leading generated tokens to scan for the score digit. With
# the "output only a single digit" instruction the score lands at position 0,
# but a stray space/prefix token is common on some tokenizers; scanning a
# short window makes the read robust without admitting rambling output.
_SCAN_TOKENS = 3
_MAX_TOKENS = 4

CONTINUOUS_SYSTEM_PROMPT = """\
You are a senior commercial-contracts attorney scoring an AI-generated contract \
redline against ONE rubric criterion. You are STRICT but fair.

Score how completely and correctly the redline satisfies the single criterion \
below, using the inline CriticMarkup rendering:

- `~~strikethrough~~` — a tracked deletion; `++insertion++` — a tracked insertion
- `~~old~~++new++` — a tracked replacement; `{cmt-N}` — a comment (body in the appendix)

Grade the OOXML state of the redline (the inline markers), not the tone of any \
comments. Comments are evidence of intent; they do not substitute for the \
structural change. Do not penalize unrelated edits outside the criterion.

Score on an integer scale where 0 = the redline does not satisfy the criterion \
at all, and {top} = it fully and unambiguously satisfies it. Partial / \
ambiguous / wrong-location edits get an intermediate value.

Output ONLY a single integer digit in [0, {top}] and nothing else — no prose, \
no punctuation, no justification.
"""


def _parse_score_token(token: str, scale: int) -> int | None:
    """Return the integer value of a score token, or None if it isn't a valid
    score on [0, scale). Tolerates surrounding whitespace / stray punctuation
    and tokenizer prefixes (e.g. " 9", "9.")."  """
    t = token.strip().strip(".,;:\"'").strip()
    m = re.match(r"-?\d+", t)
    if not m:
        return None
    v = int(m.group(0))
    return v if 0 <= v < scale else None


def expected_score(score_logprobs: dict, scale: int = DEFAULT_SCALE) -> float:
    """Expected normalized score in [0, 1] from a distribution over score tokens.

    ``score_logprobs`` maps token → log P(token) (the shape ``top_logprobs``
    gives). Non-score tokens are dropped; the retained score values are
    renormalized via softmax over their logprobs, and
    ``E[value] / (scale - 1)`` is returned in [0, 1]. Returns ``nan`` if no
    score token is present in the distribution (the caller falls back).

    Pure / deterministic — the unit-testable core of the verifier.
    """
    best: dict[int, float] = {}
    for tok, lp in score_logprobs.items():
        v = _parse_score_token(tok, scale)
        if v is None:
            continue
        if v not in best or lp > best[v]:
            best[v] = float(lp)
    if not best:
        return math.nan
    m = max(best.values())
    weights = {v: math.exp(lp - m) for v, lp in best.items()}
    z = sum(weights.values())
    return sum(v * w for v, w in weights.items()) / z / (scale - 1)


def _collect_score_logprobs(resp, scan_tokens: int = _SCAN_TOKENS) -> dict:
    """Merge ``top_logprobs`` of the first ``scan_tokens`` generated content
    tokens into a single {token: logprob} map (max logprob wins per token).
    Tolerates dict- and attribute-style litellm response objects."""
    out: dict[str, float] = {}
    try:
        content = resp.choices[0].logprobs.content
    except (AttributeError, IndexError, TypeError):
        return out
    for entry in list(content)[:scan_tokens]:
        tlps = getattr(entry, "top_logprobs", None)
        if tlps is None and isinstance(entry, dict):
            tlps = entry.get("top_logprobs")
        if not tlps:
            continue
        for item in tlps:
            if isinstance(item, dict):
                tok, lp = item.get("token"), item.get("logprob")
            else:
                tok, lp = getattr(item, "token", None), getattr(item, "logprob", None)
            if tok is None or lp is None:
                continue
            if tok not in out or lp > out[tok]:
                out[tok] = float(lp)
    return out


def _generated_text(resp) -> str:
    try:
        return resp.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError):
        return ""


def _argmax_fallback(resp, scale: int) -> float:
    """When the provider returns no usable score-token distribution (some
    backends gate logprobs behind allow-lists), parse the generated text as
    the discrete score and normalize. Continuous-path graceful degradation."""
    text = _generated_text(resp).strip()
    m = re.search(r"\d+", text)
    if not m:
        return 0.0
    v = int(m.group(0))
    v = max(0, min(scale - 1, v))
    return v / (scale - 1)


def score_criterion(
    model: str,
    user: str,
    *,
    system: str = CONTINUOUS_SYSTEM_PROMPT,
    scale: int = DEFAULT_SCALE,
    repeats: int = 1,
    temperature: float = 0.0,
    top_logprobs: int = 20,
) -> float:
    """Continuous [0, 1] score for one criterion via expectation over the
    scoring-token logits.

    ``repeats`` > 1 enables repeated evaluation: samples are drawn at a
    nonzero temperature (auto-raised from 0) and their expectations averaged,
    reducing the variance of the estimate (paper axis 2). A NaN expectation
    (no score token in the distribution) falls back to the argmax text score
    so a single provider quirk never poisons a whole run.
    """
    import litellm

    if repeats < 1:
        repeats = 1
    if repeats > 1 and temperature == 0.0:
        temperature = 0.7
    scores: list[float] = []
    for _ in range(repeats):
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system.format(top=scale - 1)},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=_MAX_TOKENS,
            logprobs=True,
            top_logprobs=top_logprobs,
            timeout=600,
        )
        lp = _collect_score_logprobs(resp)
        s = expected_score(lp, scale)
        if s != s:  # NaN
            s = _argmax_fallback(resp, scale)
        scores.append(s)
    return sum(scores) / len(scores)


def decompose_criterion(criterion: str) -> list[str]:
    """Split a compound rubric criterion into sub-clauses along explicit
    clause boundaries (criteria decomposition — paper axis 3). Conservative:
    splits only on semicolons and coordinating conjunctions that mark distinct
    requirements ("while", "but", "and also", …), never on a bare "and".
    Returns at least ``[criterion]`` unchanged when nothing splits."""
    parts = re.split(
        r"\s*;\s*|\s+\b(?:while|whilst|but|whereas|and also)\b\s+",
        criterion,
        flags=re.IGNORECASE,
    )
    cleaned = [p.strip(" .-") for p in parts if p and p.strip(" .-")]
    return cleaned or [criterion.strip()]


def build_continuous_user_prompt(task: dict, rubric: dict, criterion_text: str, annotated_doc: str) -> str:
    """Focused per-criterion scoring prompt (mirrors the context block
    `judging.build_user_prompt` builds, narrowed to ONE criterion)."""
    side_word = "vendor (provider-side)" if task["side"] == "A" else "customer-side"
    return (
        f"# Task context\n"
        f"- Scenario: {task['scenario_id']}\n"
        f"- Side being represented: {task['side']} ({side_word})\n"
        f"- Negotiation turn (level): {task['level']}\n\n"
        f"# Criterion to score\n"
        f"- id: `{rubric['id']}`\n"
        f"- importance weight: {rubric['weight']}/10\n"
        f"- **criterion**: {criterion_text.strip()}\n\n"
        f"# Annotated redlined document\n\n{annotated_doc}"
    )


def score_rubrics_continuous(
    model: str,
    task: dict,
    annotated_doc: str,
    *,
    scale: int = DEFAULT_SCALE,
    repeats: int = 1,
    decompose: bool = False,
) -> dict:
    """Score every rubric in ``task`` continuously. Returns ``{rubric_id:
    score in [0, 1]}``. With ``decompose=True`` each criterion is split into
    sub-clauses that are scored independently and averaged (axis 3)."""
    out: dict[str, float] = {}
    for r in task["rubrics"]:
        crit = r["criteria"]
        subs = decompose_criterion(crit) if decompose else [crit]
        sub_scores: list[float] = []
        for sub in subs:
            user = build_continuous_user_prompt(task, r, sub, annotated_doc)
            sub_scores.append(
                score_criterion(model, user, scale=scale, repeats=repeats)
            )
        out[r["id"]] = sum(sub_scores) / len(sub_scores)
    return out


def continuous_aggregate(per_rubric_continuous: dict, rubrics: list[dict]) -> dict:
    """Continuous analog of ``judging.aggregate`` / ``panel.weighted_score``.

    Per-rubric continuous scores in [0, 1] replace the binary PASS/FAIL:
    positive-weight rubrics contribute ``weight * score`` to the numerator;
    penalty (negative-weight) rubrics subtract ``|weight| * score``. Denominator
    is the sum of positive weights; result clamped to [0, 1] — the same shape
    and math as the discrete aggregator, generalized to fractional credit.

    Returns a grade dict compatible with ``judging.aggregate``'s output
    (``weighted``, ``per_rubric`` with a discrete ``verdict`` thresholded at
    0.5 so the existing panel majority-vote consumes it unchanged) plus a
    ``continuous`` field per rubric and a ``continuous_weighted`` aggregate.
    """
    earned = penalty = 0.0
    total_positive = 0
    per_rubric = []
    n_pass = 0
    n_total = 0
    for r in rubrics:
        w = int(r["weight"])
        raw = per_rubric_continuous.get(r["id"])
        # NaN (no signal) counts as 0 credit, matching discrete "missing = FAIL".
        s = 0.0 if raw is None or raw != raw else float(raw)
        verdict = "PASS" if s >= 0.5 else "FAIL"
        if w > 0:
            total_positive += w
            earned += w * s
            n_total += 1
            if verdict == "PASS":
                n_pass += 1
        elif w < 0:
            penalty += (-w) * s
        per_rubric.append({
            "rubric_id": r["id"],
            "verdict": verdict,
            "continuous": round(s, 6),
            "weight": w,
            "is_penalty": w < 0,
            "category": r.get("category"),
            "criteria": r["criteria"],
            "justification": f"continuous verifier score={s:.3f}",
        })
    raw_score = (earned - penalty) / total_positive if total_positive else 0.0
    weighted = max(0.0, min(1.0, raw_score))
    return {
        "weighted": weighted,
        "continuous_weighted": weighted,
        "earned_weight": round(earned, 6),
        "penalty_weight": round(penalty, 6),
        "total_weight": total_positive,
        "n_pass": n_pass,
        "n_total": n_total,
        "per_rubric": per_rubric,
    }
