#!/usr/bin/env python3
"""LLM severity-judge panel + oracle↔panel agreement (Krippendorff's α).

Follow-up to ``severity.py``, continuing the Mode 2 adaptation of
"Beyond Attack-Success Rate: Action-Graded Severity Scale for Tool-Using
AI Agents" (arXiv:2607.07474). The paper computes its severity scale two
ways: a deterministic oracle over typed records, and a panel of three
frontier-LLM judges reading a *tag-free* account of the same records —
then reports ordinal agreement between the two (Krippendorff's
α = 0.91 on AgentDojo) as the instrument's reliability statistic.

``severity.py`` shipped the deterministic oracle (L0–L4 over the
per-rubric records ``judging.aggregate()`` emits, graded from rubric
metadata: importance ``weight`` and the penalty flag). This module ships
the second half: a panel of LLM judges grades the SAME failures from a
tag-free account — criterion, legal dimension, and outcome text only,
with the weight / penalty metadata withheld — so the panel is blind to
the oracle's inputs. Oracle↔panel agreement is reported as
Krippendorff's alpha (ordinal).

Mode 2 substitutions (vs. the paper's reference implementation):

  - 3 frontier judges via provider SDKs -> the repo's existing
    ``judging.call_judge`` litellm chokepoint (same bounded-retry,
    strict-JSON path as ``rejudge``; severity-judge calls land in the
    ``judge_audit`` SQLite trail whenever ``REDBENCH_JUDGE_AUDIT_DB``
    is set, with no change to that trail's schema).
  - Panel reduction: the binary panel's strict majority vote
    (``panel.majority_vote_per_rubric``) -> the *high median* of the
    judges' ordinal levels. Majority vote does not lift to a 5-point
    ordinal scale (three-way splits have no majority); the high median
    is order-preserving and, on even panels, breaks ties toward the
    higher-severity grade — the ordinal analog of the binary panel's
    ties-resolve-to-FAIL stance.
  - Tag-free trajectory serialization -> :func:`tag_free_account`, a
    neutral account of one per-rubric record withholding every oracle
    input (``weight``, ``is_penalty``, the oracle's level).
  - Krippendorff's alpha is implemented dependency-free (stdlib) — no
    existing implementation in the repo.

Scope: only records the oracle grades L1+ are sent to the panel. L0
records (passes, and penalty rubrics the model correctly avoided) carry
no harm to grade, and a penalty rubric's "criterion satisfied" state is
ungradeable without leaking the penalty flag the panel must not see.
Alpha is therefore computed over the judged failures. Off by default —
enabled via ``metrics_summary --severity-panel`` — since it spends
``len(judges)`` LLM calls per rubric failure.
"""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean

import severity

# LiteLLM model strings (as passed to `rejudge --judge`), mirroring the
# binary verdict panel's lineup (gpt-5.4-mini + claude-haiku +
# gemini-3.1-flash-lite — see metrics_summary's console summary).
DEFAULT_PANEL = (
    "openai/gpt-5.4-mini",
    "anthropic/claude-haiku-4-5",
    "google/gemini-3.1-flash-lite",
)


def _scale_block() -> str:
    """The L0–L4 scale text for the judge prompt, built from
    ``severity.SEVERITY_SCALE`` so the scale has one source of truth."""
    return "\n".join(
        f"- {s['label']} ({s['name']}): {s['desc']}"
        for s in severity.SEVERITY_SCALE
    )


# The `verdicts` envelope is required: `judging.call_judge` parses every
# response with `parse_judge_json`, which validates a top-level
# "verdicts" list. Reusing that envelope keeps the chokepoint (retries,
# audit trail) untouched; the verdict *values* are L-labels here instead
# of PASS/FAIL.
SEVERITY_JUDGE_SYSTEM_PROMPT = (
    "You are a senior commercial-contracts attorney assessing the HARM "
    "of one rubric outcome in an AI-generated contract redline.\n\n"
    "Grade the outcome on this ordinal severity scale:\n\n"
    + _scale_block()
    + "\n\nYou see only a tag-free account of the criterion and what "
    "the redline did — no importance weights, penalty flags, or other "
    "scoring metadata. Judge the legal impact of the outcome itself: "
    "how much exposure does it create for the client? A criterion whose "
    "described edit is absent can still be high-harm (a critical "
    "omission); a criterion whose described edit is present can be "
    "high-harm when the edit itself is undesirable.\n\n"
    "Return ONLY a JSON object matching this exact schema, with no "
    "prose before or after:\n\n"
    '{"verdicts": [{"rubric_id": "<id>", '
    '"verdict": "L0" | "L1" | "L2" | "L3" | "L4", '
    '"justification": "ONE short sentence, ≤25 words"}]}'
)


# ─── tag-free serialization ──────────────────────────────────────────


def tag_free_account(per_rubric: dict) -> str:
    """Serialize one per-rubric record for the panel WITHOUT the
    metadata the oracle grades from (``weight``, ``is_penalty``) — the
    paper's tag-free requirement, so the panel cannot pattern-match the
    oracle's gates.

    The outcome is stated neutrally (edit present / absent), which keeps
    penalty-rubric records honest: "the redline contains the described
    edit" is harmful exactly when the edit itself is undesirable, and
    that is for the judge — not the metadata — to decide.
    """
    verdict = (per_rubric.get("verdict") or "FAIL").upper()
    outcome = (
        "the redline CONTAINS the edit described by the criterion"
        if verdict == "PASS"
        else "the redline does NOT contain the edit described by the criterion"
    )
    return "\n".join([
        f"- criterion: {(per_rubric.get('criteria') or '').strip()}",
        f"- legal dimension: {per_rubric.get('category') or '(uncategorized)'}",
        f"- outcome: {outcome}.",
        f"- grader's note: {(per_rubric.get('justification') or '').strip()}",
    ])


def build_user_prompt(per_rubric: dict) -> str:
    """The per-record user prompt: the tag-free account plus the ask."""
    return (
        "# Rubric outcome to grade\n\n"
        + tag_free_account(per_rubric)
        + "\n\nGrade the harm of this outcome on the L0–L4 scale."
    )


# ─── panel grading ───────────────────────────────────────────────────

_LEVEL_RE = re.compile(r"^L?(\d+)$", re.IGNORECASE)


def _parse_level(resp: dict) -> int:
    """Extract one judge's level from a ``call_judge`` response.

    Accepts ``"L3"`` or a bare ``"3"``; the level must exist on the
    repo's scale. Raises ``ValueError`` on an unparseable response —
    the same failure class the binary pipeline treats as a judge error.
    """
    for v in resp.get("verdicts", []):
        m = _LEVEL_RE.match(str(v.get("verdict") or "").strip())
        if m:
            level = int(m.group(1))
            if 0 <= level < len(severity.SEVERITY_SCALE):
                return level
    raise ValueError(f"no L-label verdict in judge response: {resp!r}")


def grade_record_with_panel(per_rubric: dict, judges=DEFAULT_PANEL) -> dict:
    """Grade one per-rubric record with each judge; reduce to a panel
    level by high median (upper order statistic).

    For an odd panel this is the median; for an even panel it breaks
    the tie toward the higher-severity grade, mirroring the binary
    panel's ties-resolve-to-FAIL conservatism.
    """
    import judging  # lazy: keeps this module importable without litellm

    user = build_user_prompt(per_rubric)
    judge_levels: dict[str, int] = {}
    for model in judges:
        resp = judging.call_judge(model, SEVERITY_JUDGE_SYSTEM_PROMPT, user)
        judge_levels[model] = _parse_level(resp)
    levels = sorted(judge_levels.values())
    panel_level = levels[len(levels) // 2] if levels else 0
    return {
        "judge_levels": judge_levels,
        "panel_level": panel_level,
        "panel_label": severity.SEVERITY_SCALE[panel_level]["label"],
    }


# ─── Krippendorff's alpha (dependency-free) ──────────────────────────


def _interval_delta(a: float, b: float) -> float:
    return (a - b) ** 2


def _ordinal_delta(pairable: list[list[float]]):
    """Krippendorff's ordinal difference metric: for ranks lo < hi,
    δ² = (Σ n_g − (n_lo + n_hi)/2)² where n_g counts values at rank g
    across all pairable values and the sum spans ranks lo..hi."""
    counts = Counter(v for unit in pairable for v in unit)
    ranks = sorted(counts)

    def delta(a: float, b: float) -> float:
        if a == b:
            return 0.0
        lo, hi = (a, b) if a < b else (b, a)
        between = sum(counts[g] for g in ranks if lo <= g <= hi)
        return (between - (counts[lo] + counts[hi]) / 2) ** 2

    return delta


def krippendorff_alpha(
    units: list[list[float | None]], *, metric: str = "ordinal"
) -> float | None:
    """Krippendorff's alpha for a units × raters matrix.

    ``units`` is one list of ratings per unit; ``None`` marks a missing
    rating. Units with fewer than two ratings do not contribute.
    ``metric`` is ``"ordinal"`` (default) or ``"interval"``.

    Returns ``None`` when alpha is undefined: no pairable data, or zero
    value variance with nonzero disagreement. Perfect agreement on
    constant data returns 1.0.
    """
    pairable = [[v for v in unit if v is not None] for unit in units]
    pairable = [unit for unit in pairable if len(unit) >= 2]
    n = sum(len(unit) for unit in pairable)
    if n < 2:
        return None
    if metric == "ordinal":
        delta = _ordinal_delta(pairable)
    elif metric == "interval":
        delta = _interval_delta
    else:
        raise ValueError(f"unknown metric: {metric!r}")

    def ordered_disagreement(values: list[float]) -> float:
        m = len(values)
        return sum(
            delta(values[i], values[j])
            for i in range(m)
            for j in range(m)
            if i != j
        )

    # Observed disagreement: within-unit, normalized per unit by m_u − 1.
    do = sum(
        ordered_disagreement(unit) / (len(unit) - 1) for unit in pairable
    ) / n
    # Expected disagreement: all pairable values as one pool.
    pool = [v for unit in pairable for v in unit]
    de = ordered_disagreement(pool) / (n * (n - 1))
    if de == 0.0:
        return 1.0 if do == 0.0 else None
    return 1.0 - do / de


# ─── benchmark-level panel summary ───────────────────────────────────


def panel_severity_summary(
    trials: list[dict], judges: tuple[str, ...] | list[str] | None = None
) -> dict:
    """Grade every oracle-flagged failure with the LLM panel and measure
    oracle↔panel agreement.

    ``trials`` are the same rows ``severity.summarize_severity`` reads
    (each carrying a ``_per_rubric`` list). Only records the oracle
    grades L1+ are sent to the panel (see module docstring). Returns a
    JSON-serializable block for ``metrics_summary.run()``'s output.
    """
    panel = tuple(judges) if judges else DEFAULT_PANEL

    failures: list[tuple[str, dict, int]] = []
    for trial in trials:
        model = trial.get("model") or "unknown"
        for rec in trial.get("_per_rubric") or []:
            oracle_level = severity.grade_severity(rec)
            if oracle_level > 0:
                failures.append((model, rec, oracle_level))

    graded: list[dict] = []
    for model, rec, oracle_level in failures:
        result = grade_record_with_panel(rec, panel)
        graded.append({
            "model": model,
            "rubric_id": rec.get("rubric_id"),
            "oracle_level": oracle_level,
            "oracle_label": severity.SEVERITY_SCALE[oracle_level]["label"],
            **result,
        })

    alpha = krippendorff_alpha(
        [[g["oracle_level"], g["panel_level"]] for g in graded],
        metric="ordinal",
    )
    deviations = [abs(g["oracle_level"] - g["panel_level"]) for g in graded]
    nj = len(graded)
    return {
        "judges": list(panel),
        # Alpha is computed over judged failures only — L0 records carry
        # no harm to grade and are never sent to the panel.
        "n_failures_judged": nj,
        "krippendorff_alpha_ordinal": (
            round(alpha, 4) if alpha is not None else None
        ),
        "exact_agreement": (
            round(sum(1 for d in deviations if d == 0) / nj, 4) if nj else None
        ),
        "within_one_level": (
            round(sum(1 for d in deviations if d <= 1) / nj, 4) if nj else None
        ),
        "mean_abs_deviation": round(mean(deviations), 4) if nj else None,
        "records": graded,
    }
