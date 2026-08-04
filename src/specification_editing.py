"""Specification-editing metrics: repair progress, unintended change
rate, and a binary specification reward.

Vector-Bench (arxiv:2607.19056v1, "Can Models Surgically Edit SVG
Code?") frames instruction-based editing as TWO simultaneous
requirements — make the requested change, AND leave everything else
alone — and shows the second half is easy to miss when an output is
judged only for whether it looks right. It captures the "leave the rest
alone" half with three signals:

  - **repair progress** — fraction of requested repairs addressed,
  - **unintended change rate (UCR)** — fraction of protected objects
    that were corrupted, and
  - **binary specification reward** — made every requested edit AND
    corrupted nothing AND produced a valid output (their headline
    metric; the strongest of 34 endpoints reaches only 15%).

RedlineBench already has the docx plumbing to score the exact same
shape for contract redlining. The human attorney's
`attorney_redlines.docx` defines which paragraphs a careful editor
SHOULD touch (the requested edits) and, by omission, which must be left
alone (the protected body). This module ports the three Vector-Bench
signals onto that plumbing — paragraph-level positional identity stands
in for Vector-Bench's SVG "object" identity, and docx load-validity
stands in for SVG validity. No judge is involved; it reads the
redline.docx files directly, exactly like
`docx_metrics.compute_surgicalness`.

Turn-1 only, for the same reason `docx_metrics.compute_verbosity_turn1`
is: at turn 1 both the expert and the model edit the *same clean
template*, so positional paragraph indices align (expert paragraph i
== model paragraph i). Later turns layer seed edits and break that
alignment.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean

# Reuses the docx-loading + touched-paragraph plumbing already proven on
# surgicalness / verbosity rather than re-walking the OOXML tree.
from docx_metrics import (
    _iter_paragraphs,
    _load_document_xml,
    _touched_paragraph_indices,
)


def _score_task(
    model_docx: Path,
    expert_docx: Path,
    *,
    model_author_substring: str | None,
    expert_author_substring: str | None,
) -> dict | None:
    """Score one (model redline, expert redline) pair at turn 1.

    Returns None when the expert reference can't be read — no spec to
    score against. Otherwise returns a dict with `valid_output` and,
    for valid outputs, `repair_progress`, `unintended_change_rate`, and
    `specification_reward` (1.0 iff every requested paragraph was
    touched AND no protected paragraph was touched).
    """
    expert_root = _load_document_xml(expert_docx)
    if expert_root is None:
        return None  # no specification reference → unscorable

    model_root = _load_document_xml(model_docx)
    if model_root is None:
        # Invalid candidate output: gates the reward to 0 and is
        # excluded from the repair-progress / UCR means (validity-
        # gated, matching Vector-Bench's valid-output UCR).
        return {"valid_output": False}

    # Requested edits = paragraphs the attorney touched. Protected body
    # = every other paragraph position in the shared template.
    requested = _touched_paragraph_indices(expert_docx, expert_author_substring)
    model_touched = _touched_paragraph_indices(model_docx, model_author_substring)

    # Tracked-change edits preserve `<w:p>` elements (a deleted
    # paragraph stays in the tree with `<w:del>` inside), so expert and
    # model paragraph counts agree at turn 1; take the union so a
    # structurally inserted paragraph also reads as collateral on the
    # protected body.
    n_universe = max(
        sum(1 for _ in _iter_paragraphs(expert_root)),
        sum(1 for _ in _iter_paragraphs(model_root)),
    )
    protected = set(range(n_universe)) - requested

    repair_progress = (
        len(model_touched & requested) / len(requested) if requested else None
    )
    ucr = len(model_touched & protected) / len(protected) if protected else None

    # Binary specification reward: addressed EVERY requested edit AND
    # corrupted NO protected paragraph (vacuously satisfied on an empty
    # side). Exact 1.0 / 0.0 equality is safe — these are ratios of
    # integers that are whole only at full coverage / zero corruption.
    fully_repaired = repair_progress in (None, 1.0)
    nothing_corrupted = ucr in (None, 0.0)
    specification_reward = 1.0 if (fully_repaired and nothing_corrupted) else 0.0

    return {
        "valid_output": True,
        "repair_progress": repair_progress,
        "unintended_change_rate": ucr,
        "specification_reward": specification_reward,
    }


def compute_specification_editing(
    by_model_turn1: dict[str, list[tuple[str, Path, Path | None]]],
    expert_turn1_docx_by_task: dict[str, Path],
    *,
    model_author_substring: str | None = "Reviewing Counsel",
    expert_author_substring: str | None = None,
) -> dict[str, dict]:
    """Per-model Vector-Bench specification signals at turn 1.

    Mirrors `docx_metrics.compute_verbosity_turn1`'s contract so it
    drops into the same wiring: `by_model_turn1` is
    `{model: [(task_name, model_docx, expert_docx_or_None), …]}`.
    `expert_turn1_docx_by_task` is kept for signature parity (the
    per-task expert docx is read from the tuples).

    For each model returns:

      - `repair_progress` — mean, over turn-1 tasks with a non-empty
        expert spec, of the fraction of attorney-touched paragraphs the
        model also touched (did it make the requested changes?). HIGHER
        is better.
      - `unintended_change_rate` — mean, over valid-output turn-1
        tasks, of the fraction of protected (attorney-untouched)
        paragraphs the model corrupted (did it leave the rest alone?).
        LOWER is better.
      - `specification_success_rate` — fraction of turn-1 tasks that
        earned the full binary reward (repaired everything, corrupted
        nothing, valid output). Vector-Bench's headline metric.
      - `valid_output_rate` — fraction of turn-1 tasks whose model
        redline.docx parsed.
      - `n_tasks` / `n_tasks_valid` — counts behind the means.

    The expert attorney redline defines the specification — the
    requested edits plus the protected body — so it is the reference
    these signals score against, not a peer actor; no `expert` baseline
    row is emitted (unlike surgicalness / verbosity).
    """
    out: dict[str, dict] = {}
    for model, items in by_model_turn1.items():
        repair_vals: list[float] = []
        ucr_vals: list[float] = []
        spec_flags: list[int] = []
        valid_flags: list[int] = []
        n_tasks = 0
        for _task_name, model_docx, expert_docx in items:
            if expert_docx is None:
                continue  # no expert reference for this task → unscorable
            scored = _score_task(
                model_docx,
                expert_docx,
                model_author_substring=model_author_substring,
                expert_author_substring=expert_author_substring,
            )
            if scored is None:
                continue
            n_tasks += 1
            valid_flags.append(1 if scored["valid_output"] else 0)
            if not scored["valid_output"]:
                # Invalid output can still earn the binary reward?
                # No — validity is a conjunct of the spec reward.
                spec_flags.append(0)
                continue
            if scored["repair_progress"] is not None:
                repair_vals.append(scored["repair_progress"])
            if scored["unintended_change_rate"] is not None:
                ucr_vals.append(scored["unintended_change_rate"])
            spec_flags.append(int(scored["specification_reward"]))
        out[model] = {
            "repair_progress": round(mean(repair_vals), 4) if repair_vals else 0.0,
            "unintended_change_rate": (
                round(mean(ucr_vals), 4) if ucr_vals else 0.0
            ),
            "specification_success_rate": (
                round(mean(spec_flags), 4) if spec_flags else 0.0
            ),
            "valid_output_rate": (
                round(mean(valid_flags), 4) if valid_flags else 0.0
            ),
            "n_tasks": n_tasks,
            "n_tasks_valid": sum(valid_flags),
        }
    return out
