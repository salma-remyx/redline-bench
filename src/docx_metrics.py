"""Docx-driven metrics: surgicalness + verbosity trap.

Two metrics that read each actor's `redline.docx` directly (no judge
involved) and characterize the *shape* of their edits rather than
their substantive correctness:

  - **Surgicalness** classifies each tracked-change event as
    *inline* (<30% of the containing paragraph's text length) or
    *block* (>=30%). A human attorney's edits land roughly half inline,
    half block; current frontier models lean ~90% block (they tend to
    gut a paragraph and rewrite it whole).

  - **Verbosity trap** (turn-1 only) compares each model's redline
    volume + clustering against the human expert's redline on the
    same source document. The expert touches many more paragraphs
    in many smaller doses; models touch fewer paragraphs in much
    bigger doses. We also include the model's paragraph-overlap rate
    against the expert (|model ∩ expert| / |model|) — at turn 1 the
    positional indices align because the input is the clean template.

This module walks the docx zip + `word/document.xml` directly with
lxml (no python-docx dependency at runtime).
"""

from __future__ import annotations

import re
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import mean

from lxml import etree


# OOXML namespace URI for the WordprocessingML schema; every revision
# element uses this prefix. We pin it as a constant so namespace-aware
# lxml queries don't have to recompute it.
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _qn(local: str) -> str:
    """Build a Clark-notation tag name (`{uri}localname`) the way lxml
    expects for namespaced lookups."""
    return f"{{{_W_NS}}}{local}"


# `INLINE_BLOCK_THRESHOLD = 0.30` is the inline/block split: edits
# touching less than 30% of their containing paragraph are *inline*
# (surgical word/phrase changes); 30%+ are *block* (sentence or
# paragraph rewrites). Validated on this corpus — expert insertions
# cluster at ~4% of paragraph length, model insertions cluster well
# above 100%.
INLINE_BLOCK_THRESHOLD = 0.30


# Elements that appear between revision elements but should not break
# del→ins pairing (comment markers, bookmarks, proof-error flags).
_TRANSPARENT_TAGS = {
    "commentRangeStart",
    "commentRangeEnd",
    "commentReference",
    "bookmarkStart",
    "bookmarkEnd",
    "proofErr",
}


# Revision element local names — used both by the paragraph-touched
# scan and the paragraph-pair walker. `moveFrom`/`moveTo` are treated
# as del/ins respectively (a move is a delete + insert).
_REVISION_TAGS = {"ins", "del", "moveFrom", "moveTo"}


# ─── docx loading ───────────────────────────────────────────────────


def _load_document_xml(docx: Path):
    """Open a docx zip and return the parsed `word/document.xml` root
    element. Returns None if the file isn't a valid docx or the part is
    missing."""
    if not docx.exists():
        return None
    try:
        with zipfile.ZipFile(docx, "r") as zf:
            blob = zf.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError, OSError):
        return None
    try:
        return etree.fromstring(blob)
    except etree.XMLSyntaxError:
        return None


def _paragraph_text_length(p_elem) -> int:
    """Length of the paragraph's UNCHANGED plain-text content — text in
    `<w:t>` elements that are NOT wrapped in any revision element
    (`<w:ins>`, `<w:del>`, `<w:moveFrom>`, `<w:moveTo>`).

    Conceptually this is the "unchanged baseline" — text that was
    neither inserted nor deleted. It's a smaller denominator than the
    pre-edit or post-edit lengths, which makes the inline/block split
    bite harder on big insertions (a 400-char insertion against a
    100-char unchanged baseline → 400% → block).

    This matches python-docx's `Paragraph.text` (top-level `<w:r>`,
    skipping revision wrappers), so callers that compute the
    denominator either way produce the same number.
    """
    total = 0

    def _walk(node, in_revision: bool) -> None:
        nonlocal total
        tag = etree.QName(node).localname
        if tag == "t" and not in_revision:
            if node.text:
                total += len(node.text)
            return
        next_in_rev = in_revision or tag in ("ins", "del", "moveFrom", "moveTo")
        for child in node.iterchildren():
            _walk(child, next_in_rev)

    _walk(p_elem, False)
    return total


def _ins_text_length(el) -> int:
    """Char count of every `<w:t>` inside an `<w:ins>`."""
    return sum(len(t.text or "") for t in el.iter(_qn("t")))


def _del_text_length(el) -> int:
    """Char count of every `<w:delText>` inside a `<w:del>`."""
    return sum(len(t.text or "") for t in el.iter(_qn("delText")))


def _iter_paragraphs(root):
    """Yield every `<w:p>` element in document order — including
    paragraphs nested inside `<w:tbl>`/`<w:tc>` (table cells).

    The walker is recursive on purpose. Table-nested paragraphs are
    legitimate contract content — schedule rows (pricing, deliverables,
    SLAs in Exhibit A/B/C), signature blocks, party-info grids — and
    the expert attorney spends substantial effort editing them. In
    this 140-task corpus the recursive walker picks up ~1058 extra
    expert events (mostly schedule + signature-block edits) compared
    to a body-only walker. Models, by contrast, barely touch tables:
    GPT-5.5, Opus 4.8, and Claude Fable 5 each made zero table edits
    across all 140 tasks; Gemini 3.5 made 44. The surgicalness metric
    should reflect this full edit corpus.
    """
    for p in root.iter(_qn("p")):
        yield p


# ─── surgicalness walker ────────────────────────────────────────────


def _walk_paragraph_pairs(
    p_elem,
    author_substring: str | None,
    *,
    paragraph_text_length: int = 0,
    inline_block_threshold: float = INLINE_BLOCK_THRESHOLD,
) -> tuple[list[int], list[int], list[float], int, list[str], list[int]]:
    """Return `(insert_lengths, delete_lengths, replace_ratios,
    n_touched, event_kinds, event_sizes)` for one paragraph.

    Walks children in document order, pairing each `<w:del>` with the
    immediately-following `<w:ins>` from the same author (transparent
    tags in between are tolerated). A paired del+ins is one *event*;
    an unpaired del or ins is one event. The event's size is the LARGER
    of its del/ins halves (so a 5-char delete paired with a 200-char
    insert is a block event because of the insert). Each event is
    classified inline vs block by `size / paragraph_text_length` against
    `INLINE_BLOCK_THRESHOLD`. `event_sizes` is the parallel list of
    those sizes in characters — used to compute "average edit length".

    `author_substring=None` returns every author's edits; passing a
    substring filters to one actor (used to scope a docx with multiple
    authors to just the model's edits).
    """
    insert_lengths: list[int] = []
    delete_lengths: list[int] = []
    replace_ratios: list[float] = []
    event_kinds: list[str] = []
    event_sizes: list[int] = []

    pending_del: tuple[int, str] | None = None  # (del_length, author)
    touched = False

    def _classify(size: int) -> str:
        if paragraph_text_length <= 0:
            # Can't classify without paragraph context; bucket as block
            # by default (so we don't undercount block events).
            return "block"
        return (
            "inline"
            if size / paragraph_text_length < inline_block_threshold
            else "block"
        )

    def _emit(size: int) -> None:
        """Record one event of size `size`: kind + size go to parallel lists."""
        event_kinds.append(_classify(size))
        event_sizes.append(size)

    for child in p_elem.iterchildren():
        tag = etree.QName(child).localname
        if tag in _TRANSPARENT_TAGS:
            continue
        if tag not in _REVISION_TAGS:
            # Plain run or other element breaks any pending pair. If we
            # had a pending standalone delete waiting for an insert that
            # never came, classify it as its own event now.
            if pending_del is not None:
                _emit(pending_del[0])
                pending_del = None
            continue
        author = child.get(_qn("author"), "") or ""
        if author_substring is not None and author_substring not in author:
            if pending_del is not None:
                _emit(pending_del[0])
                pending_del = None
            continue
        touched = True
        if tag in ("del", "moveFrom"):
            if pending_del is not None:
                _emit(pending_del[0])
            dl = _del_text_length(child)
            delete_lengths.append(dl)
            pending_del = (dl, author)
        elif tag in ("ins", "moveTo"):
            il = _ins_text_length(child)
            insert_lengths.append(il)
            if pending_del is not None and pending_del[1] == author:
                if pending_del[0] > 0:
                    replace_ratios.append(il / pending_del[0])
                _emit(max(pending_del[0], il))
                pending_del = None
            else:
                _emit(il)
                pending_del = None

    if pending_del is not None:
        _emit(pending_del[0])

    return (
        insert_lengths,
        delete_lengths,
        replace_ratios,
        (1 if touched else 0),
        event_kinds,
        event_sizes,
    )


def _collect_quality_for_docx(
    docx: Path,
    author_substring: str | None = None,
    *,
    inline_block_threshold: float = INLINE_BLOCK_THRESHOLD,
) -> dict | None:
    """Walk an entire docx and aggregate per-paragraph edit stats.

    Returns a dict with: `insert_lengths`, `delete_lengths`,
    `replace_ratios`, `event_kinds`, `edit_op_count`, `touched_paras`,
    `n_paragraphs`, `n_revisions`. None if the docx can't be opened."""
    root = _load_document_xml(docx)
    if root is None:
        return None

    insert_lengths: list[int] = []
    delete_lengths: list[int] = []
    replace_ratios: list[float] = []
    event_kinds: list[str] = []
    event_sizes: list[int] = []
    edit_op_count = 0
    touched_paras = 0
    n_paragraphs = 0

    for p in _iter_paragraphs(root):
        n_paragraphs += 1
        para_text_len = _paragraph_text_length(p)
        ins_l, del_l, ratios, t, kinds, sizes = _walk_paragraph_pairs(
            p,
            author_substring,
            paragraph_text_length=para_text_len,
            inline_block_threshold=inline_block_threshold,
        )
        # Edit ops: paired del+ins collapses to 1; standalone counts as 1.
        ops_in_para = len(del_l) + len(ins_l) - len(ratios)
        edit_op_count += ops_in_para
        insert_lengths.extend(ins_l)
        delete_lengths.extend(del_l)
        replace_ratios.extend(ratios)
        event_kinds.extend(kinds)
        event_sizes.extend(sizes)
        touched_paras += t

    return {
        "n_paragraphs": n_paragraphs,
        "n_insertions": len(insert_lengths),
        "n_deletions": len(delete_lengths),
        "n_replace_pairs": len(replace_ratios),
        # Total tracked-change events (insertions + deletions +
        # paired replacements) the actor contributed. Not used for
        # the public verbosity-trap headline (that uses
        # `_count_redline_clusters` instead — the lawyer's notion of
        # a redline); kept here for diagnostics.
        "n_revisions": len(insert_lengths) + len(delete_lengths) + len(replace_ratios),
        "edit_op_count": edit_op_count,
        "touched_paras": touched_paras,
        "event_kinds": event_kinds,
        # Per-event sizes in characters (matches event_kinds 1:1). For
        # paired del+ins the size is `max(del_len, ins_len)`; for
        # standalone events it's just the event's own char count. Used
        # to compute the verbosity-trap "average edit length".
        "event_sizes": event_sizes,
        "insert_lengths": insert_lengths,
        "delete_lengths": delete_lengths,
        "replace_ratios": replace_ratios,
    }


# ─── paragraph-touched flags (for overlap-with-expert) ──────────────


def _touched_paragraph_indices(
    docx: Path,
    author_substring: str | None = None,
) -> set[int]:
    """Positional indices of every paragraph in `docx` that contains
    at least one matching ins/del/move element."""
    root = _load_document_xml(docx)
    if root is None:
        return set()
    out: set[int] = set()
    for i, p in enumerate(_iter_paragraphs(root)):
        for el in p.iter():
            tag = etree.QName(el).localname
            if tag in _REVISION_TAGS:
                author = el.get(_qn("author"), "") or ""
                if author_substring is None or author_substring in author:
                    out.add(i)
                    break
    return out


def _count_redline_clusters(
    docx: Path,
    author_substring: str | None = None,
    *,
    max_gap: int = 0,
) -> int:
    """Count distinct "redlines" = contiguous runs of touched
    paragraphs in document order. This is the lawyer's notion of a
    redline (one logical change at the document level) — a multi-word
    rewrite of one paragraph, or edits spanning two adjacent
    paragraphs, count as ONE redline.

    With `max_gap=0` (default), paragraphs must be directly
    consecutive to belong to the same cluster.
    """
    root = _load_document_xml(docx)
    if root is None:
        return 0
    touched: list[bool] = []
    for p in _iter_paragraphs(root):
        t = False
        for el in p.iter():
            tag = etree.QName(el).localname
            if tag in _REVISION_TAGS:
                author = el.get(_qn("author"), "") or ""
                if author_substring is None or author_substring in author:
                    t = True
                    break
        touched.append(t)
    if not any(touched):
        return 0
    if max_gap < 0:
        max_gap = 0

    clusters = 0
    i = 0
    n = len(touched)
    while i < n:
        if not touched[i]:
            i += 1
            continue
        clusters += 1
        last_touched = i
        j = i + 1
        while j < n:
            if touched[j]:
                last_touched = j
                j += 1
            elif j - last_touched <= max_gap:
                j += 1
            else:
                break
        i = last_touched + 1
    return clusters


# ─── public API: surgicalness ───────────────────────────────────────


def compute_surgicalness(
    by_model_docx: dict[str, list[Path]],
    expert_docx_paths: list[Path],
    *,
    inline_block_threshold: float = INLINE_BLOCK_THRESHOLD,
    model_author_substring: str | None = "Reviewing Counsel",
    expert_author_substring: str | None = None,
) -> dict[str, dict]:
    """Aggregate inline-vs-block share per actor (each model + the
    human expert baseline).

    `by_model_docx`: `{model_name: [redline_docx_path, …]}` — one path
    per task for that model.
    `expert_docx_paths`: the human attorney's anonymized redline docx
    per task, used as the baseline row in the chart.

    Returns `{actor: {"n_inline": int, "n_block": int,
    "inline_share": float, "block_share": float, "n_tasks": int}}` with
    one key per model plus `"expert"`.

    Author filtering:

      * Model docx for turns ≥ 2 contain LAYERED edits — the prior
        turn's tracked changes from the seed (tagged "Trainer N - …")
        plus the model's own new edits (tagged "Reviewing Counsel
        (…)"). Counting both inflates the inline-share (the seed
        layer is many small attorney edits). We filter to the model's
        own edits via `model_author_substring`.
      * Expert docx (`attorney_redlines.docx`) is single-author by
        construction, so no filter is needed (default `None`).

    Pass `model_author_substring=None` to disable the model filter
    (counts every event in the docx regardless of author).
    """
    out: dict[str, dict] = {}

    def _aggregate(label: str, paths: list[Path], author_filter: str | None) -> None:
        total_inline = 0
        total_block = 0
        n_tasks = 0
        for p in paths:
            raw = _collect_quality_for_docx(
                p,
                author_substring=author_filter,
                inline_block_threshold=inline_block_threshold,
            )
            if raw is None:
                continue
            n_tasks += 1
            for kind in raw["event_kinds"]:
                if kind == "inline":
                    total_inline += 1
                elif kind == "block":
                    total_block += 1
        total_events = total_inline + total_block
        inline_share = total_inline / total_events if total_events else 0.0
        block_share = total_block / total_events if total_events else 0.0
        out[label] = {
            "n_inline": total_inline,
            "n_block": total_block,
            "inline_share": round(inline_share, 4),
            "block_share": round(block_share, 4),
            # Single-number summary of surgicalness, identical to
            # `inline_share` semantically — broken out as a distinct
            # field so the HTML and downstream consumers can grab "the
            # surgicalness number" without re-deriving it. Convention:
            # HIGHER = MORE SURGICAL (lots of small, in-place edits);
            # 1.0 = every event is a phrase-tweak, 0.0 = every event is
            # a paragraph rewrite.
            "surgicalness_score": round(inline_share, 4),
            "n_tasks": n_tasks,
        }

    _aggregate("expert", expert_docx_paths, expert_author_substring)
    for model, paths in by_model_docx.items():
        _aggregate(model, paths, model_author_substring)
    return out


# ─── public API: verbosity (turn 1) ─────────────────────────────────


def compute_verbosity_turn1(
    by_model_turn1: dict[str, list[tuple[str, Path, Path | None]]],
    expert_turn1_docx_by_task: dict[str, Path],
) -> dict[str, dict]:
    """Turn-1-only verbosity stats per actor.

    `by_model_turn1`: `{model: [(task_name, model_docx, expert_docx_or_None), …]}`.
    `expert_turn1_docx_by_task`: `{task_name: expert_docx_path}` —
    used to populate the expert baseline row (mean redlines + mean
    edits per touched paragraph across all turn-1 tasks).

    Returns `{actor: {…}}`:

      For the expert row:
        - `redlines_per_task`: mean (`n_revisions`) across turn-1 tasks
        - `edits_per_touched_para`: mean (`edit_op_count / touched_paras`),
            skipping tasks where touched_paras == 0
        - `n_tasks`: how many turn-1 tasks the expert was scanned on

      For each model row:
        - same two stats above, computed from the model's docx
        - `overlap_rate`: mean of (|model ∩ expert| / |model|) across
            tasks where both sides have a non-empty touched-paragraph
            set
        - `n_tasks_with_overlap`: count of tasks contributing to the
            overlap mean (subset of `n_tasks` — some tasks may have a
            model output but no expert docx, or vice versa)

    Restricted to turn 1 because positional paragraph indices align
    cleanly only when the input is the clean template; later turns
    add or modify paragraphs, breaking index-based overlap.
    """
    out: dict[str, dict] = {}

    # Expert baseline row — pooled across every turn-1 task with an
    # expert docx on disk.
    #
    # `redlines_per_task` counts contiguous runs of touched paragraphs
    # in document order (the lawyer's notion of one logical change at
    # the document level). Counting individual ins/del events would
    # inflate the number 10×+ on paired-heavy expert docx.
    exp_redlines: list[int] = []
    exp_edits_per_para: list[float] = []
    # Pool every event's size across the expert's turn-1 docx files —
    # `avg_edit_length` is the per-event mean (in characters), not a
    # per-task mean of per-task means, so a task with 50 events has 50×
    # the weight of a task with 1 event.
    exp_event_sizes: list[int] = []
    for _, expert_path in sorted(expert_turn1_docx_by_task.items()):
        raw = _collect_quality_for_docx(expert_path)
        if raw is None:
            continue
        exp_redlines.append(_count_redline_clusters(expert_path))
        if raw["touched_paras"] > 0:
            exp_edits_per_para.append(
                raw["edit_op_count"] / raw["touched_paras"]
            )
        exp_event_sizes.extend(raw.get("event_sizes", []))
    out["expert"] = {
        "redlines_per_task": round(mean(exp_redlines), 2) if exp_redlines else 0.0,
        "edits_per_touched_para": (
            round(mean(exp_edits_per_para), 3) if exp_edits_per_para else 0.0
        ),
        "avg_edit_length": (
            round(mean(exp_event_sizes), 1) if exp_event_sizes else 0.0
        ),
        "n_tasks": len(exp_redlines),
    }

    # Per-model stats — redlines + edits-per-para from the model docx,
    # overlap rate computed against the expert docx for the same task.
    for model, items in by_model_turn1.items():
        redlines: list[int] = []
        edits_per_para: list[float] = []
        overlaps: list[float] = []
        event_sizes_all: list[int] = []
        for task_name, model_docx, expert_docx in items:
            raw = _collect_quality_for_docx(model_docx)
            if raw is None:
                continue
            redlines.append(_count_redline_clusters(model_docx))
            if raw["touched_paras"] > 0:
                edits_per_para.append(
                    raw["edit_op_count"] / raw["touched_paras"]
                )
            event_sizes_all.extend(raw.get("event_sizes", []))
            # Paragraph overlap with expert (if expert exists for this task).
            if expert_docx is None:
                continue
            model_touched = _touched_paragraph_indices(model_docx)
            if not model_touched:
                continue
            expert_touched = _touched_paragraph_indices(expert_docx)
            if not expert_touched:
                continue
            overlaps.append(
                len(model_touched & expert_touched) / len(model_touched)
            )
        out[model] = {
            "redlines_per_task": round(mean(redlines), 2) if redlines else 0.0,
            "edits_per_touched_para": (
                round(mean(edits_per_para), 3) if edits_per_para else 0.0
            ),
            "avg_edit_length": (
                round(mean(event_sizes_all), 1) if event_sizes_all else 0.0
            ),
            "overlap_rate": round(mean(overlaps), 4) if overlaps else 0.0,
            "n_tasks": len(redlines),
            "n_tasks_with_overlap": len(overlaps),
        }
    return out


# ─── docx path resolvers ────────────────────────────────────────────


def find_model_docx_paths(
    runs_dir: Path,
    *,
    include_fable_5: bool = False,
) -> dict[str, dict[str, Path]]:
    """Walk `runs_dir` and return `{model_name: {task_name: docx_path}}`.

    Model identity follows the same auto-discovery rule used by
    `runs_reader._model_name_for`: the trace's containing directory
    name under `trajectories/` IS the model id. For Fable 5 the
    archival directory is hard-coded to `claude-fable-5` (matching the
    runs_reader override) and the docx is at
    `<task>/old_experiment_run/redline.docx` instead of alongside
    grade.json — that's the only model whose docx layout differs.
    """
    out: dict[str, dict[str, Path]] = defaultdict(dict)

    traj_root = runs_dir / "trajectories"
    if traj_root.is_dir():
        for subdir in sorted(traj_root.iterdir()):
            if not subdir.is_dir():
                continue
            model_name = subdir.name
            for task_dir in sorted(subdir.iterdir()):
                if not task_dir.is_dir():
                    continue
                docx = task_dir / "redline.docx"
                if docx.exists():
                    out[model_name][task_dir.name] = docx

    if include_fable_5:
        archival_root = runs_dir / "archival-fable5"
        if archival_root.is_dir():
            for task_dir in sorted(archival_root.iterdir()):
                docx = task_dir / "old_experiment_run" / "redline.docx"
                if docx.exists():
                    out["claude-fable-5"][task_dir.name] = docx

    return dict(out)


def find_expert_docx_paths(benchmark_dir: Path) -> dict[str, Path]:
    """`{task_name: tasks/<task>/tests/attorney_redlines.docx}` for every
    task that has an attorney redline. The golden redline lives
    verifier-side under each task's `tests/` (never mounted into the
    agent environment). Two tasks (redline-s2-t4-g03a +
    redline-s3-t4-g01a) don't have one — they're simply absent from the
    returned dict."""
    out: dict[str, Path] = {}
    tasks = Path(benchmark_dir) / "tasks"
    if not tasks.is_dir():
        return out
    for task_dir in sorted(tasks.iterdir()):
        if not task_dir.is_dir():
            continue
        expert = task_dir / "tests" / "attorney_redlines.docx"
        if expert.exists():
            out[task_dir.name] = expert
    return out


_TURN_RE = re.compile(r"-t(\d+)-")


def turn_of(task_name: str) -> int | None:
    """Extract the turn number from a `redline-sN-tT-gNNvariant` name.
    Returns None if the name doesn't parse."""
    m = _TURN_RE.search(task_name)
    return int(m.group(1)) if m else None
