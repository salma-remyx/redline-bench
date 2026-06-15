"""The "Reserved" / "Intentionally Omitted" macro.

Per the AgentCo playbook::

    If you have to delete an entire section, mark it "Reserved" to preserve
    the numbering and formatting throughout the rest of the document.

Implementation: tracked-delete the body text of every paragraph in the range,
then tracked-insert "Reserved." into the first body paragraph. The paragraphs
themselves (and their `w:numPr` list/numbering attachments) are preserved, so
downstream section numbering doesn't shift.

V1 takes paragraph_id range bounds — the LLM specifies the start and end
paragraphs explicitly. A future iteration could auto-detect section boundaries
from styles or numbering, but for the first-turn redline use case the LLM
already sees paragraph IDs in the Markdown view and can pick them.
"""

from __future__ import annotations

from docx_revisions import RevisionParagraph

from .document import DocumentView
from .ops import OpResult, _attach_comment, DEFAULT_AUTHOR


def mark_paragraph_range_reserved(
    view: DocumentView,
    start_paragraph_id: str,
    end_paragraph_id: str,
    comment: str,
    author: str = DEFAULT_AUTHOR,
) -> OpResult:
    """Replace the body of paragraphs [start..end] with a tracked 'Reserved.' insert."""
    start = view.get(start_paragraph_id)
    end = view.get(end_paragraph_id)
    if start is None or end is None:
        return OpResult(
            op="reserved",
            status="anchor_failed",
            error=f"paragraph_not_found: start={start_paragraph_id} end={end_paragraph_id}",
        )
    if start.index > end.index:
        return OpResult(
            op="reserved",
            status="anchor_failed",
            error="start paragraph comes after end paragraph",
        )

    first_para = view.runtime_paragraph(start_paragraph_id)
    try:
        for p_info in view.paragraphs[start.index : end.index + 1]:
            if p_info.is_empty:
                continue
            para = view.runtime_paragraph(p_info.id)
            rp = RevisionParagraph.from_paragraph(para)
            # Use replace_tracked_at on the whole paragraph text. For the
            # first body paragraph, the replacement is "Reserved."; for the
            # rest, replace with empty (pure deletion).
            replacement = "Reserved." if p_info.id == start_paragraph_id else ""
            rp.replace_tracked_at(
                start=0,
                end=len(p_info.text),
                replace_text=replacement,
                author=author,
                index_mode="text",
            )
    except Exception as exc:  # noqa: BLE001
        return OpResult(
            op="reserved",
            status="runtime_error",
            paragraph_id=start_paragraph_id,
            error=f"mark_reserved: {exc}",
        )

    comment_id = _attach_comment(view, first_para, comment, author)
    return OpResult(
        op="reserved",
        status="ok",
        paragraph_id=start_paragraph_id,
        comment_id=comment_id,
    )
