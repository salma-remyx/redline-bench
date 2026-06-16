"""Comments and threaded comment replies.

python-docx 1.2 has native `Document.add_comment(runs, text, author)` for top-
level comments — but it allocates IDs against existing `w:comment` IDs only,
ignoring bookmark and revision IDs that share the same namespace. We wrap it
to renumber the assigned ID against a full-document IdAllocator after the call, to avoid the silent corruption documented in the
Anthropic docx skill issue #489.

Replies are hand-rolled: python-docx has no reply API. We write the reply as a
new `<w:comment>` in `comments.xml` and a `<w15:commentEx>` entry in
`commentsExtended.xml` with `paraIdParent` pointing at the parent comment's
`w14:paraId`.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import TYPE_CHECKING

from docx.oxml.ns import nsmap
from lxml import etree

from .ooxml_utils import IdAllocator, NSMAP, W14_NS, W15_NS, qn, parse_xml

if TYPE_CHECKING:
    from .document import DocumentView


# ---------------------------------------------------------------------------
# Top-level comments
# ---------------------------------------------------------------------------


def add_top_level_comment(
    view: "DocumentView",
    paragraph,
    text: str,
    author: str,
    initials: str = "",
) -> int:
    """Add a top-level comment anchored to the whole paragraph; return its safe ID."""
    doc = view._docx

    # Allocate a safe ID by scanning every part that shares the ID space.
    safe_id = _allocate_safe_id(view)

    runs = list(paragraph.runs) or _ensure_at_least_one_run(paragraph)
    comment = doc.add_comment(
        [runs[0], runs[-1]],
        text=text,
        author=author,
        initials=initials,
    )
    assigned = int(comment._element.get(qn("w:id")))
    if assigned != safe_id:
        _renumber_comment(doc, assigned, safe_id)

    # Ensure the comment body's paragraph has a w14:paraId so replies can
    # point at it via paraIdParent. python-docx may or may not assign one;
    # set it deterministically if absent.
    _ensure_paraid(comment._element)

    return safe_id


# ---------------------------------------------------------------------------
# Comment replies
# ---------------------------------------------------------------------------


def add_comment_reply(
    view: "DocumentView",
    parent_comment_id: int,
    text: str,
    author: str,
    initials: str = "",
) -> int:
    """Append a threaded reply to an existing comment; return new comment's ID."""
    doc = view._docx
    comments_part = _comments_part(doc)
    if comments_part is None:
        raise RuntimeError(
            "Document has no comments.xml — cannot add a reply. "
            "Add at least one top-level comment first."
        )
    comments_root = comments_part._element

    parent_el = _find_comment_element(comments_root, parent_comment_id)
    if parent_el is None:
        raise ValueError(f"No existing comment with id={parent_comment_id}")
    parent_paraid = _ensure_paraid(parent_el)

    new_id = _allocate_safe_id(view)
    new_paraid = _new_paraid()

    # Build the reply <w:comment> mirroring python-docx's minimum valid shape.
    reply_xml = (
        f'<w:comment xmlns:w="{NSMAP["w"]}" '
        f'xmlns:w14="{W14_NS}" '
        f'w:id="{new_id}" w:author="{_xml_escape(author)}" '
        f'w:date="{_now_iso()}" '
        f'w:initials="{_xml_escape(initials)}">'
        f'<w:p w14:paraId="{new_paraid}" w14:textId="00000000">'
        f'<w:pPr><w:pStyle w:val="CommentText"/></w:pPr>'
        f'<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr>'
        f'<w:annotationRef/></w:r>'
        f'<w:r><w:t xml:space="preserve">{_xml_escape(text)}</w:t></w:r>'
        f'</w:p>'
        f'</w:comment>'
    )
    reply_el = etree.fromstring(reply_xml)
    comments_root.append(reply_el)

    # Add to commentsExtended.xml — creating the part if it doesn't exist.
    _ensure_commentex_entry(doc, new_paraid, parent_paraid)

    return new_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _allocate_safe_id(view: "DocumentView") -> int:
    """Scan document.xml, comments.xml, headers, footers, and footnotes/endnotes."""
    doc = view._docx
    roots: list[etree._Element] = [doc.element]
    for part_name in ("comments", "footnotes", "endnotes"):
        part = _get_named_part(doc, part_name)
        if part is None:
            continue
        root = _part_root(part)
        if root is not None:
            roots.append(root)
    return IdAllocator.from_parts(*roots).next()


def _part_root(part) -> etree._Element | None:
    """Return an XML root for a part, regardless of whether it's a specialized
    docx Part subclass or a generic `Part`.

    Specialized parts (CommentsPart, FootnotesPart in newer python-docx, ...)
    expose `_element` already parsed. Generic `Part` only exposes `blob` — we
    parse it on demand. Returns None if the part has no XML payload.
    """
    elem = getattr(part, "_element", None)
    if elem is not None:
        return elem
    blob = getattr(part, "blob", None)
    if blob:
        try:
            return parse_xml(blob)
        except Exception:  # noqa: BLE001
            return None
    return None


def _comments_part(doc) -> object | None:
    return _get_named_part(doc, "comments")


def _get_named_part(doc, suffix: str):
    """Return the part whose URI ends with `/word/<suffix>.xml`, or None."""
    target = f"{suffix}.xml"
    for rel in doc.part.rels.values():
        if rel.target_ref.endswith(target):
            return rel.target_part
    return None


def _find_comment_element(comments_root, comment_id: int):
    for c in comments_root.iter(qn("w:comment")):
        if c.get(qn("w:id")) == str(comment_id):
            return c
    return None


def _renumber_comment(doc, old_id: int, new_id: int) -> None:
    """Reassign a comment's `w:id` from `old_id` to `new_id` across all parts."""
    old, new = str(old_id), str(new_id)
    # comments.xml — the comment body
    cp = _comments_part(doc)
    if cp is not None:
        for c in cp._element.iter(qn("w:comment")):
            if c.get(qn("w:id")) == old:
                c.set(qn("w:id"), new)
    # document.xml — the range markers and reference
    for tag in ("w:commentRangeStart", "w:commentRangeEnd", "w:commentReference"):
        for el in doc.element.iter(qn(tag)):
            if el.get(qn("w:id")) == old:
                el.set(qn("w:id"), new)


def _ensure_paraid(comment_el) -> str:
    """Make sure the first paragraph inside the comment has a w14:paraId; return it."""
    first_p = comment_el.find(qn("w:p"))
    if first_p is None:
        # Comments always have at least one paragraph after python-docx adds them.
        raise RuntimeError("Comment element has no paragraph child — unexpected.")
    paraid = first_p.get(qn("w14:paraId"))
    if not paraid:
        paraid = _new_paraid()
        first_p.set(qn("w14:paraId"), paraid)
    return paraid


def _new_paraid() -> str:
    """Generate a fresh 8-hex-digit paraId."""
    return uuid.uuid4().hex[:8].upper()


def _ensure_commentex_entry(doc, paraid: str, parent_paraid: str) -> None:
    """Append a <w15:commentEx> entry, creating commentsExtended.xml if needed."""
    cep = _get_named_part(doc, "commentsExtended")
    if cep is None:
        cep = _create_commentex_part(doc)
    if cep is None:
        # If we can't create one, the reply will still be valid as a comment,
        # but Word may render it as a top-level comment in the same range
        # rather than a threaded reply. Degrade gracefully.
        return
    # commentsExtended is always a generic Part — its blob is the source of
    # truth; we must round-trip through it on every change.
    root = parse_xml(cep.blob) if getattr(cep, "blob", None) else None
    if root is None:
        # Fall back to the in-memory _element (in case this is a fresh part
        # we just created and blob isn't populated yet).
        root = getattr(cep, "_element", None)
    if root is None:
        return
    etree.SubElement(
        root,
        qn("w15:commentEx"),
        attrib={
            qn("w15:paraId"): paraid,
            qn("w15:paraIdParent"): parent_paraid,
            qn("w15:done"): "0",
        },
    )
    cep._blob = etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    cep._element = root  # keep in-memory copy in sync


def _get_root(part):
    """Best-effort: extract the XML root from a part."""
    if hasattr(part, "element"):
        return part.element
    if hasattr(part, "_element"):
        return part._element
    if hasattr(part, "blob"):
        return parse_xml(part.blob)
    return None


def _create_commentex_part(doc) -> object | None:
    """Create commentsExtended.xml if missing.

    python-docx doesn't expose a helper for this; we hand-roll the part and
    register it. If something goes wrong (unsupported package shape, etc.),
    return None and let the caller degrade gracefully.
    """
    try:
        from docx.opc.constants import CONTENT_TYPE as CT, RELATIONSHIP_TYPE as RT
        from docx.opc.part import Part
        from docx.opc.packuri import PackURI
    except Exception:  # noqa: BLE001
        return None

    blob = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        b'<w15:commentsEx '
        b'xmlns:w15="' + W15_NS.encode() + b'" '
        b'xmlns:w14="' + W14_NS.encode() + b'" '
        b'xmlns:w="' + NSMAP["w"].encode() + b'" '
        b"/>"
    )
    partname = PackURI("/word/commentsExtended.xml")
    content_type = (
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.commentsExtended+xml"
    )
    reltype = (
        "http://schemas.microsoft.com/office/2011/relationships/commentsExtended"
    )
    package = doc.part.package
    part = Part(partname, content_type, blob, package)
    # Attach _element by parsing the blob — keep API consistent with other parts.
    part._element = etree.fromstring(blob)
    package.parts.append(part) if hasattr(package, "parts") else None
    doc.part.relate_to(part, reltype)
    return part


def _new_paraid_from_uuid() -> str:
    return uuid.uuid4().hex[:8].upper()


def _ensure_at_least_one_run(paragraph):
    """If the paragraph has zero runs (rare), add one empty run and return it."""
    paragraph.add_run("")
    return list(paragraph.runs)


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
