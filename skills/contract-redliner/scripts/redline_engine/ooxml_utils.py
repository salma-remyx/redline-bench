"""Low-level OOXML helpers shared across the engine.

The most important thing in this file is `IdAllocator`. Inserting a `w:ins`,
`w:del`, or `w:commentRangeStart` with an ID that already belongs to a
`w:bookmarkStart`, `w:bookmarkEnd`, or another revision element silently
corrupts the document — Word will still open it but cross-references and
threaded comments will misbehave. So the allocator must scan every namespace
that shares the ID space and emit `max + 1` ids on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from lxml import etree

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14_NS = "http://schemas.microsoft.com/office/word/2010/wordml"
W15_NS = "http://schemas.microsoft.com/office/word/2012/wordml"
W16_NS = "http://schemas.microsoft.com/office/word/2018/wordml/cid"

NSMAP = {"w": W_NS, "w14": W14_NS, "w15": W15_NS, "w16": W16_NS}


def qn(tag: str) -> str:
    """Resolve a `prefix:local` tag to a Clark-notation qualified name."""
    prefix, _, local = tag.partition(":")
    if not local:
        return f"{{{W_NS}}}{prefix}"
    ns = NSMAP[prefix]
    return f"{{{ns}}}{local}"


# Every element type that draws from the shared revision/bookmark/comment ID
# pool. Adding to this list is the correct fix when Word complains about a new
# OOXML feature; do not narrow it without testing against a bookmark-heavy doc.
_ID_BEARING_TAGS = (
    "w:ins",
    "w:del",
    "w:moveFrom",
    "w:moveTo",
    "w:bookmarkStart",
    "w:bookmarkEnd",
    "w:commentRangeStart",
    "w:commentRangeEnd",
    "w:commentReference",
    "w:comment",  # comment bodies in comments.xml share the ID space
    "w:rPrChange",
    "w:pPrChange",
)


@dataclass
class IdAllocator:
    """Hands out unique IDs for `w:id` attributes across one Document.

    Construct via `IdAllocator.from_parts(...)` — passing every XML root that
    might contain ID-bearing elements (document, comments, footnotes, etc).
    """

    _next: int

    @classmethod
    def from_parts(cls, *roots: etree._Element | None) -> "IdAllocator":
        max_seen = -1
        for root in roots:
            if root is None:
                continue
            for tag in _ID_BEARING_TAGS:
                for el in root.iter(qn(tag)):
                    raw = el.get(qn("w:id"))
                    if raw is None:
                        continue
                    try:
                        n = int(raw)
                    except ValueError:
                        continue
                    if n > max_seen:
                        max_seen = n
        return cls(_next=max_seen + 1)

    def next(self) -> int:
        value = self._next
        self._next += 1
        return value

    def reserve_block(self, count: int) -> range:
        """Reserve `count` consecutive IDs and return them as a `range`.

        Useful when a single operation (e.g. comment range = start + end +
        reference) needs more than one ID at once and we want them contiguous
        for readability when inspecting the XML by hand.
        """
        start = self._next
        self._next += count
        return range(start, start + count)


def iter_id_attrs(root: etree._Element) -> Iterable[tuple[str, int]]:
    """Yield `(tag_localname, id_value)` for every ID-bearing element under root.

    Mainly for diagnostics / tests.
    """
    for tag in _ID_BEARING_TAGS:
        for el in root.iter(qn(tag)):
            raw = el.get(qn("w:id"))
            if raw is None:
                continue
            try:
                yield etree.QName(el).localname, int(raw)
            except ValueError:
                continue


def parse_xml(blob: bytes) -> etree._Element:
    """Parse a DOCX part's XML with the namespaces we use registered."""
    parser = etree.XMLParser(remove_blank_text=False, recover=False)
    return etree.fromstring(blob, parser=parser)


def make_element(tag: str, **attrs: str) -> etree._Element:
    """Create a standalone OOXML element.

    `tag` is `prefix:local`. Attribute kwargs use `__` to denote `:` since
    Python identifiers can't contain colons. Example::

        make_element("w:ins", w__id="42", w__author="Crosby AI")
    """
    el = etree.Element(qn(tag), nsmap=NSMAP)
    for k, v in attrs.items():
        el.set(qn(k.replace("__", ":")), v)
    return el
