#!/usr/bin/env python3
"""Print a .docx contract as Markdown with stable paragraph IDs.

Usage:
    python read_document.py CONTRACT.docx [--max-chars N]

Output: a stats line, then the document as Markdown. Each paragraph is
prefixed with its ID like `[p-027]`. Paragraphs carrying tracked changes or
comments are flagged, and an appendix lists every existing comment (cmt-N)
and tracked change (rev-N) with its author and text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from redline_engine.document import DocumentView  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx", help="path to the .docx file")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="truncate output after N characters (default: no limit)",
    )
    args = parser.parse_args()

    path = Path(args.docx)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    view = DocumentView.load(path)
    stats = view.stats()
    try:
        print(f"# stats: {stats}")
        print(view.to_markdown(max_chars=args.max_chars))
    except BrokenPipeError:  # e.g. piped to `head`
        sys.stderr.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
