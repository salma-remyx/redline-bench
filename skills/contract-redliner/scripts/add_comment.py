#!/usr/bin/env python3
"""Add a margin comment to a .docx contract, in place.

Usage:
    # New standalone comment thread on a paragraph:
    python add_comment.py CONTRACT.docx --paragraph-id p-014 \
        --comment "Can you confirm the data residency requirement applies here?" \
        --author "Jane (AgentCo Legal)"

    # Threaded reply to an existing comment (cmt-N from read_document.py).
    # --paragraph-id is the paragraph the parent comment sits on:
    python add_comment.py CONTRACT.docx --paragraph-id p-014 --reply-to 3 \
        --comment "Agreed — we accepted this change." \
        --author "Jane (AgentCo Legal)"

Use standalone comments only to flag an issue for discussion without making an
edit (edits made via propose_edits.py already carry their own comment).
Prints a JSON result with the assigned comment id.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from redline_engine.document import DocumentView  # noqa: E402
from redline_engine.ops import add_standalone_comment  # noqa: E402

DEFAULT_AUTHOR = "Counsel"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("docx", help="path to the .docx file (edited in place)")
    parser.add_argument("--paragraph-id", required=True, help="target paragraph, e.g. p-014")
    parser.add_argument("--comment", required=True, help="comment text")
    parser.add_argument(
        "--reply-to",
        type=int,
        default=None,
        help="existing comment id N (from cmt-N) to reply to instead of starting a new thread",
    )
    parser.add_argument("--author", default=DEFAULT_AUTHOR, help="comment author name")
    args = parser.parse_args()

    path = Path(args.docx)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    view = DocumentView.load(path)
    res = add_standalone_comment(
        view,
        paragraph_id=args.paragraph_id,
        comment=args.comment,
        author=args.author,
        reply_to_comment_id=args.reply_to,
    )
    view._docx.save(str(path))  # noqa: SLF001

    out = {k: v for k, v in asdict(res).items() if v not in (None, [], "")}
    print(json.dumps(out, indent=2))
    return 0 if out.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
