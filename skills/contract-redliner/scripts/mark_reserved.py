#!/usr/bin/env python3
"""Remove an entire numbered section from a .docx contract, in place,
replacing it with "Reserved." so downstream numbering stays intact.

Usage:
    python mark_reserved.py CONTRACT.docx --start p-031 --end p-035 \
        --comment "We removed the on-site audit section because ..." \
        --author "Jane (AgentCo Legal)"

Tracked-deletes the body text of every paragraph from --start through --end
(inclusive) and tracked-inserts "Reserved." into the first one. Use this for
whole-section removal instead of propose_edits.py delete ops. Prints a JSON
result.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from redline_engine.document import DocumentView  # noqa: E402
from redline_engine.reserved import mark_paragraph_range_reserved  # noqa: E402

DEFAULT_AUTHOR = "Counsel"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("docx", help="path to the .docx file (edited in place)")
    parser.add_argument("--start", required=True, help="first paragraph id of the section, e.g. p-031")
    parser.add_argument("--end", required=True, help="last paragraph id of the section, e.g. p-035")
    parser.add_argument("--comment", required=True, help="rationale comment for the removal")
    parser.add_argument("--author", default=DEFAULT_AUTHOR, help="tracked-change author name")
    args = parser.parse_args()

    path = Path(args.docx)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    view = DocumentView.load(path)
    res = mark_paragraph_range_reserved(
        view,
        start_paragraph_id=args.start,
        end_paragraph_id=args.end,
        comment=args.comment,
        author=args.author,
    )
    view._docx.save(str(path))  # noqa: SLF001

    out = {k: v for k, v in asdict(res).items() if v not in (None, [], "")}
    print(json.dumps(out, indent=2))
    return 0 if out.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
