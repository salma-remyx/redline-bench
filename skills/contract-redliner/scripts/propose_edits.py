#!/usr/bin/env python3
"""Apply a batch of tracked-change edits to a .docx contract, in place.

Usage:
    python propose_edits.py CONTRACT.docx EDITS.json --author "Jane (AgentCo Legal)"

EDITS.json is a JSON array of edit objects:

    [
      {
        "op": "replace",                  // "replace" | "delete" | "insert_after"
        "anchor": {
          "paragraph_id": "p-027",        // strongly recommended
          "text": "ninety (90) days",     // verbatim substring of that paragraph
          "context_before": "within ",    // optional disambiguators
          "context_after": " after",
          "occurrence": 1                 // optional, 1-indexed
        },
        "new_text": "thirty (30) days",   // required for replace / insert_after
        "comment": "We shortened the notice period because ..."  // required
      }
    ]

Each edit becomes a real Word tracked change (<w:ins>/<w:del>) attributed to
--author, with the comment attached to the paragraph. Results are printed as
JSON, one object per edit, in order. Edits are independent: a failed anchor
does not abort the rest of the batch.

Exit code 0 if every edit applied, 1 if any edit failed (inspect the JSON:
failed edits have status "anchor_failed" or "runtime_error", and ambiguous
anchors list candidate paragraph IDs).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from redline_engine.document import DocumentView  # noqa: E402
from redline_engine.ops import (  # noqa: E402
    delete_text,
    insert_after_text,
    replace_text,
)

DEFAULT_AUTHOR = "Counsel"


def _apply_one(view: DocumentView, edit: dict, author: str) -> dict:
    op = edit.get("op")
    anchor = edit.get("anchor") or {}
    comment = edit.get("comment", "")
    new_text = edit.get("new_text", "")
    if op == "replace":
        res = replace_text(view, anchor=anchor, new_text=new_text, comment=comment, author=author)
    elif op == "delete":
        res = delete_text(view, anchor=anchor, comment=comment, author=author)
    elif op == "insert_after":
        res = insert_after_text(
            view, anchor=anchor, new_text=new_text, comment=comment, author=author
        )
    else:
        return {"op": str(op), "status": "runtime_error", "error": f"unknown op: {op!r}"}
    return {k: v for k, v in asdict(res).items() if v not in (None, [], "")}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("docx", help="path to the .docx file (edited in place)")
    parser.add_argument("edits", help="path to a JSON file with the edits array, or '-' for stdin")
    parser.add_argument("--author", default=DEFAULT_AUTHOR, help="tracked-change author name")
    args = parser.parse_args()

    path = Path(args.docx)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    raw = sys.stdin.read() if args.edits == "-" else Path(args.edits).read_text()
    try:
        edits = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: edits file is not valid JSON: {exc}", file=sys.stderr)
        return 1
    if not isinstance(edits, list):
        print("error: edits must be a JSON array of edit objects", file=sys.stderr)
        return 1

    view = DocumentView.load(path)
    results = []
    for edit in edits:
        if not isinstance(edit, dict):
            results.append(
                {
                    "op": "unknown",
                    "status": "runtime_error",
                    "error": f"each edit must be an object; got {type(edit).__name__}",
                }
            )
            continue
        results.append(_apply_one(view, edit, args.author))

    view._docx.save(str(path))  # noqa: SLF001 — mirrors the engine's save convention

    print(json.dumps({"applied_to": str(path), "results": results}, indent=2))
    failed = sum(1 for r in results if r.get("status") != "ok")
    if failed:
        print(f"\n{failed}/{len(results)} edits FAILED — fix anchors and resubmit only those.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
