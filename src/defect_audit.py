#!/usr/bin/env python3
"""Content-level benchmark-defect audit over the judge-call trail.

``audit_reader`` (the read half of the audit-trail scaffold from
"Harnessing LLMs for Reliable Academic Supervision", arXiv:2607.14707)
already surfaces one verifier-reliability signal: ``n_unparseable_ok`` —
judge calls recorded as ``ok`` whose raw response does not parse into
``verdicts``. That is a *mechanical* (clerical) defect: the verifier's own
output format broke.

This module extends that thread to the *content-level* defect class — the
one "SciCode-Verified: How Benchmark Defects Underestimated the
Scientific-Coding Ability of Language Models" (arXiv:2608.04975) found
dominates score suppression: defects that need domain knowledge to spot,
not proofreading, and that cause correct, instruction-following solutions
to be *wrongly rejected*. SciCode-Verified's deliverable was a manual
domain-expert audit producing classified defects + verdict *flips* + a
re-grade summary. ``defect_audit`` adapts that contribution to
RedlineBench (Mode 2 — adapted port):

  * the paper's per-problem domain-expert estimator is replaced by a
    parameter-free proxy over the multi-judge panel's own per-rubric
    verdicts — when the panel splits PASS/FAIL on the same rubric and
    document, the rubric behaves like the self-contradictory /
    over-tight specification the paper describes, and the minority PASS
    is a wrongly-rejected correct redline (a verdict flip);
  * the paper's full re-evaluation of frontier models on corrected specs
    is cut (a downstream benchmark run); the re-grade here is a
    verdict-space recovery estimate computed from the existing trail.

It reads the same ``judge_calls`` SQLite trail ``judge_audit`` writes and
reuses ``judging.parse_judge_json``, so it slots into the audit thread
without touching the judging path. Leaf-reader shape + ``python -m
defect_audit`` CLI mirror ``audit_reader``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from collections import defaultdict

from judge_audit import audit_path
from judging import parse_judge_json

#: Columns read for the panel-defect sweep, in a stable order. Unlike
#: ``audit_reader.read_rows`` (counts + parse check) this also needs
#: ``user_prompt`` (the task-turn key that groups a panel) and ``model``
#: (which judge returned which verdict).
_PANEL_COLUMNS = ("ts", "model", "user_prompt", "raw_response", "ok")

# Defect classes — mirror SciCode-Verified's taxonomy.
PARSE_MISMATCH = "parse_mismatch"  # mechanical: ok call w/ unparseable raw
CONTRADICTORY_SPEC = "contradictory_spec"  # content-level: panel splits on a rubric


def _read_panel_rows(db_path: str) -> list[dict]:
    """Return the judge-call rows the defect sweep needs, oldest first.

    Returns ``[]`` when the ``judge_calls`` table does not exist yet (an
    audit DB that was configured but never written to).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cols = ", ".join(_PANEL_COLUMNS)
        cur = conn.execute(f"SELECT {cols} FROM judge_calls ORDER BY ts")
        return [dict(r) for r in cur.fetchall()]
    except sqlite3.OperationalError:
        # No judge_calls table => nothing has been audited yet.
        return []
    finally:
        conn.close()


def _task_key(user_prompt: str | None) -> str:
    """Stable short key for one judged task-turn.

    The user prompt (the document + rubrics being judged) uniquely
    identifies a task-turn, so calls sharing it form one panel.
    """
    if not user_prompt:
        return "<no-prompt>"
    return hashlib.sha1(user_prompt.encode("utf-8")).hexdigest()[:12]


def _verdicts_of(raw: str | None) -> list[dict] | None:
    """Parse a raw judge response into its ``verdicts`` list, or ``None``.

    ``None`` (not ``[]``) marks the parse-mismatch class so the caller can
    tell "parsed, returned no rubrics" apart from "did not parse at all".
    """
    if raw is None:
        return None
    try:
        data = parse_judge_json(raw)
    except Exception:  # noqa: BLE001 — any parse failure is the mismatch class.
        return None
    out = []
    for v in data.get("verdicts", []):
        if isinstance(v, dict) and v.get("rubric_id") is not None:
            out.append({"rubric_id": v["rubric_id"], "verdict": v.get("verdict", "FAIL")})
    return out


def detect_defects(db_path: str) -> dict:
    """Sweep the audit trail for benchmark/verifier defects and the verdict
    flips they imply (SciCode-Verified, adapted).

    Returns a dict with:
      * ``n_tasks`` / ``n_calls`` — coverage of the sweep.
      * ``defects`` — records by class: ``parse_mismatch`` (one per
        ok-but-unparseable call) and ``contradictory_spec`` (one per rubric
        the panel split PASS/FAIL on).
      * ``by_class`` — defect counts keyed by class.
      * ``flips`` — verdict-flip candidates a corrected benchmark would
        re-grade. ``fail_to_pass`` entries are the wrongly-rejected correct
        redlines the paper centers on.
      * ``regrade`` — verdict-space recovery estimate (rubrics that would
        re-pass) if the ``fail_to_pass`` flips were honored.
    """
    rows = _read_panel_rows(db_path)

    defects: list[dict] = []
    # task_key -> rubric_id -> verdicts seen across the panel's judges.
    panel: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    tasks: set[str] = set()

    for r in rows:
        task = _task_key(r.get("user_prompt"))
        tasks.add(task)
        parsed = _verdicts_of(r.get("raw_response"))
        if r.get("ok") and parsed is None:
            # Mechanical class: the verifier said ok but produced no usable
            # verdict, so every rubric on this task is under-judged.
            defects.append({
                "task": task, "model": r.get("model"),
                "class": PARSE_MISMATCH,
                "detail": "ok-recorded judge call whose raw response did not parse",
            })
            continue
        if parsed:
            for v in parsed:
                panel[task][v["rubric_id"]].append(v["verdict"])

    flips: list[dict] = []
    # Content-level class: rubrics the panel split on. Competent judges
    # disagreeing on the same rubric + document is the parameter-free proxy
    # for the self-contradictory / over-tight specification.
    for task, rubrics in sorted(panel.items()):
        for rid, vs in sorted(rubrics.items()):
            n_pass = sum(1 for x in vs if x == "PASS")
            n_fail = len(vs) - n_pass
            if n_pass == 0 or n_fail == 0 or len(vs) < 2:
                continue  # consensus (or a lone judge) is not a defect
            majority = "PASS" if n_pass > n_fail else "FAIL"
            defects.append({
                "task": task, "rubric_id": rid, "class": CONTRADICTORY_SPEC,
                "split": f"{n_pass}P/{n_fail}F", "majority": majority,
            })
            # The minority verdict is the flip candidate.
            if majority == "FAIL":
                # Dissenting PASSes => a possibly-correct redline wrongly rejected.
                flips.append({
                    "task": task, "rubric_id": rid,
                    "direction": "fail_to_pass", "minority": n_pass,
                })
            else:
                # Dissenting FAILs => a possibly-incorrect redline wrongly accepted.
                flips.append({
                    "task": task, "rubric_id": rid,
                    "direction": "pass_to_fail", "minority": n_fail,
                })

    n_wrongly_rejected = sum(1 for f in flips if f["direction"] == "fail_to_pass")
    by_class = {
        PARSE_MISMATCH: sum(1 for d in defects if d["class"] == PARSE_MISMATCH),
        CONTRADICTORY_SPEC: sum(1 for d in defects if d["class"] == CONTRADICTORY_SPEC),
    }
    return {
        "n_tasks": len(tasks),
        "n_calls": len(rows),
        "defects": defects,
        "n_defects": len(defects),
        "by_class": by_class,
        "flips": flips,
        "n_flip_candidates": len(flips),
        "regrade": {
            # Verdict-space recovery: up to this many additional rubrics would
            # PASS if the wrongly-rejected majority-FAIL verdicts were corrected.
            "wrongly_rejected": n_wrongly_rejected,
            "wrongly_accepted": len(flips) - n_wrongly_rejected,
            "estimated_pass_recovery": n_wrongly_rejected,
        },
    }


def format_defects(summary: dict) -> str:
    """Render :func:`detect_defects`'s dict as a one-line defect report."""
    bc = summary["by_class"]
    rg = summary["regrade"]
    return (
        f"defect audit: {summary['n_defects']} defect(s) across "
        f"{summary['n_tasks']} task(s) "
        f"({bc[PARSE_MISMATCH]} parse-mismatch, "
        f"{bc[CONTRADICTORY_SPEC]} contradictory-spec), "
        f"{summary['n_flip_candidates']} flip candidate(s) — "
        f"{rg['wrongly_rejected']} wrongly-rejected redline(s) would re-pass"
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Audit the judge-call trail for benchmark defects that "
                    "wrongly reject correct redlines.",
    )
    ap.add_argument(
        "--db", default=None,
        help="audit SQLite path (default: $REDBENCH_JUDGE_AUDIT_DB)",
    )
    ap.add_argument("--json", action="store_true", help="emit the full defect report as JSON")
    args = ap.parse_args()

    db = args.db or audit_path()
    if not db:
        print(
            "no audit DB configured (set REDBENCH_JUDGE_AUDIT_DB or pass --db)",
            file=sys.stderr,
        )
        return 2
    if not os.path.exists(db):
        print(f"audit DB not found: {db}", file=sys.stderr)
        return 2

    summary = detect_defects(db)
    print(json.dumps(summary, indent=2) if args.json else format_defects(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
