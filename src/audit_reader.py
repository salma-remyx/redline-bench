#!/usr/bin/env python3
"""Inspection/summary over the judge-call SQLite audit trail.

The write half of the *persistent-state / audit-trail* harness scaffold
(from "Harnessing LLMs for Reliable Academic Supervision: A Comparative
Study", arXiv:2607.14707) already lives in ``judge_audit.log_judge_call``,
wired into ``judging.call_judge``. That records every repo-level judge call
(prompt, raw response, retry count, latency, outcome) into SQLite.

This module is the *read* half: it turns that write-only log into a
traceability summary, which is the point of an audit trail — post-hoc
debuggability, not just capture. It is the single source of truth for
reading ``judge_calls`` rows, mirroring the repo's other reader modules
(``runs_reader`` / ``panel_reader``).

The headline signal is ``n_unparseable_ok``: judge calls the trail recorded
as ``ok`` whose ``raw_response`` does NOT parse into a valid ``verdicts``
object (reusing ``judging.parse_judge_json``). That is exactly the
verdict-format-mismatch class that silently zeroed panel scores upstream —
the concrete regression that motivates capturing raw responses in the first
place. Surfacing it here means the failure is visible in a one-line report
instead of buried in raw SQL.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys

from judge_audit import audit_path
from judging import parse_judge_json

#: Columns read from the ``judge_calls`` table, in a stable order.
_COLUMNS = ("ts", "model", "raw_response", "attempts", "latency_ms", "ok", "error")


def read_rows(db_path: str) -> list[dict]:
    """Return every ``judge_calls`` row as a dict, oldest first.

    Returns an empty list when the table does not exist yet (an audit DB
    that was configured but never written to).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cols = ", ".join(_COLUMNS)
        cur = conn.execute(f"SELECT {cols} FROM judge_calls ORDER BY ts")
        return [dict(r) for r in cur.fetchall()]
    except sqlite3.OperationalError:
        # No judge_calls table => nothing has been audited yet.
        return []
    finally:
        conn.close()


def _parses(raw: str | None) -> bool:
    """True iff ``raw`` parses into a valid judge ``verdicts`` object."""
    if raw is None:
        return False
    try:
        parse_judge_json(raw)
        return True
    except Exception:  # noqa: BLE001 — any parse failure means "unparseable".
        return False


def summarize(db_path: str) -> dict:
    """Aggregate the audit trail into a traceability summary.

    Keys:
      * ``n_calls`` / ``n_ok`` / ``n_failed`` — call counts by outcome.
      * ``n_unparseable_ok`` — calls recorded as ``ok`` whose raw response
        does not parse into ``verdicts`` (the verdict-format-mismatch class).
      * ``n_retried`` / ``max_attempts`` / ``mean_attempts`` — retry pressure.
      * ``mean_latency_ms`` — mean judge-call latency.
    """
    rows = read_rows(db_path)
    n_calls = len(rows)
    n_ok = sum(1 for r in rows if r["ok"])
    attempts = [r["attempts"] for r in rows if r["attempts"] is not None]
    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    n_unparseable = sum(1 for r in rows if r["ok"] and not _parses(r["raw_response"]))
    return {
        "n_calls": n_calls,
        "n_ok": n_ok,
        "n_failed": n_calls - n_ok,
        "n_unparseable_ok": n_unparseable,
        "n_retried": sum(1 for a in attempts if a > 1),
        "max_attempts": max(attempts) if attempts else 0,
        "mean_attempts": round(sum(attempts) / len(attempts), 3) if attempts else 0.0,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
    }


def format_summary(summary: dict) -> str:
    """Render :func:`summarize`'s dict as a one-line traceability report."""
    return (
        f"judge audit: {summary['n_calls']} call(s), "
        f"{summary['n_ok']} ok / {summary['n_failed']} failed, "
        f"{summary['n_unparseable_ok']} ok-but-unparseable, "
        f"{summary['n_retried']} retried (max {summary['max_attempts']} attempts), "
        f"mean latency {summary['mean_latency_ms']} ms"
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Summarize the judge-call audit trail.")
    ap.add_argument(
        "--db", default=None,
        help="audit SQLite path (default: $REDBENCH_JUDGE_AUDIT_DB)",
    )
    ap.add_argument("--json", action="store_true", help="emit the summary as JSON")
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

    summary = summarize(db)
    print(json.dumps(summary, indent=2) if args.json else format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
