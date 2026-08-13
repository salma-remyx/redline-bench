#!/usr/bin/env python3
"""SQLite audit trail for judge LLM calls.

This module implements the *persistent-state / audit-trail* scaffold from
the harness-engineering framework described in "Harnessing LLMs for
Reliable Academic Supervision: A Comparative Study" (arXiv:2607.14707).
That paper's central claim is that reliability comes from wrapping an LLM
core in deterministic scaffolding -- one piece of which is a per-node
SQLite audit log of every LLM call (prompt, raw response, retry count,
latency, outcome) for traceability and post-hoc debugging.

RedlineBench is itself an LLM-as-judge harness that already has
schema-typed I/O and bounded retry but lacks this audit scaffold. Wiring
the log into the shared ``judging.call_judge`` chokepoint records every
judge call that flows through the repo-level judging path -- in practice
the re-judging tool (``rejudge.py``), the sole caller of ``call_judge`` --
without changing its return contract. Scope note: the judge panel
(``panel.py``, a majority vote over stored grade JSONs) makes no LLM call,
and reproduction runs a vendored copy of the judging logic inside the
Harbor container, so neither of those paths is covered by this hook.

The read/inspection half of this scaffold lives in ``audit_reader``.

Opt-in: logging activates only when ``REDBENCH_JUDGE_AUDIT_DB`` points at a
writable SQLite path, so the default judging path is unchanged and
existing tests are unaffected. Logging is best-effort: a storage failure
is swallowed and never propagates into the judging path.
"""

from __future__ import annotations

import os
import sqlite3
import time

import inference_provenance

#: Environment variable holding the audit-DB path. Unset => auditing off.
AUDIT_DB_ENV = "REDBENCH_JUDGE_AUDIT_DB"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS judge_calls (
    ts REAL NOT NULL,
    model TEXT,
    system_prompt TEXT,
    user_prompt TEXT,
    raw_response TEXT,
    attempts INTEGER,
    latency_ms REAL,
    ok INTEGER NOT NULL,
    error TEXT,
    backend TEXT,
    backend_version TEXT
);
"""

#: Columns added after the original schema. ``CREATE TABLE IF NOT EXISTS``
#: won't add them to a DB created by an older release, so ``_migrate`` does
#: it via ``ALTER TABLE``. Each is the per-call backend disclosure
#: (arXiv:2608.04714): which inference framework + version produced the
#: verdict on this row.
_ADDED_COLUMNS = ("backend", "backend_version")


def audit_path() -> str | None:
    """Return the configured audit-DB path, or ``None`` when auditing is off."""
    return os.environ.get(AUDIT_DB_ENV) or None


def _migrate(conn: sqlite3.Connection) -> None:
    """Add backend-disclosure columns to judge-call tables created before
    they existed. Idempotent: only adds columns that are genuinely missing."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(judge_calls)")}
    for col in _ADDED_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE judge_calls ADD COLUMN {col} TEXT")
    conn.commit()


def _connect() -> sqlite3.Connection | None:
    path = audit_path()
    if not path:
        return None
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def log_judge_call(
    *,
    model: str,
    system: str,
    user: str,
    raw_response: str | None,
    attempts: int,
    latency_ms: float,
    ok: bool,
    error: str | None = None,
) -> None:
    """Persist one judge call to the SQLite audit trail.

    No-op when auditing is disabled (no env var). Best-effort: any storage
    error is swallowed so the judging path can never be broken by auditing.
    """
    conn = _connect()
    if conn is None:
        return
    # Stamp the inference-backend disclosure (arXiv:2608.04714) onto the
    # row: which framework + version produced this verdict. Best-effort,
    # like the rest of the audit write.
    provenance = inference_provenance.judge_call_backend()
    try:
        conn.execute(
            "INSERT INTO judge_calls "
            "(ts, model, system_prompt, user_prompt, raw_response, "
            " attempts, latency_ms, ok, error, backend, backend_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                model,
                system,
                user,
                raw_response,
                attempts,
                latency_ms,
                1 if ok else 0,
                error,
                provenance["backend"],
                provenance["backend_version"],
            ),
        )
        conn.commit()
    except sqlite3.Error:
        # Audit trail is best-effort; never break the judge call.
        pass
    finally:
        conn.close()
