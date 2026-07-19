#!/usr/bin/env python3
"""Re-grade existing rollout outputs with a chosen judge model.

Reads each trial's saved `verifier/annotated_view.md` and the rubric set it was
graded against (from `verifier/grade.json`), re-judges with --judge, and writes
grades to <out>/<model>/<task>.json. Resume-safe (skips existing). No sandboxes,
no .docx re-rendering — judging is a single LLM call per output.

Used to run additional judge families for the 3-judge panel and the
judge-sensitivity analysis.

Usage:
    python -m rejudge --jobs jobs/ref1-* --judge openai/gpt-5.4-mini \
        --out results/judge/gpt-5.4-mini [--workers 12] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from judging import (
    JUDGE_SYSTEM_PROMPT, aggregate, build_user_prompt, call_judge,
)

_NAME_RE = re.compile(r"redline-s(\d+)-t(\d+)-g(\d+)([a-z])")
_lock = threading.Lock()


def _model_for_trial(trial: Path, override: str | None) -> str:
    if override:
        return override
    rp = trial / "result.json"
    if rp.exists():
        try:
            info = json.loads(rp.read_text()).get("agent_info") or {}
            return (info.get("model_info") or {}).get("name") or "unknown"
        except Exception:  # noqa: BLE001
            pass
    return "unknown"


def regrade_one(trial: Path, judge_model: str, out_dir: Path, model_override: str | None) -> str:
    m = _NAME_RE.search(trial.name)
    if not m:
        return "skip"
    task = m.group(0)
    model = _model_for_trial(trial, model_override)
    out = out_dir / model / f"{task}.json"
    if out.exists():
        return "skip"
    grade_p = trial / "verifier/grade.json"
    if not grade_p.exists():
        return "skip"
    grade = json.loads(grade_p.read_text())
    out.parent.mkdir(parents=True, exist_ok=True)

    # Gate failures have no judge-gradable output — carry through as 0.
    view_p = trial / "verifier/annotated_view.md"
    if not grade.get("gate", {}).get("passed", True) or not view_p.exists():
        out.write_text(json.dumps({
            "task_id": grade.get("task_id"), "model": model, "judge_model": judge_model,
            "gate": grade.get("gate", {"passed": False}),
            "score": {"weighted": 0.0, "per_rubric": []}, "gate_failure": True,
        }, indent=2))
        return "gate0"

    task_ctx = {
        "scenario_id": grade["scenario_id"], "side": grade["side"], "level": grade["level"],
        "rubrics": [
            {"id": p["rubric_id"], "criteria": p["criteria"], "weight": p["weight"],
             "category": p.get("category"), "justification": ""}
            for p in grade["score"]["per_rubric"]
        ],
    }
    user = build_user_prompt(task_ctx, view_p.read_text())
    resp = call_judge(judge_model, JUDGE_SYSTEM_PROMPT, user)
    score = aggregate(resp["verdicts"], task_ctx["rubrics"])
    out.write_text(json.dumps({
        "task_id": grade["task_id"], "scenario_id": grade["scenario_id"],
        "side": grade["side"], "level": grade["level"], "model": model,
        "judge_model": judge_model, "gate": {"passed": True}, "score": score,
    }, indent=2))
    return "graded"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--judge", required=True, help="LiteLLM judge model string")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="model-label override (else from result.json)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    trials = []
    for j in args.jobs:
        jd = Path(j)
        if jd.is_dir():
            trials += sorted(jd.glob("*__*"))
    if args.limit:
        trials = trials[: args.limit]
    out_dir = Path(args.out)
    print(f"{len(trials)} trials -> judge {args.judge} -> {out_dir}", flush=True)

    counts = {"graded": 0, "skip": 0, "gate0": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(regrade_one, t, args.judge, out_dir, args.model): t for t in trials}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                counts[f.result()] += 1
            except Exception as exc:  # noqa: BLE001
                counts["error"] += 1
                with _lock:
                    print(f"ERROR {futs[f].name}: {str(exc)[:120]}", flush=True)
            if i % 50 == 0:
                with _lock:
                    print(f"  {i}/{len(trials)} {counts}", flush=True)
    print("done:", counts)

    # Best-effort: when the opt-in judge audit trail is active, surface a
    # one-line traceability summary of this run's judge calls (the read half
    # of the audit-trail harness scaffold — see audit_reader / README). Never
    # fails the run.
    try:
        from judge_audit import audit_path
        db = audit_path()
        if db:
            import audit_reader
            print(audit_reader.format_summary(audit_reader.summarize(db)))
    except Exception as exc:  # noqa: BLE001
        print(f"(audit summary unavailable: {exc})", flush=True)

    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
