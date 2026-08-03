#!/usr/bin/env python3
"""Judge-time-compute scaling for the single-judge path.

Adapted (Mode 2) from *Verdict: A Library for Scaling Judge-Time Compute*
(Stanfield et al., arXiv:2502.18018). Verdict's central claim is that an
LLM judge becomes more reliable when you compose modular reasoning units
-- verification, debate, aggregation -- and spend more inference-time
compute, rather than trusting a single greedy call.

This module ports Verdict's *aggregation* unit at full fidelity onto the
repo's native single-judge chokepoint: sample ``judging.call_judge`` N
times on the same prompt and resolve a per-rubric consensus by majority
vote (self-consistency). It is complementary to the existing 3-MODEL
panel in ``panel.py`` (which resolves *cross-model* disagreement over
stored grade JSONs and makes no LLM call of its own); this scales compute
*within one model* on the live single-judge path that ``rejudge`` drives.

What is substituted (Mode 2) to stay target-native, with the core intact:

* Verdict's Unit/Graph DSL and library runtime -> a plain module wrapping
  the repo's existing ``judging.call_judge`` / ``judging.aggregate``.
* Verdict's debate protocol -> cut. Cross-model disagreement is already
  the panel's job; intra-model debate adds little on top of aggregation.
* Verdict's learned verifiers -> a parameter-free per-rubric agreement
  margin. This carries the *verification* unit's interpretability signal
  (it flags rubrics the judge is unreliable on) into the score payload the
  metrics/panel readers already consume, without a second learned model.
* Verdict's separate benchmark suite -> cut; evaluation is downstream.

The result keeps the ``{"verdicts": [...]}`` shape so it is a drop-in for
``judging.aggregate``, and adds a ``consensus`` block (per-rubric
agreement + contested-rubric flags) for the metrics tools to surface.
"""

from __future__ import annotations

import judging

#: Default sample count. 1 reproduces the plain single judge call.
DEFAULT_SAMPLES = 1

#: A rubric is "contested" (the verification interpretability flag) when
#: the winning faction holds less than this share of the samples -- i.e.
#: the judge is unreliable on it and a human should glance at the verdict.
CONTESTED_THRESHOLD = 2 / 3


def majority_verdict(entries: list[dict]) -> dict:
    """Resolve one rubric's consensus from its per-sample verdict entries.

    ``entries`` is the list of ``{"verdict", "justification"}`` dicts the
    judge produced for a single ``rubric_id`` across N samples. Majority
    vote; ties (and the no-data case) resolve to FAIL, because the rubric
    was not *clearly* satisfied -- matching the judge prompt's "PASS means
    clearly and unambiguously."

    Returns the consensus verdict plus an agreement diagnostic:
    ``n_samples``, ``n_pass``, ``agreement`` (winning-faction share in
    ``[0, 1]``) and ``contested`` (``agreement < CONTESTED_THRESHOLD``).
    """
    valid = [e for e in entries if e.get("verdict") in ("PASS", "FAIL")]
    denom = len(valid) or 1
    n_pass = sum(1 for e in valid if e.get("verdict") == "PASS")
    n_fail = len(valid) - n_pass
    verdict = "PASS" if n_pass > n_fail else "FAIL"
    agreement = max(n_pass, n_fail) / denom
    justification = ""
    for e in valid:
        if e.get("verdict") == verdict and e.get("justification"):
            justification = e["justification"]
            break
    return {
        "verdict": verdict,
        "justification": justification,
        "n_samples": len(entries),
        "n_pass": n_pass,
        "agreement": agreement,
        "contested": agreement < CONTESTED_THRESHOLD,
    }


def consensus_verdicts(samples: list[list[dict]]) -> dict:
    """Majority-vote N judge samples into one consensus verdict set.

    ``samples`` is a list of verdict lists (each the ``verdicts`` field of
    one ``call_judge`` result). Verdicts are aligned by ``rubric_id``
    (first occurrence per sample wins, matching ``judging.aggregate``) in
    first-seen order. Returns ``{"verdicts": [...], "consensus": {...}}``
    whose ``verdicts`` list is a drop-in for ``judging.aggregate``.
    """
    order: list[str] = []
    by_rubric: dict[str, list[dict]] = {}
    for verdicts in samples:
        seen: set[str] = set()
        for v in verdicts:
            rid = v.get("rubric_id")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            by_rubric.setdefault(rid, []).append(v)
            if rid not in order:
                order.append(rid)

    rows: list[dict] = []
    n_contested = 0
    for rid in order:
        cons = majority_verdict(by_rubric[rid])
        if cons["contested"]:
            n_contested += 1
        rows.append({"rubric_id": rid, **cons})

    return {
        "verdicts": [
            {
                "rubric_id": r["rubric_id"],
                "verdict": r["verdict"],
                "justification": r["justification"],
            }
            for r in rows
        ],
        "consensus": {
            "n_samples": len(samples),
            "n_rubrics": len(order),
            "n_contested": n_contested,
            "contested_threshold": CONTESTED_THRESHOLD,
            "per_rubric": rows,
        },
    }


def scale_judge(
    model: str, system: str, user: str, samples: int = DEFAULT_SAMPLES
) -> dict:
    """Judge-time-compute scaling: N ``call_judge`` samples -> consensus.

    Calls the shared ``judging.call_judge`` chokepoint ``samples`` times on
    the same prompt and majority-votes the per-rubric verdicts. Per-sample
    failures (network blip, unparseable response, non-retriable 4xx) are
    tolerated: the consensus is built from whichever samples succeeded and
    the failure count is recorded in the ``consensus`` block. Raises only
    if every sample failed.
    """
    samples = max(1, int(samples))
    collected: list[list[dict]] = []
    n_failures = 0
    last_exc: Exception | None = None
    for _ in range(samples):
        try:
            resp = judging.call_judge(model, system, user)
            collected.append(resp.get("verdicts") or [])
        except Exception as exc:  # noqa: BLE001 - tolerate one bad sample
            n_failures += 1
            last_exc = exc
    if not collected:
        raise RuntimeError(
            f"consensus judge failed on all {samples} samples: {last_exc!r}"
        )
    out = consensus_verdicts(collected)
    out["consensus"]["n_failures"] = n_failures
    return out
