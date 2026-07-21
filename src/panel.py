#!/usr/bin/env python3
"""Judge-panel scoring + sensitivity analysis for RedlineBench.

Takes the per-rubric verdicts from several judges (each a directory tree of
grade JSONs produced by `rejudge`, laid out as
<dir>/<model>/<task>.json) and produces:

  1. Official panel score  — rubric-level MAJORITY VOTE across the judges
     (binary labels + odd judge count => no ties), re-aggregated with the
     weighted/penalty score math, input-group averaged, mean over groups.
  2. Per-judge sensitivity — each judge family's standalone leaderboard, so we
     can see whether the model ranking depends on the judge family.
  3. Judge agreement      — pairwise rubric-level agreement rates + overall.

Writes <out>/panel_summary.json and <out>/panel_leaderboard.csv.

Usage:
    python -m panel \
        --judge "gpt-5.4-mini=results/judge/gpt-5.4-mini" \
        --judge "gemini-3.5-flash=results/judge/gemini-3.5-flash" \
        --judge "claude-haiku=results/judge/claude-haiku" \
        --out results/panel
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

from judge_confusion import confusion_metrics, format_confusion

_NAME_RE = re.compile(r"redline-s(\d+)-t(\d+)-g(\d+)([a-z])")


def _input_group(task: str) -> str:
    m = _NAME_RE.search(task)
    return f"s{m.group(1)}-t{m.group(2)}-g{m.group(3)}"


def load_judge(root: Path) -> dict:
    """(model, task) -> grade dict, for one judge's output tree."""
    out = {}
    for f in root.rglob("*.json"):
        d = json.loads(f.read_text())
        out[(f.parent.name, f.stem)] = d
    return out


def _rubric_rows(grade: dict) -> dict:
    """rubric_id -> (verdict, weight, category) for a single grade."""
    return {
        p["rubric_id"]: (p["verdict"], int(p["weight"]), p.get("category"))
        for p in grade.get("score", {}).get("per_rubric", [])
    }


def weighted_score(verdicts: dict, weights: dict) -> float:
    """Penalty-aware weighted score, clamped to [0, 1].

    `verdicts`: rubric_id → "PASS" / "FAIL"
    `weights`:  rubric_id → int (positive = reward, negative = penalty)

    Positive-weight rubrics with PASS contribute their weight to `earned`.
    Negative-weight rubrics with PASS subtract |weight| as penalty.
    Denominator is the sum of positive weights; score clamps to [0, 1].

    Single source of truth for the weighted-score formula across:
      - `panel.main()` (post-hoc panel aggregation CLI)
      - `panel_reader.collect_panel_rows()` (metrics pipeline reader)
      - `harbor/tasks/*/tests/judge.py` (live verifier; can't import,
        so it carries an inline mirror — keep in sync if either
        changes).
    """
    earned = penalty = total_pos = 0
    for rid, w in weights.items():
        if w > 0:
            total_pos += w
        if verdicts.get(rid) == "PASS":
            if w > 0:
                earned += w
            elif w < 0:
                penalty += -w
    raw = (earned - penalty) / total_pos if total_pos else 0.0
    return max(0.0, min(1.0, raw))


# Back-compat private alias — older callers (and the verifier mirror)
# still reference `_weighted`; keep the symbol so external code doesn't
# break.
_weighted = weighted_score


def majority_vote_per_rubric(
    rubric_sets_per_judge: list[dict[str, tuple[str, int, str | None]]],
) -> tuple[dict[str, str], dict[str, int]]:
    """Reduce N judges' per-rubric verdicts to a single panel verdict
    per rubric by strict majority vote.

    Input: list of N maps (one per judge), each rubric_id → (verdict,
    weight, category) tuple — the shape `_rubric_rows()` produces.

    Returns `(panel_verdicts, weights)`:
      - `panel_verdicts[rid]`: "PASS" iff `n_pass * 2 > n_voters` (strict
        majority among the judges who actually graded that rubric),
        else "FAIL". With 3 judges this never ties; with an even count
        ties resolve to "FAIL".
      - `weights[rid]`: the rubric's weight, taken from the first judge
        that scored it (weight is judge-invariant — same rubric, same
        weight, regardless of who graded).

    This is the same vote `panel.main()` runs inline, factored out so
    both the post-hoc panel CLI and `panel_reader` use
    one implementation.
    """
    all_rids: set[str] = set().union(
        *[set(rs.keys()) for rs in rubric_sets_per_judge]
    ) if rubric_sets_per_judge else set()
    panel_verdicts: dict[str, str] = {}
    weights: dict[str, int] = {}
    for rid in all_rids:
        votes = [rs[rid][0] for rs in rubric_sets_per_judge if rid in rs]
        weights[rid] = next(
            rs[rid][1] for rs in rubric_sets_per_judge if rid in rs
        )
        n_pass = sum(1 for v in votes if v == "PASS")
        panel_verdicts[rid] = "PASS" if n_pass * 2 > len(votes) else "FAIL"
    return panel_verdicts, weights


def _leaderboard(per_model_group_scores: dict) -> dict:
    """{model: {group: score}} -> {model: overall mean-over-groups}."""
    return {m: round(mean(g.values()), 4) for m, g in per_model_group_scores.items() if g}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", action="append", required=True,
                    help="label=path/to/judge/output/tree (repeatable; use an odd count)")
    ap.add_argument("--reference", default=None,
                    help="optional label=path for the reference judge (e.g. gpt-5.5 from rollout grades) "
                         "— compared against the panel but NOT part of the vote")
    ap.add_argument("--out", default="results/panel")
    args = ap.parse_args()

    judges = {}
    for spec in args.judge:
        label, path = spec.split("=", 1)
        judges[label] = load_judge(Path(path))
    if len(judges) % 2 == 0:
        print(f"WARNING: {len(judges)} judges is even — ties possible in majority vote",
              file=sys.stderr)

    # union of (model, task) keys present in ALL judges
    common = set.intersection(*[set(j.keys()) for j in judges.values()])
    print(f"{len(judges)} judges, {len(common)} (model,task) pairs graded by all", flush=True)

    # --- per-judge standalone leaderboards (sensitivity) ---
    per_judge_scores = {}            # label -> {model: {group: score}}
    for label, jg in judges.items():
        pmg = defaultdict(dict)
        by_mt = defaultdict(list)
        for (model, task) in common:
            by_mt[(model, _input_group(task))].append(jg[(model, task)])
        for (model, group), grades in by_mt.items():
            pmg[model][group] = mean(g.get("score", {}).get("weighted", 0.0) for g in grades)
        per_judge_scores[label] = pmg
    sensitivity = {label: _leaderboard(s) for label, s in per_judge_scores.items()}

    # --- panel: rubric-level majority vote ---
    panel_pmg = defaultdict(dict)    # model -> {group: score}
    by_mt = defaultdict(list)
    for (model, task) in common:
        by_mt[(model, _input_group(task))].append((model, task))
    for (model, group), keys in by_mt.items():
        task_scores = []
        for (m, task) in keys:
            rubric_sets = [_rubric_rows(judges[label][(m, task)]) for label in judges]
            panel_verdicts, weights = majority_vote_per_rubric(rubric_sets)
            task_scores.append(weighted_score(panel_verdicts, weights))
        panel_pmg[model][group] = mean(task_scores)
    panel_leaderboard = _leaderboard(panel_pmg)

    # --- judge agreement (pairwise rubric-level) ---
    agreement = {}
    for a, b in combinations(judges, 2):
        agree = total = 0
        for (model, task) in common:
            ra, rb = _rubric_rows(judges[a][(model, task)]), _rubric_rows(judges[b][(model, task)])
            for rid in set(ra) & set(rb):
                total += 1
                if ra[rid][0] == rb[rid][0]:
                    agree += 1
        agreement[f"{a} vs {b}"] = round(agree / total, 4) if total else None

    # --- reference-judge comparison (optional, not part of vote) ---
    reference = None
    if args.reference:
        rlabel, rpath = args.reference.split("=", 1)
        ref = load_judge(Path(rpath))
        ref_common = common & set(ref.keys())
        pmg = defaultdict(dict)
        by = defaultdict(list)
        for (model, task) in ref_common:
            by[(model, _input_group(task))].append(ref[(model, task)])
        for (model, group), grades in by.items():
            pmg[model][group] = mean(g.get("score", {}).get("weighted", 0.0) for g in grades)
        reference = {"label": rlabel, "leaderboard": _leaderboard(pmg)}

    # --- per-judge directional confusion (FPR / FNR / pass-rate drift) ---
    # `sensitivity_per_judge` above collapses each judge to one scalar and
    # hides error *direction* — an over-crediting judge and an over-rejecting
    # one can tie. We break that apart against a gold reference: the
    # designated --reference judge when supplied, otherwise a leave-one-out
    # majority of the *other* panel judges (consensus-as-gold, the stand-in
    # for the paper's human gold labels). This is the prescriptive half of
    # the sensitivity analysis — it tells `rejudge` which judge family is
    # safe to cheap out on, not just which ranks highest.
    # (Adapted from arXiv:2607.08700 — directional-metric mechanism at full
    # fidelity; only the gold source is substituted.)
    labels = list(judges)
    sensitivity_confusion = {}
    for label in labels:
        judge_lv: list[str] = []
        gold_lv: list[str] = []
        for (model, task) in common:
            judge_v = {rid: v[0] for rid, v in
                       _rubric_rows(judges[label][(model, task)]).items()}
            if reference is not None and (model, task) in ref_common:
                gold_v = {rid: v[0] for rid, v in
                          _rubric_rows(ref[(model, task)]).items()}
            else:
                others = [_rubric_rows(judges[o][(model, task)])
                          for o in labels if o != label]
                gold_v, _ = majority_vote_per_rubric(others)
            for rid in set(judge_v) & set(gold_v):
                judge_lv.append(judge_v[rid])
                gold_lv.append(gold_v[rid])
        if judge_lv:
            sensitivity_confusion[label] = confusion_metrics(judge_lv, gold_lv)

    def ranked(lb):
        return [m for m, _ in sorted(lb.items(), key=lambda kv: -kv[1])]

    summary = {
        "judges": list(judges),
        "n_pairs": len(common),
        "panel_leaderboard": dict(sorted(panel_leaderboard.items(), key=lambda kv: -kv[1])),
        "panel_ranking": ranked(panel_leaderboard),
        "sensitivity_per_judge": sensitivity,
        "sensitivity_rankings": {lbl: ranked(lb) for lbl, lb in sensitivity.items()},
        "sensitivity_confusion": sensitivity_confusion,
        "judge_agreement": agreement,
        "ranking_stable_across_judges": len({tuple(ranked(lb)) for lb in sensitivity.values()}) == 1,
    }
    if reference:
        summary["reference_judge"] = reference
        summary["panel_matches_reference_ranking"] = ranked(panel_leaderboard) == ranked(reference["leaderboard"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "panel_summary.json").write_text(json.dumps(summary, indent=2))

    models = sorted(panel_leaderboard, key=lambda m: -panel_leaderboard[m])
    with (out / "panel_leaderboard.csv").open("w", newline="") as f:
        w = csv.writer(f)
        cols = ["model", "panel_majority"] + [f"judge:{lbl}" for lbl in judges]
        if reference:
            cols.append(f"reference:{reference['label']}")
        w.writerow(cols)
        for m in models:
            row = [m, panel_leaderboard[m]] + [sensitivity[lbl].get(m) for lbl in judges]
            if reference:
                row.append(reference["leaderboard"].get(m))
            w.writerow(row)

    print(f"\npanel (majority vote): {summary['panel_leaderboard']}")
    print(f"ranking stable across judge families: {summary['ranking_stable_across_judges']}")
    if sensitivity_confusion:
        source = (f"reference {reference['label']}" if reference
                  else "leave-one-out panel consensus")
        print(f"judge directional bias (vs {source}):")
        for label in labels:
            if label in sensitivity_confusion:
                print("  " + format_confusion(label, sensitivity_confusion[label]))
    if reference:
        print(f"panel matches reference ({reference['label']}) ranking: "
              f"{summary['panel_matches_reference_ranking']}")
    print(f"judge agreement: {agreement}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
