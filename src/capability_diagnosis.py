#!/usr/bin/env python3
"""Capability diagnosis over rubric-scored redlining verdicts.

Treats each rubric criterion as a *capability probe* (the criterion text IS
the capability description), groups probes into a small capability tree,
scores the model at every node, and reports the nodes where the model is
weakest — i.e. *why* a model fails, not just *where*.

This is the diagnosis step of CRAFT (Khasseh et al., "CRAFT: Clustering
Rubrics to Diagnose Weak LLM Capabilities and Generate Targeted Fine-Tuning
Data", arXiv:2607.16122), adapted to RedlineBench:

  * CRAFT extracts a capability description from each prompt-rubric pair with
    an LLM. RedlineBench rubrics already carry a ``criteria`` field, so the
    LLM extraction step is replaced by that field — the rubric IS the probe.
  * CRAFT clusters descriptions with embeddings into a hierarchical tree.
    Here the tree is a parameter-free lexical proxy: root -> category ->
    keyword sub-cluster (rubrics probing the same capability share
    vocabulary), approximating the embedding-similarity signal with no model.
  * CRAFT selects low-performing nodes dynamically across tree levels, at the
    granularity where each failure is clearest. That selection rule is kept
    verbatim (most-specific node with enough support, greedy de-dup).
  * CRAFT then generates targeted supervised fine-tuning data from the
    selected weak capabilities. That step needs training infrastructure this
    harness does not host, so it is intentionally out of scope; instead each
    weak node carries the failing rubrics' judge justifications — the seed
    material that would direct such generation downstream.

Penalty rubrics (negative weight — an edit the attorney flagged as
undesirable) are scored with their verdict inverted: avoiding the bad edit
is the capability success, matching the harness's own scoring.
"""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import mean

# Domain-generic tokens that don't discriminate between capabilities;
# dropped from the keyword sub-cluster key so the key reflects the *substance*
# of a probe rather than boilerplate ("inserts", "section", "clause", ...).
_STOP = frozenset(
    (
        "the a an and or of to in for on with at by from as is are be it this "
        "that section clause contract redline edit edits rubric provision "
        "language text word paragraph new existing following must should "
        "insert inserts inserted insertion delete deletes deletion "
        "replace replaces replacement reject rejects accept accepts "
        "preserve preserves maintain maintains retain retains add adds "
        "remove removes use used using via per etc example also"
    ).split()
)
_WORD = re.compile(r"[a-z][a-z0-9-]{2,}")


def _tokens(text: str) -> list[str]:
    """Significant lowercase word tokens from a criterion string."""
    return [t for t in _WORD.findall((text or "").lower()) if t not in _STOP]


def _capability_success(verdict: str, weight: int) -> bool:
    """Did the model do the right thing on this probe?

    Normal (positive-weight) rubric: a PASS is success. Penalty
    (negative-weight) rubric — an edit flagged as undesirable — the model
    succeeds by *not* making it, so a FAIL verdict is the success.
    """
    if weight < 0:
        return verdict != "PASS"
    return verdict == "PASS"


def _cluster_label(category: str | None, criteria: str) -> tuple[str, str]:
    """Sub-cluster key + human label within a category.

    The key is the sorted union of significant tokens — rubrics probing the
    same capability share vocabulary, so equal keys land together, while two
    rubrics with disjoint substance get distinct keys even in the same
    category.
    """
    toks = sorted(set(_tokens(criteria)))
    key = " ".join(toks) or "(generic)"
    core = " ".join(toks[:3]) if toks else "(generic)"
    cat = category or "(uncategorized)"
    return f"{cat} :: {key}", f"{cat} / {core}"


def _build_tree(rows: list[dict]) -> list[dict]:
    """Group rows into root / category / keyword capability-tree nodes."""
    nodes: list[dict] = [
        {"level": 0, "key": "__root__", "label": "all rubrics", "rows": list(rows)}
    ]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    by_key: dict[str, dict] = {}  # key -> {"label", "rows"}
    for r in rows:
        cat = r.get("category") or "(uncategorized)"
        by_cat[cat].append(r)
        k, lbl = _cluster_label(r.get("category"), r.get("criteria") or "")
        slot = by_key.setdefault(k, {"label": lbl, "rows": []})
        slot["rows"].append(r)
    for cat, cat_rows in by_cat.items():
        nodes.append({"level": 1, "key": f"cat::{cat}", "label": cat, "rows": cat_rows})
    for k, slot in by_key.items():
        nodes.append({"level": 2, "key": k, "label": slot["label"], "rows": slot["rows"]})
    return nodes


def _pass_rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return mean(
        1 if _capability_success(r.get("verdict", "FAIL"), int(r.get("weight", 0))) else 0
        for r in rows
    )


def _summarize_node(node: dict) -> dict:
    rows = node["rows"]
    fails = [
        r
        for r in rows
        if not _capability_success(r.get("verdict", "FAIL"), int(r.get("weight", 0)))
    ]
    return {
        "capability": node["label"],
        "level": {0: "root", 1: "category", 2: "keyword"}.get(node["level"], str(node["level"])),
        "pass_rate": round(_pass_rate(rows), 4),
        "n_rubrics": len(rows),
        "categories": sorted({(r.get("category") or "(uncategorized)") for r in rows}),
        "example_criteria": [r.get("criteria", "") for r in rows[:2]],
        # Judge justifications for the failures in this node — the seed
        # material for targeted SFT data generation downstream (CRAFT's final
        # step, cut here for lack of training infrastructure in this harness).
        "example_failures": [
            (r.get("justification") or "").strip()
            for r in fails
            if (r.get("justification") or "").strip()
        ][:3],
    }


def diagnose_weak_capabilities(
    panel_rows_by_model: dict[str, list[dict]],
    *,
    min_support: int = 3,
    weak_pass_rate: float = 0.5,
    top_k: int = 8,
) -> dict[str, dict]:
    """Diagnose each model's weak capabilities from its panel rubric rows.

    ``panel_rows_by_model``: model -> list of per-rubric rows, each carrying
    ``rubric_id``, ``verdict`` ("PASS"/"FAIL"), ``weight`` (int; negative =
    penalty), and optionally ``category``, ``criteria``, ``justification``.

    Selection (CRAFT): consider nodes from most-specific (keyword) to least
    (category); a node is selectable once it has >= ``min_support`` rubrics;
    weak nodes (pass_rate < ``weak_pass_rate``) are reported greedily, most
    specific first, and a node is dropped once its rubrics are fully covered
    by a more-specific report — so each failure is reported at the
    granularity where it is clearest, without redundancy.
    """
    return {
        model: _diagnose_one(rows, min_support, weak_pass_rate, top_k)
        for model, rows in panel_rows_by_model.items()
    }


def _diagnose_one(
    rows: list[dict], min_support: int, weak_pass_rate: float, top_k: int
) -> dict:
    rows = list(rows or [])
    summary: dict = {
        "n_rubrics": len(rows),
        "overall_pass_rate": round(_pass_rate(rows), 4),
        "weak_capabilities": [],
        "strongest_capabilities": [],
    }
    if not rows:
        return summary

    nodes = [n for n in _build_tree(rows) if n["level"] >= 1]  # drop root as a capability

    weak_candidates = [
        n for n in nodes if len(n["rows"]) >= min_support and _pass_rate(n["rows"]) < weak_pass_rate
    ]
    weak_candidates.sort(key=lambda n: (-n["level"], _pass_rate(n["rows"])))

    covered: set[str] = set()
    for n in weak_candidates:
        rids = {r.get("rubric_id") for r in n["rows"]}
        if rids <= covered:
            continue  # wholly explained by a more-specific report
        summary["weak_capabilities"].append(_summarize_node(n))
        covered |= rids
        if len(summary["weak_capabilities"]) >= top_k:
            break

    strong_candidates = [
        n
        for n in nodes
        if len(n["rows"]) >= min_support and _pass_rate(n["rows"]) >= max(weak_pass_rate, 0.75)
    ]
    strong_candidates.sort(key=lambda n: (-_pass_rate(n["rows"]), -n["level"]))
    seen: set[str] = set()
    for n in strong_candidates:
        rids = {r.get("rubric_id") for r in n["rows"]}
        if rids <= seen:
            continue
        summary["strongest_capabilities"].append(_summarize_node(n))
        seen |= rids
        if len(summary["strongest_capabilities"]) >= 3:
            break
    return summary
