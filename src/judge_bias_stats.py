"""Judge-panel agreement and same-provider bias statistics.

Adapted from *Evaluating medical AI under missing information: same-provider
judges and human raters change apparent safety* (arXiv:2607.18828), whose two
evaluator-facing findings are that (1) judge choice materially changes apparent
safety — inter-judge agreement is only moderate — and (2) after adjusting for
each judge's general leniency a same-provider association remains. This module
ports those two evaluator-facing statistics onto RedlineBench's existing
per-rubric-verdict panel contract:

  * ``fleiss_kappa``               — multi-rater agreement across the whole
                                     panel (the panel currently emits only
                                     pairwise agreement rates).
  * ``same_provider_association``  — the leniency-adjusted same-provider boost
                                     on P(PASS), via a vote-level regression
                                     with judge fixed effects + a same-provider
                                     indicator, tested by a Monte-Carlo
                                     permutation test.

Mode-2 adaptation, cited: the paper fits a *logistic* regression; we use a
*linear probability model* (OLS with judge fixed effects) so the same-provider
slope is reported directly on the probability scale — the scale the paper
itself uses ("~ +0.10 on the probability scale") — and so the permutation null
is tractable in pure Python: each permutation is O(votes) via
Frisch-Waugh-Lovell residualization on judge means (no numpy/scipy needed).
The core mechanism — a leniency-adjusted, judge-fixed-effect-netted
same-provider association, significance-tested by permutation — is preserved
at full fidelity. The paper's medical-AI experiment, clinician reference, and
benchmark suite are out of scope (downstream); this module consumes the
panel's own per-rubric verdicts.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

# (lowercased token -> provider key); first match wins. Best-effort — override
# via the ``provider_of`` argument when a deployment uses non-standard names.
_PROVIDER_RULES = (
    ("openai", "openai"), ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"),
    ("anthropic", "anthropic"), ("claude", "anthropic"),
    ("gemini", "google"), ("google", "google"),
    ("grok", "xai"), ("xai", "xai"),
    ("llama", "meta"), ("meta", "meta"),
    ("mistral", "mistral"), ("mixtral", "mistral"),
    ("deepseek", "deepseek"),
    ("qwen", "alibaba"), ("alibaba", "alibaba"),
    ("phi", "microsoft"),
)


def provider_of(name: str) -> str | None:
    """Best-effort provider key for a judge/model label.

    ``"gpt-5.5"`` -> ``"openai"``; ``"claude-haiku-4-5"`` -> ``"anthropic"``;
    ``"gemini-3.5-flash"`` -> ``"google"``. Returns ``None`` when no provider
    token is recognised, in which case the label never counts as
    same-provider.
    """
    s = name.lower()
    for token, provider in _PROVIDER_RULES:
        if token in s:
            return provider
    return None


def fleiss_kappa(vote_matrix: list[list[int]]) -> dict:
    """Fleiss' kappa for binary votes (1 = PASS, 0 = FAIL).

    ``vote_matrix`` is one row per rubric-item, each row a list of 0/1 votes
    from the panel's judges (all rows should come from items scored by the
    same set of judges). Returns ``{"kappa", "p_observed", "p_expected",
    "n_items", "n_raters"}``; ``kappa`` is ``None`` when agreement is
    undefined (fewer than two items, fewer than two raters, or every vote
    landing in one category so chance agreement is degenerate).
    """
    items = [row for row in vote_matrix if row]
    n_items = len(items)
    n_raters = len(items[0]) if items else 0
    none = {"kappa": None, "p_observed": None, "p_expected": None,
            "n_items": n_items, "n_raters": n_raters}
    if n_items < 2 or n_raters < 2:
        return none

    total_pass = sum(sum(row) for row in items)
    total_votes = n_items * n_raters
    p_pass = total_pass / total_votes
    p_fail = 1.0 - p_pass
    p_expected = p_pass * p_pass + p_fail * p_fail
    if p_expected >= 1.0:
        none["p_expected"] = p_expected
        return none

    p_observed = 0.0
    for row in items:
        n_pass = sum(row)
        n_fail = n_raters - n_pass
        p_observed += (n_pass * n_pass + n_fail * n_fail) / (n_raters * n_raters)
    p_observed /= n_items
    kappa = (p_observed - p_expected) / (1.0 - p_expected)
    return {"kappa": round(kappa, 4), "p_observed": round(p_observed, 4),
            "p_expected": round(p_expected, 4), "n_items": n_items,
            "n_raters": n_raters}


def _judge_demean(values: list[float], judges: list[str]) -> list[float]:
    """Residualize ``values`` on judge fixed effects (subtract each judge's
    mean) — the Frisch-Waugh-Lovell projection for one-way judge effects."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for v, j in zip(values, judges):
        sums[j] += v
        counts[j] += 1
    means = {j: sums[j] / counts[j] for j in sums}
    return [v - means[j] for v, j in zip(values, judges)]


def same_provider_association(
    observations: list[dict],
    *,
    provider_of=provider_of,
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict:
    """Leniency-adjusted same-provider association on P(PASS).

    Each observation is a ``{"judge", "model", "verdict"}`` dict (verdict
    ``"PASS"``/``"FAIL"``). We fit a linear probability model

        PASS ~ judge fixed effects + same_provider

    so the same-provider slope is the boost (on the probability scale) a
    judge grants models from its own provider, *net of its general leniency*
    (the judge fixed effects absorb it). Significance comes from a
    Monte-Carlo permutation test that shuffles the same-provider indicator
    across votes and recomputes the slope, judge fixed effects held fixed.

    Returns ``{"coef", "perm_p", "se", "wald_z", "ci95", "n",
    "n_same_provider", "n_judges", "n_models"}``; ``coef``/``perm_p`` are
    ``None`` with a ``note`` when the effect is not identifiable (fewer than
    two judges, no same-provider votes, or same-provider collinear with the
    judge partition).
    """
    base = {
        "coef": None, "perm_p": None, "se": None, "wald_z": None,
        "ci95": None, "n": 0, "n_same_provider": 0, "n_judges": 0,
        "n_models": 0,
    }
    if not observations:
        base["note"] = "no votes"
        return base

    y: list[int] = []
    judges: list[str] = []
    models: list[str] = []
    s: list[int] = []
    n_same = 0
    for obs in observations:
        verdict = obs.get("verdict")
        if verdict not in ("PASS", "FAIL"):
            continue
        judge, model = obs["judge"], obs["model"]
        jp, mp = provider_of(judge), provider_of(model)
        same = 1 if (jp is not None and jp == mp) else 0
        n_same += same
        y.append(1 if verdict == "PASS" else 0)
        judges.append(judge)
        models.append(model)
        s.append(same)

    n = len(y)
    base.update(n=n, n_same_provider=n_same,
                n_judges=len(set(judges)), n_models=len(set(models)))
    if len(set(judges)) < 2:
        base["note"] = "fewer than two judges"
        return base
    if n_same == 0:
        base["note"] = ("no same-provider votes — set provider_of or rename "
                        "labels so a judge shares a provider with a model")
        return base

    y_res = _judge_demean([float(v) for v in y], judges)
    s_res = _judge_demean([float(v) for v in s], judges)
    den = sum(si * si for si in s_res)
    if den <= 1e-12:
        base["note"] = "same-provider collinear with the judge partition"
        return base
    coef = sum(yr * sr for yr, sr in zip(y_res, s_res)) / den

    # Heteroskedasticity-robust (HC1) SE for the same-provider slope. Under
    # perfect separation the LPM residuals are zero -> SE collapses to 0
    # (a degenerate fit); inference then rests on the permutation test below.
    resid = [yr - coef * sr for yr, sr in zip(y_res, s_res)]
    num_var = sum(e * e * sr * sr for e, sr in zip(resid, s_res))
    p_params = len(set(judges)) + 1  # judge means + slope
    df = max(n - p_params, 1)
    var = (n / df) * num_var / (den * den)
    se = math.sqrt(max(var, 0.0))
    wald_z = coef / se if se > 0 else None
    ci = [coef - 1.96 * se, coef + 1.96 * se]

    # Monte-Carlo permutation null: shuffle the same-provider indicator,
    # re-residualize on judge means, recompute the slope.
    judge_idx: dict[str, list[int]] = defaultdict(list)
    for i, j in enumerate(judges):
        judge_idx[j].append(i)
    obs_abs = abs(coef)
    count = 0
    rng = random.Random(seed)
    perm = list(s)
    reps = max(1, n_permutations)
    for _ in range(reps):
        rng.shuffle(perm)
        pmean = {j: sum(perm[i] for i in idx) / len(idx)
                 for j, idx in judge_idx.items()}
        pnum = pden = 0.0
        for i in range(n):
            sri = perm[i] - pmean[judges[i]]
            pnum += y_res[i] * sri
            pden += sri * sri
        bstar = pnum / pden if pden > 1e-12 else 0.0
        if abs(bstar) >= obs_abs:
            count += 1
    perm_p = (1 + count) / (reps + 1)

    return {
        "coef": round(coef, 6), "perm_p": round(perm_p, 6),
        "se": round(se, 6) if se is not None else None,
        "wald_z": round(wald_z, 4) if wald_z is not None else None,
        "ci95": [round(c, 6) for c in ci] if ci else None,
        "n": n, "n_same_provider": n_same,
        "n_judges": len(set(judges)), "n_models": len(set(models)),
    }


def summarize_panel_bias(
    judges: dict,
    common: set,
    rubric_rows,
    *,
    provider_of=provider_of,
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict:
    """Aggregate the panel's agreement + same-provider bias statistics.

    ``judges``: ``label -> {(model, task): grade}`` (the shape ``panel.main``
    builds). ``common``: the ``(model, task)`` pairs every judge graded.
    ``rubric_rows(grade) -> {rubric_id: (verdict, weight, category)}`` — pass
    ``panel._rubric_rows``.

    Fleiss' kappa uses only rubric-items scored by *every* judge (constant
    rater count); the regression uses all available per-rubric votes.
    """
    labels = list(judges)

    complete_rows: list[list[int]] = []
    for (model, task) in common:
        per_judge = {l: rubric_rows(judges[l][(model, task)]) for l in labels}
        shared = (set.intersection(*[set(r) for r in per_judge.values()])
                  if per_judge else set())
        for rid in shared:
            complete_rows.append(
                [1 if per_judge[l][rid][0] == "PASS" else 0 for l in labels])
    fleiss = fleiss_kappa(complete_rows)

    observations: list[dict] = []
    for (model, task) in common:
        for l in labels:
            for rid, row in rubric_rows(judges[l][(model, task)]).items():
                observations.append(
                    {"judge": l, "model": model, "verdict": row[0]})
    spa = same_provider_association(observations, provider_of=provider_of,
                                    n_permutations=n_permutations, seed=seed)

    return {"fleiss_kappa": fleiss, "same_provider_association": spa,
            "n_judges": len(labels), "n_rubric_items": fleiss["n_items"]}
