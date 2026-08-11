"""Pivotal-rubric (affected-set) analysis for the multi-judge panel.

Implements the single-ballot-substitution "affected set" from *Blind to
the Pivotal Vote: Aggregate Independence Metrics Miss Where Verification
Actually Helps* (arXiv:2608.06940v1).

The paper's core arithmetic: when you add one extra ballot to a panel
(a 4th judge, an external verifier signal), only rubrics decided by a
one-vote margin can possibly flip — everything else is locked in. So:

  * the panel's entire sensitivity to that extra ballot sits on the
    "pivotal" rubrics (``|n_pass - n_fail| == 1``);
  * a deeper-verification / 4th-judge pass only needs to *run* on those
    rubrics — a call-reduction rule. The paper finds the accuracy gain
    from verification concentrates entirely on pivotal queries and is
    ~zero elsewhere.

This is a Mode-2 adapted port. The paper's full substitution experiment
— actually executing a verifier signal / calling a real 4th judge and
measuring the accuracy lift — is cut (that is downstream evaluation on
the paper's own code benchmarks). What survives at full fidelity is the
affected-set characterization and the call-reduction rule, computed from
the per-rubric vote counts ``panel.majority_vote_per_rubric`` already
produces internally, with no extra judge calls and using the panel's own
``weighted_score`` denominator in place of the paper's accuracy metric.

Inputs/outputs consume the exact shapes ``panel.py`` and
``panel_reader.py`` already use, so this slots in alongside the existing
rubric-level majority vote without changing its signature.
"""

from __future__ import annotations


def per_rubric_margins(
    rubric_sets_per_judge: list[dict[str, tuple[str, int, str | None]]],
) -> dict[str, dict]:
    """Per-rubric vote tallies for a panel.

    Input: the same shape ``panel.majority_vote_per_rubric`` takes — a
    list of N judge maps, each ``rubric_id -> (verdict, weight, category)``.

    Returns ``rubric_id -> {n_pass, n_fail, n_voters, margin, pivotal}``
    where ``margin = |n_pass - n_fail|`` and ``pivotal`` is True iff that
    margin is exactly 1 (a single extra ballot can flip the verdict).
    Votes are counted only among judges that actually graded each rubric,
    matching ``majority_vote_per_rubric``.
    """
    all_rids: set[str] = set().union(
        *[set(rs.keys()) for rs in rubric_sets_per_judge]
    ) if rubric_sets_per_judge else set()
    out: dict[str, dict] = {}
    for rid in all_rids:
        votes = [rs[rid][0] for rs in rubric_sets_per_judge if rid in rs]
        n_pass = sum(1 for v in votes if v == "PASS")
        n_voters = len(votes)
        n_fail = n_voters - n_pass
        margin = abs(n_pass - n_fail)
        out[rid] = {
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_voters": n_voters,
            "margin": margin,
            "pivotal": margin == 1,
        }
    return out


def _pivotal_counts(
    rubric_sets_per_judge: list[dict[str, tuple[str, int, str | None]]],
    weights: dict[str, int],
) -> tuple[int, int, int, int]:
    """Raw pivotal tallies: ``(n_rubrics, n_pivotal, total_pos_weight,
    pivotal_pos_weight)``.

    Shared by :func:`pivotal_task_stats` (rich per-task dict) and
    :class:`PivotalTally` (streaming panel-wide rollup) so the margin
    loop lives in one place. ``total_pos_weight`` is the panel score's
    denominator (Σ positive weights); ``pivotal_pos_weight`` is the
    slice of it sitting on pivotal rubrics.
    """
    margins = per_rubric_margins(rubric_sets_per_judge)
    n_pivotal = pivotal_pos = 0
    for rid, m in margins.items():
        if m["pivotal"]:
            n_pivotal += 1
            w = weights.get(rid, 0)
            if w > 0:
                pivotal_pos += w
    total_pos = sum(w for w in weights.values() if w > 0)
    return len(margins), n_pivotal, total_pos, pivotal_pos


def pivotal_task_stats(
    rubric_sets_per_judge: list[dict[str, tuple[str, int, str | None]]],
    weights: dict[str, int],
) -> dict:
    """Pivotal-vote statistics for one task's panel vote.

    ``weights`` is the second return value of
    ``panel.majority_vote_per_rubric``, so the two share a rubric set.
    """
    n_rubrics, n_pivotal, total_pos, pivotal_pos = _pivotal_counts(
        rubric_sets_per_judge, weights,
    )
    margins = per_rubric_margins(rubric_sets_per_judge)
    pivotal_rids = sorted(r for r, m in margins.items() if m["pivotal"])

    fraction_pivotal = n_pivotal / n_rubrics if n_rubrics else 0.0
    pivotal_weight_share = pivotal_pos / total_pos if total_pos else 0.0

    return {
        "n_rubrics": n_rubrics,
        "n_pivotal": n_pivotal,
        # Call-reduction rule (arXiv:2608.06940v1): an extra ballot /
        # verifier signal can only move pivotal rubrics, so a
        # pivotal-only verification pass invokes the signal on exactly
        # this fraction of rubrics and may skip the rest.
        "fraction_pivotal": round(fraction_pivotal, 4),
        "verification_invocation_rate": round(fraction_pivotal, 4),
        # Fraction of the panel's positive rubric weight — i.e. of the
        # weighted-score denominator — sitting on rubrics one ballot can
        # flip. That is the share of the score hinging on the pivotal set.
        "pivotal_weight_share": round(pivotal_weight_share, 4),
        "total_pos_weight": total_pos,
        "pivotal_pos_weight": pivotal_pos,
        "pivotal_rubric_ids": pivotal_rids,
    }


def pivotal_score_swing(
    panel_verdicts: dict[str, str],
    weights: dict[str, int],
    rubric_sets_per_judge: list[dict[str, tuple[str, int, str | None]]],
) -> float:
    """Upper bound on ``|Δscore|`` a single extra ballot could force on
    this task: the weighted-score change if every pivotal rubric flipped
    against the panel.

    Adversarial — it assumes all pivotal flips break the panel's way at
    once — so it *bounds* rather than predicts the realized swing. Uses
    the shared :func:`panel.weighted_score` so the delta is in the same
    units the panel leaderboard scores in.
    """
    # Imported lazily to avoid a top-level cycle: ``panel`` imports this
    # module at load time, and ``weighted_score`` is the only symbol we
    # need from it (and only here).
    from panel import weighted_score

    margins = per_rubric_margins(rubric_sets_per_judge)
    flipped = dict(panel_verdicts)
    for rid, m in margins.items():
        if m["pivotal"] and rid in flipped:
            flipped[rid] = "FAIL" if flipped[rid] == "PASS" else "PASS"
    return round(
        abs(weighted_score(flipped, weights)
            - weighted_score(panel_verdicts, weights)),
        4,
    )


class PivotalTally:
    """Streaming accumulator for pivotal-vote stats across many tasks.

    ``panel.main()`` votes on hundreds of (model, task) panels; this
    rolls the per-task counts and weights up to a panel-wide fraction
    without holding every task's stats in memory. Used as a
    ``defaultdict`` factory: ``defaultdict(PivotalTally)``.
    """

    def __init__(self) -> None:
        self.n_rubrics = 0
        self.n_pivotal = 0
        self.total_pos_weight = 0
        self.pivotal_pos_weight = 0

    def add(self, stats: dict) -> None:
        """Fold one :func:`pivotal_task_stats` result into the tally."""
        self.n_rubrics += stats["n_rubrics"]
        self.n_pivotal += stats["n_pivotal"]
        self.total_pos_weight += stats["total_pos_weight"]
        self.pivotal_pos_weight += stats["pivotal_pos_weight"]

    def summary(self) -> dict:
        fraction_pivotal = (
            self.n_pivotal / self.n_rubrics if self.n_rubrics else 0.0
        )
        pivotal_weight_share = (
            self.pivotal_pos_weight / self.total_pos_weight
            if self.total_pos_weight else 0.0
        )
        return {
            "n_rubrics": self.n_rubrics,
            "n_pivotal": self.n_pivotal,
            "fraction_pivotal": round(fraction_pivotal, 4),
            "verification_invocation_rate": round(fraction_pivotal, 4),
            "pivotal_weight_share": round(pivotal_weight_share, 4),
        }
