#!/usr/bin/env python3
"""Build the single-file, on-brand RedlineBench HTML research report.

Reads `report/report_data.json` (from `report_metrics`)
and the micro1 logo, emits a self-contained `report/index.html` with a
dark/light toggle on the micro1-report house style. This is the
report's figure-generation logic — the .html it produces is the
deliverable.

The 8 sections rendered (matching the user's spec):

  1. Overall score (turn-weighted, 12-cell scenario×turn mean)
  2. Score by side (turn-weighted)
  3. Score by scenario (turn-weighted)
  4. Score by evaluation dimension (rubric category)
  5. Score breakdown by turn (per model)
  6. best@k (max reward per (model, task) across trials; collapses
     to overall_turn_weighted when there's only 1 trial per task)
  7. Verbosity trap (turn 1 only — paragraph-index alignment is only
     reliable when the input is the clean template)
  8. Surgicalness (inline vs block share per model + human baseline)

The "failure modes", "more edits not better edits", "where attorneys
disagree", and "judge-panel robustness" sections from the previous
version are gone. The CSS + dark/light toggle from the prior version
is preserved.

Usage:
    python -m build_report_html \\
        --data report/report_data.json \\
        --logo <path/to/micro1-logo.svg> \\
        --out report/index.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

# Per-model accent colors. Up to 5 models displayed in distinct hues;
# overflow recycles the palette.
C = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--accent)"]


def esc(s) -> str:
    return html.escape(str(s))


def pct(x) -> str:
    """Render a 0..1 number as a percent like '34.2%'. None → em-dash."""
    if x is None:
        return "—"
    return f"{x * 100:.1f}%"


def fnum(x, places: int = 3) -> str:
    """Render a number with N decimal places. None → em-dash."""
    if x is None:
        return "—"
    return f"{x:.{places}f}"


def short(m: str) -> str:
    """Display-pretty version of a model id. Maps the dir-name model
    identifiers (`opus48`, `gpt55`, `gemini35`) to short labels, and
    strips `claude-` prefixes."""
    return (
        m.replace("claude-", "")
        .replace("-3.5-flash", "-flash")
        .replace("-4-8", "-4.8")
        .replace("opus48", "opus-4.8")
        .replace("gpt55", "gpt-5.5")
        .replace("gemini35", "gemini-3.5-flash")
    )


def bar_row(label, frac, num, color, sub=""):
    w = max(0.0, min(1.0, frac)) * 100
    sublab = f'<div class="bar-sub">{esc(sub)}</div>' if sub else ""
    return (
        f'<div class="bar-row"><div class="bar-label">{esc(label)}{sublab}</div>'
        f'<div class="bar-wrap"><div class="bar-fill" style="width:{w:.1f}%;background:{color}"></div></div>'
        f'<div class="bar-num">{esc(num)}</div></div>'
    )


def heat_color(v, lo, hi):
    """Smooth red→amber→green ramp at ~55% opacity for the category
    heatmap; precomputed so no JS is needed."""
    if v is None:
        return "rgba(120,120,120,0.2)"
    if hi == lo:
        t = 0.5
    else:
        t = (v - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    stops = [(239, 68, 68), (251, 191, 36), (34, 197, 94)]
    if t < 0.5:
        a, b, f = stops[0], stops[1], t / 0.5
    else:
        a, b, f = stops[1], stops[2], (t - 0.5) / 0.5
    r = int(a[0] + (b[0] - a[0]) * f)
    g = int(a[1] + (b[1] - a[1]) * f)
    bl = int(a[2] + (b[2] - a[2]) * f)
    return f"rgba({r},{g},{bl},0.5)"


# ─── individual section renderers ──────────────────────────────────


def _render_findings(data: dict, lb: list[dict]) -> str:
    """A small panel above the first section summarizing the headline
    numbers. We intentionally keep it short — the detailed breakdowns
    are in the sections below."""
    if not lb:
        return ""
    top = lb[0]
    bullets = [
        f"<b>{esc(short(top['model']))}</b> leads at "
        f"<b>{pct(top['overall_turn_weighted'])}</b> (turn-weighted, "
        f"averaged equally across the 12 scenario × turn cells).",
        "Every model is weakest on turn 1 (clean-template opening markup) "
        "and strongest on later turns — see the per-turn breakdown below.",
        "best@k matches the turn-weighted overall here because each "
        "task has exactly 1 trial in this run; with multiple trials per "
        "task, best@k would diverge (max-reward-per-task aggregation).",
    ]
    if data.get("include_fable_5"):
        bullets.append(
            "Claude Fable 5's rubric performance and overall score "
            "come from the same Harbor pipeline as the other models "
            "— apples-to-apples on the leaderboard. The <code>.docx</code> "
            "outputs from that Harbor run were not preserved, so the "
            "behavioral metrics (surgicalness, edit verbosity) use docx "
            "files borrowed from an earlier experiment on the same "
            "benchmark. See <code>docs/REPORT-METRICS.md</code> §4 "
            "for details."
        )
    out = ['<div class="findings-panel"><div class="kicker">Headline</div><ul>']
    for b in bullets:
        out.append(f"<li>{b}</li>")
    out.append("</ul></div>")
    return "".join(out)


def _render_stat_cards(data: dict, lb: list[dict]) -> str:
    """A row of stat cards above the leaderboard."""
    if not lb:
        return ""
    top = lb[0]
    return (
        '<div class="stats-row">'
        f'<div class="stat-card"><div class="label">Tasks</div>'
        f'<div class="value">140</div>'
        f'<div class="detail">3 scenarios &middot; 4 turns</div></div>'
        f'<div class="stat-card"><div class="label">Trials</div>'
        f'<div class="value">{data["n_trials"]}</div>'
        f'<div class="detail">{data["n_models"]} model(s)</div></div>'
        f'<div class="stat-card hi"><div class="label">Top score</div>'
        f'<div class="value">{pct(top["overall_turn_weighted"])}</div>'
        f'<div class="detail">{esc(short(top["model"]))}, turn-weighted</div></div>'
        f'<div class="stat-card"><div class="label">Best@k top</div>'
        f'<div class="value">{pct(top["best_at_k_turn_weighted"])}</div>'
        f'<div class="detail">{esc(short(top["model"]))}, best-per-task</div></div>'
        "</div>"
    )


def _render_section_1_overall(lb: list[dict]) -> str:
    """Section 1: Overall score (turn-weighted) — leaderboard."""
    out = ['<div class="section"><h2>1. Overall score (turn-weighted)</h2>']
    out.append(
        '<p class="prose">Per-task scores are dedup-averaged within input '
        "groups, then per (scenario, turn) cell, then the 12 cells are "
        "averaged equally. This stops late turns (which have 4× as many "
        "input groups as turn 1) from dominating the headline. The 95% "
        "bootstrap CI is computed over the same 12-cell sample.</p>"
    )
    out.append('<div class="chart-card"><div class="sub">Turn-weighted overall score</div>')
    mx = max((r["overall_turn_weighted"] or 0) for r in lb) or 1.0
    for i, r in enumerate(lb):
        ci = r["ci"]
        out.append(
            bar_row(
                short(r["model"]),
                (r["overall_turn_weighted"] or 0) / mx,
                pct(r["overall_turn_weighted"]),
                C[i % len(C)],
                sub=f"95% CI {ci[0]:.2f}–{ci[1]:.2f}",
            )
        )
    out.append(
        '<div class="footnote">12-cell scenario × turn mean. CIs are '
        "wide because there are only 12 cells — interpret as cluster, "
        "not podium.</div></div></div>"
    )
    return "".join(out)


def _render_section_2_by_side(lb: list[dict]) -> str:
    """Section 2: Score by side (turn-weighted)."""
    out = ['<div class="section"><h2>2. Score by side (turn-weighted)</h2>']
    out.append(
        '<p class="prose">For each side, we compute per-turn averages '
        "within that side first, then average over turns. This isolates "
        "side asymmetry from the turn-distribution effect.</p>"
    )
    out.append(
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Model</th><th>Side A</th><th>Side B</th><th>A − B</th>"
        "</tr></thead><tbody>"
    )
    for i, r in enumerate(lb):
        a = r["by_side_turn_weighted"].get("A", 0)
        b = r["by_side_turn_weighted"].get("B", 0)
        gap = a - b
        hi = " class='win'" if i == 0 else ""
        out.append(
            f"<tr{hi}><td>{esc(short(r['model']))}</td>"
            f"<td class='mono'>{pct(a)}</td>"
            f"<td class='mono'>{pct(b)}</td>"
            f"<td class='mono'>{'+' if gap >= 0 else ''}{gap*100:.1f} pts</td></tr>"
        )
    out.append("</tbody></table></div></div>")
    return "".join(out)


def _render_section_3_by_scenario(lb: list[dict]) -> str:
    """Section 3: Score by scenario (turn-weighted)."""
    out = ['<div class="section"><h2>3. Score by scenario (turn-weighted)</h2>']
    out.append(
        '<p class="prose">For each scenario, per-turn averages within '
        "that scenario, then averaged over turns. Each scenario "
        "contributes one fully-turn-equal-weighted number.</p>"
    )
    # Find every scenario present (typically 1, 2, 3).
    scenarios: set[str] = set()
    for r in lb:
        scenarios.update(r["by_scenario_turn_weighted"].keys())
    scenario_order = sorted(scenarios, key=lambda s: int(s) if s.isdigit() else 99)

    out.append(
        '<div class="table-wrap"><table><thead><tr><th>Model</th>'
        + "".join(f"<th>Scenario {esc(s)}</th>" for s in scenario_order)
        + "</tr></thead><tbody>"
    )
    for i, r in enumerate(lb):
        hi = " class='win'" if i == 0 else ""
        cells = "".join(
            f"<td class='mono'>{pct(r['by_scenario_turn_weighted'].get(s))}</td>"
            for s in scenario_order
        )
        out.append(f"<tr{hi}><td>{esc(short(r['model']))}</td>{cells}</tr>")
    out.append("</tbody></table></div></div>")
    return "".join(out)


def _render_section_4_by_dimension(lb: list[dict]) -> str:
    """Section 4: Score by evaluation dimension (rubric category).

    This is the section the OLD repo called "Where the score lives" —
    same metric (pooled weighted pass rate per rubric category),
    just renamed per the user's spec.
    """
    # Discover the categories actually present.
    cats: list[str] = []
    for r in lb:
        for c in r["by_category"].keys():
            if c not in cats and c != "(uncategorized)":
                cats.append(c)
    # Canonical category display order.
    preferred = [
        "Legal correctness",
        "Negotiation Quality",
        "Commercial Context",
        "Counterparty Acceptance Prediction",
        "Deal-closing Orientation",
    ]
    ordered = [c for c in preferred if c in cats] + [c for c in cats if c not in preferred]

    out = ['<div class="section"><h2>4. Score by evaluation dimension</h2>']
    out.append(
        '<p class="prose">Rubric categories test different skills. '
        "Deal-closing rewards restraint; legal-correctness and "
        "negotiation-quality reward substantive judgment. Each cell is "
        'the model\'s pooled weighted pass rate (Σ(weight × PASS) / Σ|weight|) '
        "for rubrics in that category, across every trial. "
        "<em>Not</em> turn-weighted — this is the same formula the old "
        "report's &ldquo;Where the score lives&rdquo; section uses.</p>"
    )
    all_vals = [
        r["by_category"].get(c, 0) for r in lb for c in ordered
        if r["by_category"].get(c) is not None
    ]
    lo, hi = (min(all_vals), max(all_vals)) if all_vals else (0, 1)
    out.append('<div class="chart-card"><table class="heat-table"><thead><tr><th></th>')
    for c in ordered:
        short_label = (
            c.replace(" Orientation", "")
             .replace(" Prediction", "")
        )
        out.append(f"<th>{esc(short_label)}</th>")
    out.append("</tr></thead><tbody>")
    for r in lb:
        out.append(f"<tr><td class='heat-row-label'>{esc(short(r['model']))}</td>")
        for c in ordered:
            v = r["by_category"].get(c)
            label = "—" if v is None else f"{v:.2f}"
            out.append(
                f"<td class='heat-cell' style='background:{heat_color(v, lo, hi)}'>"
                f"{label}</td>"
            )
        out.append("</tr>")
    out.append(
        "</tbody></table>"
        '<div class="footnote">Weighted pass rate per rubric category, pooled per model. '
        "Heat ramp: red = low, green = high (per-table normalization).</div></div></div>"
    )
    return "".join(out)


def _render_section_5_by_turn(lb: list[dict]) -> str:
    """Section 5: Score breakdown by turn (per model) — one card per
    model with 4 horizontal bars showing turn 1 → 4 progression."""
    out = ['<div class="section"><h2>5. Score breakdown by turn</h2>']
    out.append(
        '<p class="prose">Per-turn group-score averages, pooled across '
        "scenarios. Turn 1 is universally the weakest turn (clean-template "
        "opening markup demands the most edits from scratch); turn 4 the "
        "strongest (deal-closing is closer to leaving things alone). Each "
        "model gets its own card so the turn-1 → turn-4 trajectory "
        "is visible at a glance.</p>"
    )
    turns = ["1", "2", "3", "4"]
    palette = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)"]
    out.append('<div class="turn-grid">')
    for i, r in enumerate(lb):
        color = palette[i % len(palette)]
        model_label = esc(short(r["model"]))
        overall = r.get("overall_turn_weighted") or 0.0
        out.append(
            '<div class="turn-card">'
            f'<div class="turn-card-head">'
            f'<span class="turn-card-name">{model_label}</span>'
            f'<span class="turn-card-overall">overall {overall*100:.1f}%</span>'
            f'</div>'
        )
        for t in turns:
            v = r["by_turn"].get(t) or 0.0
            w = max(0.0, min(1.0, v)) * 100
            out.append(
                '<div class="turn-bar-row">'
                f'<div class="turn-bar-label">Turn {t}</div>'
                f'<div class="bar-wrap"><div class="bar-fill" '
                f'style="width:{w:.1f}%;background:{color}"></div></div>'
                f'<div class="bar-num">{v*100:.1f}%</div>'
                '</div>'
            )
        out.append('</div>')
    out.append('</div>')
    out.append(
        '<div class="footnote">Each row is the mean of per-input-group '
        "scores within that turn, pooled across all scenarios. The card "
        "header shows the model's overall (12-cell turn-weighted) score "
        "for reference.</div></div>"
    )
    return "".join(out)


def _render_section_6_best_at_k(lb: list[dict]) -> str:
    """Section 6: best@k vs overall."""
    out = ['<div class="section"><h2>6. best@k</h2>']
    out.append(
        '<p class="prose">For each <code>(model, task)</code> pair, keep '
        "the highest-reward trial, then run the same turn-weighted "
        "12-cell aggregation. With one trial per task (as in this run) "
        "best@k equals the turn-weighted overall by construction. With "
        "multiple trials per task, best@k bounds how high the model "
        "could score if you ran it k times and kept the winner per task.</p>"
    )
    out.append(
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Model</th><th>Overall (turn-weighted)</th>"
        "<th>best@k (turn-weighted)</th><th>Δ</th>"
        "</tr></thead><tbody>"
    )
    for i, r in enumerate(lb):
        hi = " class='win'" if i == 0 else ""
        overall = r["overall_turn_weighted"] or 0
        best = r["best_at_k_turn_weighted"] or 0
        gap = best - overall
        out.append(
            f"<tr{hi}><td>{esc(short(r['model']))}</td>"
            f"<td class='mono'>{pct(overall)}</td>"
            f"<td class='mono'>{pct(best)}</td>"
            f"<td class='mono'>{'+' if gap >= 0 else ''}{gap*100:.1f} pts</td></tr>"
        )
    out.append("</tbody></table></div></div>")
    return "".join(out)


def _render_section_7_verbosity(verb: dict) -> str:
    """Section 7: Verbosity trap (turn 1)."""
    if not verb:
        return ""
    out = ['<div class="section"><h2>7. Verbosity trap (turn 1)</h2>']
    out.append(
        '<p class="prose">Each row reports per-model edit volume + clustering on '
        "turn-1 tasks (where every model receives the same clean-template "
        "input, so positional paragraph indices align across actors). "
        "<b>Mean redlines per task</b> counts contiguous runs of touched "
        "paragraphs (one logical change at the document level — a paired "
        "delete+insert at the same anchor is one redline; standalone "
        "deletes or inserts count one each). <b>Mean edits per touched "
        "paragraph</b> measures clustering. <b>Avg edit length</b> is "
        "the per-event mean size in characters (for a paired del+ins, "
        "the larger of the two halves) — pooled across every event the "
        "actor emitted, so it captures whether each edit is a "
        "word-tweak or a paragraph rewrite. <b>Paragraph overlap with "
        "expert</b> is the fraction of the model's touched paragraphs "
        "that the human attorney ALSO touched on the same source — a "
        "coarse alignment signal.</p>"
    )
    out.append(
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Actor</th><th>Mean redlines / task</th>"
        "<th>Mean edits / touched para</th>"
        "<th>Avg edit length (chars)</th>"
        "<th>Paragraph overlap with expert</th>"
        "</tr></thead><tbody>"
    )
    # Expert baseline first.
    exp = verb.get("expert") or {}
    out.append(
        "<tr class='archival'><td><b>Expert (human baseline)</b></td>"
        f"<td class='mono'>{fnum(exp.get('redlines_per_task'), 1)}</td>"
        f"<td class='mono'>{fnum(exp.get('edits_per_touched_para'))}</td>"
        f"<td class='mono'>{fnum(exp.get('avg_edit_length'), 1)}</td>"
        f"<td class='mono'>—</td></tr>"
    )
    for actor, stats in verb.items():
        if actor == "expert":
            continue
        out.append(
            f"<tr><td>{esc(short(actor))}</td>"
            f"<td class='mono'>{fnum(stats.get('redlines_per_task'), 1)}</td>"
            f"<td class='mono'>{fnum(stats.get('edits_per_touched_para'))}</td>"
            f"<td class='mono'>{fnum(stats.get('avg_edit_length'), 1)}</td>"
            f"<td class='mono'>{pct(stats.get('overlap_rate'))}</td></tr>"
        )
    out.append("</tbody></table></div></div>")
    return "".join(out)


def _render_section_8_surgicalness(surg: dict, threshold: float = 0.30) -> str:
    """Section 8: Surgicalness — a single composite "Surgicalness Score"
    per actor (sorted leaderboard) followed by the inline/block
    breakdown chart for context."""
    if not surg:
        return ""
    out = ['<div class="section"><h2>8. Surgicalness</h2>']
    pct_thr = int(round(threshold * 100))
    out.append(
        f'<p class="prose">Every tracked-change event is classified as '
        f"<b>inline</b> (touches less than {pct_thr}% of its containing "
        f"paragraph's unchanged baseline text) or <b>block</b> "
        f"(touches at least {pct_thr}% — typically a sentence or paragraph rewrite). "
        f"<b>Surgicalness score</b> = inline share — the fraction of "
        f"this actor's edits that are inline (small in-place tweaks "
        f"rather than paragraph rewrites). Higher = more surgical. "
        f"Pooled across all 140 tasks per actor.</p>"
    )

    # ─── Headline: single-number surgicalness score per actor ───
    INLINE_COLOR = "var(--c1)"  # blue/teal-ish — surgical
    BLOCK_COLOR = "var(--c3)"   # warmer — bigger rewrites
    SCORE_COLOR = "var(--c2)"   # green — the composite "score"

    # Order actors by score (highest = most surgical), expert first
    # regardless of rank since it's the reference baseline.
    expert_entry = surg.get("expert")
    model_entries = [(k, v) for k, v in surg.items() if k != "expert"]
    model_entries.sort(key=lambda kv: -(kv[1].get("surgicalness_score") or 0.0))

    out.append('<div class="chart-card">')
    out.append('<div class="sub">Surgicalness score (= inline share). Higher = more surgical.</div>')

    def _score_row(label: str, stats: dict, archival: bool = False) -> str:
        score = stats.get("surgicalness_score") or 0.0
        w = max(0.0, min(1.0, score)) * 100
        delta = ""
        if not archival and expert_entry:
            exp_score = expert_entry.get("surgicalness_score") or 0.0
            d = (score - exp_score) * 100
            sign = "+" if d > 0 else ""
            delta = (
                f'<span class="score-delta">{sign}{d:.1f}pp vs expert</span>'
            )
        cls = " score-row-expert" if archival else ""
        # Expert row uses neutral accent so it visually reads as the baseline.
        bar_color = "var(--accent)" if archival else SCORE_COLOR
        return (
            f'<div class="score-row{cls}">'
            f'<div class="score-label">{esc(label)}</div>'
            f'<div class="bar-wrap"><div class="bar-fill" '
            f'style="width:{w:.1f}%;background:{bar_color}"></div></div>'
            f'<div class="score-num">{score*100:.1f}%</div>'
            f'<div class="score-delta-wrap">{delta}</div>'
            f'</div>'
        )

    if expert_entry:
        out.append(_score_row("Expert (human baseline)", expert_entry, archival=True))
    for actor, stats in model_entries:
        out.append(_score_row(short(actor), stats))

    out.append('</div>')  # /chart-card (score strip)

    # ─── Detail chart: inline vs block breakdown (vertical bars) ───
    out.append(
        '<div class="chart-card">'
        '<div class="sub">Inline vs block share per actor</div>'
        '<div class="legend">'
        f'<span class="legend-swatch" style="background:{INLINE_COLOR}"></span> '
        f'Inline (&lt;{pct_thr}% of paragraph) '
        f'&nbsp;&nbsp;<span class="legend-swatch" style="background:{BLOCK_COLOR}"></span> '
        f'Block (&ge;{pct_thr}% of paragraph)</div>'
    )

    # Build the actor order: expert first, then the rest in their
    # natural dict order (which mirrors the leaderboard ordering).
    actors: list[tuple[str, dict, bool]] = []
    if "expert" in surg:
        actors.append(("Expert", surg["expert"], True))
    for k, v in surg.items():
        if k == "expert":
            continue
        actors.append((short(k), v, False))

    out.append('<div class="vbar-area">')
    # Y-axis ticks (0% / 25% / 50% / 75% / 100%). Rendered as horizontal
    # rules behind the bars via CSS gradients on the bar plot itself.
    out.append('<div class="vbar-yaxis">'
               '<span>100%</span><span>75%</span><span>50%</span>'
               '<span>25%</span><span>0%</span></div>')
    out.append('<div class="vbar-plot">')
    for label, stats, is_expert in actors:
        inline = stats.get("inline_share") or 0.0
        block = stats.get("block_share") or 0.0
        cls = " vbar-group-expert" if is_expert else ""
        out.append(
            f'<div class="vbar-group{cls}">'
            f'<div class="vbar-bars">'
            f'<div class="vbar">'
            f'<div class="vbar-num">{inline*100:.1f}%</div>'
            f'<div class="vbar-fill" style="height:{inline*100:.1f}%;background:{INLINE_COLOR}"></div>'
            f'<div class="vbar-tag">Inline</div>'
            f'</div>'
            f'<div class="vbar">'
            f'<div class="vbar-num">{block*100:.1f}%</div>'
            f'<div class="vbar-fill" style="height:{block*100:.1f}%;background:{BLOCK_COLOR}"></div>'
            f'<div class="vbar-tag">Block</div>'
            f'</div>'
            f'</div>'
            f'<div class="vbar-label">{esc(label)}</div>'
            f'</div>'
        )
    out.append('</div>')  # /vbar-plot
    out.append('</div>')  # /vbar-area
    out.append('</div></div>')  # /chart-card /section
    return "".join(out)


# ─── top-level builder ─────────────────────────────────────────────


def build(data: dict, logo: str) -> str:
    lb = data.get("leaderboard", [])

    out: list[str] = []
    A = out.append

    A("<!doctype html><html data-theme='dark' lang='en'><head><meta charset='utf-8'>")
    A("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    A("<title>RedlineBench — Reference Results</title>")
    A(
        "<script>document.documentElement.dataset.theme="
        "localStorage.getItem('m1-theme')||'dark';</script>"
    )
    A("<style>")
    A(THEME_CSS)
    A(COMPONENT_CSS)
    A("</style></head><body><div class='container'>")

    # header
    fable_sub = " (incl. Fable 5)" if data.get("include_fable_5") else ""
    judge_sub = (
        "3-judge panel (gpt-5.4-mini + claude-haiku + gemini-3.1-flash-lite, majority vote)"
        if data.get("judge_method") == "panel"
        else "GPT-5.5 single judge"
    )
    A(
        "<header class='header'><div class='brand'>"
        f"<span class='logo'>{logo}</span><div>"
        "<h1>RedlineBench &middot; Reference Results</h1>"
        f"<div class='sub'>Frontier-model contract redlining &mdash; 140 tasks, "
        f"{data.get('n_models', 0)} model(s){fable_sub} &middot; judged by {judge_sub}</div>"
        "</div></div><div class='right'><span class='meta'>trial 1</span>"
        "<button class='theme-toggle' aria-label='Toggle theme' onclick=\""
        "const r=document.documentElement;r.dataset.theme=r.dataset.theme==='dark'?'light':'dark';"
        "localStorage.setItem('m1-theme',r.dataset.theme);\">"
        "<span class='when-dark'>&#9728;</span><span class='when-light'>&#9790;</span></button></div></header>"
    )

    A(_render_findings(data, lb))
    A(_render_stat_cards(data, lb))
    A(_render_section_1_overall(lb))
    A(_render_section_2_by_side(lb))
    A(_render_section_3_by_scenario(lb))
    A(_render_section_4_by_dimension(lb))
    A(_render_section_5_by_turn(lb))
    A(_render_section_6_best_at_k(lb))
    A(_render_section_7_verbosity(data.get("verbosity_turn1", {})))
    A(_render_section_8_surgicalness(
        data.get("surgicalness", {}),
        threshold=float(data.get("surgicalness_threshold", 0.30)),
    ))

    A(
        "<footer class='footer'>RedlineBench reference run &middot; "
        f"{data.get('n_trials', 0)} trials, {data.get('n_models', 0)} model(s) &middot; "
        "weighted attorney-rubric pass rate, input-group averaged, "
        "12-cell turn-weighted overall &middot; "
        "see <code>docs/REPORT-METRICS.md</code> for formulas.</footer>"
    )
    A("</div></body></html>")
    return "\n".join(out)


THEME_CSS = """
:root,:root[data-theme="dark"]{color-scheme:dark;--bg:#050505;--panel:#111;--panel-2:#181818;--row-alt:#0d0d0d;--thead:#141414;--border:rgba(255,255,255,.10);--border-soft:rgba(255,255,255,.06);--text:#f6f6f6;--muted:rgba(255,255,255,.62);--dim:rgba(255,255,255,.38);--accent:#aab2ff;--accent-strong:#aab2ff;--accent-bg:rgba(170,178,255,.12);--accent-border:rgba(170,178,255,.30);--pass:#64e6c3;--pass-bg:rgba(100,230,195,.12);--pass-border:rgba(100,230,195,.30);--warn:#ffc86f;--warn-bg:rgba(255,200,111,.12);--warn-border:rgba(255,200,111,.30);--fail:#ee9eb6;--fail-bg:rgba(238,158,182,.12);--fail-border:rgba(238,158,182,.30);--shadow:none;--c1:#aab2ff;--c2:#64e6c3;--c3:#ffc86f;--c4:#ee9eb6;}
:root[data-theme="light"]{color-scheme:light;--bg:#f8fafc;--panel:#fff;--panel-2:#f1f5f9;--row-alt:#f8fafc;--thead:#f1f5f9;--border:#e5e7eb;--border-soft:#f1f5f9;--text:#0f172a;--muted:#64748b;--dim:#94a3b8;--accent:#2563eb;--accent-strong:#1d4ed8;--accent-bg:#eff6ff;--accent-border:#93c5fd;--pass:#047857;--pass-bg:#dcfce7;--pass-border:#86efac;--warn:#b45309;--warn-bg:#fef3c7;--warn-border:#fcd34d;--fail:#b91c1c;--fail-bg:#fee2e2;--fail-border:#fca5a5;--shadow:0 1px 2px rgba(15,23,42,.04);--c1:#3b82f6;--c2:#10b981;--c3:#f59e0b;--c4:#ec4899;}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:Outfit,system-ui,sans-serif;line-height:1.6;min-height:100vh;}
code,.mono{font-family:"JetBrains Mono",ui-monospace,monospace;}
"""

COMPONENT_CSS = """
.container{max-width:1180px;margin:0 auto;padding:1.5rem;}
.section{margin:1.5rem 0 2.5rem;}
.section>h2{padding-bottom:.4rem;border-bottom:1px solid var(--border);margin-bottom:.7rem;font-size:1.4rem;font-weight:600;}
h1{font-size:1.4rem;font-weight:600;letter-spacing:-.01em;}
p.prose{color:var(--text);opacity:.92;font-size:.95rem;line-height:1.7;max-width:82ch;margin-bottom:.85rem;}
code{background:var(--panel-2);border:1px solid var(--border);border-radius:3px;padding:.05rem .35rem;font-size:.85em;}
.header{display:flex;justify-content:space-between;align-items:flex-end;padding-bottom:1.25rem;border-bottom:1px solid var(--border);margin-bottom:1.5rem;gap:1rem;flex-wrap:wrap;}
.brand{display:flex;gap:.8rem;align-items:flex-end;}
.brand .logo{display:block;width:96px;height:22px;}
.brand .logo svg{width:100%;height:100%;}
:root[data-theme="light"] .brand .logo .m1-word{fill:var(--text);}
.header .sub{color:var(--muted);font-size:.85rem;margin-top:.2rem;}
.header .right{display:flex;align-items:center;gap:.6rem;}
.header .meta{color:var(--dim);font-size:.78rem;}
.theme-toggle{background:var(--panel-2);color:var(--muted);border:1px solid var(--border);border-radius:999px;width:34px;height:34px;cursor:pointer;font-size:.95rem;}
.theme-toggle:hover{color:var(--text);border-color:var(--accent-border);}
:root[data-theme="dark"] .when-light{display:none;}:root[data-theme="light"] .when-dark{display:none;}
.kicker{color:var(--accent);font-size:.7rem;text-transform:uppercase;letter-spacing:.07em;font-weight:600;margin-bottom:.4rem;}
.findings-panel{background:linear-gradient(180deg,var(--accent-bg),var(--panel));border:1px solid var(--accent-border);border-radius:10px;padding:1.25rem 1.5rem;margin-bottom:1.5rem;}
.findings-panel ul{margin-left:1.1rem;}
.findings-panel li{margin-bottom:.45rem;line-height:1.6;font-size:.9rem;}
.stats-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.85rem;margin:1.25rem 0 1.75rem;}
.stat-card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:.95rem;box-shadow:var(--shadow);}
.stat-card .label{color:var(--muted);font-size:.68rem;text-transform:uppercase;letter-spacing:.07em;}
.stat-card .value{font-size:1.5rem;font-weight:700;margin-top:.2rem;}
.stat-card .detail{color:var(--dim);font-size:.75rem;margin-top:.2rem;}
.stat-card.hi{background:var(--accent-bg);border-color:var(--accent-border);}
.stat-card.hi .value{color:var(--accent-strong);}
:root[data-theme="dark"] .stat-card.hi .value{background:linear-gradient(90deg,#aab2ff,#64e6c3);-webkit-background-clip:text;background-clip:text;color:transparent;}
.chart-card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:1.1rem;margin-bottom:1rem;box-shadow:var(--shadow);}
.chart-card .sub{color:var(--muted);font-size:.78rem;margin-bottom:.8rem;}
.chart-card .footnote{color:var(--dim);font-size:.72rem;margin-top:.55rem;font-style:italic;}
.bar-row{display:grid;grid-template-columns:200px 1fr 90px;gap:.75rem;align-items:center;padding:.4rem 0;border-bottom:1px solid var(--border-soft);}
.bar-label{font-size:.86rem;}
.bar-label .bar-sub{color:var(--dim);font-size:.7rem;font-family:"JetBrains Mono",monospace;}
.bar-wrap{background:var(--panel-2);height:14px;border-radius:4px;overflow:hidden;}
.bar-fill{height:100%;border-radius:4px;transition:width .4s ease;}
.bar-num{text-align:right;font-family:"JetBrains Mono",monospace;font-size:.82rem;font-weight:600;}
.legend{display:flex;align-items:center;gap:.4rem;font-size:.78rem;color:var(--muted);margin-bottom:.9rem;flex-wrap:wrap;}
.legend-swatch{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:middle;margin-right:.3rem;}

/* Section 5 — per-model turn cards (horizontal bars). */
.turn-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem;margin-top:.4rem;}
.turn-card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:.9rem 1rem;box-shadow:var(--shadow);}
.turn-card-head{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:.65rem;padding-bottom:.5rem;border-bottom:1px solid var(--border-soft);}
.turn-card-name{font-weight:600;font-size:.95rem;color:var(--text);}
.turn-card-overall{font-size:.74rem;color:var(--muted);font-family:"JetBrains Mono",monospace;}
.turn-bar-row{display:grid;grid-template-columns:60px 1fr 60px;gap:.6rem;align-items:center;padding:.25rem 0;}
.turn-bar-label{font-size:.78rem;color:var(--muted);font-family:"JetBrains Mono",monospace;}

/* Section 8 — Surgicalness Score leaderboard (single-number headline). */
.score-row{display:grid;grid-template-columns:200px 1fr 70px 140px;gap:.75rem;align-items:center;padding:.4rem 0;border-bottom:1px solid var(--border-soft);}
.score-row:last-child{border-bottom:none;}
.score-row.score-row-expert{background:var(--panel-2);border-radius:4px;padding:.45rem .55rem;border-bottom:1px dashed var(--border);font-style:italic;color:var(--muted);}
.score-label{font-size:.86rem;font-weight:500;}
.score-row-expert .score-label{font-weight:600;}
.score-num{text-align:right;font-family:"JetBrains Mono",monospace;font-size:.86rem;font-weight:600;}
.score-delta-wrap{text-align:right;font-size:.72rem;color:var(--dim);font-family:"JetBrains Mono",monospace;}
.score-delta{display:inline-block;}

/* Section 8 — vertical grouped bar chart for surgicalness. */
.vbar-area{display:flex;gap:.6rem;align-items:stretch;margin-top:.2rem;}
.vbar-yaxis{display:flex;flex-direction:column;justify-content:space-between;width:36px;font-size:.68rem;color:var(--dim);font-family:"JetBrains Mono",monospace;text-align:right;padding-bottom:34px;padding-top:18px;}
.vbar-plot{flex:1;display:flex;align-items:flex-end;justify-content:space-around;gap:.8rem;padding:18px 0 0;border-left:1px solid var(--border-soft);position:relative;}
.vbar-plot::before{content:"";position:absolute;left:0;right:0;top:18px;bottom:34px;background-image:linear-gradient(to top,transparent calc(25% - 1px),var(--border-soft) calc(25% - 1px),var(--border-soft) 25%,transparent 25%,transparent calc(50% - 1px),var(--border-soft) calc(50% - 1px),var(--border-soft) 50%,transparent 50%,transparent calc(75% - 1px),var(--border-soft) calc(75% - 1px),var(--border-soft) 75%,transparent 75%);pointer-events:none;}
.vbar-group{display:flex;flex-direction:column;align-items:center;flex:1;min-width:0;height:240px;padding:0 .35rem;border-radius:6px;position:relative;}
.vbar-group-expert{background:var(--panel-2);border:1px dashed var(--border);}
.vbar-bars{display:flex;align-items:flex-end;justify-content:center;flex:1;width:100%;gap:.45rem;padding-bottom:.3rem;}
.vbar{display:flex;flex-direction:column;align-items:center;justify-content:flex-end;flex:1;min-width:0;max-width:50px;height:100%;position:relative;}
.vbar-num{font-size:.7rem;font-family:"JetBrains Mono",monospace;color:var(--text);position:absolute;top:-2px;transform:translateY(-100%);white-space:nowrap;}
.vbar-fill{width:100%;border-radius:3px 3px 0 0;transition:height .4s ease;min-height:2px;}
.vbar-tag{position:absolute;bottom:-18px;font-size:.64rem;color:var(--dim);font-family:"JetBrains Mono",monospace;}
.vbar-label{margin-top:22px;font-size:.78rem;color:var(--text);text-align:center;font-weight:500;padding-top:.25rem;border-top:1px solid var(--border-soft);width:100%;}
.vbar-group-expert .vbar-label{color:var(--muted);font-style:italic;}
.turn-group{margin-bottom:1rem;}
.turn-model{font-size:.8rem;font-weight:600;color:var(--accent);margin:.6rem 0 .2rem;}
.table-wrap{background:var(--panel);border:1px solid var(--border);border-radius:8px;overflow-x:auto;box-shadow:var(--shadow);margin-bottom:1rem;}
table{width:100%;border-collapse:collapse;font-size:.85rem;}
th{background:var(--thead);padding:.5rem .7rem;text-align:left;border-bottom:1px solid var(--border);font-weight:600;}
td{padding:.45rem .7rem;border-bottom:1px solid var(--border-soft);}
tr:hover td{background:var(--row-alt);}
tr.win td{background:var(--accent-bg);color:var(--accent-strong);font-weight:600;}
tr.archival td{background:var(--panel-2);color:var(--muted);font-style:italic;border-top:1px dashed var(--border);}
.ast{color:var(--warn);font-weight:700;font-style:normal;}
.heat-table{border-collapse:separate;border-spacing:4px;font-size:.82rem;width:100%;}
.heat-table th{background:transparent;color:var(--muted);font-weight:500;text-align:center;border:0;font-size:.74rem;}
.heat-row-label{font-size:.82rem;font-weight:600;padding-right:.6rem;white-space:nowrap;}
.heat-cell{font-family:"JetBrains Mono",monospace;font-weight:600;text-align:center;padding:.5rem .65rem;border-radius:4px;min-width:56px;color:var(--text);}
:root[data-theme="light"] .heat-cell{color:#0f172a;}
.callout{border-radius:6px;padding:.85rem 1rem;margin:.75rem 0;font-size:.88rem;line-height:1.65;border:1px solid;}
.callout.note{background:var(--accent-bg);border-color:var(--accent-border);color:var(--text);}
.footer{color:var(--dim);font-size:.76rem;border-top:1px solid var(--border);padding-top:1rem;margin-top:2rem;line-height:1.7;}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="report/report_data.json")
    ap.add_argument("--logo", required=True)
    ap.add_argument("--out", default="report/index.html")
    args = ap.parse_args()
    data = json.loads(Path(args.data).read_text())
    logo = Path(args.logo).read_text()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(data, logo))
    print(f"wrote {out} ({out.stat().st_size//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
