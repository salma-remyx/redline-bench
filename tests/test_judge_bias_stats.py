"""Integration + unit tests for the judge-panel agreement & bias statistics.

Exercises the wiring at ``panel.main()`` (a NON-NEW module): building fake
judge output trees, running the panel CLI end-to-end, and asserting the new
``judge_bias_stats`` block lands in ``panel_summary.json`` with a detectable
same-provider leniency effect. The unit tests pin the three statistics
ported from arXiv:2607.18828 — Fleiss' kappa, provider detection, and the
leniency-adjusted (judge-fixed-effect-netted) same-provider association with
its Monte-Carlo permutation test.
"""

import importlib
import json
import sys

panel = importlib.import_module("panel")
judge_bias_stats = importlib.import_module("judge_bias_stats")

provider_of = judge_bias_stats.provider_of
fleiss_kappa = judge_bias_stats.fleiss_kappa
same_provider_association = judge_bias_stats.same_provider_association


def _grade(rubrics):
    """Build a grade JSON like ``rejudge`` emits: per_rubric verdicts + weighted."""
    verdicts = [v for _, v in rubrics]
    weighted = 1.0 if verdicts and all(v == "PASS" for v in verdicts) else 0.0
    return {"score": {
        "weighted": weighted,
        "per_rubric": [{"rubric_id": r, "verdict": v, "weight": 1,
                        "category": "x"} for r, v in rubrics],
    }}


def _write_judge_tree(root, verdicts_by_mt):
    """``{(model, task): [(rid, verdict), ...]}`` -> <root>/<model>/<task>.json."""
    for (model, task), rubrics in verdicts_by_mt.items():
        d = root / model
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{task}.json").write_text(json.dumps(_grade(rubrics)))


# --- provider_of -----------------------------------------------------------

def test_provider_of_known_families():
    assert provider_of("gpt-5.5") == "openai"
    assert provider_of("claude-haiku-4-5") == "anthropic"
    assert provider_of("gemini-3.5-flash") == "google"
    assert provider_of("grok-4.3") == "xai"
    assert provider_of("llama-3.1") == "meta"
    assert provider_of("deepseek-v3") == "deepseek"
    assert provider_of("unknown-llm") is None


# --- fleiss_kappa ----------------------------------------------------------

def test_fleiss_perfect_agreement_is_one():
    # perfect agreement with both categories present -> kappa = 1.0
    assert fleiss_kappa([[1, 1, 1], [0, 0, 0]])["kappa"] == 1.0


def test_fleiss_degenerate_and_too_few():
    # every vote in one category -> chance agreement degenerate -> None
    assert fleiss_kappa([[1, 1, 1], [1, 1, 1]])["kappa"] is None
    # single item -> not enough to define multi-item agreement
    assert fleiss_kappa([[1, 0]])["kappa"] is None


def test_fleiss_is_bounded():
    res = fleiss_kappa([[1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0]])
    assert res["kappa"] is not None and -1.0 <= res["kappa"] <= 1.0


# --- same_provider_association ---------------------------------------------

def test_same_provider_strong_effect():
    # same-provider votes always PASS, cross-provider always FAIL.
    obs = []
    for judge in ["gpt-j", "claude-j"]:
        for model in ["gpt-model", "claude-model"]:
            same = provider_of(judge) == provider_of(model)
            verdict = "PASS" if same else "FAIL"
            for rid in ["r1", "r2"]:
                obs.append({"judge": judge, "model": model, "verdict": verdict})
    res = same_provider_association(obs, n_permutations=200, seed=0)
    assert res["coef"] > 0.9          # +~1.0 on the probability scale
    assert res["perm_p"] < 0.05       # the observed effect is not chance
    assert res["n_same_provider"] == 4
    assert res["ci95"] is not None and res["ci95"][0] > 0.0


def test_same_provider_balanced_null():
    # verdicts balanced identically for same- and cross-provider -> coef 0,
    # and (since |coef_obs| = 0) every permutation meets-or-exceeds it.
    obs = []
    for judge in ["gpt-j", "claude-j"]:
        for model in ["gpt-model", "claude-model"]:
            for verdict in ["PASS", "FAIL"]:
                obs.append({"judge": judge, "model": model, "verdict": verdict})
    res = same_provider_association(obs, n_permutations=200, seed=0)
    assert abs(res["coef"]) < 1e-6
    assert res["perm_p"] > 0.99


def test_same_provider_not_identifiable():
    # a single judge -> nothing to partial out.
    one = same_provider_association(
        [{"judge": "gpt-j", "model": "gpt-model", "verdict": "PASS"}])
    assert one["coef"] is None and "note" in one
    # two judges but no same-provider votes.
    none = same_provider_association([
        {"judge": "gpt-j", "model": "claude-model", "verdict": "PASS"},
        {"judge": "claude-j", "model": "gpt-model", "verdict": "FAIL"}])
    assert none["coef"] is None and "note" in none


# --- panel.main() end-to-end wiring ----------------------------------------

def test_panel_emits_bias_stats(tmp_path, monkeypatch):
    task = "redline-s1-t1-g01a"
    judges = ["gpt-5.4-mini", "claude-haiku-4-5", "gemini-3.5-flash"]
    models = ["gpt-5.5", "claude-opus-4-8"]
    rubrics = [("r1", None), ("r2", None)]

    trees = {}
    for label in judges:
        jp = provider_of(label)
        by_mt = {}
        for model in models:
            same = jp == provider_of(model)
            verdict = "PASS" if same else "FAIL"
            by_mt[(model, task)] = [(r, verdict) for r, _ in rubrics]
        root = tmp_path / label
        _write_judge_tree(root, by_mt)
        trees[label] = root

    out = tmp_path / "panel"
    monkeypatch.setattr(sys, "argv", [
        "panel",
        f"--judge={judges[0]}={trees[judges[0]]}",
        f"--judge={judges[1]}={trees[judges[1]]}",
        f"--judge={judges[2]}={trees[judges[2]]}",
        f"--out={out}",
    ])
    assert panel.main() == 0

    summary = json.loads((out / "panel_summary.json").read_text())
    assert "judge_bias_stats" in summary
    bias = summary["judge_bias_stats"]

    fk = bias["fleiss_kappa"]
    assert fk["n_raters"] == 3
    assert fk["kappa"] is None or -1.0 <= fk["kappa"] <= 1.0

    spa = bias["same_provider_association"]
    assert spa["coef"] is not None and spa["coef"] > 0.5
    assert spa["perm_p"] < 0.05
    assert spa["n_same_provider"] == 4
    assert spa["n_judges"] == 3 and spa["n_models"] == 2
