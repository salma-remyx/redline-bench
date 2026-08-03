"""Integration tests for judge-time-compute scaling (judge_consensus).

Drives the consensus path through the REAL shared chokepoint
(``judging.call_judge`` -- a NON-NEW module) with ``litellm`` faked via
``sys.modules``, then asserts the per-rubric majority-vote consensus and
its agreement / contested diagnostic. Mirrors the ``test_judge_audit.py``
pattern of faking litellm and exercising ``judging.call_judge`` directly.
"""

import sys
import types

import judging
import judge_consensus


def _verdicts(rubric_id: str, verdict: str, justification: str = "ok") -> str:
    return (
        '{"verdicts": [{"rubric_id": "%s", "verdict": "%s", '
        '"justification": "%s"}]}' % (rubric_id, verdict, justification)
    )


def _install_litellm_queue(monkeypatch, payloads):
    """Fake ``litellm.completion`` to return successive payloads (or raise)."""
    calls = []
    fake = types.ModuleType("litellm")
    remaining = list(payloads)

    def _completion(**kw):
        calls.append(kw)
        assert remaining, "litellm.completion called more times than provisioned"
        item = remaining.pop(0)
        if isinstance(item, BaseException):
            raise item
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content=item))]
        )

    fake.completion = _completion
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return calls


class _HttpError(Exception):
    """Stand-in for a provider error carrying a status code."""

    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_consensus_majority_vote_and_contested_flag():
    # 3 samples: r1 seen in all three, r2 seen in only two (and split).
    samples = [
        [{"rubric_id": "r1", "verdict": "PASS", "justification": "p1"},
         {"rubric_id": "r2", "verdict": "PASS", "justification": "p2"}],
        [{"rubric_id": "r1", "verdict": "PASS", "justification": "p1b"},
         {"rubric_id": "r2", "verdict": "FAIL", "justification": "f2"}],
        [{"rubric_id": "r1", "verdict": "FAIL", "justification": "f1"}],
    ]
    out = judge_consensus.consensus_verdicts(samples)
    verdicts = {v["rubric_id"]: v["verdict"] for v in out["verdicts"]}
    assert verdicts == {"r1": "PASS", "r2": "FAIL"}  # r2 tie -> FAIL

    diag = {r["rubric_id"]: r for r in out["consensus"]["per_rubric"]}
    assert diag["r1"]["n_pass"] == 2
    assert diag["r1"]["agreement"] == 2 / 3
    assert diag["r1"]["contested"] is False      # 2/3 is not < 2/3
    assert diag["r2"]["contested"] is True       # 1/2 < 2/3
    assert out["consensus"]["n_contested"] == 1


def test_scale_judge_drives_real_chokepoint(monkeypatch):
    payloads = [
        _verdicts("r1", "PASS"),
        _verdicts("r1", "PASS"),
        _verdicts("r1", "FAIL"),
    ]
    calls = _install_litellm_queue(monkeypatch, payloads)

    resp = judge_consensus.scale_judge("m", "SYS", "USER", samples=3)

    assert len(calls) == 3                          # 3 real call_judge invocations
    assert resp["verdicts"][0]["verdict"] == "PASS"  # majority (2 PASS / 3)
    assert resp["consensus"]["n_samples"] == 3
    # Drop-in: the consensus verdicts feed the repo's own scorer unchanged.
    rubrics = [{"id": "r1", "weight": 5, "criteria": "c"}]
    score = judging.aggregate(resp["verdicts"], rubrics)
    assert score["weighted"] == 1.0 and score["n_pass"] == 1


def test_scale_judge_tolerates_one_failed_sample(monkeypatch):
    # A non-retriable 4xx makes call_judge raise immediately (no retry/sleep,
    # so the test stays fast). The consensus is built from the survivors.
    payloads = [
        _HttpError(400),
        _verdicts("r1", "PASS"),
        _verdicts("r1", "PASS"),
    ]
    _install_litellm_queue(monkeypatch, payloads)

    resp = judge_consensus.scale_judge("m", "SYS", "USER", samples=3)

    assert resp["consensus"]["n_failures"] == 1
    assert resp["consensus"]["n_samples"] == 2       # built from the 2 survivors
    assert resp["verdicts"][0]["verdict"] == "PASS"  # unanimous among survivors
