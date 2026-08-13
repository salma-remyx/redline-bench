#!/usr/bin/env python3
"""Disclose the inference backend, its version, and generation config.

Benchmark scores are reported as properties of a model, yet the inference
framework that produced them (HuggingFace transformers, vLLM, Ollama,
LiteLLM, ...) is treated as non-influential and almost never disclosed.
"What We Observe as LLM Behavior Can Be a Side-effect of Inference Backend"
(arXiv:2608.04714) measures this directly: in a fully-crossed study the
backend moves scores *even under greedy, sampling-noise-free decoding*,
and roughly 39% of the out-of-the-box variance a practitioner sees stems
from the backend / its default generation parameters. Its prescription is
to **disclose the backend, its version, and the full generation
configuration**, and to use deterministic decoding for cross-backend
comparison.

RedlineBench has two inference surfaces, and this module stamps the
provenance the paper asks for onto both:

  * **The judge path** runs every grading call through LiteLLM
    (``judging.call_judge``). That wrapper *is* the backend the paper
    warns about, so its name + installed version + the judge decoding
    config are the disclosure that belongs on the benchmark output and
    on each audited judge call.
  * **The model-under-test** runs out-of-process through the Harbor agent
    harness (``reproduce.py``). Its agent / model / environment are the
    backend-side facts visible to the scorer.

The decoder here is *not* pinned deterministic: the judge path leaves
``temperature`` unset on purpose because reasoning judge models reject a
pinned value, so each backend's default applies — precisely the
backend-dependence the paper measures. Recording that as an explicit,
machine-readable ``deterministic: False`` (rather than letting it stay
implicit) is the core of this module.
"""

from __future__ import annotations

import importlib.metadata

#: The inference framework behind every repo-level judge call. LiteLLM is
#: the wrapper that routes ``judging.call_judge`` to the provider backend,
#: so it is the backend whose name + version the paper asks to disclose.
JUDGE_BACKEND = "litellm"

#: Canonical decoding settings of the judge path (``judging.call_judge``).
#: ``None`` means *deliberately unset* and is recorded as such — an unset
#: sampling parameter inherits the backend's default, which is exactly the
#: backend-dependence arXiv:2608.04714 isolates. ``temperature`` is unset
#: because reasoning judge models reject a pinned value, so the judge path
#: is **not** deterministic.
JUDGE_GENERATION_CONFIG: dict[str, object] = {
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_tokens": None,
    "seed": None,
    "response_format": "json_object",
}


def backend_version() -> str | None:
    """Installed distribution version of the LiteLLM judge backend.

    Best-effort: ``None`` when LiteLLM is not installed (e.g. a test
    environment where it is faked via ``sys.modules``). Never imports the
    library at module load, so this module stays a light dependency-free
    import.
    """
    try:
        return importlib.metadata.version(JUDGE_BACKEND)
    except importlib.metadata.PackageNotFoundError:
        return None


def provider_of(model: str | None) -> str | None:
    """LiteLLM ``provider/model-id`` → provider.

    ``"anthropic/claude-opus-4-8"`` → ``"anthropic"``. Returns ``None``
    when the model string carries no provider prefix.
    """
    if not model or "/" not in model:
        return None
    return model.split("/", 1)[0]


def is_deterministic(generation_config: dict) -> bool:
    """Greedy / sampling-noise-free per arXiv:2608.04714: temperature
    pinned to ``0``.

    An *unset* (``None``) temperature is **not** deterministic — it
    inherits the backend's default, which is the backend-dependence the
    paper measures, so this returns ``False`` rather than guessing.
    """
    return generation_config.get("temperature") == 0


def judge_disclosure() -> dict:
    """Provenance record for the judge path: backend, version, decoding
    config, and the deterministic flag the paper asks practitioners to
    verify before comparing scores across runs or backends."""
    return {
        "backend": JUDGE_BACKEND,
        "backend_version": backend_version(),
        "generation_config": dict(JUDGE_GENERATION_CONFIG),
        "deterministic": is_deterministic(JUDGE_GENERATION_CONFIG),
        "note": (
            "Judge decoding is not pinned deterministic: temperature is "
            "unset because reasoning judge models reject it, so each "
            "backend's default applies and cross-run / cross-backend "
            "verdict variance is expected (arXiv:2608.04714)."
        ),
    }


def judge_call_backend() -> dict:
    """Per-call backend stamp for the audit trail: just the backend name +
    installed version observed at call time. The decoding config is uniform
    across judge calls (see ``JUDGE_GENERATION_CONFIG``) and is recorded
    once on the metrics summary rather than repeated on every row."""
    return {"backend": JUDGE_BACKEND, "backend_version": backend_version()}


def agent_harness_disclosure(
    agent: str | None = None,
    agent_model: str | None = None,
    harbor_env: str | None = None,
) -> dict | None:
    """Provenance record for the model-under-test, which runs out-of-process
    through the Harbor agent harness.

    Returns ``None`` when no agent context is available (e.g. the metrics
    summary built standalone from a pre-existing ``runs/`` tree, where the
    harness that produced the traces is no longer knowable) — recorded as
    absent rather than fabricated."""
    if not agent and not agent_model:
        return None
    return {
        "harness": "harbor",
        "agent": agent,
        "model": agent_model,
        "provider": provider_of(agent_model),
        "env": harbor_env,
        "note": (
            "The model-under-test runs out-of-process via the Harbor agent "
            "harness; its inference backend/version is not visible to the "
            "scorer, so capture it at run time when reproducibility matters "
            "(arXiv:2608.04714)."
        ),
    }


def benchmark_disclosure(
    *,
    judge_method: str,
    agent: str | None = None,
    agent_model: str | None = None,
    harbor_env: str | None = None,
) -> dict:
    """Top-level provenance block for the metrics summary JSON: the judge
    backend disclosure (always known) plus the agent-harness disclosure
    (known when built via ``reproduce``). Stamps the benchmark output with
    exactly the backend / version / generation-config disclosure
    arXiv:2608.04714 recommends."""
    return {
        "judge": judge_disclosure(),
        "judge_method": judge_method,
        "agent_harness": agent_harness_disclosure(agent, agent_model, harbor_env),
    }
