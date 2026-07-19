"""Smoke test: every flattened module imports cleanly.

Catches any missed `redlinebench.` → flat-module import rewrite. Modules
that need heavy/optional deps (litellm) are imported defensively.
"""

import importlib

import pytest

# Pure-stdlib / light modules — must import with no extra deps.
LIGHT = ["dataset", "aggregate", "panel", "runs_reader", "audit_reader"]

# Modules that pull in optional deps (lxml, litellm); skip if unavailable.
HEAVY = [
    "docx_metrics",
    "judging",
    "panel_reader",
    "metrics_summary",
    "rejudge",
    "reproduce",
]


@pytest.mark.parametrize("mod", LIGHT)
def test_light_imports(mod):
    importlib.import_module(mod)


@pytest.mark.parametrize("mod", HEAVY)
def test_heavy_imports(mod):
    try:
        importlib.import_module(mod)
    except ImportError as exc:
        pytest.skip(f"optional dependency missing for {mod}: {exc}")
