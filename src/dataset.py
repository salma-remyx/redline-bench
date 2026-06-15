"""Resolve the RedlineBench benchmark directory.

The benchmark data (the `tasks/` tree) is NOT committed to this GitHub
repo — it lives on HuggingFace at `crosbylegal/RedlineBench` and is
downloaded on demand. Resolution precedence:

    1. a local `./benchmark/` directory (e.g. a manual clone), then
    2. the `$REDLINEBENCH_BENCHMARK_DIR` environment variable, then
    3. a HuggingFace snapshot download (cached by huggingface_hub).

The returned directory always has `tasks/` as a child.
"""

from __future__ import annotations

import os
from pathlib import Path

HF_REPO_ID = "crosbylegal/RedlineBench"
HF_REPO_TYPE = "dataset"
# Pin to a commit SHA or tag for byte-stable reproduction; "main" tracks
# the latest published revision.
HF_REVISION = "main"

_LOCAL_DIRNAME = "benchmark"
_ENV_VAR = "REDLINEBENCH_BENCHMARK_DIR"


def get_benchmark_dir() -> Path:
    """Return the benchmark root (a directory containing `tasks/`).

    Resolution order: local ./benchmark → $REDLINEBENCH_BENCHMARK_DIR →
    HuggingFace download.
    """
    local = Path(_LOCAL_DIRNAME)
    if local.is_dir():
        return local.resolve()

    env = os.environ.get(_ENV_VAR)
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(
                f"{_ENV_VAR}={p} does not exist or is not a directory."
            )
        return p

    return _download_from_hf()


def tasks_dir() -> Path:
    """Path to the `tasks/` tree inside the resolved benchmark root."""
    return get_benchmark_dir() / "tasks"


def _download_from_hf() -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "huggingface_hub is required to download the benchmark. "
            "Install it with `pip install huggingface_hub`, or point "
            f"${_ENV_VAR} at a local copy of the benchmark."
        ) from exc

    path = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        revision=HF_REVISION,
    )
    return Path(path)
