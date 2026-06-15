"""Tests for the benchmark dataset resolver (src/dataset.py).

Covers the resolution precedence (local ./benchmark → env var → HF) without
ever hitting the network: the HF path is monkeypatched.
"""

import importlib

import pytest

dataset = importlib.import_module("dataset")


def test_local_benchmark_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "benchmark" / "tasks").mkdir(parents=True)
    monkeypatch.setenv(dataset._ENV_VAR, "/nonexistent/should/be/ignored")
    monkeypatch.setattr(dataset, "_download_from_hf",
                        lambda: pytest.fail("HF download must not run"))

    resolved = dataset.get_benchmark_dir()
    assert resolved == (tmp_path / "benchmark").resolve()
    assert dataset.tasks_dir() == (tmp_path / "benchmark").resolve() / "tasks"


def test_env_var_used_when_no_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)            # no ./benchmark here
    bench = tmp_path / "elsewhere"
    (bench / "tasks").mkdir(parents=True)
    monkeypatch.setenv(dataset._ENV_VAR, str(bench))
    monkeypatch.setattr(dataset, "_download_from_hf",
                        lambda: pytest.fail("HF download must not run"))

    assert dataset.get_benchmark_dir() == bench.resolve()


def test_env_var_missing_dir_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(dataset._ENV_VAR, str(tmp_path / "does-not-exist"))
    with pytest.raises(FileNotFoundError):
        dataset.get_benchmark_dir()


def test_falls_back_to_hf(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(dataset._ENV_VAR, raising=False)
    sentinel = tmp_path / "hf-cache"
    sentinel.mkdir()
    monkeypatch.setattr(dataset, "_download_from_hf", lambda: sentinel)

    assert dataset.get_benchmark_dir() == sentinel


def test_hf_constants():
    assert dataset.HF_REPO_ID == "crosbylegal/RedlineBench"
    assert dataset.HF_REPO_TYPE == "dataset"
