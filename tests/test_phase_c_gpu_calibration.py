"""Guardrails for the non-experimental Phase C GPU calibration."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from tools import phase_c_gpu_calibration as calibration
from benchmarks import run_phase_c_confirm as phase_c
from benchmarks import run_research_experiments as h


def test_training_prefix_uses_ordered_complete_documents():
    texts = ["aa", "bbb", "cccc", "z"]
    assert calibration.training_prefix(texts, 6) == (["aa", "bbb"], 5)
    with pytest.raises(ValueError, match="no complete"):
        calibration.training_prefix(texts, 1)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_runtime_uses_harness_extension_signature_not_raw_binary_hash(monkeypatch, device):
    raw = "a" * 64
    signature = h.digest([raw])
    assert signature != raw
    monkeypatch.setattr(calibration, "EXPECTED_EXTENSION_SHA256", signature)
    monkeypatch.setattr(calibration, "extension_binary", lambda: "binary.so")
    monkeypatch.setattr(h, "file_hash", lambda _: raw)
    monkeypatch.setattr(calibration.platform, "python_version", lambda: "3.10.17")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    nvidia_calls = []
    def fake_nvidia(*args, **kwargs):
        nvidia_calls.append(args)
        return SimpleNamespace(stdout="NVIDIA L4, driver")
    monkeypatch.setattr(calibration.subprocess, "run", fake_nvidia)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="2.6.0+cu124", version=SimpleNamespace(cuda="12.4"),
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "NVIDIA L4"),
        get_num_threads=lambda: 4))
    fingerprint = calibration.runtime_fingerprint({"extension_hash": signature, "versions": {}}, device)
    assert fingerprint["extension_binary_sha256"] == raw
    assert bool(nvidia_calls) is (device == "cuda")
    assert fingerprint["gpu_name"] == ("NVIDIA L4" if device == "cuda" else None)


def test_projection_is_training_only_lower_bound_and_three_seed():
    records = [{"tokenizer": name, "training_steps": calibration.STEPS_PER_TOKENIZER,
                "sample_normalized_utf8_bytes": 1_000_000,
                "model_and_optimizer_init_seconds": 2.0,
                "tokenization_seconds": 1.0, "training_seconds": 256.0}
               for name in phase_c.NAMES]
    feasibility = {"records": [{"tokenizer": name, "training_steps": 1000}
                               for name in phase_c.NAMES]}
    result = calibration.extrapolate_cost(records, feasibility, 2.0)
    expected_seconds = len(phase_c.NAMES) * len(phase_c.SEEDS) * (2 + 300 + 1000)
    assert result["extrapolated_wall_seconds"] == expected_seconds
    assert result["extrapolated_cost_usd"] == pytest.approx(expected_seconds / 1800)


@pytest.mark.parametrize("bad_records", [[], [{"tokenizer": "sp_unigram"}],
                                          [{"tokenizer": name, "training_steps": 0,
                                            "sample_normalized_utf8_bytes": 1}
                                           for name in phase_c.NAMES]])
def test_projection_refuses_incomplete_or_uncapped_records(bad_records):
    feasibility = {"records": [{"tokenizer": name, "training_steps": 1000}
                               for name in phase_c.NAMES]}
    with pytest.raises((ValueError, KeyError)):
        calibration.extrapolate_cost(bad_records, feasibility, 1.0)


def test_run_uses_training_prefix_only_and_never_scores_validation(monkeypatch, tmp_path):
    plan = {"identity": {"commit_hash": "a" * 40, "working_tree_dirty": False},
            "provenance": {"source": "frozen"}}
    phase_a = {"records": [{"tokenizer": name, "vocab_budget": phase_c.VOCAB,
                            "artifact_hashes": {"model": name}} for name in phase_c.NAMES]}
    docs = {"train": ["a" * 600_000, "b" * 500_000], "validation": ["held out"]}
    monkeypatch.setattr(phase_c, "prepare", lambda args, execution: (plan, phase_a, docs, {}))
    monkeypatch.setattr(calibration, "runtime_fingerprint", lambda identity, device: {"identity": identity,
                                                                                    "device": device})
    monkeypatch.setattr(h, "load_tokenizer", lambda source, root: source["tokenizer"])
    seen = []

    def fake_measure(tok, texts, seed, device):
        seen.append((tok, texts, seed, device))
        return {"sample_documents": len(texts), "sample_normalized_utf8_bytes": 600_000,
                "training_steps": calibration.STEPS_PER_TOKENIZER,
                "model_and_optimizer_init_seconds": 1.0, "tokenization_seconds": 1.0,
                "training_seconds": 1.0, "measured_training_steps_per_second": 256.0}

    monkeypatch.setattr(calibration, "measure_condition", fake_measure)
    monkeypatch.setattr(h, "evaluate", lambda *args, **kwargs: pytest.fail("validation evaluated"))
    feasibility = {"conditions": 9, "experiments_started": False, "provenance": plan["provenance"],
                   "records": [{"tokenizer": name, "training_steps": 1000} for name in phase_c.NAMES]}
    feasibility["content_sha256"] = h.digest(feasibility)
    receipt_path = tmp_path / "feasibility.json"
    h.write_new_json_atomic(receipt_path, feasibility)
    args = SimpleNamespace(device="cuda", feasibility_receipt=receipt_path,
                           phase_a=tmp_path / "phase-a" / "ledger.json", hourly_rate=1.5,
                           output=tmp_path / "calibration.json")
    result = calibration.run(args)
    assert len(seen) == 3
    assert all(texts == ["a" * 600_000] and seed == 0 and device == "cuda"
               for _, texts, seed, device in seen)
    assert result["sample_normalized_utf8_bytes"] == 600_000
    assert result["projection"]["cost_with_margin_usd"] == 1.5 * result["projection"]["extrapolated_cost_usd"]
    assert result["validation_scored"] is False and result["test_split_opened"] is False
    assert h.read_json(args.output) == result
