"""Guardrails for the non-experimental Phase C GPU calibration."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from tools import phase_c_gpu_calibration as calibration
from tools import phase_c_exposure as exposure_gate
from benchmarks import run_phase_c_confirm as phase_c
from benchmarks import run_phase_b_screen as phase_b
from benchmarks import run_research_experiments as h


def test_training_prefix_uses_ordered_complete_documents():
    texts = ["aa", "bbb", "cccc", "z"]
    assert calibration.training_prefix(texts, 6) == (["aa", "bbb"], 5)
    with pytest.raises(ValueError, match="no complete"):
        calibration.training_prefix(texts, 1)


def test_calibration_plan_reads_only_selected_prefix_not_phase_c_validation(monkeypatch, tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "train.jsonl").write_text("\n".join('{"text":"' + t + '"}' for t in "abc") + "\n")
    source = {"splits": {"train": {"path": "train.jsonl", "sha256": "hash"}}}
    h.write_new_json_atomic(source_dir / "manifest.json", source)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    exposure = {
        "status": "phase_c_exposure_frozen",
        "normalized_utf8_bytes": phase_c.EXPOSURE_BYTES,
        "test_split_opened": False,
        "ordered_documents": [
            {"train_row_index": i, "normalized_utf8_bytes": 1, "position": p} for p, i in enumerate((0, 2))
        ],
    }
    exposure["content_sha256"] = h.digest(exposure)
    h.write_new_json_atomic(frozen / "exposure.json", exposure)
    h.write_new_json_atomic(
        tmp_path / "receipt.json",
        {
            "status": "phase_c_exposure_frozen",
            "exposure_sha256": "hash",
            "confirmation_validation_sha256": "hash",
            "test_split_opened": False,
        },
    )
    h.write_new_json_atomic(frozen / "verification.json", {"status": "PASS", "test_split_opened": False})
    monkeypatch.setattr(calibration, "EXPECTED_SAMPLE_DOCUMENTS", 2)
    monkeypatch.setattr(phase_b, "verify_phase_a", lambda *a: ({"records": []}, None, None))
    monkeypatch.setattr(h, "runtime_identity", lambda: {"working_tree_dirty": False})
    monkeypatch.setattr(
        phase_b,
        "check_selection",
        lambda *a: {
            "phase_a_identity": {},
            "conditions": [{"tokenizer": name, "vocab_budget": phase_c.VOCAB} for name in phase_c.NAMES],
        },
    )
    monkeypatch.setattr(phase_b, "check_runtime", lambda *a: None)
    monkeypatch.setattr(phase_c, "require_hash", lambda *a: None)
    monkeypatch.setattr(h, "file_hash", lambda *a: "hash")
    monkeypatch.setattr(
        exposure_gate, "document", lambda row, index: {"train_row_index": index, "normalized_utf8_bytes": 1}
    )
    args = SimpleNamespace(
        phase_a=tmp_path / "phase-a",
        dataset=tmp_path / "dataset",
        selection=tmp_path / "selection",
        selection_sha256="hash",
        device="cuda",
        protocol=tmp_path / "protocol",
        phase_b_report=tmp_path / "report",
        phase_b_ledger=tmp_path / "ledger",
        source_manifest=source_dir / "manifest.json",
        exposure_manifest=frozen / "exposure.json",
        exposure_manifest_sha256="hash",
        validation_receipt=frozen / "confirmation-validation.json",
        validation_receipt_sha256="hash",
    )
    _, _, texts = calibration.calibration_plan(args)
    assert texts == ["a", "c"]


@pytest.mark.parametrize(
    "device,expected_gpu,gpu_name,memory",
    [
        ("cuda", "T4", "Tesla T4", 15109),
        ("cuda", "A100-40GB", "NVIDIA A100-SXM4-40GB", 40960),
        ("cpu", None, None, None),
    ],
)
def test_runtime_uses_harness_extension_signature_not_raw_binary_hash(
    monkeypatch, device, expected_gpu, gpu_name, memory
):
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
        return SimpleNamespace(stdout=f"{gpu_name}, 580.95.05, {memory} MiB")

    monkeypatch.setattr(calibration.subprocess, "run", fake_nvidia)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.6.0+cu124",
            version=SimpleNamespace(cuda="12.4"),
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: gpu_name),
            get_num_threads=lambda: 4,
        ),
    )
    fingerprint = calibration.runtime_fingerprint({"extension_hash": signature, "versions": {}}, device, expected_gpu)
    assert fingerprint["extension_binary_sha256"] == raw
    assert bool(nvidia_calls) is (device == "cuda")
    assert fingerprint["gpu_name"] == gpu_name
    assert fingerprint["gpu_memory_total_mib"] == memory


def test_runtime_rejects_wrong_gpu_or_memory(monkeypatch):
    raw = "a" * 64
    signature = h.digest([raw])
    monkeypatch.setattr(calibration, "EXPECTED_EXTENSION_SHA256", signature)
    monkeypatch.setattr(calibration, "extension_binary", lambda: "binary.so")
    monkeypatch.setattr(h, "file_hash", lambda _: raw)
    monkeypatch.setattr(calibration.platform, "python_version", lambda: "3.10.17")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(
        calibration.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="NVIDIA A100-SXM4-80GB, 580.95.05, 81251 MiB"),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.6.0+cu124",
            version=SimpleNamespace(cuda="12.4"),
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "NVIDIA A100-SXM4-80GB"),
            get_num_threads=lambda: 4,
        ),
    )
    with pytest.raises(ValueError, match="A100 40GB"):
        calibration.runtime_fingerprint({"extension_hash": signature, "versions": {}}, "cuda", "A100-40GB")


def test_projection_is_point_extrapolation_and_three_seed():
    records = [
        {
            "tokenizer": name,
            "training_steps": calibration.STEPS_PER_TOKENIZER,
            "sample_normalized_utf8_bytes": 1_000_000,
            "model_and_optimizer_init_seconds": 2.0,
            "tokenization_seconds": 1.0,
            "training_seconds": 256.0,
        }
        for name in phase_c.NAMES
    ]
    feasibility = {"records": [{"tokenizer": name, "training_steps": 1000} for name in phase_c.NAMES]}
    result = calibration.extrapolate_cost(records, feasibility, 2.0)
    expected_seconds = len(phase_c.NAMES) * len(phase_c.SEEDS) * (2 + 300 + 1000)
    assert result["extrapolated_wall_seconds"] == expected_seconds
    assert result["extrapolated_cost_usd"] == pytest.approx(expected_seconds / 1800)


@pytest.mark.parametrize(
    "bad_records",
    [
        [],
        [{"tokenizer": "sp_unigram"}],
        [{"tokenizer": name, "training_steps": 0, "sample_normalized_utf8_bytes": 1} for name in phase_c.NAMES],
    ],
)
def test_projection_refuses_incomplete_or_uncapped_records(bad_records):
    feasibility = {"records": [{"tokenizer": name, "training_steps": 1000} for name in phase_c.NAMES]}
    with pytest.raises((ValueError, KeyError)):
        calibration.extrapolate_cost(bad_records, feasibility, 1.0)


def test_run_uses_training_prefix_only_and_never_scores_validation(monkeypatch, tmp_path):
    plan = {"identity": {"commit_hash": "a" * 40, "working_tree_dirty": False}, "provenance": {"source": "frozen"}}
    phase_a = {
        "records": [
            {"tokenizer": name, "vocab_budget": phase_c.VOCAB, "artifact_hashes": {"model": name}}
            for name in phase_c.NAMES
        ]
    }
    docs = ["a" * 600_000, "b" * 500_000]
    monkeypatch.setattr(calibration, "calibration_plan", lambda args: (plan, phase_a, docs))
    monkeypatch.setattr(calibration, "EXPECTED_SAMPLE_DOCUMENTS", 1)
    monkeypatch.setattr(calibration, "EXPECTED_SAMPLE_BYTES", 600_000)
    monkeypatch.setattr(calibration, "EXPECTED_SAMPLE_HASH", h.digest([h.digest(docs[0])]))
    monkeypatch.setattr(
        calibration,
        "runtime_fingerprint",
        lambda identity, device, expected_gpu: {"identity": identity, "device": device, "expected_gpu": expected_gpu},
    )
    monkeypatch.setattr(h, "load_tokenizer", lambda source, root: source["tokenizer"])
    seen = []

    def fake_measure(tok, texts, seed, device, check_budget):
        check_budget()
        seen.append((tok, texts, seed, device))
        return {
            "sample_documents": len(texts),
            "sample_normalized_utf8_bytes": 600_000,
            "training_steps": calibration.STEPS_PER_TOKENIZER,
            "model_and_optimizer_init_seconds": 1.0,
            "tokenization_seconds": 1.0,
            "training_seconds": 1.0,
            "measured_training_steps_per_second": 256.0,
        }

    monkeypatch.setattr(calibration, "measure_condition", fake_measure)
    monkeypatch.setattr(h, "evaluate", lambda *args, **kwargs: pytest.fail("validation evaluated"))
    feasibility = {
        "conditions": 9,
        "experiments_started": False,
        "provenance": plan["provenance"],
        "records": [{"tokenizer": name, "training_steps": 1000} for name in phase_c.NAMES],
    }
    feasibility["content_sha256"] = h.digest(feasibility)
    receipt_path = tmp_path / "feasibility.json"
    h.write_new_json_atomic(receipt_path, feasibility)
    args = SimpleNamespace(
        device="cuda",
        expected_gpu="T4",
        feasibility_receipt=receipt_path,
        phase_a=tmp_path / "phase-a" / "ledger.json",
        hourly_rate=1.5,
        max_wall_seconds=480,
        max_compute_cost_usd=0.35,
        modal_hard_timeout_seconds=540,
        price_source="https://modal.com/pricing",
        price_observed_date="2026-09-20",
        output=tmp_path / "calibration.json",
    )
    result = calibration.run(args)
    assert len(seen) == 3
    assert all(texts == ["a" * 600_000] and seed == 0 and device == "cuda" for _, texts, seed, device in seen)
    assert result["sample_normalized_utf8_bytes"] == 600_000
    assert result["projection"]["cost_with_margin_usd"] == 1.5 * result["projection"]["extrapolated_cost_usd"]
    assert result["projection"]["scope"] == "training_and_tokenization_point_extrapolation"
    assert result["validation_scored"] is False and result["test_split_opened"] is False
    assert h.read_json(args.output) == result


def test_run_rejects_unsafe_compute_cap_before_data_access(monkeypatch):
    monkeypatch.setattr(calibration, "calibration_plan", lambda args: pytest.fail("data accessed"))
    args = SimpleNamespace(
        hourly_rate=1.0, max_wall_seconds=481, max_compute_cost_usd=0.35, modal_hard_timeout_seconds=540
    )
    with pytest.raises(ValueError, match="wall cap"):
        calibration.run(args)
