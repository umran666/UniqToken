"""Non-experimental Phase C stack calibration on a fixed training-only prefix."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
from itertools import islice
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

from benchmarks import run_phase_c_confirm as phase_c
from benchmarks import run_phase_b_screen as phase_b
from benchmarks import run_research_experiments as h
from tools import phase_c_exposure as exposure_gate


VERSION = 1
MAX_SLICE_BYTES = 1_000_000
STEPS_PER_TOKENIZER = 256
EXPECTED_EXTENSION_SHA256 = "b98f262df1c63e1b4fd0cfa38f5b673ce4affd8f8349bd6f642c13ef2c47db42"
EXPECTED_SAMPLE_DOCUMENTS = 203
EXPECTED_SAMPLE_BYTES = 999_753
EXPECTED_SAMPLE_HASH = "76c8f5a5701860c4291468f8bb145f168704bd93b474aaa08e4db4b7ec29f028"


def training_prefix(texts, max_bytes=MAX_SLICE_BYTES):
    h.require(type(max_bytes) is int and max_bytes > 0, "invalid calibration byte cap")
    selected, used = [], 0
    for text in texts:
        size = len(text.encode("utf-8"))
        if used + size > max_bytes:
            break
        selected.append(text)
        used += size
    h.require(bool(selected), "no complete training document fits calibration cap")
    return selected, used


def calibration_plan(args):
    """Verify frozen pins, then read only the prefix, never Phase C validation/test."""
    phase_a, _, _ = phase_b.verify_phase_a(args.phase_a, args.dataset)
    identity = h.runtime_identity()
    selection = phase_b.check_selection(args.selection, args.selection_sha256,
                                        phase_a, args.phase_a, identity)
    phase_b.check_runtime(identity, selection["phase_a_identity"], args.device)
    for path, expected, label in (
        (args.protocol, exposure_gate.PHASE_C_PROTOCOL_SHA256, "Phase C protocol"),
        (args.phase_b_report, exposure_gate.PHASE_B_REPORT_SHA256, "Phase B report"),
        (args.phase_b_ledger, exposure_gate.PHASE_B_LEDGER_SHA256, "Phase B ledger"),
        (args.source_manifest, exposure_gate.SOURCE_SHA256, "source manifest"),
        (args.exposure_manifest, args.exposure_manifest_sha256, "Phase C exposure"),
        (args.validation_receipt, args.validation_receipt_sha256, "confirmation validation receipt"),
    ):
        phase_c.require_hash(path, expected, label)
    exposure = h.read_json(args.exposure_manifest)
    body = dict(exposure)
    content = body.pop("content_sha256")
    h.require(h.digest(body) == content and exposure["status"] == "phase_c_exposure_frozen"
              and exposure["normalized_utf8_bytes"] == phase_c.EXPOSURE_BYTES
              and exposure["test_split_opened"] is False, "invalid frozen Phase C exposure")
    receipt_path = args.exposure_manifest.parent.parent / "receipt.json"
    receipt = h.read_json(receipt_path)
    h.require(receipt["status"] == "phase_c_exposure_frozen"
              and receipt["exposure_sha256"] == h.file_hash(args.exposure_manifest)
              and receipt["confirmation_validation_sha256"] == h.file_hash(args.validation_receipt)
              and receipt["test_split_opened"] is False, "invalid exposure/validation receipt")
    verification = h.read_json(args.exposure_manifest.with_name("verification.json"))
    h.require(verification["status"] == "PASS" and verification["test_split_opened"] is False,
              "independent exposure verification missing")
    selected = [row for row in selection["conditions"]
                if row["vocab_budget"] == phase_c.VOCAB and row["tokenizer"] in phase_c.NAMES]
    h.require([(row["tokenizer"], row["vocab_budget"]) for row in selected]
              == [(name, phase_c.VOCAB) for name in phase_c.NAMES], "frozen tokenizer cohort mismatch")
    source = h.read_json(args.source_manifest)
    source_root = args.source_manifest.resolve().parent
    train_path = (source_root / source["splits"]["train"]["path"]).resolve()
    h.require(train_path.is_relative_to(source_root), "training path escapes frozen source")
    h.require(h.file_hash(train_path) == source["splits"]["train"]["sha256"], "source training file changed")
    first = exposure["ordered_documents"][:EXPECTED_SAMPLE_DOCUMENTS]
    h.require(len(first) == EXPECTED_SAMPLE_DOCUMENTS, "calibration prefix missing")
    indices = [row["train_row_index"] for row in first]
    h.require(indices == sorted(set(indices)), "calibration source order changed")
    texts = []
    with train_path.open(encoding="utf-8") as stream:
        selected = iter(first)
        expected = next(selected)
        for index, line in enumerate(stream):
            if index != expected["train_row_index"]:
                continue
            row = json.loads(line)
            actual = exposure_gate.document(row, index)
            for key, value in expected.items():
                if key != "position":
                    h.require(actual[key] == value, "calibration source record changed")
            texts.append(h.normalize(row["text"]))
            expected = next(selected, None)
            if expected is None:
                break
    h.require(len(texts) == EXPECTED_SAMPLE_DOCUMENTS, "calibration prefix source records missing")
    pins = {
        "phase_c_protocol_sha256": exposure_gate.PHASE_C_PROTOCOL_SHA256,
        "phase_b_report_sha256": exposure_gate.PHASE_B_REPORT_SHA256,
        "phase_b_ledger_sha256": exposure_gate.PHASE_B_LEDGER_SHA256,
        "source_manifest_sha256": exposure_gate.SOURCE_SHA256,
        "exposure_manifest_sha256": args.exposure_manifest_sha256,
        "exposure_receipt_sha256": h.file_hash(receipt_path),
        "validation_receipt_sha256": args.validation_receipt_sha256,
    }
    return {"identity": identity, "provenance": pins}, phase_a, texts


def extrapolate_cost(records, feasibility, hourly_rate):
    """Short-slice timing extrapolation, not a mathematical bound."""
    h.require(math.isfinite(hourly_rate) and hourly_rate > 0, "invalid metered hourly rate")
    by_name = {row["tokenizer"]: row for row in records}
    h.require(set(by_name) == set(phase_c.NAMES), "incomplete calibration cohort")
    seconds = 0.0
    for condition in feasibility["records"]:
        row = by_name[condition["tokenizer"]]
        h.require(row["training_steps"] == STEPS_PER_TOKENIZER, "calibration step cap changed")
        h.require(row["sample_normalized_utf8_bytes"] > 0, "empty calibration sample")
        per_seed = (
            row["model_and_optimizer_init_seconds"]
            + row["tokenization_seconds"] * phase_c.EXPOSURE_BYTES / row["sample_normalized_utf8_bytes"]
            + row["training_seconds"] / row["training_steps"] * condition["training_steps"]
        )
        seconds += len(phase_c.SEEDS) * per_seed
    return {"extrapolated_wall_seconds": seconds,
            "extrapolated_wall_hours": seconds / 3600,
            "extrapolated_cost_usd": seconds / 3600 * hourly_rate}


def extension_binary():
    spec = importlib.util.find_spec("uniqtoken_core")
    h.require(spec is not None, "Rust extension unavailable")
    paths = [Path(spec.origin)] if spec.origin else []
    if spec.submodule_search_locations:
        paths += [p for root in spec.submodule_search_locations for p in Path(root).rglob("*")]
    binaries = [p for p in paths if p.suffix in (".so", ".pyd")]
    h.require(len(binaries) == 1, "expected one installed Rust extension binary")
    return binaries[0]


def runtime_fingerprint(identity, device, expected_gpu=None):
    import torch

    binary_hash = h.file_hash(extension_binary())
    h.require(h.digest([binary_hash]) == identity["extension_hash"], "installed Rust extension identity mismatch")
    h.require(identity["extension_hash"] == EXPECTED_EXTENSION_SHA256, "Phase A Rust extension hash changed")
    h.require(platform.python_version() == "3.10.17", "Phase A Python version changed")
    h.require(torch.__version__ == "2.6.0+cu124", "pinned PyTorch changed")
    h.require(torch.version.cuda == "12.4", "pinned PyTorch CUDA build changed")
    h.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "deterministic CUDA setting changed")
    h.require(device in ("cuda", "cpu"), "unsupported calibration device")
    if device == "cuda":
        h.require(expected_gpu in ("T4", "A100-40GB"), "explicit calibration GPU required")
        h.require(torch.cuda.is_available(), "CUDA unavailable")
        gpu_name = torch.cuda.get_device_name(0)
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        h.require(len(driver.splitlines()) == 1, "expected one calibration GPU")
        reported_name, driver_version, reported_memory = [part.strip() for part in driver.split(",")]
        memory_mib = int(reported_memory.split()[0])
        if expected_gpu == "T4":
            h.require("T4" in gpu_name and "T4" in reported_name and 14_000 <= memory_mib <= 17_000,
                      "calibration requires NVIDIA T4")
        else:
            h.require("A100" in gpu_name and "A100" in reported_name and 38_000 <= memory_mib <= 42_000,
                      "calibration requires NVIDIA A100 40GB")
    else:
        h.require(expected_gpu is None, "CPU calibration cannot declare a GPU")
        driver = gpu_name = None
        driver_version = memory_mib = None
    return {
        "identity": identity, "os": platform.platform(), "python": platform.python_version(),
        "packages": identity["versions"], "extension_binary_sha256": binary_hash,
        "device": device, "torch_cuda_version": torch.version.cuda, "gpu_name": gpu_name,
        "expected_gpu": expected_gpu, "nvidia_smi": driver, "driver_version": driver_version,
        "gpu_memory_total_mib": memory_mib, "torch_num_threads": torch.get_num_threads(),
        "environment": {key: os.environ.get(key) for key in (
            "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM")},
    }


def measure_condition(tok, texts, seed, device, check_budget=lambda: None):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    condition_started = time.perf_counter()
    check_budget()
    started = time.perf_counter()
    model = h.CausalMiniTransformer(len(tok.vocab), h.SCREEN, 128).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=h.SCREEN.lr, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.01)
    if device == "cuda":
        torch.cuda.synchronize()
    init_seconds = time.perf_counter() - started
    h.require(sum(p.numel() for p in model.parameters()) == h.parameter_count(len(tok.vocab), h.SCREEN, 128)[0],
              "calibration model differs from Phase C")

    started = time.perf_counter()
    encoded = []
    for text in texts:
        check_budget()
        encoded.append(tok.encode(text))
    encoded_tokens = sum(len(ids) for ids in encoded)
    tokenization_seconds = time.perf_counter() - started
    windows = islice((window for ids in encoded for window in h.windows(ids, 128)), STEPS_PER_TOKENIZER)
    completed = targets = 0
    started = time.perf_counter()
    for x, y in windows:
        check_budget()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(torch.tensor([x], device=device))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), torch.tensor(y, device=device))
        h.require(bool(torch.isfinite(loss)), "non-finite calibration computation")
        loss.backward()
        optimizer.step()
        completed += 1
        targets += len(y)
    if device == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    h.require(completed == STEPS_PER_TOKENIZER, "training-only slice has too few windows")
    return {
        "sample_documents": len(texts), "sample_normalized_utf8_bytes": sum(len(t.encode("utf-8")) for t in texts),
        "encoded_tokens_excluding_eos": encoded_tokens, "model_and_optimizer_init_seconds": init_seconds,
        "tokenization_seconds": tokenization_seconds, "training_seconds": training_seconds,
        "total_condition_seconds": time.perf_counter() - condition_started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "training_steps": completed, "training_target_tokens": targets,
        "measured_training_steps_per_second": completed / training_seconds,
    }


def run(args):
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    h.require(math.isfinite(args.hourly_rate) and args.hourly_rate > 0, "invalid price")
    h.require(math.isfinite(args.max_wall_seconds) and 0 < args.max_wall_seconds <= 480,
              "calibration wall cap must be at most 480 seconds")
    h.require(math.isfinite(args.max_compute_cost_usd) and 0 < args.max_compute_cost_usd <= 0.35,
              "calibration compute-cost cap must be at most $0.35")
    h.require(args.modal_hard_timeout_seconds <= 540 and args.max_wall_seconds < args.modal_hard_timeout_seconds,
              "invalid outer timeout")
    def check_budget():
        elapsed = time.perf_counter() - started
        h.require(elapsed < args.max_wall_seconds and elapsed * args.hourly_rate / 3600 < args.max_compute_cost_usd,
                  "calibration safety cap reached")

    check_budget()
    plan, phase_a, calibration_texts = calibration_plan(args)
    h.require(plan["identity"]["working_tree_dirty"] is False, "calibration requires a clean committed harness")
    h.require(args.device in ("cuda", "cpu"), "unsupported calibration device")
    fingerprint = runtime_fingerprint(plan["identity"], args.device, args.expected_gpu)
    feasibility = h.read_json(args.feasibility_receipt)
    body = dict(feasibility)
    expected = body.pop("content_sha256")
    h.require(h.digest(body) == expected, "feasibility receipt content changed")
    h.require(feasibility["provenance"] == plan["provenance"], "feasibility provenance changed")
    h.require(feasibility["conditions"] == 9 and feasibility["experiments_started"] is False,
              "invalid feasibility receipt")
    texts, sample_bytes = training_prefix(calibration_texts)
    sample_hash = h.digest([h.digest(t) for t in texts])
    h.require(len(texts) == EXPECTED_SAMPLE_DOCUMENTS and sample_bytes == EXPECTED_SAMPLE_BYTES
              and sample_hash == EXPECTED_SAMPLE_HASH,
              "calibration prefix differs from prior probes")
    records = []
    for name in phase_c.NAMES:
        source = next(row for row in phase_a["records"]
                      if (row["tokenizer"], row["vocab_budget"]) == (name, phase_c.VOCAB))
        tok = h.load_tokenizer(source, Path(args.phase_a).parent)
        row = {"tokenizer": name, "vocab_budget": phase_c.VOCAB,
               "tokenizer_artifact_hash": h.digest(source["artifact_hashes"]),
               **measure_condition(tok, texts, seed=0, device=args.device, check_budget=check_budget)}
        records.append(row)
        check_budget()
        print(f"Calibrated {name}: {row['measured_training_steps_per_second']:.2f} steps/s", flush=True)
    projection = extrapolate_cost(records, feasibility, args.hourly_rate)
    projection.update({
        "overhead_margin_factor": 1.5,
        "cost_with_margin_usd": 1.5 * projection["extrapolated_cost_usd"],
        "wall_with_margin_hours": 1.5 * projection["extrapolated_wall_hours"],
        "interpretation": "Point extrapolation from a small training slice; the margin is not a guaranteed upper bound.",
    })
    receipt = {
        "schema_version": VERSION, "status": "CALIBRATION_COMPLETE_COST_GATE_NOT_CLEARED",
        "purpose": "non_experimental_training_stack_calibration", "experiments_started": False,
        "device": args.device,
        "validation_scored": False, "test_split_opened": False,
        "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started, "hard_step_cap_per_tokenizer": STEPS_PER_TOKENIZER,
        "sample_max_normalized_utf8_bytes": MAX_SLICE_BYTES,
        "sample_normalized_utf8_bytes": sample_bytes, "sample_ordered_text_hash": sample_hash,
        "sample_scope": "first_complete_documents_of_frozen_phase_c_training_exposure",
        "safety_limits": {"max_wall_seconds": args.max_wall_seconds,
                          "max_compute_cost_usd": args.max_compute_cost_usd,
                          "modal_hard_timeout_seconds": args.modal_hard_timeout_seconds},
        "runtime": fingerprint, "provenance": plan["provenance"],
        "feasibility_receipt_sha256": h.file_hash(args.feasibility_receipt),
        "records": records, "hourly_rate_usd": args.hourly_rate,
        "published_price_source": args.price_source,
        "published_price_observed_date": args.price_observed_date,
        "projection": {**projection, "scope": "training_and_tokenization_point_extrapolation",
                       "unmeasured": "validation, orchestration, checkpointing, billing variance"},
    }
    receipt["content_sha256"] = h.digest(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    h.write_new_json_atomic(args.output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("phase-a", "dataset", "selection", "source-manifest", "exposure-manifest",
                 "validation-receipt", "phase-b-ledger", "phase-b-report", "protocol", "feasibility-receipt"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("selection-sha256", "exposure-manifest-sha256", "validation-receipt-sha256"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-gpu", choices=("T4", "A100-40GB"))
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--price-source", required=True)
    parser.add_argument("--price-observed-date", required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--max-compute-cost-usd", type=float, required=True)
    parser.add_argument("--modal-hard-timeout-seconds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
