"""Independent post-process for non-experimental T4/A100 calibration evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


NAMES = ("sp_unigram", "boundary_bpe", "uniq_superbpe")
RATES = {"T4": 0.906912, "A100-40GB": 2.415312}
SAMPLE_HASH = "76c8f5a5701860c4291468f8bb145f168704bd93b474aaa08e4db4b7ec29f028"
EXTENSION_HASH = "b98f262df1c63e1b4fd0cfa38f5b673ce4affd8f8349bd6f642c13ef2c47db42"
EXPOSURE_HASH = "88c6661699025bb90c1a42b9b24adbc3b523b79282f3925e6a40653f7e4fa5f8"
SOURCE_HASH = "af11dc30fc96c683f92434d27c79fc45dd649dfb3787c6d9b02f81075aedd0c5"
PROTOCOL_HASH = "2aac9a7d6d80f1c26a8716316d2458b4261c9b6219c18a5b78b49a55f0d6f335"
FEASIBILITY_HASH = "813ad3300a10451ddf3c0097b4c05f8e6bcefb4df519c30abaee4cd79ae7866d"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_file(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def sha_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_immutable(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    body = dict(data)
    expected = body.pop("content_sha256")
    require(sha_object(body) == expected, f"content hash mismatch: {path}")
    return data


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7), f"{label} mismatch")


def verify_one(gpu, measurement_path, billing_path, feasibility, commit):
    measurement = read_immutable(measurement_path)
    billing = read_immutable(billing_path)
    require(
        measurement["status"] == "CALIBRATION_COMPLETE_COST_GATE_NOT_CLEARED"
        and measurement["purpose"] == "non_experimental_training_stack_calibration"
        and measurement["device"] == "cuda"
        and measurement["experiments_started"] is False
        and measurement["validation_scored"] is False
        and measurement["test_split_opened"] is False,
        "calibration scope violated",
    )
    require(
        measurement["runtime"]["identity"]["commit_hash"] == commit
        and measurement["runtime"]["identity"]["working_tree_dirty"] is False,
        "calibration commit or tree mismatch",
    )
    require(
        measurement["runtime"]["identity"]["extension_hash"] == EXTENSION_HASH
        and measurement["runtime"]["python"] == "3.10.17"
        and measurement["runtime"]["packages"]["torch"] == "2.6.0+cu124"
        and measurement["runtime"]["torch_cuda_version"] == "12.4",
        "runtime package or extension mismatch",
    )
    runtime = measurement["runtime"]
    gpu_name_ok = (
        "T4" in runtime["gpu_name"]
        if gpu == "T4"
        else ("A100" in runtime["gpu_name"] and "40GB" in runtime["gpu_name"])
    )
    require(
        runtime["expected_gpu"] == gpu
        and gpu_name_ok
        and runtime["environment"]["OMP_NUM_THREADS"] == "4"
        and runtime["environment"]["MKL_NUM_THREADS"] == "4"
        and runtime["environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8",
        "GPU or thread runtime mismatch",
    )
    memory = runtime["gpu_memory_total_mib"]
    require((14_000 <= memory <= 17_000) if gpu == "T4" else (38_000 <= memory <= 42_000), "GPU memory class mismatch")
    pins = measurement["provenance"]
    require(
        pins["exposure_manifest_sha256"] == EXPOSURE_HASH
        and pins["source_manifest_sha256"] == SOURCE_HASH
        and pins["phase_c_protocol_sha256"] == PROTOCOL_HASH
        and measurement["feasibility_receipt_sha256"] == FEASIBILITY_HASH,
        "frozen provenance changed",
    )
    require(
        measurement["sample_normalized_utf8_bytes"] == 999_753
        and measurement["sample_ordered_text_hash"] == SAMPLE_HASH
        and measurement["hard_step_cap_per_tokenizer"] == 256,
        "calibration slice or step cap changed",
    )
    limits = measurement["safety_limits"]
    require(
        limits == {"max_wall_seconds": 480.0, "max_compute_cost_usd": 0.35, "modal_hard_timeout_seconds": 540}
        and measurement["wall_seconds"] < limits["max_wall_seconds"],
        "safety controls not evidenced",
    )
    require(
        measurement["published_price_source"] == "https://modal.com/pricing"
        and measurement["published_price_observed_date"] == "2026-09-20"
        and measurement["hourly_rate_usd"] == RATES[gpu],
        "published price mismatch",
    )
    rows = measurement["records"]
    require([row["tokenizer"] for row in rows] == list(NAMES), "calibration cohort/order changed")
    by_name = {row["tokenizer"]: row for row in rows}
    for row in rows:
        require(
            row["sample_documents"] == 203
            and row["sample_normalized_utf8_bytes"] == 999_753
            and row["training_steps"] == 256
            and row["peak_gpu_memory_bytes"] > 0,
            "incomplete tokenizer probe",
        )
        require(
            abs(
                row["model_and_optimizer_init_seconds"]
                + row["tokenization_seconds"]
                + row["training_seconds"]
                - row["total_condition_seconds"]
            )
            < 0.02,
            "condition time mismatch",
        )
    require(
        [row["tokenizer"] for row in feasibility["records"]] == list(NAMES) and feasibility["conditions"] == 9,
        "feasibility cohort mismatch",
    )
    seconds = 0.0
    for condition in feasibility["records"]:
        row = by_name[condition["tokenizer"]]
        per_seed = (
            row["model_and_optimizer_init_seconds"]
            + row["tokenization_seconds"] * 300_000_000 / 999_753
            + row["training_seconds"] * condition["training_steps"] / 256
        )
        seconds += 3 * per_seed
    projection = measurement["projection"]
    close(projection["extrapolated_wall_seconds"], seconds, "projected wall seconds")
    close(projection["extrapolated_wall_hours"], seconds / 3600, "projected hours")
    close(projection["extrapolated_cost_usd"], seconds * RATES[gpu] / 3600, "projected cost")
    require(projection["overhead_margin_factor"] == 1.5, "50 percent allowance missing")
    close(projection["cost_with_margin_usd"], seconds * RATES[gpu] / 3600 * 1.5, "margin cost")
    close(projection["wall_with_margin_hours"], seconds / 3600 * 1.5, "margin hours")
    require(
        billing["measurement_sha256"] == sha_file(measurement_path)
        and billing["measurement_content_sha256"] == measurement["content_sha256"],
        "billing is not bound to measurement",
    )
    require(
        billing["modal_app_url"].startswith("https://modal.com/apps/")
        and billing["gpu"] == gpu
        and billing["app_metered_usage_usd"] >= 0,
        "missing Modal app charge",
    )
    require(
        billing["published_price_source"] == "https://modal.com/pricing"
        and billing["published_price_observed_date"] == "2026-09-20"
        and billing["published_hourly_rate_usd"] == RATES[gpu],
        "billing price mismatch",
    )
    before, after = billing["workspace_before"], billing["workspace_after"]
    for snapshot in (before, after):
        require(
            snapshot["hard_usage_limit_usd"] == 30.0
            and snapshot["post_credit_spend_limit_usd"] == 13.0
            and snapshot["charges_usd"] == 0.0,
            "workspace limit changed",
        )
        close(snapshot["remaining_to_hard_limit_usd"], 30.0 - snapshot["usage_usd"] + 0.0, "remaining budget")
    require(after["usage_usd"] >= before["usage_usd"], "usage moved backwards")
    remaining = after["remaining_to_hard_limit_usd"]
    decision = "PASS" if projection["cost_with_margin_usd"] <= remaining else "FAIL"
    return {
        "gpu": gpu,
        "status": "VERIFIED",
        "decision": decision,
        "measurement_file_sha256": sha_file(measurement_path),
        "billing_file_sha256": sha_file(billing_path),
        "sample_documents": 203,
        "sample_normalized_utf8_bytes": 999_753,
        "steps_per_tokenizer": 256,
        "measured_wall_seconds": measurement["wall_seconds"],
        "app_metered_usage_usd": billing["app_metered_usage_usd"],
        "projected_grid_hours": seconds / 3600,
        "projected_grid_cost_usd": seconds * RATES[gpu] / 3600,
        "projected_grid_hours_with_50pct_allowance": seconds / 3600 * 1.5,
        "projected_grid_cost_with_50pct_allowance_usd": seconds * RATES[gpu] / 3600 * 1.5,
        "verified_remaining_workspace_capacity_usd": remaining,
    }


def verify(t4, t4_billing, a100, a100_billing, feasibility_path, commit):
    require(sha_file(feasibility_path) == FEASIBILITY_HASH, "feasibility artifact changed")
    feasibility = read_immutable(feasibility_path)
    records = [
        verify_one("T4", t4, t4_billing, feasibility, commit),
        verify_one("A100-40GB", a100, a100_billing, feasibility, commit),
    ]
    return {
        "schema_version": 1,
        "status": "INDEPENDENT_VERIFICATION_COMPLETE",
        "phase_c_conditions_started": 0,
        "validation_scored": False,
        "test_split_opened": False,
        "launch_authorized": False,
        "calibration_commit": commit,
        "feasibility_receipt_sha256": FEASIBILITY_HASH,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("t4", "t4-billing", "a100", "a100-billing", "feasibility", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    result = verify(args.t4, args.t4_billing, args.a100, args.a100_billing, args.feasibility, args.commit)
    result["content_sha256"] = sha_object(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
