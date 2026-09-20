"""Tokenizer-only analytical compute, wall-time, and cost gate for Phase C."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

_worker_tokenizer = None


def initialize_encoder(source, root):
    global _worker_tokenizer
    _worker_tokenizer = h.load_tokenizer(source, Path(root))


def encode_length(text):
    return len(_worker_tokenizer.encode(text))


def document_lengths(source, root, texts, workers):
    """Ordered process map; workers use the unchanged tokenizer and roundtrip checks."""
    h.require(type(workers) is int and 1 <= workers <= 32, "invalid worker count")
    if workers == 1:
        tok = h.load_tokenizer(source, Path(root))
        return [len(tok.encode(text)) for text in texts]
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_encoder,
                             initargs=(source, str(root))) as pool:
        result = []
        for length in pool.map(encode_length, texts, chunksize=8):
            result.append(length)
            if len(result) % 1000 == 0:
                print(f"{source['tokenizer']}: {len(result)}/{len(texts)} documents", flush=True)
        return result

from benchmarks import run_phase_c_confirm as phase_c
from benchmarks import run_research_experiments as h


VERSION = 1
PRICE_OBSERVED_AT = "2026-09-16"
PRICE_SOURCE = "https://modal.com/pricing"
L4_PER_SECOND = 0.000222
CPU_CORE_PER_SECOND = 0.00003942
MEMORY_GIB_PER_SECOND = 0.00000222
CPU_CORES = 4
MEMORY_GIB = 16


def analytical_schedule(encoded, vocab=phase_c.VOCAB, context=128):
    return schedule_lengths([len(ids) for ids in encoded], vocab, context)


def schedule_lengths(lengths, vocab=phase_c.VOCAB, context=128):
    targets = squares = steps = 0
    for count in lengths:
        remaining = count + 1  # EOS is a causal target.
        while remaining:
            length = min(context, remaining)
            targets += length
            squares += length * length
            steps += 1
            remaining -= length
    return {"training_steps": steps, "training_target_tokens": targets,
            "training_sequence_length_squared_sum": squares,
            **h.flop_accounting(vocab, h.SCREEN, targets, squares)}


def hourly_cost():
    components = {
        "l4_gpu": 3600 * L4_PER_SECOND,
        "four_cpu_cores": 3600 * CPU_CORES * CPU_CORE_PER_SECOND,
        "sixteen_gib_memory": 3600 * MEMORY_GIB * MEMORY_GIB_PER_SECOND,
    }
    return components, sum(components.values())


def validate_record(row, name):
    h.require(row["tokenizer"] == name and row["vocab_budget"] == phase_c.VOCAB, "feasibility condition mismatch")
    h.require(row["documents"] == 42_897 and row["normalized_utf8_bytes"] == phase_c.EXPOSURE_BYTES,
              "feasibility exposure mismatch")
    h.require(row["training_steps"] > 0 and row["training_target_tokens"] > 0,
              "empty feasibility accounting")
    h.require(row["actual_analytical_flops"] ==
              row["core_analytical_flops"] + row["output_projection_flops"], "FLOP decomposition mismatch")
    h.digest(row)
    return row


def run(args):
    plan, phase_a, docs, _ = phase_c.prepare(args)
    phase_b = h.read_json(args.phase_b_ledger)
    output = args.output.resolve()
    execution_plan = {
        "schema_version": VERSION, "phase_c_runner_identity": plan["identity"],
        "provenance": plan["provenance"], "tokenizers": list(phase_c.NAMES),
        "normalized_utf8_bytes": phase_c.EXPOSURE_BYTES,
        "capacity_upper_bound_usd": args.available_workspace_capacity,
        "pricing_observed_at": PRICE_OBSERVED_AT, "pricing_source": PRICE_SOURCE,
        "encoding_workers": args.workers,
    }
    if output.exists():
        h.require(args.resume and output.is_dir(), "output exists; use --resume for exact feasibility provenance")
        h.require(h.read_json(output / "plan.json") == execution_plan, "feasibility resume provenance mismatch")
    else:
        h.require(not args.resume, "resume requires an existing feasibility directory")
        output.mkdir(parents=True)
        h.write_new_json_atomic(output / "plan.json", execution_plan)
    records = {}
    for index, name in enumerate(phase_c.NAMES):
        checkpoint = output / f"condition-{index:03d}.json"
        if checkpoint.exists():
            envelope = h.read_json(checkpoint)
            h.require(envelope.get("status") == "condition_complete", "incomplete feasibility checkpoint")
            records[name] = validate_record(envelope["record"], name)
            continue
        source = next(row for row in phase_a["records"]
                      if (row["tokenizer"], row["vocab_budget"]) == (name, phase_c.VOCAB))
        baseline = next(row for row in phase_b["records"]
                        if (row["tokenizer"], row["vocab_budget"], row["budget_regime"])
                        == (name, phase_c.VOCAB, "bytes"))
        started = time.monotonic()
        lengths = document_lengths(source, Path(args.phase_a).parent, docs["train"], args.workers)
        accounting = schedule_lengths(lengths)
        seconds_per_step = baseline["wall_clock_seconds"] / baseline["training_steps"]
        estimated_seconds = accounting["training_steps"] * seconds_per_step
        record = {
            "tokenizer": name, "vocab_budget": phase_c.VOCAB,
            "documents": len(lengths), "normalized_utf8_bytes": phase_c.EXPOSURE_BYTES,
            "encoded_tokens_excluding_eos": sum(lengths),
            "ordered_document_lengths_sha256": h.digest(lengths),
            "encoding_workers": args.workers,
            **accounting, "tokenizer_preflight_seconds": time.monotonic() - started,
            "phase_b_reference_wall_seconds": baseline["wall_clock_seconds"],
            "phase_b_reference_training_steps": baseline["training_steps"],
            "wall_time_estimator": "phase_b_same_architecture_l4_seconds_per_training_step_v1",
            "estimated_wall_seconds_per_seed": estimated_seconds,
            "estimated_wall_seconds_three_seeds": 3 * estimated_seconds,
        }
        validate_record(record, name)
        h.write_new_json_atomic(checkpoint, {"status": "condition_complete", "record": record})
        records[name] = record
        print(f"Completed feasibility {index + 1}/3: {name}", flush=True)
    ordered = [records[name] for name in phase_c.NAMES]
    components, per_hour = hourly_cost()
    total_seconds = sum(row["estimated_wall_seconds_three_seeds"] for row in ordered)
    estimated_cost = total_seconds / 3600 * per_hour
    body = {
        "phase_c_feasibility_schema_version": VERSION,
        "status": "BLOCKED_COST_AND_WALL_TIME" if estimated_cost > args.available_workspace_capacity else "FEASIBLE",
        "experiments_started": False, "test_split_opened": False,
        "conditions": 9, "records": ordered,
        "total_estimated_wall_seconds": total_seconds,
        "total_estimated_wall_hours": total_seconds / 3600,
        "pricing": {"observed_at": PRICE_OBSERVED_AT, "source": PRICE_SOURCE,
                    "modal_configuration": {"gpu": "L4", "cpu_cores": CPU_CORES, "memory_gib": MEMORY_GIB},
                    "hourly_components_usd": components, "total_per_hour_usd": per_hour},
        "estimated_modal_cost_usd": estimated_cost,
        "conservative_workspace_capacity_upper_bound_usd": args.available_workspace_capacity,
        "estimator_limitations": [
            "Wall time scales Phase B observed total condition time by training-step count.",
            "It includes Phase B tokenization and validation overhead proportionally and is an estimate, not a runtime result.",
            "Pricing is the public rate observed on the recorded date; launch requires a fresh dashboard capacity receipt.",
            "The capacity input is the last pre-Phase-B value, so current remaining capacity can only be lower.",
        ],
        "provenance": plan["provenance"], "runner_commit": plan["identity"]["commit_hash"],
        "runner_working_tree_dirty": plan["identity"]["working_tree_dirty"],
    }
    result = {**body, "content_sha256": h.digest(body)}
    receipt = output / "receipt.json"
    h.require(not receipt.exists(), "completed feasibility receipt cannot resume")
    h.write_new_json_atomic(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--available-workspace-capacity", type=float, required=True)
    # Reuse the runner's exact provenance CLI surface.
    parser.add_argument("--phase-a", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--exposure-manifest", type=Path, required=True)
    parser.add_argument("--exposure-manifest-sha256", required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--validation-receipt-sha256", required=True)
    parser.add_argument("--phase-b-ledger", type=Path, required=True)
    parser.add_argument("--phase-b-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
