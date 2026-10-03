"""Paired release-build comparison of current, prefix-only and compact Viterbi."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

from benchmarks import profile_residual_native as p
from benchmarks import run_research_experiments as h

ROUNDS = 2
TARGET_BLOCK_NS = 50_000_000
BOOTSTRAP_SEED = 9509699


def interval(ratios, repeats=5000):
    """Percentile bootstrap interval for the median paired throughput ratio."""
    h.require(ratios and all(math.isfinite(x) and x > 0 for x in ratios), "invalid paired ratios")
    rng = random.Random(BOOTSTRAP_SEED)
    medians = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(repeats))
    return [medians[math.floor(0.025 * repeats)], medians[math.ceil(0.975 * repeats) - 1]]


def synthetic_model(path):
    tokens = ["<pad>", "<unk>", "<s>", "</s>"]
    tokens += [f"<0x{byte:02X}>" for byte in range(256)]
    tokens += ["a" * length for length in range(1, 17)]
    h.write_new_json(
        path,
        {
            "vocab": {token: (-len(token) if token.startswith("a") else -10.0) for token in tokens},
            "token_to_id": {token: index for index, token in enumerate(tokens)},
            "max_subword_len": 16,
            "scope": "synthetic dense prefix stress fixture; not a trained vocabulary",
        },
    )


def specs(models, stress):
    yield from p.specifications(models)
    for length in (32, 256, 4096, 16384):
        yield {
            "surface": "segmentation",
            "vocab_budget": 276,
            "model": str(stress),
            "length": "stress",
            "characters": length,
            "script": "dense_ascii",
            "batch_size": 1,
            "output": "spans",
            "cache": "cold",
        }


def fixture(spec):
    return ["a" * spec["characters"]] if spec["length"] == "stress" else p.fixture(spec)


def paired_worker(spec):
    texts = fixture(spec)
    cold = spec["cache"] == "cold"
    calls, tries, streams, scores, errors = {}, {}, {}, {}, {}
    for name, binary in spec["variants"].items():
        native = p.load_native(Path(binary))
        h.require(not hasattr(native, "rust_allocation_profile_viterbi"), "timings must be uninstrumented")
        trie, ids, vocabulary = p.model_trie(native, Path(spec["model"]))
        tries[name] = trie
        if spec["surface"] == "segmentation":
            calls[name] = lambda native=native, trie=trie: native.rust_viterbi_decode(texts[0], trie, True)
            streams[name] = p.span_stream(calls[name]())
            scores[name] = p.validate_spans(texts[0], streams[name], ids, vocabulary)
            # Raw batch spans independently preserve every character offset.
            batched = native.rust_viterbi_decode_batch(texts, trie, True)
            h.require([p.span_stream(row) for row in batched] == [streams[name]], "raw single/batch span parity")
        else:
            api = (
                native.rust_encode_text_native_ids_batch
                if spec["output"] == "ids"
                else native.rust_encode_text_native_batch
            )
            calls[name] = lambda api=api, trie=trie: api(texts, trie)
            streams[name] = calls[name]()
            tokens = native.rust_encode_text_native_batch(texts, trie)
            integers = native.rust_encode_text_native_ids_batch(texts, trie)
            h.require([[ids[t] for t in row] for row in tokens] == integers, "string/ID parity")
            single = (
                native.rust_encode_text_native(texts[0], trie),
                native.rust_encode_text_native_ids(texts[0], trie),
            )
            h.require(all((row, integer) == single for row, integer in zip(tokens, integers)), "single/batch parity")
            h.require(
                all(p.decode_pieces(row) == native.rust_normalize(text) for row, text in zip(tokens, texts)),
                "decode parity",
            )
        errors[name] = p.validate_errors(native, trie)
    baseline = streams["baseline"]
    h.require(all(stream == baseline for stream in streams.values()), "cross-build stream mismatch")
    h.require(len(set(scores.values())) <= 1, "path score mismatch")
    h.require(all(error == errors["baseline"] for error in errors.values()), "error parity mismatch")
    for name, call in calls.items():
        for _ in range(p.WARMUP):
            if cold:
                tries[name].clear_seg_cache()
            call()
    calibration = []
    for name, call in calls.items():
        observations = []
        for _ in range(3):
            if cold:
                tries[name].clear_seg_cache()
            started = time.perf_counter_ns()
            observed = call()
            del observed
            observations.append(time.perf_counter_ns() - started)
        calibration.append(statistics.median(observations))
    iterations = min(80000, max(1, math.ceil(TARGET_BLOCK_NS / min(calibration))))
    wall = {name: [] for name in calls}
    cpu, phase_wall = {name: 0 for name in calls}, {name: 0 for name in calls}
    order = list(calls)
    for repetition in range(p.REPETITIONS):
        # Round and repetition alternate ordering to reduce temperature/drift bias.
        ordering = order if (repetition + spec["round"]) % 2 == 0 else list(reversed(order))
        for name in ordering:
            elapsed = 0
            phase_start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
            for _ in range(iterations):
                if cold:
                    tries[name].clear_seg_cache()
                started = time.perf_counter_ns()
                observed = calls[name]()
                del observed
                elapsed += time.perf_counter_ns() - started
            cpu[name] += time.process_time_ns() - cpu_start
            phase_wall[name] += time.perf_counter_ns() - phase_start
            wall[name].append(elapsed / iterations)
    normalized_bytes = sum(len(h.normalize(text).encode("utf-8")) for text in texts)
    tokens = len(baseline) if spec["surface"] == "segmentation" else sum(map(len, baseline))
    variants = {}
    for name in calls:
        samples = wall[name]
        median = statistics.median(samples)
        variants[name] = {
            "wall_ns_samples": samples,
            "latency_p50_ns": median,
            "latency_p95_ns": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
            "normalized_mb_per_second": normalized_bytes * 1000 / median,
            "tokens_per_second": tokens * 1e9 / median,
            "cpu_observation_wall_ns": phase_wall[name],
            "process_cpu_ns": cpu[name],
            "process_cpu_percent": cpu[name] / phase_wall[name] * 100,
            "native_sha256": h.file_hash(Path(spec["variants"][name])),
            "output_sha256": h.digest(streams[name]),
            "error_sha256": h.digest(errors[name]),
            "path_score_float_hex": scores.get(name),
        }
    comparisons = {}
    for name in calls:
        if name != "baseline":
            ratios = [a / b for a, b in zip(wall["baseline"], wall[name])]
            comparisons[name] = {
                "paired_ratios": ratios,
                "speedup_median": statistics.median(ratios),
                "bootstrap_95_interval": interval(ratios),
            }
    return {
        **spec,
        "iterations": iterations,
        "fixture_sha256": h.digest(texts),
        "model_sha256": h.file_hash(Path(spec["model"])),
        "normalized_input_bytes": normalized_bytes,
        "tokens": tokens,
        "measurements": variants,
        "comparisons": comparisons,
        "memory_note": "shared-process timing RSS is not attributed; isolated allocation workers report memory",
    }


def allocation_worker(spec):
    if spec["length"] != "stress":
        return p.worker({**spec, "mode": "allocations"})
    text = fixture(spec)[0]
    native = p.load_native(Path(spec["native"]))
    trie, ids, vocabulary = p.model_trie(native, Path(spec["model"]))
    before = p.peak_rss()
    spans, counts = native.rust_allocation_profile_viterbi(text, trie, True)
    stream = p.span_stream(spans)
    return {
        **spec,
        "mode": "allocations",
        "fixture_sha256": h.digest([text]),
        "model_sha256": h.file_hash(Path(spec["model"])),
        "native_sha256": h.file_hash(Path(spec["native"])),
        "output_sha256": h.digest(stream),
        "path_score_float_hex": p.validate_spans(text, stream, ids, vocabulary),
        "normalized_input_bytes": len(text),
        "tokens": len(spans),
        "process_peak_before_measurement_bytes": before,
        "process_peak_rss_bytes": p.peak_rss(),
        "rust_allocations": dict(
            zip(("requests", "requested_bytes", "peak_live_bytes", "live_before_bytes", "live_after_bytes"), counts)
        ),
    }


def run(args):
    output = args.output
    h.require(not output.exists(), "output must be new")
    identity = h.runtime_identity()
    h.require(not identity["working_tree_dirty"], "commit comparison protocol before measuring")
    output.mkdir(parents=True)
    stress = output / "dense-prefix-model.json"
    synthetic_model(stress)
    models = {budget: args.models / f"{budget}-baseline" / "tokenizer.json" for budget in (8192, 16384, 32768)}
    variants = {
        name: str(path.resolve())
        for name, path in (("baseline", args.baseline), ("prefix", args.prefix), ("compact", args.compact))
    }
    records, allocations = [], []
    for index, base in enumerate(specs(models, stress)):
        operations = [
            ("paired", {**base, "variants": variants, "round": round_number}) for round_number in range(ROUNDS)
        ]
        operations += [
            ("allocations", {**base, "variant": name, "native": str(native.resolve())})
            for name, native in (("baseline", args.baseline_allocation), ("compact", args.compact_allocation))
        ]
        for operation, spec in operations:
            suffix = str(spec["round"]) if operation == "paired" else spec["variant"]
            request, response = (
                output / f"{index}-{operation}-{suffix}-request.json",
                output / f"{index}-{operation}-{suffix}.json",
            )
            h.write_new_json(request, spec)
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.compare_residual_native",
                    "--worker",
                    str(request),
                    "--operation",
                    operation,
                    "--output",
                    str(response),
                ],
                capture_output=True,
                timeout=900,
                env={**os.environ, "RAYON_NUM_THREADS": "1", "PYTHONHASHSEED": "0"},
            )
            h.require(process.returncode == 0, process.stderr.decode("utf-8", errors="replace"))
            row = h.read_json(response)
            (records if operation == "paired" else allocations).append(row)
            print(
                f"{index} {operation} {suffix} {base['surface']} {base['vocab_budget']} {base['length']} batch={base['batch_size']}",
                flush=True,
            )
    h.require(h.runtime_identity() == identity, "comparison source/runtime changed")
    payload = {
        "schema_version": 1,
        "identity": identity,
        "records": records,
        "allocations": allocations,
        "build_sources": {
            "baseline": args.baseline_commit,
            "prefix": args.prefix_commit,
            "compact": args.compact_commit,
        },
        "compiler": subprocess.check_output(["rustc", "--version", "--verbose"], text=True).strip(),
        "hardware": {
            "platform": __import__("platform").platform(),
            "processor": __import__("platform").processor(),
            "logical_cpus": os.cpu_count(),
        },
        "configuration": {
            "rounds": ROUNDS,
            "repetitions": p.REPETITIONS,
            "warmup": p.WARMUP,
            "rayon_threads": 1,
            "target_block_ns": TARGET_BLOCK_NS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "build_flags": {
            "profile": "release",
            "opt_level": 3,
            "lto": True,
            "codegen_units": 1,
            "timing_features": ["python", "c_abi"],
            "allocation_features": ["python", "c_abi", "allocation-profile"],
        },
        "timing_method": "paired default releases, shared calibrated iteration count, alternating order, cold clear outside call wall; result disposal included",
        "cpu_method": "pooled process CPU/wall for eleven blocks per variant; includes cache clear and loop overhead; Windows clock remains quantized",
        "uncertainty_method": "95 percent percentile bootstrap of median paired ratios per independent fresh-worker round; local fixture uncertainty, no population inference",
        "allocation_method": "two isolated feature workers per cell; Rust System allocator request/live counters, Python and C++ heaps excluded",
        "memory_method": "fresh-worker high-water RSS including common imports/model/validation; temporary Rust live peak minus live before separately available",
    }
    h.write_new_json(output / "results.json", payload)
    h.write_new_json(output / "manifest.json", {"status": "complete", "artifacts": h.artifact_hashes(output)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline",
        "prefix",
        "compact",
        "baseline-allocation",
        "compact-allocation",
        "models",
        "worker",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=name == "output")
    for name in ("baseline-commit", "prefix-commit", "compact-commit", "operation"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if args.worker:
        specification = h.read_json(args.worker)
        result = paired_worker(specification) if args.operation == "paired" else allocation_worker(specification)
        h.write_new_json(args.output, result)
    else:
        run(args)


if __name__ == "__main__":
    main()
