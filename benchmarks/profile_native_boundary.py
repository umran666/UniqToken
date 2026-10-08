"""Issue #101: direct PyO3 stages plus separate-process default-build comparisons.

Only the embedded engineering fixtures are used. No research dataset is opened.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

from benchmarks.profile_boundary_overhead import (
    digest,
    python_memory,
    run_exact_parity_gate,
    source_sha256,
    timing_summary,
)
from benchmarks.profile_hot_paths import FIXTURES, ROOT, git_value, make_tokenizer, sha256, tool_version, workload_cases
from benchmarks.profile_residual_native import load_native, peak_rss

STAGES = ("input_utf8_access", "native_compute", "result_materialization", "input_release", "root_list_incref_decref")


def select_native(path):
    """Replace every loaded package alias before training or constructing tries."""
    native = load_native(path)
    for name, module in list(sys.modules.items()):
        if name.startswith("uniqtoken."):
            for alias in ("_native_core", "uniqtoken_core", "_uniqtoken_core"):
                if hasattr(module, alias):
                    setattr(module, alias, native)
    return native


def observed_calls(native, call):
    counts = Counter()
    with ExitStack() as stack:
        for name in dir(native):
            if name.startswith("rust_") and callable(getattr(native, name)):
                original = getattr(native, name)

                def wrapper(*args, _name=name, _original=original, **kwargs):
                    counts[_name] += 1
                    return _original(*args, **kwargs)

                stack.enter_context(patch.object(native, name, wrapper))
        value = call()
    return dict(counts), value


def api_calls(tok, texts, single):
    ids = tok.encode_to_ids_batch(texts)
    if single:
        return {
            "tokens": lambda: tok.encode(texts[0]),
            "ids": lambda: tok.encode_to_ids(texts[0]),
            "offsets": lambda: [(t.text, t.id, t.raw_span) for t in tok.encode_with_offsets(texts[0])],
            "decode": lambda: tok.decode(ids[0]),
        }
    return {
        "tokens": lambda: tok.encode_batch(texts),
        "ids": lambda: tok.encode_to_ids_batch(texts),
        "offsets": lambda: [[(t.text, t.id, t.raw_span) for t in row] for row in tok.encode_with_offsets_batch(texts)],
        "decode": lambda: tok.decode_batch(ids),
        "iterative_tokens": lambda: [tok.encode(t) for t in texts],
        "iterative_ids": lambda: [tok.encode_to_ids(t) for t in texts],
    }


def public_worker(native, tok, repetitions, warmup):
    rows = []
    for workload, texts in workload_cases():
        single = workload.endswith("_single")
        size = sum(len(tok.normalizer.normalize(t).encode("utf-8")) for t in texts)
        for api, call in api_calls(tok, texts, single).items():
            counts, expected = observed_calls(native, call)
            if api in ("tokens", "ids"):
                function = f"rust_encode_text_native{'_ids' if api == 'ids' else ''}{'' if single else '_batch'}"
                if counts != {function: 1}:
                    raise AssertionError(f"native fallback: {workload}/{api}: {counts}")
            if api.startswith("iterative_") and sum(counts.values()) != len(texts):
                raise AssertionError("iterative crossing count drift")
            if api == "decode" and counts:
                raise AssertionError("decode unexpectedly crossed FFI")
            for _ in range(warmup):
                call()
            samples = []
            for _ in range(repetitions):
                start = time.perf_counter_ns()
                result = call()
                samples.append((time.perf_counter_ns() - start) / 1e6)
                del result  # Destruction outside latency; no trial averaging.
            rows.append(
                {
                    "workload": workload,
                    "api": api,
                    "normalized_utf8_bytes": size,
                    "samples_ms": samples,
                    "native_calls": counts,
                    "output_sha256": digest(expected),
                    "python_heap": python_memory(call),
                    "process_peak_rss_bytes": peak_rss(),
                }
            )
    return rows


def stages_worker(native, tok, repetitions, warmup):
    if not hasattr(native, "rust_profile_boundary"):
        raise RuntimeError("build the diagnostic extension with --features allocation-profile")
    trie, kwargs = tok.model._get_rust_trie(), tok._native_pipeline_kwargs()
    rows = []
    for workload, texts in workload_cases():
        single = workload.endswith("_single")
        for ids in (False, True):
            expected = api_calls(tok, texts, single)["ids" if ids else "tokens"]()

            def call(reference, allocations=False):
                return native.rust_profile_boundary(
                    texts[0] if single else texts,
                    trie,
                    reference=reference,
                    output_ids=ids,
                    allocations=allocations,
                    single=single,
                    **kwargs,
                )

            for _ in range(warmup):
                call(True)
                call(False)
            samples = {True: [], False: []}
            for trial in range(repetitions):
                for reference in (True, False) if trial % 2 == 0 else (False, True):
                    output, ns, _, input_bytes, copied_bytes, input_copies = call(reference)
                    if output != expected:
                        raise AssertionError("native stage replay output drift")
                    samples[reference].append(ns)
            for reference in (True, False):
                output, _, counts, input_bytes, copied_bytes, input_copies = call(reference, allocations=True)
                if output != expected:
                    raise AssertionError("allocation replay output drift")
                rows.append(
                    {
                        "workload": workload,
                        "output": "ids" if ids else "tokens",
                        "reference": reference,
                        "samples_ns": samples[reference],
                        "rust_allocation_counts": dict(zip(STAGES, counts)),
                        "input_utf8_bytes": input_bytes,
                        "rust_input_bytes_copied": input_copies,
                        "intermediate_token_bytes_copied": copied_bytes,
                        "python_input_copied_bytes": None,
                        "python_heap_diagnostic": python_memory(lambda: call(reference)),
                        "process_peak_rss_bytes": peak_rss(),
                    }
                )
    return rows


def worker(args):
    native = select_native(args.native)
    if not args.stages and hasattr(native, "rust_allocation_profile_native_batch"):
        raise RuntimeError("end-to-end timings require a default build without allocator instrumentation")
    tok = make_tokenizer()
    return {
        "native_sha256": sha256(args.native),
        "model_sha256": digest(sorted((t, p, tok.model.token_to_id[t]) for t, p in tok.model.vocab.items())),
        "parity": run_exact_parity_gate(tok),
        "public_native_signatures": {
            # PyO3's existing Unicode char default is not parsed by Python 3.10
            # inspect.signature. Preserve and compare the actual binding text.
            name: getattr(native, name).__text_signature__
            for name in (
                "rust_encode_text_native",
                "rust_encode_text_native_batch",
                "rust_encode_text_native_ids",
                "rust_encode_text_native_ids_batch",
            )
        },
        "rows": (stages_worker if args.stages else public_worker)(native, tok, args.repetitions, args.warmup),
    }


def run_worker(path, args, stages=False):
    command = [
        sys.executable,
        "-m",
        "benchmarks.profile_native_boundary",
        "--worker",
        "--native",
        str(path),
        "--repetitions",
        str(args.repetitions),
        "--warmup",
        str(args.warmup),
        "--threads",
        str(args.threads),
    ]
    if stages:
        command.append("--stages")
    payload = json.loads(subprocess.check_output(command, cwd=ROOT))
    if args.raw_workers:
        with args.raw_workers.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"stages": stages, "payload": payload}) + "\n")
    return payload


def compare_runs(runs):
    reference = runs[0][1]
    for _, run in runs:
        if any(run[key] != reference[key] for key in ("model_sha256", "parity", "public_native_signatures")):
            raise AssertionError("before/after model or exact public API parity differs")
    records = {}
    for variant, run in runs:
        for row in run["rows"]:
            key = row["workload"], row["api"]
            record = records.setdefault(key, {"workload": key[0], "api": key[1], "variants": {}})
            previous = record["variants"].get(variant)
            if previous:
                if row["output_sha256"] != previous["output_sha256"] or row["native_calls"] != previous["native_calls"]:
                    raise AssertionError("replicate output or call-count drift")
                previous["samples_ms"].extend(row["samples_ms"])
                previous["replicate_python_heap"].append(row["python_heap"])
                previous["replicate_peak_rss_bytes"].append(row["process_peak_rss_bytes"])
            else:
                row = dict(row)
                row["samples_ms"] = list(row["samples_ms"])
                row["replicate_python_heap"] = [row.pop("python_heap")]
                row["replicate_peak_rss_bytes"] = [row.pop("process_peak_rss_bytes")]
                record["variants"][variant] = row
    for record in records.values():
        before, after = (record["variants"][variant] for variant in ("before", "after"))
        if before["output_sha256"] != after["output_sha256"] or before["native_calls"] != after["native_calls"]:
            raise AssertionError("public API output or crossing-count drift")
        for row in (before, after):
            row["latency"] = timing_summary(row.pop("samples_ms"))
            row["normalized_MB_per_s"] = row["normalized_utf8_bytes"] / row["latency"]["p50_ms"] / 1000
        record["speed_ratio"] = before["latency"]["p50_ms"] / after["latency"]["p50_ms"]
    return list(records.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stages", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--native", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--before-native", type=Path)
    parser.add_argument("--after-native", type=Path)
    parser.add_argument("--profile-native", type=Path)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-workers", type=Path, help="optional new local file retaining each completed worker")
    parser.add_argument("--repetitions", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.repetitions < 3 or args.warmup < 0 or args.threads < 1:
        parser.error("repetitions >= 3, warmup >= 0, threads >= 1 required")
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    logging.getLogger("uniqtoken").setLevel(logging.ERROR)
    if args.worker:
        print(json.dumps(worker(args)))
        return
    if not all((args.before_native, args.after_native, args.profile_native, args.baseline_commit, args.output)):
        parser.error("three binaries, baseline commit, and a new output directory are required")
    if args.output.exists():
        parser.error("output exists; evidence cannot be overwritten")
    if git_value("status", "--porcelain", "--untracked-files=no"):
        parser.error("commit tracked source before recording evidence")
    compiler_version = tool_version("rustc", "--version")
    if args.raw_workers:
        args.raw_workers.parent.mkdir(parents=True, exist_ok=True)
        with args.raw_workers.open("x", encoding="utf-8"):
            pass
    runs = []
    for variant in ("before", "after", "after", "before"):
        print(f"Measuring default release: {variant}", flush=True)
        runs.append((variant, run_worker(getattr(args, f"{variant}_native"), args)))
    print("Measuring opt-in native stages and allocations", flush=True)
    stages = run_worker(args.profile_native, args, stages=True)
    if stages["parity"] != runs[0][1]["parity"] or stages["model_sha256"] != runs[0][1]["model_sha256"]:
        raise AssertionError("instrumented build parity differs")
    metadata = {
        "schema_version": 1,
        "issue": 101,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_value("rev-parse", "HEAD"),
        "baseline_commit": git_value("rev-parse", args.baseline_commit),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "rustc": compiler_version,
        "build": "maturin develop --release; profiler adds --features allocation-profile; abi3-py39 unchanged",
        "threads": args.threads,
        "warmup": args.warmup,
        "repetitions_per_process": args.repetitions,
        "process_order": [v for v, _ in runs],
        "fixture_sha256": digest(FIXTURES),
        "model_sha256": stages["model_sha256"],
        "parity": stages["parity"],
        "public_native_signatures": runs[0][1]["public_native_signatures"],
        "native_sha256": {
            "before": runs[0][1]["native_sha256"],
            "after": runs[1][1]["native_sha256"],
            "profile": stages["native_sha256"],
        },
        "source_sha256": {
            name: source_sha256(name)
            for name in (
                "benchmarks/profile_native_boundary.py",
                "benchmarks/profile_hot_paths.py",
                "benchmarks/profile_residual_native.py",
                "benchmarks/profile_boundary_overhead.py",
                "crates/uniqtoken_core/src/pipeline.rs",
                "crates/uniqtoken_core/src/boundary_profile.rs",
                "crates/uniqtoken_core/src/lib.rs",
                "crates/uniqtoken_core/Cargo.toml",
                "crates/uniqtoken_core/Cargo.lock",
            )
        },
        "baseline_source_sha256": {
            name: source_sha256(name, args.baseline_commit)
            for name in (
                "crates/uniqtoken_core/src/pipeline.rs",
                "crates/uniqtoken_core/src/lib.rs",
                "crates/uniqtoken_core/Cargo.toml",
                "crates/uniqtoken_core/Cargo.lock",
            )
        },
    }
    records = compare_runs(runs)
    args.output.mkdir(parents=True)
    for filename, payload in (
        ("metadata.json", metadata),
        ("public_api.jsonl", records),
        ("native_stages.jsonl", stages["rows"]),
    ):
        content = (
            json.dumps(payload, indent=2) + "\n"
            if isinstance(payload, dict)
            else "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in payload)
        )
        (args.output / filename).write_bytes(content.encode())
    manifest = {
        name: sha256(args.output / name) for name in ("metadata.json", "public_api.jsonl", "native_stages.jsonl")
    }
    (args.output / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
    print(args.output)


if __name__ == "__main__":
    main()
