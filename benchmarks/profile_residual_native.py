"""Cold segmentation and full native batch matrix, with separate untimed Rust allocation counts."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import itertools
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from benchmarks import run_research_experiments as h
from benchmarks.analyze_tokenizer_failures import csv_text

SCRIPTS = {
    "latin": "counterproductive",
    "cjk": "\u4e2d\u6587\u6d4b\u8bd5\u7cfb\u7edf",
    "indic": "\u0928\u092e\u0938\u094d\u0924\u0947\u092d\u093e\u0930\u0924",
    "arabic": "\u0627\u0644\u0639\u0631\u0628\u064a\u0629",
    "dense_ascii": "a",
}
LENGTHS = {"short": 32, "medium": 256, "long": 4096}
BATCHES = (1, 8, 32, 128)
WARMUP, REPETITIONS = 3, 11


def load_native(path):
    name = f"_residual_{h.file_hash(path)[:16]}.uniqtoken_core"
    specification = importlib.util.spec_from_file_location(name, path)
    h.require(specification is not None and specification.loader is not None, "native extension loader unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def model_trie(native, model):
    payload = h.read_json(model)
    metadata = payload
    vocab = metadata["vocab"]
    ids = metadata["token_to_id"]
    trie = native.RustPrefixTrie(metadata.get("max_subword_len", 16))
    for token, score in vocab.items():
        trie.insert(token, score, ids[token])
    return trie, ids, vocab


def peak_rss():
    if os.name != "nt":
        import resource

        count = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(count if sys.platform == "darwin" else count * 1024)
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("peak", "current", "pp", "p", "pnp", "np", "pf", "ppf")
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    api = ctypes.WinDLL("psapi", use_last_error=True)
    api.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    api.GetProcessMemoryInfo.restype = wintypes.BOOL
    counts = Counters()
    counts.cb = ctypes.sizeof(counts)
    h.require(api.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counts), counts.cb), "RSS unavailable")
    return int(counts.peak)


def fixture(spec):
    size, word = LENGTHS[spec["length"]], SCRIPTS[spec["script"]]
    text = (word * math.ceil(size / len(word)))[:size]
    if spec["surface"] == "batch":
        # Long is one word exceeding the cache cutoff. Short/medium use ordinary
        # words plus punctuation, exercising the full regex/output pipeline.
        if spec["length"] != "long":
            text = ((word + " example, value! ") * math.ceil(size / (len(word) + 17)))[:size]
        return [text] * spec["batch_size"]
    return [text]


def span_stream(spans):
    return [(span.token, span.token_id, span.start, span.end) for span in spans]


def decode_pieces(tokens):
    pieces, pending = [], bytearray()
    for token in tokens:
        if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
            pending.append(int(token[3:5], 16))
        else:
            if pending:
                pieces.append(pending.decode("utf-8"))
                pending.clear()
            pieces.append(token)
    if pending:
        pieces.append(pending.decode("utf-8"))
    return "".join(pieces)


def validate_spans(text, stream, ids, scores):
    position, score = 0, 0.0
    for (start, end), grouped in itertools.groupby(stream, key=lambda span: span[2:]):
        group = list(grouped)
        h.require(start == position and start < end <= len(text), "span stream does not tile source characters")
        h.require(all(identifier == ids[token] for token, identifier, _, _ in group), "span token/ID mismatch")
        h.require(decode_pieces([row[0] for row in group]) == text[start:end], "span text mismatch")
        edge_score = 0.0
        for token, _, _, _ in group:
            edge_score += scores[token]
        score += edge_score
        position = end
    h.require(position == len(text), "incomplete source coverage")
    return score.hex()


def validate_errors(native, trie):
    observed = []
    for text, fallback, limit in (("\U0001f9ec", False, 1), ("x", True, 0)):
        try:
            native.rust_viterbi_decode(text, trie, fallback, limit)
        except Exception as error:
            observed.append((type(error).__name__, str(error)))
        else:
            observed.append(("success", None))
    for bad in ("single string", [1], ["x", None]):
        try:
            native.rust_encode_text_native_ids_batch(bad, trie)
        except Exception as error:
            observed.append((type(error).__name__, str(error)))
        else:
            observed.append(("success", None))
    return observed


def worker(spec):
    texts = fixture(spec)
    native = load_native(Path(spec["native"]))
    trie, ids, scores = model_trie(native, Path(spec["model"]))
    cold = spec["cache"] == "cold"
    if spec["surface"] == "segmentation":
        call = lambda: native.rust_viterbi_decode(texts[0], trie, True)
        project = span_stream
    elif spec["output"] == "ids":
        call = lambda: native.rust_encode_text_native_ids_batch(texts, trie)
        project = lambda result: result
    else:
        call = lambda: native.rust_encode_text_native_batch(texts, trie)
        project = lambda result: result
    expected = project(call())
    if spec["surface"] == "batch":
        tokens = native.rust_encode_text_native_batch(texts, trie)
        integer_rows = native.rust_encode_text_native_ids_batch(texts, trie)
        h.require([[ids[t] for t in row] for row in tokens] == integer_rows, "token/ID batch parity")
        singles = {}
        for text, row, integer_row in zip(texts, tokens, integer_rows):
            if text not in singles:
                singles[text] = (
                    native.rust_encode_text_native(text, trie),
                    native.rust_encode_text_native_ids(text, trie),
                )
            h.require(singles[text] == (row, integer_row), "single/batch token/ID parity")
            h.require(decode_pieces(row) == native.rust_normalize(text), "normalized decode mismatch")
        token_count = sum(map(len, integer_rows))
    else:
        token_count = len(expected)
        path_score = validate_spans(texts[0], expected, ids, scores)
    errors = validate_errors(native, trie)
    if cold:
        trie.clear_seg_cache()
    result = {
        **spec,
        "fixture_sha256": h.digest(texts),
        "output_sha256": h.digest(expected),
        "error_sha256": h.digest(errors),
        "normalized_input_bytes": sum(len(h.normalize(t).encode("utf-8")) for t in texts),
        "tokens": token_count,
        "native_sha256": h.file_hash(Path(spec["native"])),
        "model_sha256": h.file_hash(Path(spec["model"])),
        "process_peak_before_measurement_bytes": peak_rss(),
    }
    if spec["surface"] == "segmentation":
        result["path_score_float_hex"] = path_score
    if spec["mode"] == "allocations":
        if spec["surface"] == "segmentation":
            spans, counts = native.rust_allocation_profile_viterbi(texts[0], trie, True)
            h.require(span_stream(spans) == expected, "allocation replay span parity")
        else:
            strings, integers, counts = native.rust_allocation_profile_native_batch(
                texts, trie, spec["output"] == "ids"
            )
            h.require((integers if spec["output"] == "ids" else strings) == expected, "allocation replay batch parity")
        result["rust_allocations"] = dict(
            zip(("requests", "requested_bytes", "peak_live_bytes", "live_before_bytes", "live_after_bytes"), counts)
        )
        result["process_peak_rss_bytes"] = peak_rss()
        return result
    h.require(
        not hasattr(native, "rust_allocation_profile_viterbi"),
        "primary timings require default build without allocation instrumentation",
    )
    if spec["surface"] == "segmentation":
        diagnostic = native.rust_diagnostic_viterbi(texts[0], trie, True)
        result["baseline_stage_replay"] = {
            "owned_prefix_seconds": diagnostic[0],
            "dp_seconds": diagnostic[1],
            "edges": diagnostic[2],
            "states": diagnostic[3],
        }
    for _ in range(WARMUP):
        if cold:
            trie.clear_seg_cache()
        h.require(project(call()) == expected, "unstable native stream")
    iterations = max(1, 12000 // (LENGTHS[spec["length"]] * len(texts)))
    samples, cpu_samples, cpu_percent_samples = [], [], []
    for _ in range(REPETITIONS):
        elapsed = 0
        wall_start = time.perf_counter_ns()
        cpu_start = time.process_time_ns()
        for _ in range(iterations):
            if cold:
                trie.clear_seg_cache()
            started = time.perf_counter_ns()
            observed = call()
            del observed
            elapsed += time.perf_counter_ns() - started
        cpu_elapsed = time.process_time_ns() - cpu_start
        cpu_samples.append(cpu_elapsed / iterations)
        cpu_percent_samples.append(cpu_elapsed / (time.perf_counter_ns() - wall_start) * 100)
        samples.append(elapsed / iterations)
    median = statistics.median(samples)
    result.update(
        iterations=iterations,
        wall_ns_samples=samples,
        cpu_ns_samples=cpu_samples,
        latency_p50_ns=median,
        latency_p95_ns=sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
        normalized_mb_per_second=result["normalized_input_bytes"] * 1000 / median,
        tokens_per_second=token_count * 1e9 / median,
        process_cpu_percent=statistics.median(cpu_percent_samples),
        process_peak_rss_bytes=peak_rss(),
        cache_entries=trie.seg_cache_len(),
    )
    return result


def specifications(models):
    for budget, model in models.items():
        for length in LENGTHS:
            for script in SCRIPTS:
                yield {
                    "surface": "segmentation",
                    "vocab_budget": budget,
                    "model": str(model),
                    "length": length,
                    "script": script,
                    "batch_size": 1,
                    "output": "spans",
                    "cache": "cold",
                }
    for length in LENGTHS:
        for batch_size in BATCHES:
            for output in ("strings", "ids"):
                for cache in ("warm", "cold"):
                    yield {
                        "surface": "batch",
                        "vocab_budget": 8192,
                        "model": str(models[8192]),
                        "length": length,
                        "script": "latin",
                        "batch_size": batch_size,
                        "output": output,
                        "cache": cache,
                    }


def run(args):
    output = args.output
    h.require(not output.exists(), "output must be new")
    identity = h.runtime_identity()
    h.require(not identity["working_tree_dirty"], "commit profiler protocol before running")
    models = {budget: args.models / f"{budget}-baseline" / "tokenizer.json" for budget in (8192, 16384, 32768)}
    output.mkdir(parents=True)
    records = []
    for index, base in enumerate(specifications(models)):
        for mode, native in (("timing", args.native), ("allocations", args.allocation_native)):
            spec = {**base, "mode": mode, "native": str(native.resolve())}
            request, response = output / f"{index}-{mode}-request.json", output / f"{index}-{mode}.json"
            h.write_new_json(request, spec)
            command = [
                sys.executable,
                "-m",
                "benchmarks.profile_residual_native",
                "--worker",
                str(request),
                "--output",
                str(response),
            ]
            process = subprocess.run(
                command,
                capture_output=True,
                env={**os.environ, "RAYON_NUM_THREADS": "1", "PYTHONHASHSEED": "0"},
                timeout=300,
            )
            h.require(process.returncode == 0, process.stderr.decode("utf-8", errors="replace"))
            record = h.read_json(response)
            records.append(record)
            print(
                f"{index} {mode} {base['surface']} {base['vocab_budget']} {base['length']} {base['script']} batch={base['batch_size']}",
                flush=True,
            )
    h.require(h.runtime_identity() == identity, "profiler source/runtime changed")
    payload = {
        "schema_version": 1,
        "identity": identity,
        "build_source_commit": args.build_commit,
        "compiler": subprocess.check_output(["rustc", "--version", "--verbose"], text=True).strip(),
        "cargo": subprocess.check_output(["cargo", "--version"], text=True).strip(),
        "build_flags": {
            "profile": "release",
            "opt_level": 3,
            "lto": True,
            "codegen_units": 1,
            "timing_features": ["python", "c_abi"],
            "allocation_features": ["python", "c_abi", "allocation-profile"],
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "configuration": {"warmup": WARMUP, "repetitions": REPETITIONS, "rayon_threads": 1, "python_hash_seed": "0"},
        "allocation_method": "untimed opt-in Rust System allocator requests, realloc counts as one full-size request; excludes Python and external-library heaps",
        "memory_method": "fresh worker process high-water RSS, includes imports/model loading; Rust live requested bytes reported separately",
        "timing_method": "default release build, per-call wall including result materialization/disposal; cold cache clear excluded from wall; CPU samples include cache clears",
        "records": records,
    }
    h.write_new_json(output / "results.json", payload)
    (output / "metrics.csv").write_text(csv_text(records), encoding="utf-8")
    h.write_new_json(output / "manifest.json", {"status": "complete", "artifacts": h.artifact_hashes(output)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--allocation-native", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--build-commit")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        h.write_new_json(args.output, worker(h.read_json(args.worker)))
    else:
        run(args)


if __name__ == "__main__":
    main()
