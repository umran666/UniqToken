"""Reproducible profiler for the current fused UniqToken encode and decode paths.

Run from the repository root after `maturin develop --release`:
    python -m benchmarks.profile_hot_paths --output benchmarks/profiles/issue95
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
import platform
import pstats
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "security_gate",
    "normalization",
    "pretokenization_regex",
    "unicode_grapheme_and_chunk_copy",
    "cached_trie_viterbi_byte_fallback",
    "token_and_id_copy",
)
FIXTURES = {
    "short": "The quick brown fox jumps over 42 lazy dogs.",
    "medium": "Tokenizer throughput depends on normalization, Unicode handling, and lattice search. " * 12,
    "long": "Reliable measurements use fixed inputs and repeated trials across tokenization paths. " * 55,
    "multilingual": "Cafe\u0301 \u0928\u092e\u0938\u094d\u0924\u0947 \u4e16\u754c \ud55c\uad6d\uc5b4 \U0001f469\u200d\U0001f4bb data 2026. "
    * 12,
    "source_code": "def encode_text(items: list[str]) -> list[int]:\n    return [len(item.encode('utf-8')) for item in items]\n"
    * 10,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def tool_version(*command: str) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def observed_range(values: list[float]) -> list[float]:
    return [min(values), max(values)]


def benchmark(call, iterations: int, warmup: int, repetitions: int) -> list[float]:
    for _ in range(warmup):
        for _ in range(iterations):
            call()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            call()
        samples.append((time.perf_counter_ns() - start) / iterations)
    return samples


def make_tokenizer():
    from uniqtoken.tokenizer import CustomTokenizer

    corpus = list(FIXTURES.values()) * 3
    return CustomTokenizer.train_from_corpus(corpus, target_vocab_size=400, min_frequency=1, verbose=False)


def workload_cases():
    return [
        (f"{name}_{mode}", [text] if mode == "single" else [f"{text} {i:02d}" for i in range(32)])
        for name, text in FIXTURES.items()
        for mode in ("single", "batch")
    ]


def reference_raw_spans(tok, text, tokens):
    """Align token text to normalized characters without using tokenizer offset APIs."""
    prepared, prepared_alignment = tok._prepare_text_with_alignment(
        text,
        allowed_special="none",
        disallowed_special_action="escape",
    )
    normalized, normalization_alignment = tok.normalizer.normalize_with_alignment(prepared)
    source_spans = tok._compose_alignment(normalization_alignment, prepared_alignment)
    spans = []
    pending = bytearray()
    pending_count = 0
    position = 0
    for token in tokens:
        if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
            pending.append(int(token[3:5], 16))
            pending_count += 1
            try:
                piece = pending.decode("utf-8")
            except UnicodeDecodeError as error:
                if error.reason == "unexpected end of data":
                    continue
                raise AssertionError("invalid byte fallback sequence") from error
            pending.clear()
            if len(piece) != 1:
                raise AssertionError("byte fallback must complete one normalized character")
            count = pending_count
            pending_count = 0
        else:
            if pending:
                raise AssertionError("incomplete byte fallback sequence")
            piece = token
            count = 1
        end = position + len(piece)
        if normalized[position:end] != piece:
            raise AssertionError("token stream does not tile normalized text")
        raw = source_spans[position:end]
        if not raw:
            raise AssertionError("token has no source span")
        spans.extend([(min(start for start, _ in raw), max(stop for _, stop in raw))] * count)
        position = end
    if pending or position != len(normalized):
        raise AssertionError("token stream does not cover normalized text")
    covered = {index for start, stop in spans for index in range(start, stop)}
    expected = {index for start, stop in prepared_alignment for index in range(start, stop)}
    if covered != expected:
        raise AssertionError("token spans do not cover prepared text")
    return spans


def validate(tok, native, cases):
    kwargs = tok._native_pipeline_kwargs()
    trie = tok.model._get_rust_trie()
    if kwargs is None or trie is None or not hasattr(native, "rust_profile_native_batch"):
        raise RuntimeError("release native profiler and fused tokenizer path are required")
    for name, texts in cases:
        expected_tokens = [tok.encode(text) for text in texts]
        expected_ids = [tok.encode_to_ids(text) for text in texts]
        offsets = [tok.encode_with_offsets(text) for text in texts]
        if [tok.encode(text) for text in texts] != expected_tokens:
            raise AssertionError(f"unstable token strings: {name}")
        if tok.encode_batch(texts) != expected_tokens or tok.encode_to_ids_batch(texts) != expected_ids:
            raise AssertionError(f"batch token/ID parity: {name}")
        for text, tokens, ids, spans in zip(texts, expected_tokens, expected_ids, offsets):
            if [(item.text, item.id) for item in spans] != list(zip(tokens, ids)):
                raise AssertionError(f"offset token/ID parity: {name}")
            if [item.raw_span for item in spans] != reference_raw_spans(tok, text, tokens):
                raise AssertionError(f"exact offset parity: {name}")
            if tok.decode(ids) != tok.decode_tokens(tokens):
                raise AssertionError(f"decode parity: {name}")
        if tok.decode_batch(expected_ids) != [tok.decode(ids) for ids in expected_ids]:
            raise AssertionError(f"batch decode parity: {name}")
        observed = native.rust_profile_native_batch(texts, trie, tok.model.byte_fallback, **kwargs)
        if [(tokens, ids) for tokens, ids, _ in observed] != list(zip(expected_tokens, expected_ids)):
            raise AssertionError(f"diagnostic native replay parity: {name}")
    return kwargs, trie


def profile(args):
    if args.threads < 1 or args.warmup < 0 or args.repetitions < 3 or args.iterations < 1:
        raise ValueError("threads and iterations must be positive; repetitions >= 3; warmup >= 0")
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    import uniqtoken_core as native

    tok = make_tokenizer()
    cases = workload_cases()
    kwargs, trie = validate(tok, native, cases)
    records = []
    for name, texts in cases:
        mode = name.rsplit("_", 1)[-1]
        encode = (lambda: tok.encode(texts[0])) if mode == "single" else (lambda: tok.encode_batch(texts))
        ids = tok.encode_to_ids_batch(texts)
        decode = (lambda: tok.decode(ids[0])) if mode == "single" else (lambda: tok.decode_batch(ids))
        diagnostic = lambda: native.rust_profile_native_batch(texts, trie, tok.model.byte_fallback, **kwargs)
        encode_ns = benchmark(encode, args.iterations, args.warmup, args.repetitions)
        decode_ns = benchmark(decode, args.iterations, args.warmup, args.repetitions)
        for _ in range(args.warmup * args.iterations):
            diagnostic()
        stage_samples = []
        for _ in range(args.repetitions):
            totals = [0] * len(STAGES)
            for _ in range(args.iterations):
                rows = diagnostic()
                for i in range(len(STAGES)):
                    totals[i] += sum(row[2][i] for row in rows)
            stage_samples.append([total / args.iterations for total in totals])
        normalized_bytes = sum(len(tok.normalizer.normalize(text).encode("utf-8")) for text in texts)
        token_count = sum(len(row) for row in ids)
        median_encode = statistics.median(encode_ns)
        records.append(
            {
                "workload": name,
                "mode": mode,
                "rows": len(texts),
                "normalized_input_bytes": normalized_bytes,
                "tokens": token_count,
                "encode_wall_ns": {
                    "median": median_encode,
                    "observed_min_max": observed_range(encode_ns),
                    "samples": encode_ns,
                },
                "decode_wall_ns": {
                    "median": statistics.median(decode_ns),
                    "observed_min_max": observed_range(decode_ns),
                    "samples": decode_ns,
                },
                "input_mb_per_s": normalized_bytes / median_encode * 1000,
                "tokens_per_s": token_count / median_encode * 1e9,
                "stage_elapsed_ns": {
                    stage: {
                        "median": statistics.median(sample[i] for sample in stage_samples),
                        "observed_min_max": observed_range([sample[i] for sample in stage_samples]),
                        "samples": [sample[i] for sample in stage_samples],
                    }
                    for i, stage in enumerate(STAGES)
                },
            }
        )
    profiler = cProfile.Profile()
    profiler.enable()
    for _, texts in cases:
        if len(texts) == 1:
            tok.encode(texts[0])
            tok.decode(tok.encode_to_ids(texts[0]))
        else:
            tok.encode_batch(texts)
            tok.decode_batch(tok.encode_to_ids_batch(texts))
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats()
    return records, stream.getvalue(), tok


def write_results(args, records, callstack, tok):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    commit = git_value("rev-parse", "HEAD")
    dirty = bool(git_value("status", "--porcelain", "--untracked-files=no"))
    vocab = sorted((token, score, tok.model.token_to_id[token]) for token, score in tok.model.vocab.items())
    vocab_hash = hashlib.sha256(json.dumps(vocab, ensure_ascii=True).encode()).hexdigest()
    fixture_hash = hashlib.sha256(json.dumps(workload_cases(), ensure_ascii=True).encode()).hexdigest()
    metadata = {
        "schema_version": 2,
        "issue": 95,
        "parity_gate": "passed_before_measurements",
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "build_mode": args.build_mode,
        "python": sys.version,
        "rustc": tool_version("rustc", "--version"),
        "cargo": tool_version("cargo", "--version"),
        "maturin": tool_version("maturin", "--version"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "rayon_threads": args.threads,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "iterations_per_repetition": args.iterations,
        "batch_rows": 32,
        "native_module": Path(native_file()).name,
        "native_module_sha256": sha256(Path(native_file())),
        "fixture_sha256": fixture_hash,
        "vocabulary_sha256": vocab_hash,
        "timing_note": "Stage values are diagnostic replay elapsed nanoseconds summed across rows; end-to-end encode/decode values are public API wall nanoseconds. Bounds are observed minimum and maximum over repetitions, not confidence intervals.",
        "unisolated": [
            "trie_lookup_vs_viterbi_vs_byte_fallback",
            "ffi_boundary_vs_python_dispatch",
            "allocation_vs_copy",
            "merge_application_inactive_for_fused_path",
        ],
    }
    results = output / "results.json"
    stack = output / "python_callstack.txt"
    report = output / "REPORT.md"
    manifest = output / "manifest.json"
    results.write_text(json.dumps({"metadata": metadata, "records": records}, indent=2) + "\n", encoding="utf-8")
    stack.write_text(callstack.rstrip() + "\n", encoding="utf-8")
    lines = [
        "# UniqToken hot-path profile",
        "",
        f"Commit: `{commit}`; build: `{args.build_mode}`; Rayon threads: `{args.threads}`.",
        f"Warmup: {args.warmup}; repetitions: {args.repetitions}; iterations per repetition: {args.iterations}.",
        "",
        "Exact parity gate passed before timings. Input MB/s uses normalized UTF-8 bytes (decimal MB).",
        "",
        "| Workload | Normalized bytes | Encode MB/s | Encode ms | Decode ms | Tokens/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            f"| {record['workload']} | {record['normalized_input_bytes']} | {record['input_mb_per_s']:.2f} | "
            f"{record['encode_wall_ns']['median'] / 1e6:.3f} | {record['decode_wall_ns']['median'] / 1e6:.3f} | "
            f"{record['tokens_per_s']:.0f} |"
        )
    lines.extend(
        [
            "",
            "Diagnostic stage elapsed time (ms, summed across rows):",
            "",
            "| Workload | Gate | Normalize | Regex | Grapheme + chunks | Cached segmentation | Output copy |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in records:
        values = [record["stage_elapsed_ns"][stage]["median"] / 1e6 for stage in STAGES]
        lines.append(f"| {record['workload']} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines.extend(
        [
            "",
            "Largest measured stage per workload (median and observed min/max, ms):",
            "",
            "| Workload | Stage | Median | Observed range |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for record in records:
        stage, value = max(record["stage_elapsed_ns"].items(), key=lambda item: item[1]["median"])
        low, high = value["observed_min_max"]
        lines.append(
            f"| {record['workload']} | {stage} | {value['median'] / 1e6:.3f} | {low / 1e6:.3f}-{high / 1e6:.3f} |"
        )
    lines.extend(
        [
            "",
            "All stage ranges and raw wall samples are in `results.json`.",
            "`python_callstack.txt` is a cProfile call-stack report; native Rust frames are opaque there.",
            "The native diagnostic replays the exact normalization, regex, grapheme, cached segmentation, and token-copy calls.",
            "Cached segmentation combines trie lookup, Viterbi/lattice work, and byte fallback. Merge application is inactive for the fused path.",
            "Python/Rust conversion and allocation cannot be subtracted reliably from parallel wall time; no separate percentage is claimed.",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    hashes = {path.name: sha256(path) for path in (results, stack, report)}
    manifest.write_text(json.dumps({"schema_version": 1, "sha256": hashes}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote verified profile to {output}")


def native_file():
    import uniqtoken_core

    return uniqtoken_core._native.__file__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/profiles/issue95"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--build-mode", choices=("release", "debug"), default="release")
    args = parser.parse_args()
    records, callstack, tok = profile(args)
    write_results(args, records, callstack, tok)


if __name__ == "__main__":
    main()
