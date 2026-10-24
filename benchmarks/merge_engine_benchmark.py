"""Canonical merge-engine benchmark: production vs FastMergeEngine vs Reference (Issue #111).

Compares the current production SuperBPE runtime merge path against
FastMergeEngine (and ReferenceMergeEngine as the oracle) under identical
merge tables, constraints, inputs, and worker configuration, following the
canonical methodology consolidated in issue #105:

- **Exact parity gate before timing.** Every workload is checked for
  bit-identical outputs across all compared paths before any timing data is
  accepted; a parity failure aborts the benchmark (fail-loud).
- **Workload matrix.** Text-domain fixtures (the canonical five from #105
  plus ``punctuation_heavy``, ``cross_word_heavy`` and ``pathological``) run
  through a genuinely CEM-trained cross-word tokenizer, in both the
  pieces-only and span-preserving output paths, plus engine-level synthetic
  sequence-length x merge-regime cells for controlled worst-case analysis.
- **Metrics.** Normalized input decimal MB/s is the primary metric
  (decimal MB = 10^6 bytes). Secondary: tokens/s, encode p50/p95 latency (ms),
  tracemalloc allocation count/bytes, peak process RSS (MB), CPU utilization.
- **Repetitions.** 3 warmup calls, 7 timed repetitions (``--quick``: 1/3),
  ``time.perf_counter_ns``; individual workloads stay visible — no aggregate
  hides a per-workload regression. Geometric means are reported *alongside*.
- **Fingerprint.** Git commit/dirty state, platform, CPU, RAM, Python
  version, warmup/repetition policy, and fixture hashes are recorded so
  results are only comparable on matching hardware/software.

Interpretation rules (inherited from #105 / benchmarks/PERF_GATE.md):
- No performance claim is based on one run or generalized beyond the
  measured environment.
- This benchmark never replaces correctness evidence (#110/#114).

Run with::

    python -m benchmarks.merge_engine_benchmark --output benchmarks/merge_engine/issue111
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import psutil

from benchmarks.canonical_benchmark import (
    FIXTURES as CANONICAL_FIXTURES,
    get_git_commit,
    get_peak_rss_bytes,
    geomean,
    is_git_dirty,
    sha256_bytes,
)
from benchmarks.fast_merge_engine_benchmark import build_synthetic_benchmark_fixture
from uniqtoken.cem_merger import CrossEntropyMerging
from uniqtoken.merge_engine import (
    FastMergeEngine,
    ReferenceMergeEngine,
    apply_engine_to_pieces,
    apply_engine_to_tokens,
    cross_word_membership_table,
    production_constraints,
)
from uniqtoken.tokenizer import CustomTokenizer, Token

ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "1.0"
BENCHMARK_ISSUE = "111"
WARMUP = 3
REPETITIONS = 7
QUICK_WARMUP = 1
QUICK_REPETITIONS = 3

# Additional text-domain fixtures required by the #111 workload matrix.
EXTRA_TEXT_FIXTURES = {
    "punctuation_heavy": "wait... really?! (yes; no: maybe) [a, b] {c: d} -> e != f != g; " * 40,
    "cross_word_heavy": "the quick brown fox jumps over the lazy dog near the riverbank " * 60,
    "pathological": "the system " * 900,  # ~10.8 KB, dense repeated cross-word candidates
}

TEXT_FIXTURES: Dict[str, str] = {**CANONICAL_FIXTURES, **EXTRA_TEXT_FIXTURES}

# Engine-level synthetic matrix (sequence length x merge regime).
SYNTHETIC_CELLS: Tuple[Tuple[int, str], ...] = (
    (1024, "sparse"),
    (1024, "medium"),
    (1024, "dense"),
    (1024, "hierarchical"),
    (4096, "sparse"),
    (4096, "dense"),
)

# Corpus reused from the #114 full-semantics fixture (fixed recorded inputs).
TRAIN_CORPUS = [
    "hello world and welcome to the system",
    "in the beginning of the process to the end",
    "the quick brown fox jumps over the lazy dog",
    "def calculate_loss(predictions, targets) -> float:",
    "processing... wait...... result = ([{a: (b + c)}]);",
    "人工智能和自然语言处理技术发展迅速",
    "प्रणाली और विश्वविद्यालय में अनुसंधान कार्य",
    "المعلوماتية والاسـتراتيجية في معالجة اللغات",
    "v1.2.3 and parseXMLDocument with calculate_total_score",
    "the system processes the process systematically",
]

COMPARISON_PATHS = ("production", "fast", "reference")


class ParityGateError(AssertionError):
    """Raised when the pre-timing parity gate fails (fail-loud, #111)."""


# ---------------------------------------------------------------------------
# Fixture / tokenizer construction
# ---------------------------------------------------------------------------


def train_superbpe_tokenizer() -> CustomTokenizer:
    """Train a CEM cross-word (SuperBPE) tokenizer deterministically."""
    base = CustomTokenizer.train_from_corpus(
        TRAIN_CORPUS,
        target_vocab_size=400,
        min_frequency=1,
        verbose=False,
        special_tokens=["<|endoftext|>"],
    )
    pretok_chunks = [
        chunk
        for doc in TRAIN_CORPUS
        for chunk in base.pre_tokenizer.pre_tokenize(base.normalizer.normalize(doc))
    ]
    cem = CrossEntropyMerging(max_merges=30, cross_word=True, verbose=False)
    model = cem.optimize(base.model, chunks=pretok_chunks)
    return CustomTokenizer(normalizer=base.normalizer, pre_tokenizer=base.pre_tokenizer, model=model)


def text_workload_leaves(tokenizer: CustomTokenizer, text: str) -> List[str]:
    """Pre-tokenize a text fixture into the leaf pieces fed to the merge stage."""
    normalized = tokenizer.normalizer.normalize(text)
    return list(tokenizer.pre_tokenizer.pre_tokenize(normalized))


def text_workload_leaf_tokens(tokenizer: CustomTokenizer, text: str) -> List[Token]:
    """Leaf pieces as Tokens with synthetic positional spans (merge-stage input)."""
    pieces = text_workload_leaves(tokenizer, text)
    unk = tokenizer.model.token_to_id.get(tokenizer.model.unk_token, -1)
    return [
        Token(text=piece, id=tokenizer.model.token_to_id.get(piece, unk), raw_span=(i, i + 1))
        for i, piece in enumerate(pieces)
    ]


def build_synthetic_leaf_tokens(tokenizer: CustomTokenizer, pieces: Sequence[str]) -> List[Token]:
    """Leaf pieces of a synthetic fixture as Tokens with positional spans."""
    unk = tokenizer.model.token_to_id.get(tokenizer.model.unk_token, -1)
    return [
        Token(text=piece, id=tokenizer.model.token_to_id.get(piece, unk), raw_span=(i, i + 1))
        for i, piece in enumerate(pieces)
    ]


# ---------------------------------------------------------------------------
# Compared paths (identical inputs, identical tables)
# ---------------------------------------------------------------------------


def _run_production_pieces(tokenizer: CustomTokenizer, pieces: List[str]) -> List[str]:
    return tokenizer._apply_cross_word_merges(list(pieces), dropout_prob=0.0)


def _run_production_spans(tokenizer: CustomTokenizer, tokens: List[Token]) -> List[Token]:
    return tokenizer._apply_cross_word_merges_with_spans(list(tokens), dropout_prob=0.0)


def _run_engine_pieces(engine: Any, pieces: List[str], table: Any, constraints: Any) -> List[str]:
    out, _ = apply_engine_to_pieces(engine, pieces, table, constraints)
    return out


def _run_engine_spans(engine: Any, tokens: List[Token], table: Any, constraints: Any) -> List[Token]:
    out, _ = apply_engine_to_tokens(engine, tokens, table, constraints)
    return out


def run_parity_gate(
    tokenizer: CustomTokenizer,
    pieces: List[str],
    tokens: List[Token],
    table: Any,
    constraints: Any,
) -> Dict[str, Any]:
    """Exact 3-way parity across production/fast/reference on both output paths.

    Runs BEFORE any timing (issue #111: no timing data accepted otherwise).
    """
    outputs: Dict[str, Any] = {
        "production_pieces": _run_production_pieces(tokenizer, pieces),
        "fast_pieces": _run_engine_pieces(FastMergeEngine(), pieces, table, constraints),
        "reference_pieces": _run_engine_pieces(ReferenceMergeEngine(), pieces, table, constraints),
        "production_spans": _run_production_spans(tokenizer, tokens),
        "fast_spans": _run_engine_spans(FastMergeEngine(), tokens, table, constraints),
        "reference_spans": _run_engine_spans(ReferenceMergeEngine(), tokens, table, constraints),
    }
    failures = []
    if not (outputs["production_pieces"] == outputs["fast_pieces"] == outputs["reference_pieces"]):
        failures.append("pieces path mismatch")
    for key in ("production_spans", "fast_spans", "reference_spans"):
        outputs[f"{key}_view"] = [
            (t.text, t.id, t.raw_span[0], t.raw_span[1]) for t in outputs[key]
        ]
    if not (
        outputs["production_spans_view"] == outputs["fast_spans_view"] == outputs["reference_spans_view"]
    ):
        failures.append("span path mismatch")
    if failures:
        raise ParityGateError(f"parity gate failed ({failures}); no timing data may be accepted")
    applied = len(pieces) - len(outputs["production_pieces"])
    return {"parity": "pass", "leaf_pieces": len(pieces), "applied_merges": applied}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile on an ascending-sorted list."""
    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[rank - 1]


def measure_call(fn: Callable[[], Any], warmup: int, repetitions: int) -> Dict[str, float]:
    """Time ``fn`` with warmup + repetitions; returns ms statistics."""
    for _ in range(warmup):
        fn()
    gc.collect()
    samples_ms: List[float] = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        fn()
        samples_ms.append((time.perf_counter_ns() - start) / 1e6)
    ordered = sorted(samples_ms)
    mean = statistics.fmean(samples_ms)
    return {
        "mean_ms": mean,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p50_ms": percentile(ordered, 0.50),
        "p95_ms": percentile(ordered, 0.95),
        "std_ms": statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "repetitions": float(repetitions),
    }


def measure_allocations(fn: Callable[[], Any]) -> Dict[str, int]:
    """tracemalloc allocation count/bytes for a single call (#111 measurement contract)."""
    import tracemalloc

    gc.collect()
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    first = tracemalloc.take_snapshot()
    fn()
    second = tracemalloc.take_snapshot()
    _, peak2 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    count_diff = sum(s.count_diff for s in second.compare_to(first, "lineno"))
    return {
        "alloc_count_per_call": max(0, count_diff),
        "alloc_peak_bytes": max(peak, peak2),
    }


def run_benchmark(
    output_dir: Path,
    quick: bool = False,
) -> Tuple[Path, Path, Path]:
    """Execute the full matrix and write the receipt triple. Returns their paths."""
    warmup = QUICK_WARMUP if quick else WARMUP
    repetitions = QUICK_REPETITIONS if quick else REPETITIONS

    process = psutil.Process()
    tokenizer = train_superbpe_tokenizer()
    table = cross_word_membership_table(tokenizer)
    constraints = production_constraints(table)
    fast_engine = FastMergeEngine()
    reference_engine = ReferenceMergeEngine()

    fixture_hashes = {name: sha256_bytes(text.encode("utf-8")) for name, text in TEXT_FIXTURES.items()}

    process.cpu_percent(None)  # reset counter
    records: List[Dict[str, Any]] = []

    def add_record(record: Dict[str, Any]) -> None:
        records.append(record)

    # ---- Text-domain workloads (pieces + span output paths) ----
    for name, text in TEXT_FIXTURES.items():
        pieces = text_workload_leaves(tokenizer, text)
        tokens = text_workload_leaf_tokens(tokenizer, text)
        input_bytes = len(text.encode("utf-8"))
        parity = run_parity_gate(tokenizer, pieces, tokens, table, constraints)

        piece_fns = {
            "production": lambda p=list(pieces): _run_production_pieces(tokenizer, p),
            "fast": lambda p=list(pieces): _run_engine_pieces(fast_engine, p, table, constraints),
            "reference": lambda p=list(pieces): _run_engine_pieces(reference_engine, p, table, constraints),
        }
        span_fns = {
            "production": lambda t=list(tokens): _run_production_spans(tokenizer, t),
            "fast": lambda t=list(tokens): _run_engine_spans(fast_engine, t, table, constraints),
            "reference": lambda t=list(tokens): _run_engine_spans(reference_engine, t, table, constraints),
        }
        for output_path, fns in (("pieces", piece_fns), ("spans", span_fns)):
            for path_name, fn in fns.items():
                stats = measure_call(fn, warmup, repetitions)
                allocs = measure_allocations(fn)
                tokens_out = (
                    len(_run_production_pieces(tokenizer, list(pieces)))
                    if output_path == "pieces"
                    else len(_run_production_spans(tokenizer, list(tokens)))
                )
                add_record(
                    {
                        "tokenizer": path_name,
                        "workload": name,
                        "output_path": output_path,
                        "batch_size": 1,
                        "input_bytes": input_bytes,
                        "leaf_pieces": len(pieces),
                        "tokens_out": tokens_out,
                        "throughput_mb_s": (input_bytes / 1e6) / (stats["mean_ms"] / 1e3)
                        if stats["mean_ms"] > 0
                        else 0.0,
                        "tokens_per_s": tokens_out / (stats["mean_ms"] / 1e3) if stats["mean_ms"] > 0 else 0.0,
                        **{k: v for k, v in stats.items()},
                        **allocs,
                        "parity_verified": True,
                    }
                )

    # ---- Engine-level synthetic cells (pieces path) ----
    for seq_len, regime in SYNTHETIC_CELLS:
        synth_pieces, synth_table, synth_tok = build_synthetic_benchmark_fixture(seq_len, regime)
        synth_constraints = production_constraints(synth_table)
        synth_tokens = build_synthetic_leaf_tokens(synth_tok, synth_pieces)
        input_bytes = sum(len(p.encode("utf-8")) for p in synth_pieces)
        parity = run_parity_gate(synth_tok, list(synth_pieces), synth_tokens, synth_table, synth_constraints)
        synth_fns = {
            "production": lambda p=list(synth_pieces), t=synth_tok: _run_production_pieces(t, p),
            "fast": lambda p=list(synth_pieces): _run_engine_pieces(
                fast_engine, p, synth_table, synth_constraints
            ),
            "reference": lambda p=list(synth_pieces): _run_engine_pieces(
                reference_engine, p, synth_table, synth_constraints
            ),
        }
        for path_name, fn in synth_fns.items():
            stats = measure_call(fn, warmup, repetitions)
            allocs = measure_allocations(fn)
            tokens_out = len(_run_production_pieces(synth_tok, list(synth_pieces)))
            add_record(
                {
                    "tokenizer": path_name,
                    "workload": f"synthetic_{seq_len}_{regime}",
                    "output_path": "pieces",
                    "batch_size": 1,
                    "input_bytes": input_bytes,
                    "leaf_pieces": len(synth_pieces),
                    "tokens_out": tokens_out,
                    "throughput_mb_s": (input_bytes / 1e6) / (stats["mean_ms"] / 1e3)
                    if stats["mean_ms"] > 0
                    else 0.0,
                    "tokens_per_s": tokens_out / (stats["mean_ms"] / 1e3) if stats["mean_ms"] > 0 else 0.0,
                    **{k: v for k, v in stats.items()},
                    **allocs,
                    "parity_verified": True,
                }
            )

    cpu_util = process.cpu_percent(None)

    metadata = build_metadata(fixture_hashes, warmup, repetitions, cpu_util, quick)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    report_path = output_dir / "REPORT.md"
    manifest_path = output_dir / "manifest.json"

    results = {"metadata": metadata, "records": records, "summaries": build_summaries(records)}
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(generate_report(metadata, results["summaries"], records), encoding="utf-8")
    manifest = {
        "status": "complete",
        "artifacts": {
            "results.json": sha256_bytes(results_path.read_bytes()),
            "REPORT.md": sha256_bytes(report_path.read_bytes()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return results_path, report_path, manifest_path


def build_metadata(
    fixture_hashes: Dict[str, str],
    warmup: int,
    repetitions: int,
    cpu_util: float,
    quick: bool,
) -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_issue": BENCHMARK_ISSUE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "git_dirty": is_git_dirty(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "total_ram_gb": round(vm.total / 1e9, 2),
        },
        "workers": 1,
        "warmup": warmup,
        "repetitions": repetitions,
        "quick_mode": quick,
        "primary_metric": "throughput_mb_s (normalized input decimal MB/s)",
        "secondary_metrics": [
            "tokens_per_s",
            "p50_ms",
            "p95_ms",
            "alloc_count_per_call",
            "alloc_peak_bytes",
            "peak_rss_mb",
            "cpu_util_pct",
        ],
        "fixture_hashes": fixture_hashes,
        "peak_rss_mb": round(get_peak_rss_bytes() / 1e6, 2),
        "cpu_util_pct": cpu_util,
        "interpretation_rules": [
            "Exact parity gate passed before any timing data was accepted.",
            "Individual workloads remain visible; geomean summaries never hide a per-workload regression.",
            "No performance claim is based on one run or generalized beyond the measured environment.",
            "Correctness evidence lives in #110/#114; this benchmark makes no correctness claims.",
        ],
    }


def build_summaries(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-path geomean of text-workload MB/s (reported alongside, not instead of, cells)."""
    text_names = set(TEXT_FIXTURES)
    summaries: Dict[str, Any] = {}
    for path_name in COMPARISON_PATHS:
        text_mb_s = [
            r["throughput_mb_s"]
            for r in records
            if r["tokenizer"] == path_name and r["workload"] in text_names and r["output_path"] == "pieces"
        ]
        summaries[path_name] = {
            "text_workloads_geomean_mb_s": round(geomean(text_mb_s), 4) if text_mb_s else None,
            "text_workload_count": len(text_mb_s),
        }
    # Speedup of fast vs production per workload (pieces path).
    by_cell: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in records:
        if r["output_path"] == "pieces":
            by_cell.setdefault((r["workload"], "mb_s"), {})[r["tokenizer"]] = r["throughput_mb_s"]
            by_cell.setdefault((r["workload"], "p50"), {})[r["tokenizer"]] = r["p50_ms"]
    speedups = {}
    for workload in sorted({r["workload"] for r in records}):
        cell = by_cell.get((workload, "mb_s"), {})
        if "production" in cell and "fast" in cell and cell["fast"] > 0:
            speedups[workload] = round(cell["production"] / cell["fast"], 4)
    summaries["fast_vs_production_speedup_mb_s"] = speedups
    return summaries


def generate_report(metadata: Dict[str, Any], summaries: Dict[str, Any], records: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Merge-Engine Benchmark Report (Issue #111)")
    lines.append("")
    lines.append(f"- Generated: {metadata['generated_at']}")
    lines.append(f"- Commit: `{metadata['git_commit']}` (dirty={metadata['git_dirty']})")
    lines.append(
        f"- Platform: {metadata['platform']['system']} {metadata['platform']['release']} / "
        f"{metadata['platform']['python']} / {metadata['platform']['cpu_count']} logical CPUs / "
        f"{metadata['platform']['total_ram_gb']} GB"
    )
    lines.append(
        f"- Policy: workers=1, warmup={metadata['warmup']}, repetitions={metadata['repetitions']}, "
        f"peak RSS {metadata['peak_rss_mb']} MB"
    )
    lines.append("- Parity gate: exact 3-way (production/fast/reference), both output paths, passed before timing.")
    lines.append("")
    lines.append("## Geometric means over text workloads (pieces path, primary metric)")
    lines.append("")
    lines.append("| Path | Text-workload geomean MB/s | Cells |")
    lines.append("|---|---|---|")
    for path_name in COMPARISON_PATHS:
        s = summaries[path_name]
        lines.append(f"| {path_name} | {s['text_workloads_geomean_mb_s']} | {s['text_workload_count']} |")
    lines.append("")
    lines.append("## Fast vs production throughput ratio (pieces path, per workload)")
    lines.append("")
    lines.append("| Workload | production MB/s | fast MB/s | speedup |")
    lines.append("|---|---|---|---|")
    by_cell: Dict[str, Dict[str, float]] = {}
    for r in records:
        if r["output_path"] == "pieces":
            by_cell.setdefault(r["workload"], {})[r["tokenizer"]] = r["throughput_mb_s"]
    for workload in sorted(by_cell):
        cell = by_cell[workload]
        if "production" in cell and "fast" in cell:
            speedup = cell["production"] / cell["fast"] if cell["fast"] > 0 else float("inf")
            lines.append(
                f"| {workload} | {cell['production']:.4f} | {cell['fast']:.4f} | {speedup:.3f}x |"
            )
    lines.append("")
    lines.append("## Individual workload records")
    lines.append("")
    lines.append(
        "| Workload | Path | Output | MB/s | p50 ms | p95 ms | tokens/s | allocs/call | alloc peak B |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in records:
        lines.append(
            f"| {r['workload']} | {r['tokenizer']} | {r['output_path']} | {r['throughput_mb_s']:.4f} "
            f"| {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['tokens_per_s']:.0f} "
            f"| {r['alloc_count_per_call']} | {r['alloc_peak_bytes']} |"
        )
    lines.append("")
    lines.append("## Interpretation rules")
    lines.append("")
    for rule in metadata["interpretation_rules"]:
        lines.append(f"- {rule}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "merge_engine" / "issue111",
        help="Receipt output directory",
    )
    parser.add_argument("--quick", action="store_true", help="1 warmup / 3 repetitions (smoke runs)")
    args = parser.parse_args()
    results_path, report_path, manifest_path = run_benchmark(args.output, quick=args.quick)
    print(f"results:  {results_path}")
    print(f"report:   {report_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
