# Merge-Engine Benchmark (Issue #111)

Canonical-methodology benchmark comparing the current **production** SuperBPE
runtime merge path against **FastMergeEngine** (with ReferenceMergeEngine as
the correctness oracle), per the measurement contract in issue #111.

## Run

```bash
python -m benchmarks.merge_engine_benchmark --output benchmarks/merge_engine/issue111
# smoke run (1 warmup / 3 repetitions):
python -m benchmarks.merge_engine_benchmark --output /tmp/mebench --quick
```

## Methodology

- **Exact parity gate before timing** (`run_parity_gate`): 3-way
  production/fast/reference, on both the pieces-only and span-preserving
  output paths, per workload. Any mismatch raises `ParityGateError` and no
  timing data is accepted (fail-loud).
- **Workload matrix**:
  - Text-domain fixtures: the canonical five from #105 (`short`, `medium`,
    `long`, `multilingual`, `source_code`) plus `punctuation_heavy`,
    `cross_word_heavy`, `pathological`, run through a genuinely CEM-trained
    cross-word (SuperBPE) tokenizer, in both output paths.
  - Engine-level synthetic cells: sequence length 1024/4096 x merge regimes
    sparse/medium/dense/hierarchical for controlled worst-case analysis.
- **Metrics**: normalized input decimal MB/s (primary), tokens/s, encode
  p50/p95 latency (nearest-rank percentiles over repetitions), tracemalloc
  allocation count/bytes per call, peak process RSS, CPU utilization.
- **Repetitions**: 3 warmup calls, 7 timed repetitions (`--quick`: 1/3) with
  `time.perf_counter_ns`; per-workload records stay visible, and geomean
  summaries are reported *alongside*, never instead of, individual cells.
- **Fingerprint**: schema 1.0 metadata (git commit + dirty flag, platform,
  CPU, RAM, Python, worker/warmup/repetition policy, fixture hashes) so
  results are only compared on matching hardware/software.

## Receipt

`benchmarks/merge_engine/issue111/` holds the committed evidence:
`results.json`, `REPORT.md`, and a `manifest.json` binding SHA-256 hashes of
both. Records use canonical schema-1.0 field names (`tokenizer`, `workload`,
`throughput_mb_s`, `peak_rss_mb`, `repetitions`, ...) for compatibility with
`benchmarks/canonical_gate.py` consumers.

## Interpretation rules

- No performance claim is based on one run or generalized beyond the
  measured environment.
- This benchmark never replaces correctness evidence (#110 differential
  verification, #114 full-semantics validation).
- Comparison against a different machine's receipt requires a matching
  hardware/software fingerprint (see `benchmarks/PERF_GATE.md`).
