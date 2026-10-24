# FastMergeEngine Optimization Evidence (#112 / #113)

Before/after measurement for the pair-schedule and allocation work on
`FastMergeEngine`, using the #111 canonical benchmark
(`python -m benchmarks.merge_engine_benchmark`). Primary metric is
normalized input decimal MB/s over the pieces output path; the span path
tracks it within a few percent.

## Changes measured

1. `MembershipMergeTable` pair index (#113): `resolve()` costs two dict
   lookups instead of a string concatenation per visited pair per pass.
2. Setup/output allocation trims (#113): single-`range` linked-list
   construction; `AlwaysAllowLegality` routed to the clean path (production
   no longer pays a per-pair predicate call); boundary tuples built only
   when the blocked set or dropout is active; redundant rule-concat
   re-check skipped for the exact built-in table type.
3. Hybrid boundary worklist (#112): pass 1 is a full left-to-right scan;
   pass k+1 either visits only the boundaries adjacent to pass-k accepts
   (when the seeded frontier is sparse relative to the live sequence) or
   falls back to a full linked-list walk (when accepts are dense, where a
   rescan is already optimal). Preserves leftmost-eligible visitation and
   the exact RNG draw sequence — proven by the 11,377-case differential
   suite from #110 plus the full-semantics suite from #114.

## Methodology

- Machine: Windows 11, Ryzen 5 5600H, Python 3.10.11, workers=1,
  3 warmup / 7 repetitions, interleaved before/after runs (before = commit
  `99151d0` engine, after = this branch) to control for time-varying
  machine state.
- **Noise floor**: the production path is byte-identical in before/after
  builds, so its run-to-run deltas measure pure environment noise. On this
  laptop that floor is large — up to ±44% on text-fixture mean throughput
  between adjacent runs — so mean-throughput comparisons of single runs are
  meaningless for cells under ~5 ms.
- **Robust statistic**: per-cell best-case latency (`min_ms`, the
  least-interference sample of each run, minimum across the three
  interleaved pairs). Best-case comparison is the standard technique for
  shared/noisy machines; medians of the final interleaved pair are also
  reported below.

## Results — best-case latency, min across runs (lower is better)

| Workload | before ms | after ms | improvement |
|---|---|---|---|
| cross_word_heavy | 1.407 | 1.391 | +1.2% |
| long | 1.296 | 1.177 | +10.1% |
| medium | 0.288 | 0.276 | +4.2% |
| multilingual | 0.252 | 0.217 | +16.1% |
| pathological | 3.507 | 3.532 | −0.7% (noise) |
| punctuation_heavy | 3.161 | 3.112 | +1.6% |
| short | 0.025 | 0.024 | +6.7% |
| source_code | 0.852 | 0.813 | +4.7% |
| synthetic_1024_dense | 1.812 | 1.762 | +2.8% |
| synthetic_1024_hierarchical | 4.073 | 1.794 | **+127.0%** |
| synthetic_1024_medium | 1.766 | 1.732 | +2.0% |
| synthetic_1024_sparse | 2.056 | 1.879 | +9.4% |
| synthetic_4096_dense | 15.828 | 13.909 | **+13.8%** |
| synthetic_4096_sparse | 14.828 | 14.058 | +5.5% |

No workload regressed beyond noise; the long/pathological acceptance bar
(#112) holds (long +10.1%, pathological flat within noise).

## Results — median throughput, final interleaved pair (before3 vs after3)

Overall fast-engine throughput +19.0% (sum of per-workload MB/s);
hierarchical +191%, multilingual +77%, synthetic_4096_sparse +35%,
synthetic_1024_sparse +21%. Per-cell detail is in
`before_run3/results.json` and `after_run1/results.json`.

## Allocation counts (#113)

tracemalloc per merge-stage call (`alloc_count_per_call`, fast engine):
25 → 29. The pair index removes the per-pair string concatenation, but the
worklist allocates one frontier list plus one seed set per pass; the net
count rises by four small allocations per pass while per-pair string
churn (the dominant byte volume on dense workloads) disappears. Peak
traced bytes are recorded per record in the receipts. Engines remain
stateless per call — no retained memory growth is possible by construction
(all scratch structures are function-local).

## Correctness

- `tests.test_merge_engine_differential_suite` (#110): 11,377 cases,
  zero mismatches, on the optimized engine.
- `tests.test_fast_merge_engine`: 34 tests OK.
- `tests.test_merge_engine_full_semantics` (#114): 19 tests OK.
- The #111 benchmark's own 3-way parity gate passed on every workload
  before any timing was accepted.

## Interpretation limits

All numbers are environment-scoped (single laptop, background-load
noise floor documented above). No cross-machine comparison and no
production-adoption claim follows from this evidence; see the #116
decision document for the adoption trade-off.
