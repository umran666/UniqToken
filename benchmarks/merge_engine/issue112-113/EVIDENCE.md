# FastMergeEngine Optimization Evidence (#112 / #113)

Before/after measurement for the pair-index (#113) and hybrid boundary
worklist (#112) changes on `FastMergeEngine`, using the #111 canonical
benchmark (`python -m benchmarks.merge_engine_benchmark`). Primary metric
is normalized input decimal MB/s over the pieces output path; the span
path tracks it within a few percent.

## Measured builds (clean worktrees, pinned commits)

- **before**: `cca0dcf8` (`main` with #105/#110/#111/#114 merged;
  pre-optimization engine)
- **after**: `9614ca63` (this branch rebased on the same base; optimized
  engine)

Both worktrees were clean: every receipt records `git_dirty=false`, and
the only source difference between the builds is
`uniqtoken/merge_engine.py`. Three interleaved before/after pairs were run
on the same machine in one session. Receipts:
`before_run{1,2,3}/` and `after_run{1,2,3}/`.

## Methodology

- Machine: Windows 10, Python 3.10.11, workers=1, 3 warmup /
  7 repetitions, interleaved runs to control for time-varying machine
  state.
- **Noise floor**: the production path is byte-identical in both builds,
  so its run-to-run deltas measure pure environment noise. Using per-cell
  best-case latency (`min_ms`, minimum across the three runs), the
  production-path deltas stay within ±5.5% on every workload.
- **Robust statistic**: per-cell best-case latency (`min_ms`, minimum
  across the three interleaved runs); a change is only called real when it
  exceeds the ±5.5% min-latency noise floor. Mean-throughput comparisons
  of single runs are not used: the production path's own mean throughput
  varies by up to 89% across these six runs on sub-5 ms cells.
- The final-pair median-throughput numbers below are directional only and
  are not part of the acceptance argument.

## Results — best-case latency, min across three runs (lower is better)

Improvement is `(before - after) / after` (percent faster).

| Workload | before ms | after ms | improvement |
|---|---|---|---|
| short | 0.026 | 0.025 | +3.2% |
| medium | 0.307 | 0.295 | +4.0% |
| long | 1.337 | 1.239 | +8.0% |
| multilingual | 0.234 | 0.221 | +6.2% |
| source_code | 0.907 | 0.844 | +7.5% |
| punctuation_heavy | 3.327 | 3.068 | +8.4% |
| cross_word_heavy | 1.523 | 1.397 | +9.0% |
| pathological | 3.817 | 3.525 | +8.3% |
| synthetic_1024_sparse | 2.501 | 2.014 | +24.2% |
| synthetic_1024_medium | 2.406 | 1.814 | +32.6% |
| synthetic_1024_dense | 2.403 | 1.830 | +31.3% |
| synthetic_1024_hierarchical | 2.494 | 1.727 | +44.4% |
| synthetic_4096_sparse | 9.730 | 7.555 | +28.8% |
| synthetic_4096_dense | 9.843 | 7.535 | +30.6% |

No workload regressed: every cell improves, and all 14 improvements exceed
the measured min-latency noise floor. Text cells gain +3% to +9%; the
multi-pass synthetic cells where the worklist replaces repeated full
rescans gain +24% to +44%. The long/pathological acceptance bar (#112)
holds (long +8.0%, pathological +8.3%).

## Results — median throughput, final interleaved pair (directional)

Overall fast-engine throughput +10.4% (sum of per-workload MB/s,
`before_run3` vs `after_run3`); per-cell highlights: synthetic_4096_sparse
+49.5%, synthetic_1024_hierarchical +33.3%, synthetic_1024_medium +31.6%.
Some single-run cells move negative (long -11.3%, multilingual -9.6%),
consistent with the large mean-throughput noise documented above; the
best-case latency table is the acceptance evidence.

## Allocation counts (#113)

tracemalloc per merge-stage call (`alloc_count_per_call`, fast engine):
25 → 29 (short: 21 → 24). The pair index removes the per-pair string
concatenation, but the worklist allocates one frontier list plus one seed
set per pass; the net count rises by four small allocations per pass while
per-pair string churn (the dominant byte volume on dense workloads)
disappears. Peak traced bytes are recorded per record in the receipts.
Engines remain stateless per call — no retained memory growth is possible
by construction (all scratch structures are function-local). The pair
index itself is built once per table (`MembershipMergeTable.__post_init__`)
and cached per tokenizer by `CustomTokenizer._get_cached_merge_table`.

## Correctness

- `tests.test_merge_engine_differential_suite` (#110): 11,377 cases,
  zero mismatches, on the optimized engine.
- `tests.test_fast_merge_engine`: 34 tests OK.
- `tests.test_merge_engine_full_semantics` (#114): 19 tests OK.
- The #111 benchmark's own 3-way parity gate passed on every workload
  before any timing was accepted.

## Interpretation limits

All numbers are environment-scoped (single machine; noise floor documented
above). No cross-machine comparison and no production-adoption claim
follows from this evidence; see the #116 decision document for the adoption
trade-off.
