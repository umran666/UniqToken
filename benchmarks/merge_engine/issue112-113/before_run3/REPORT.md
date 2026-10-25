# Merge-Engine Benchmark Report (Issue #111)

- Generated: 2026-10-25T02:21:09.804769+00:00
- Commit: `cca0dcf899007b312fc2e50eeac82472c9149380` (dirty=False)
- Platform: Windows 10 / 3.10.11 / 16 logical CPUs / 16.85 GB
- Policy: workers=1, warmup=3, repetitions=7, peak RSS 42.34 MB
- Parity gate: exact 3-way (production/fast/reference), both output paths, passed before timing.

## Geometric means over text workloads (pieces path, primary metric)

| Path | Text-workload geomean MB/s | Cells |
|---|---|---|
| production | 8.7569 | 8 |
| fast | 1.8504 | 8 |
| reference | 1.4896 | 8 |

## Fast vs production throughput ratio (pieces path, per workload)

| Workload | production MB/s | fast MB/s | speedup |
|---|---|---|---|
| cross_word_heavy | 11.6728 | 2.4029 | 4.858x |
| long | 16.1363 | 3.1484 | 5.125x |
| medium | 14.5004 | 2.5437 | 5.700x |
| multilingual | 14.4337 | 3.2122 | 4.493x |
| pathological | 12.9033 | 2.3527 | 5.484x |
| punctuation_heavy | 3.0701 | 0.6639 | 4.624x |
| short | 4.3564 | 1.2759 | 3.414x |
| source_code | 5.0823 | 1.1155 | 4.556x |
| synthetic_1024_dense | 6.4544 | 2.2515 | 2.867x |
| synthetic_1024_hierarchical | 7.2592 | 2.1164 | 3.430x |
| synthetic_1024_medium | 8.4263 | 2.2649 | 3.720x |
| synthetic_1024_sparse | 7.0383 | 2.1808 | 3.227x |
| synthetic_4096_dense | 7.1660 | 2.1731 | 3.298x |
| synthetic_4096_sparse | 8.1125 | 1.8515 | 4.382x |

## Individual workload records

| Workload | Path | Output | MB/s | p50 ms | p95 ms | tokens/s | allocs/call | alloc peak B |
|---|---|---|---|---|---|---|---|---|
| short | production | pieces | 4.3564 | 0.006 | 0.030 | 990099 | 12 | 3134 |
| short | fast | pieces | 1.2759 | 0.026 | 0.075 | 289975 | 21 | 7072 |
| short | reference | pieces | 1.1724 | 0.032 | 0.071 | 266464 | 21 | 7936 |
| short | production | spans | 4.3937 | 0.007 | 0.027 | 998573 | 10 | 1886 |
| short | fast | spans | 1.1733 | 0.029 | 0.083 | 266667 | 21 | 7256 |
| short | reference | spans | 1.1294 | 0.034 | 0.066 | 256692 | 21 | 8120 |
| medium | production | pieces | 14.5004 | 0.065 | 0.098 | 2231925 | 10 | 10192 |
| medium | fast | pieces | 2.5437 | 0.325 | 0.802 | 391535 | 25 | 70960 |
| medium | reference | pieces | 2.3525 | 0.428 | 0.490 | 362097 | 22 | 88256 |
| medium | production | spans | 12.9771 | 0.073 | 0.105 | 1997455 | 10 | 10192 |
| medium | fast | spans | 2.9389 | 0.337 | 0.408 | 452356 | 26 | 72448 |
| medium | reference | spans | 2.1139 | 0.478 | 0.528 | 325379 | 23 | 89744 |
| long | production | pieces | 16.1363 | 0.291 | 0.315 | 2254983 | 10 | 53160 |
| long | fast | pieces | 3.1484 | 1.431 | 2.034 | 439984 | 25 | 333900 |
| long | reference | pieces | 2.6225 | 1.785 | 1.880 | 366481 | 22 | 386100 |
| long | production | spans | 10.7361 | 0.451 | 0.603 | 1500324 | 10 | 53160 |
| long | fast | spans | 3.1669 | 1.468 | 1.617 | 442568 | 26 | 339388 |
| long | reference | spans | 2.4638 | 1.921 | 1.976 | 344312 | 23 | 391588 |
| multilingual | production | pieces | 14.4337 | 0.050 | 0.083 | 2205155 | 10 | 9904 |
| multilingual | fast | pieces | 3.2122 | 0.239 | 0.290 | 490758 | 25 | 55280 |
| multilingual | reference | pieces | 2.5512 | 0.301 | 0.363 | 389766 | 22 | 68472 |
| multilingual | production | spans | 12.8126 | 0.057 | 0.084 | 1957476 | 10 | 9904 |
| multilingual | fast | spans | 2.6166 | 0.293 | 0.375 | 399755 | 26 | 56416 |
| multilingual | reference | spans | 2.3294 | 0.328 | 0.405 | 355882 | 23 | 69608 |
| source_code | production | pieces | 5.0823 | 0.199 | 0.254 | 2129719 | 10 | 40152 |
| source_code | fast | pieces | 1.1155 | 0.923 | 1.031 | 467446 | 25 | 213740 |
| source_code | reference | pieces | 0.8450 | 1.205 | 1.438 | 354096 | 22 | 252884 |
| source_code | production | spans | 3.3500 | 0.246 | 0.568 | 1403829 | 10 | 40152 |
| source_code | fast | spans | 1.0242 | 1.014 | 1.077 | 429197 | 26 | 217500 |
| source_code | reference | spans | 0.5988 | 1.693 | 2.548 | 250908 | 23 | 256644 |
| punctuation_heavy | production | pieces | 3.0701 | 0.723 | 1.336 | 1920026 | 10 | 198152 |
| punctuation_heavy | fast | pieces | 0.6639 | 3.688 | 5.291 | 415197 | 25 | 848076 |
| punctuation_heavy | reference | pieces | 0.4884 | 5.007 | 6.134 | 305411 | 22 | 951932 |
| punctuation_heavy | production | spans | 3.0940 | 0.809 | 0.907 | 1934977 | 10 | 198152 |
| punctuation_heavy | fast | spans | 0.4650 | 5.368 | 7.032 | 290815 | 26 | 862492 |
| punctuation_heavy | reference | spans | 0.5239 | 4.880 | 5.207 | 327641 | 23 | 966348 |
| cross_word_heavy | production | pieces | 11.6728 | 0.311 | 0.360 | 2226487 | 10 | 92416 |
| cross_word_heavy | fast | pieces | 2.4029 | 1.554 | 1.630 | 458339 | 25 | 366988 |
| cross_word_heavy | reference | pieces | 1.6938 | 2.141 | 2.713 | 323082 | 22 | 422068 |
| cross_word_heavy | production | spans | 10.4128 | 0.356 | 0.406 | 1986148 | 10 | 92416 |
| cross_word_heavy | fast | spans | 2.2758 | 1.647 | 1.760 | 434095 | 26 | 373180 |
| cross_word_heavy | reference | spans | 1.7659 | 2.127 | 2.332 | 336830 | 23 | 428260 |
| pathological | production | pieces | 12.9033 | 0.762 | 0.793 | 2347366 | 10 | 199752 |
| pathological | fast | pieces | 2.3527 | 4.145 | 4.657 | 428002 | 25 | 957324 |
| pathological | reference | pieces | 1.8789 | 5.268 | 5.465 | 341813 | 22 | 1074124 |
| pathological | production | spans | 10.5751 | 0.939 | 1.031 | 1923822 | 10 | 199752 |
| pathological | fast | spans | 2.0651 | 4.637 | 5.692 | 375681 | 26 | 973564 |
| pathological | reference | spans | 1.7903 | 5.546 | 5.913 | 325689 | 23 | 1090364 |
| synthetic_1024_sparse | production | pieces | 7.0383 | 0.793 | 0.858 | 909010 | 10 | 128444 |
| synthetic_1024_sparse | fast | pieces | 2.1808 | 2.537 | 2.837 | 281650 | 25 | 474380 |
| synthetic_1024_sparse | reference | pieces | 1.3231 | 4.015 | 5.464 | 170876 | 23 | 504720 |
| synthetic_1024_medium | production | pieces | 8.4263 | 0.657 | 0.714 | 776913 | 10 | 126402 |
| synthetic_1024_medium | fast | pieces | 2.2649 | 2.475 | 2.557 | 208829 | 25 | 433330 |
| synthetic_1024_medium | reference | pieces | 1.4278 | 3.896 | 4.298 | 131649 | 22 | 529566 |
| synthetic_1024_dense | production | pieces | 6.4544 | 0.791 | 1.060 | 587079 | 10 | 126870 |
| synthetic_1024_dense | fast | pieces | 2.2515 | 2.474 | 2.769 | 204788 | 25 | 432062 |
| synthetic_1024_dense | reference | pieces | 1.3614 | 4.016 | 5.056 | 123825 | 22 | 530902 |
| synthetic_1024_hierarchical | production | pieces | 7.2592 | 0.750 | 0.874 | 330140 | 10 | 126870 |
| synthetic_1024_hierarchical | fast | pieces | 2.1164 | 2.594 | 3.014 | 96254 | 25 | 379364 |
| synthetic_1024_hierarchical | reference | pieces | 1.0740 | 4.898 | 6.889 | 48845 | 22 | 516566 |
| synthetic_4096_sparse | production | pieces | 8.1125 | 2.711 | 2.964 | 745659 | 10 | 544620 |
| synthetic_4096_sparse | fast | pieces | 1.8515 | 11.332 | 15.050 | 170176 | 25 | 1778916 |
| synthetic_4096_sparse | reference | pieces | 1.1578 | 18.328 | 26.029 | 106423 | 22 | 2150676 |
| synthetic_4096_dense | production | pieces | 7.1660 | 3.081 | 3.811 | 649464 | 10 | 546534 |
| synthetic_4096_dense | fast | pieces | 2.1731 | 10.237 | 11.380 | 196950 | 25 | 1772942 |
| synthetic_4096_dense | reference | pieces | 1.3003 | 16.567 | 21.255 | 117849 | 22 | 2157810 |

## Interpretation rules

- Exact parity gate passed before any timing data was accepted.
- Individual workloads remain visible; geomean summaries never hide a per-workload regression.
- No performance claim is based on one run or generalized beyond the measured environment.
- Correctness evidence lives in #110/#114; this benchmark makes no correctness claims.
