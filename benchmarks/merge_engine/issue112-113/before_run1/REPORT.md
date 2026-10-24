# Merge-Engine Benchmark Report (Issue #111)

- Generated: 2026-10-24T09:29:08.211022+00:00
- Commit: `6e6e4d33b969308a0f2770f447667ff9b9add0eb` (dirty=True)
- Platform: Windows 10 / 3.10.11 / 16 logical CPUs / 16.85 GB
- Policy: workers=1, warmup=3, repetitions=7, peak RSS 45.31 MB
- Parity gate: exact 3-way (production/fast/reference), both output paths, passed before timing.

## Geometric means over text workloads (pieces path, primary metric)

| Path | Text-workload geomean MB/s | Cells |
|---|---|---|
| production | 7.929 | 8 |
| fast | 1.8874 | 8 |
| reference | 1.3401 | 8 |

## Fast vs production throughput ratio (pieces path, per workload)

| Workload | production MB/s | fast MB/s | speedup |
|---|---|---|---|
| cross_word_heavy | 9.9012 | 2.6119 | 3.791x |
| long | 16.2423 | 3.3692 | 4.821x |
| medium | 13.9290 | 3.2316 | 4.310x |
| multilingual | 8.7459 | 2.3349 | 3.746x |
| pathological | 10.9177 | 2.6124 | 4.179x |
| punctuation_heavy | 3.6769 | 0.7370 | 4.989x |
| short | 4.4767 | 1.2291 | 3.642x |
| source_code | 4.4371 | 1.0248 | 4.330x |
| synthetic_1024_dense | 8.5666 | 3.0428 | 2.815x |
| synthetic_1024_hierarchical | 3.3000 | 1.2411 | 2.659x |
| synthetic_1024_medium | 8.3808 | 3.0937 | 2.709x |
| synthetic_1024_sparse | 6.7700 | 2.0027 | 3.380x |
| synthetic_4096_dense | 3.5557 | 1.3354 | 2.663x |
| synthetic_4096_sparse | 3.7342 | 1.4275 | 2.616x |

## Individual workload records

| Workload | Path | Output | MB/s | p50 ms | p95 ms | tokens/s | allocs/call | alloc peak B |
|---|---|---|---|---|---|---|---|---|
| short | production | pieces | 4.4767 | 0.006 | 0.027 | 1017442 | 12 | 3134 |
| short | fast | pieces | 1.2291 | 0.028 | 0.068 | 279330 | 24 | 7595 |
| short | reference | pieces | 1.0934 | 0.032 | 0.091 | 248491 | 21 | 7936 |
| short | production | spans | 4.3937 | 0.006 | 0.029 | 998573 | 10 | 1886 |
| short | fast | spans | 1.0799 | 0.033 | 0.079 | 245442 | 24 | 7779 |
| short | reference | spans | 1.0189 | 0.036 | 0.081 | 231558 | 21 | 8120 |
| medium | production | pieces | 13.9290 | 0.068 | 0.101 | 2143972 | 10 | 10192 |
| medium | fast | pieces | 3.2316 | 0.302 | 0.426 | 497420 | 29 | 72886 |
| medium | reference | pieces | 2.1421 | 0.432 | 0.631 | 329713 | 22 | 88256 |
| medium | production | spans | 12.8718 | 0.073 | 0.113 | 1981251 | 10 | 10192 |
| medium | fast | spans | 2.0126 | 0.491 | 0.753 | 309778 | 30 | 74374 |
| medium | reference | spans | 2.3456 | 0.426 | 0.476 | 361038 | 23 | 89744 |
| long | production | pieces | 16.2423 | 0.284 | 0.340 | 2269806 | 10 | 53160 |
| long | fast | pieces | 3.3692 | 1.367 | 1.611 | 470831 | 29 | 351674 |
| long | reference | pieces | 2.4627 | 1.838 | 2.361 | 344158 | 22 | 386100 |
| long | production | spans | 13.7011 | 0.332 | 0.435 | 1914674 | 10 | 53160 |
| long | fast | spans | 2.7987 | 1.650 | 1.954 | 391111 | 30 | 357162 |
| long | reference | spans | 2.2810 | 1.823 | 2.746 | 318765 | 23 | 391588 |
| multilingual | production | pieces | 8.7459 | 0.084 | 0.122 | 1336173 | 10 | 9904 |
| multilingual | fast | pieces | 2.3349 | 0.333 | 0.420 | 356722 | 29 | 56882 |
| multilingual | reference | pieces | 2.3075 | 0.334 | 0.403 | 352535 | 22 | 68472 |
| multilingual | production | spans | 12.2764 | 0.061 | 0.094 | 1875554 | 10 | 9904 |
| multilingual | fast | spans | 2.7339 | 0.254 | 0.442 | 417673 | 30 | 58018 |
| multilingual | reference | spans | 2.0955 | 0.368 | 0.439 | 320142 | 23 | 69608 |
| source_code | production | pieces | 4.4371 | 0.233 | 0.286 | 1859342 | 10 | 40152 |
| source_code | fast | pieces | 1.0248 | 0.874 | 1.272 | 429454 | 29 | 223313 |
| source_code | reference | pieces | 0.5333 | 2.091 | 2.714 | 223488 | 22 | 252884 |
| source_code | production | spans | 2.3436 | 0.430 | 0.574 | 982080 | 10 | 40152 |
| source_code | fast | spans | 0.6034 | 1.849 | 2.083 | 252842 | 30 | 227073 |
| source_code | reference | spans | 0.8410 | 1.215 | 1.448 | 352403 | 23 | 256644 |
| punctuation_heavy | production | pieces | 3.6769 | 0.687 | 0.747 | 2299485 | 10 | 198152 |
| punctuation_heavy | fast | pieces | 0.7370 | 3.402 | 4.003 | 460939 | 29 | 900630 |
| punctuation_heavy | reference | pieces | 0.4378 | 5.254 | 7.766 | 273804 | 22 | 951932 |
| punctuation_heavy | production | spans | 3.1644 | 0.792 | 0.904 | 1978986 | 10 | 198152 |
| punctuation_heavy | fast | spans | 0.7077 | 3.513 | 3.916 | 442599 | 30 | 915046 |
| punctuation_heavy | reference | spans | 0.4649 | 5.499 | 7.010 | 290733 | 23 | 966348 |
| cross_word_heavy | production | pieces | 9.9012 | 0.370 | 0.514 | 1888565 | 10 | 92416 |
| cross_word_heavy | fast | pieces | 2.6119 | 1.440 | 1.516 | 498189 | 29 | 386982 |
| cross_word_heavy | reference | pieces | 1.9596 | 1.929 | 1.988 | 373769 | 22 | 422068 |
| cross_word_heavy | production | spans | 7.7418 | 0.524 | 0.606 | 1476681 | 10 | 92416 |
| cross_word_heavy | fast | spans | 2.0801 | 1.742 | 2.595 | 396764 | 30 | 393174 |
| cross_word_heavy | reference | spans | 1.7069 | 2.159 | 2.464 | 325573 | 23 | 428260 |
| pathological | production | pieces | 10.9177 | 0.761 | 1.496 | 1986136 | 10 | 199752 |
| pathological | fast | pieces | 2.6124 | 3.673 | 4.451 | 475248 | 29 | 1017278 |
| pathological | reference | pieces | 1.7080 | 5.591 | 7.130 | 310712 | 22 | 1074124 |
| pathological | production | spans | 11.5799 | 0.847 | 0.912 | 2106609 | 10 | 199752 |
| pathological | fast | spans | 2.5349 | 3.833 | 4.141 | 461139 | 30 | 1033518 |
| pathological | reference | spans | 1.6270 | 6.327 | 7.058 | 295975 | 23 | 1090364 |
| synthetic_1024_sparse | production | pieces | 6.7700 | 0.815 | 0.909 | 874369 | 10 | 128444 |
| synthetic_1024_sparse | fast | pieces | 2.0027 | 2.546 | 4.379 | 258660 | 29 | 487701 |
| synthetic_1024_sparse | reference | pieces | 1.6601 | 3.394 | 3.468 | 214404 | 23 | 477356 |
| synthetic_1024_medium | production | pieces | 8.3808 | 0.664 | 0.733 | 772716 | 10 | 126402 |
| synthetic_1024_medium | fast | pieces | 3.0937 | 1.808 | 1.923 | 285243 | 29 | 431977 |
| synthetic_1024_medium | reference | pieces | 1.6826 | 3.349 | 3.427 | 155140 | 22 | 482980 |
| synthetic_1024_dense | production | pieces | 8.5666 | 0.642 | 0.729 | 779198 | 10 | 126870 |
| synthetic_1024_dense | fast | pieces | 3.0428 | 1.844 | 1.962 | 276770 | 29 | 430185 |
| synthetic_1024_dense | reference | pieces | 1.6078 | 3.430 | 4.073 | 146237 | 22 | 483652 |
| synthetic_1024_hierarchical | production | pieces | 3.3000 | 1.676 | 1.991 | 150080 | 10 | 126870 |
| synthetic_1024_hierarchical | fast | pieces | 1.2411 | 4.488 | 5.246 | 56444 | 29 | 396313 |
| synthetic_1024_hierarchical | reference | pieces | 0.7211 | 7.666 | 8.285 | 32793 | 22 | 469316 |
| synthetic_4096_sparse | production | pieces | 3.7342 | 5.441 | 7.757 | 343226 | 10 | 544620 |
| synthetic_4096_sparse | fast | pieces | 1.4275 | 15.594 | 17.043 | 131210 | 29 | 1781881 |
| synthetic_4096_sparse | reference | pieces | 0.7709 | 27.209 | 38.971 | 70861 | 22 | 1964696 |
| synthetic_4096_dense | production | pieces | 3.5557 | 6.474 | 7.521 | 322257 | 10 | 546534 |
| synthetic_4096_dense | fast | pieces | 1.3354 | 16.852 | 18.295 | 121030 | 29 | 1773993 |
| synthetic_4096_dense | reference | pieces | 0.7264 | 30.354 | 35.829 | 65837 | 22 | 1969104 |

## Interpretation rules

- Exact parity gate passed before any timing data was accepted.
- Individual workloads remain visible; geomean summaries never hide a per-workload regression.
- No performance claim is based on one run or generalized beyond the measured environment.
- Correctness evidence lives in #110/#114; this benchmark makes no correctness claims.
