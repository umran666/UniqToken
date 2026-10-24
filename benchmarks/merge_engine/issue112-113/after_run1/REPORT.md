# Merge-Engine Benchmark Report (Issue #111)

- Generated: 2026-10-24T09:32:23.129730+00:00
- Commit: `6e6e4d33b969308a0f2770f447667ff9b9add0eb` (dirty=True)
- Platform: Windows 10 / 3.10.11 / 16 logical CPUs / 16.85 GB
- Policy: workers=1, warmup=3, repetitions=7, peak RSS 44.99 MB
- Parity gate: exact 3-way (production/fast/reference), both output paths, passed before timing.

## Geometric means over text workloads (pieces path, primary metric)

| Path | Text-workload geomean MB/s | Cells |
|---|---|---|
| production | 7.671 | 8 |
| fast | 1.9355 | 8 |
| reference | 1.6033 | 8 |

## Fast vs production throughput ratio (pieces path, per workload)

| Workload | production MB/s | fast MB/s | speedup |
|---|---|---|---|
| cross_word_heavy | 11.6216 | 2.4759 | 4.694x |
| long | 15.9106 | 2.5417 | 6.260x |
| medium | 7.8058 | 3.2634 | 2.392x |
| multilingual | 14.2666 | 2.9444 | 4.845x |
| pathological | 7.2076 | 2.5416 | 2.836x |
| punctuation_heavy | 3.4929 | 0.8064 | 4.331x |
| short | 4.3812 | 1.3473 | 3.252x |
| source_code | 5.2790 | 1.1793 | 4.476x |
| synthetic_1024_dense | 7.9445 | 2.2192 | 3.580x |
| synthetic_1024_hierarchical | 7.2495 | 3.0649 | 2.365x |
| synthetic_1024_medium | 7.7789 | 2.9750 | 2.615x |
| synthetic_1024_sparse | 6.8189 | 2.6684 | 2.555x |
| synthetic_4096_dense | 3.8414 | 1.3481 | 2.849x |
| synthetic_4096_sparse | 3.8956 | 1.3501 | 2.885x |

## Individual workload records

| Workload | Path | Output | MB/s | p50 ms | p95 ms | tokens/s | allocs/call | alloc peak B |
|---|---|---|---|---|---|---|---|---|
| short | production | pieces | 4.3812 | 0.007 | 0.029 | 995733 | 12 | 3134 |
| short | fast | pieces | 1.3473 | 0.026 | 0.066 | 306212 | 24 | 7643 |
| short | reference | pieces | 1.2232 | 0.031 | 0.064 | 277998 | 21 | 7936 |
| short | production | spans | 4.3750 | 0.007 | 0.026 | 994318 | 10 | 1886 |
| short | fast | spans | 1.1823 | 0.029 | 0.080 | 268714 | 24 | 7827 |
| short | reference | spans | 1.0349 | 0.036 | 0.079 | 235215 | 21 | 8120 |
| medium | production | pieces | 7.8058 | 0.120 | 0.185 | 1201487 | 10 | 10192 |
| medium | fast | pieces | 3.2634 | 0.299 | 0.390 | 502308 | 29 | 72998 |
| medium | reference | pieces | 2.4963 | 0.402 | 0.450 | 384239 | 22 | 88256 |
| medium | production | spans | 12.9700 | 0.073 | 0.110 | 1996367 | 10 | 10192 |
| medium | fast | spans | 3.0853 | 0.318 | 0.397 | 474894 | 30 | 74486 |
| medium | reference | spans | 2.3578 | 0.426 | 0.462 | 362922 | 23 | 89744 |
| long | production | pieces | 15.9106 | 0.290 | 0.344 | 2223450 | 10 | 53160 |
| long | fast | pieces | 2.5417 | 1.819 | 2.357 | 355188 | 29 | 340442 |
| long | reference | pieces | 2.7835 | 1.673 | 1.828 | 388980 | 22 | 386100 |
| long | production | spans | 12.5913 | 0.341 | 0.479 | 1759583 | 10 | 53160 |
| long | fast | spans | 3.0735 | 1.377 | 1.846 | 429516 | 30 | 345930 |
| long | reference | spans | 2.3214 | 1.933 | 2.591 | 324410 | 23 | 391588 |
| multilingual | production | pieces | 14.2666 | 0.050 | 0.083 | 2179619 | 10 | 9904 |
| multilingual | fast | pieces | 2.9444 | 0.265 | 0.303 | 449838 | 29 | 56930 |
| multilingual | reference | pieces | 2.5878 | 0.297 | 0.367 | 395351 | 22 | 68472 |
| multilingual | production | spans | 12.5288 | 0.057 | 0.088 | 1914124 | 10 | 9904 |
| multilingual | fast | spans | 2.9171 | 0.259 | 0.325 | 445672 | 30 | 58066 |
| multilingual | reference | spans | 2.4880 | 0.310 | 0.365 | 380110 | 23 | 69608 |
| source_code | production | pieces | 5.2790 | 0.194 | 0.225 | 2212167 | 10 | 40152 |
| source_code | fast | pieces | 1.1793 | 0.865 | 1.048 | 494176 | 29 | 218317 |
| source_code | reference | pieces | 0.9182 | 1.129 | 1.218 | 384755 | 22 | 252884 |
| source_code | production | spans | 4.7926 | 0.212 | 0.255 | 2008346 | 10 | 40152 |
| source_code | fast | spans | 1.0758 | 0.959 | 1.040 | 450800 | 30 | 222077 |
| source_code | reference | spans | 0.4652 | 2.341 | 2.364 | 194932 | 23 | 256644 |
| punctuation_heavy | production | pieces | 3.4929 | 0.708 | 0.911 | 2184430 | 10 | 198152 |
| punctuation_heavy | fast | pieces | 0.8064 | 3.167 | 3.250 | 504320 | 29 | 864486 |
| punctuation_heavy | reference | pieces | 0.6206 | 4.110 | 4.311 | 388090 | 22 | 951932 |
| punctuation_heavy | production | spans | 3.1917 | 0.787 | 0.868 | 1996082 | 10 | 198152 |
| punctuation_heavy | fast | spans | 0.7657 | 3.332 | 3.410 | 478878 | 30 | 878902 |
| punctuation_heavy | reference | spans | 0.5685 | 4.400 | 5.083 | 355514 | 23 | 966348 |
| cross_word_heavy | production | pieces | 11.6216 | 0.319 | 0.363 | 2216708 | 10 | 92416 |
| cross_word_heavy | fast | pieces | 2.4759 | 1.508 | 1.717 | 472256 | 29 | 374294 |
| cross_word_heavy | reference | pieces | 1.9332 | 1.975 | 2.009 | 368731 | 22 | 422068 |
| cross_word_heavy | production | spans | 10.2905 | 0.360 | 0.417 | 1962820 | 10 | 92416 |
| cross_word_heavy | fast | spans | 2.4182 | 1.560 | 1.618 | 461246 | 30 | 380486 |
| cross_word_heavy | reference | spans | 1.6953 | 2.106 | 2.997 | 323354 | 23 | 428260 |
| pathological | production | pieces | 7.2076 | 1.430 | 1.613 | 1311194 | 10 | 199752 |
| pathological | fast | pieces | 2.5416 | 3.744 | 5.058 | 462369 | 29 | 975758 |
| pathological | reference | pieces | 1.8028 | 5.421 | 6.672 | 327964 | 22 | 1074124 |
| pathological | production | spans | 10.5018 | 0.917 | 1.109 | 1910470 | 10 | 199752 |
| pathological | fast | spans | 1.7527 | 5.411 | 7.592 | 318857 | 30 | 991998 |
| pathological | reference | spans | 1.9708 | 4.984 | 5.147 | 358519 | 23 | 1090364 |
| synthetic_1024_sparse | production | pieces | 6.8189 | 0.814 | 0.909 | 880678 | 10 | 128444 |
| synthetic_1024_sparse | fast | pieces | 2.6684 | 2.101 | 2.237 | 344629 | 29 | 454637 |
| synthetic_1024_sparse | reference | pieces | 1.6975 | 3.300 | 3.406 | 219241 | 23 | 477356 |
| synthetic_1024_medium | production | pieces | 7.7789 | 0.713 | 0.828 | 717219 | 10 | 126402 |
| synthetic_1024_medium | fast | pieces | 2.9750 | 1.873 | 2.006 | 274294 | 29 | 392597 |
| synthetic_1024_medium | reference | pieces | 1.6554 | 3.344 | 3.637 | 152628 | 22 | 482980 |
| synthetic_1024_dense | production | pieces | 7.9445 | 0.694 | 0.757 | 722610 | 10 | 126870 |
| synthetic_1024_dense | fast | pieces | 2.2192 | 2.596 | 3.029 | 201852 | 29 | 390665 |
| synthetic_1024_dense | reference | pieces | 1.7027 | 3.322 | 3.427 | 154871 | 22 | 483652 |
| synthetic_1024_hierarchical | production | pieces | 7.2495 | 0.765 | 0.837 | 329697 | 10 | 126870 |
| synthetic_1024_hierarchical | fast | pieces | 3.0649 | 1.809 | 2.010 | 139389 | 29 | 357001 |
| synthetic_1024_hierarchical | reference | pieces | 1.5496 | 3.629 | 3.693 | 70474 | 22 | 469316 |
| synthetic_4096_sparse | production | pieces | 3.8956 | 5.746 | 6.199 | 358060 | 10 | 544620 |
| synthetic_4096_sparse | fast | pieces | 1.3501 | 16.554 | 17.729 | 124091 | 29 | 1615877 |
| synthetic_4096_sparse | reference | pieces | 0.6389 | 30.869 | 50.669 | 58729 | 22 | 1964696 |
| synthetic_4096_dense | production | pieces | 3.8414 | 5.911 | 5.951 | 348150 | 10 | 546534 |
| synthetic_4096_dense | fast | pieces | 1.3481 | 16.834 | 17.374 | 122183 | 29 | 1607177 |
| synthetic_4096_dense | reference | pieces | 0.8083 | 27.413 | 33.306 | 73259 | 22 | 1969104 |

## Interpretation rules

- Exact parity gate passed before any timing data was accepted.
- Individual workloads remain visible; geomean summaries never hide a per-workload regression.
- No performance claim is based on one run or generalized beyond the measured environment.
- Correctness evidence lives in #110/#114; this benchmark makes no correctness claims.
