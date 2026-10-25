# Merge-Engine Benchmark Report (Issue #111)

- Generated: 2026-10-25T02:21:04.471821+00:00
- Commit: `9614ca63f7239576321c2c3d443a0f6448bb42ff` (dirty=False)
- Platform: Windows 10 / 3.10.11 / 16 logical CPUs / 16.85 GB
- Policy: workers=1, warmup=3, repetitions=7, peak RSS 41.23 MB
- Parity gate: exact 3-way (production/fast/reference), both output paths, passed before timing.

## Geometric means over text workloads (pieces path, primary metric)

| Path | Text-workload geomean MB/s | Cells |
|---|---|---|
| production | 8.1141 | 8 |
| fast | 1.8491 | 8 |
| reference | 1.3715 | 8 |

## Fast vs production throughput ratio (pieces path, per workload)

| Workload | production MB/s | fast MB/s | speedup |
|---|---|---|---|
| cross_word_heavy | 9.8182 | 2.5184 | 3.899x |
| long | 16.2407 | 3.6874 | 4.404x |
| medium | 14.5477 | 2.9404 | 4.947x |
| multilingual | 9.6083 | 3.3960 | 2.829x |
| pathological | 12.9916 | 2.5147 | 5.166x |
| punctuation_heavy | 3.3453 | 0.7797 | 4.291x |
| short | 4.1678 | 0.6604 | 6.311x |
| source_code | 4.6540 | 1.1385 | 4.088x |
| synthetic_1024_dense | 6.2979 | 2.7957 | 2.253x |
| synthetic_1024_hierarchical | 7.4045 | 2.9261 | 2.531x |
| synthetic_1024_medium | 8.4854 | 2.5974 | 3.267x |
| synthetic_1024_sparse | 6.3156 | 1.9935 | 3.168x |
| synthetic_4096_dense | 7.4631 | 2.7792 | 2.685x |
| synthetic_4096_sparse | 7.4818 | 2.7059 | 2.765x |

## Individual workload records

| Workload | Path | Output | MB/s | p50 ms | p95 ms | tokens/s | allocs/call | alloc peak B |
|---|---|---|---|---|---|---|---|---|
| short | production | pieces | 4.1678 | 0.006 | 0.032 | 947226 | 12 | 3134 |
| short | fast | pieces | 0.6604 | 0.085 | 0.089 | 150086 | 24 | 7643 |
| short | reference | pieces | 1.1874 | 0.032 | 0.065 | 269854 | 21 | 7936 |
| short | production | spans | 4.9123 | 0.007 | 0.020 | 1116427 | 10 | 1886 |
| short | fast | spans | 1.4026 | 0.026 | 0.053 | 318761 | 24 | 7827 |
| short | reference | spans | 1.0822 | 0.033 | 0.070 | 245959 | 21 | 8120 |
| medium | production | pieces | 14.5477 | 0.063 | 0.110 | 2239201 | 10 | 10192 |
| medium | fast | pieces | 2.9404 | 0.305 | 0.470 | 452599 | 29 | 72998 |
| medium | reference | pieces | 2.4061 | 0.421 | 0.465 | 370358 | 22 | 88256 |
| medium | production | spans | 13.3708 | 0.071 | 0.106 | 2058052 | 10 | 10192 |
| medium | fast | spans | 2.7161 | 0.376 | 0.444 | 418061 | 30 | 74486 |
| medium | reference | spans | 2.1262 | 0.458 | 0.587 | 327268 | 23 | 89744 |
| long | production | pieces | 16.2407 | 0.287 | 0.330 | 2269584 | 10 | 53160 |
| long | fast | pieces | 3.6874 | 1.264 | 1.346 | 515296 | 29 | 340442 |
| long | reference | pieces | 2.6111 | 1.822 | 1.979 | 364894 | 22 | 386100 |
| long | production | spans | 12.3388 | 0.343 | 0.469 | 1724305 | 10 | 53160 |
| long | fast | spans | 2.3731 | 1.819 | 2.661 | 331628 | 30 | 345930 |
| long | reference | spans | 2.3798 | 1.945 | 2.320 | 332567 | 23 | 391588 |
| multilingual | production | pieces | 9.6083 | 0.073 | 0.151 | 1467938 | 10 | 9904 |
| multilingual | fast | pieces | 3.3960 | 0.223 | 0.274 | 518836 | 29 | 56930 |
| multilingual | reference | pieces | 1.9732 | 0.372 | 0.519 | 301456 | 22 | 68472 |
| multilingual | production | spans | 12.7360 | 0.057 | 0.092 | 1945785 | 10 | 9904 |
| multilingual | fast | spans | 3.1742 | 0.240 | 0.292 | 484942 | 30 | 58066 |
| multilingual | reference | spans | 1.3884 | 0.524 | 0.953 | 212111 | 23 | 69608 |
| source_code | production | pieces | 4.6540 | 0.191 | 0.370 | 1950231 | 10 | 40152 |
| source_code | fast | pieces | 1.1385 | 0.881 | 1.095 | 477083 | 29 | 218317 |
| source_code | reference | pieces | 0.8199 | 1.242 | 1.563 | 343577 | 22 | 252884 |
| source_code | production | spans | 4.7001 | 0.220 | 0.259 | 1969561 | 10 | 40152 |
| source_code | fast | spans | 0.8889 | 0.943 | 2.562 | 372494 | 30 | 222077 |
| source_code | reference | spans | 0.8779 | 1.192 | 1.243 | 367880 | 23 | 256644 |
| punctuation_heavy | production | pieces | 3.3453 | 0.695 | 0.936 | 2092146 | 10 | 198152 |
| punctuation_heavy | fast | pieces | 0.7797 | 3.206 | 3.929 | 487613 | 29 | 864486 |
| punctuation_heavy | reference | pieces | 0.4943 | 4.942 | 6.694 | 309131 | 22 | 951932 |
| punctuation_heavy | production | spans | 2.7030 | 0.857 | 1.410 | 1690449 | 10 | 198152 |
| punctuation_heavy | fast | spans | 0.7217 | 3.519 | 3.757 | 451326 | 30 | 878902 |
| punctuation_heavy | reference | spans | 0.5270 | 4.871 | 5.228 | 329556 | 23 | 966348 |
| cross_word_heavy | production | pieces | 9.8182 | 0.389 | 0.398 | 1872727 | 10 | 92416 |
| cross_word_heavy | fast | pieces | 2.5184 | 1.474 | 1.619 | 480369 | 29 | 374294 |
| cross_word_heavy | reference | pieces | 1.4952 | 2.543 | 3.118 | 285202 | 22 | 422068 |
| cross_word_heavy | production | spans | 10.3359 | 0.362 | 0.395 | 1971484 | 10 | 92416 |
| cross_word_heavy | fast | spans | 2.4221 | 1.557 | 1.635 | 461989 | 30 | 380486 |
| cross_word_heavy | reference | spans | 1.6741 | 2.248 | 2.550 | 319325 | 23 | 428260 |
| pathological | production | pieces | 12.9916 | 0.757 | 0.816 | 2363428 | 10 | 199752 |
| pathological | fast | pieces | 2.5147 | 3.817 | 4.418 | 457467 | 29 | 975758 |
| pathological | reference | pieces | 1.4036 | 6.612 | 9.637 | 255345 | 22 | 1074124 |
| pathological | production | spans | 10.9361 | 0.905 | 0.957 | 1989490 | 10 | 199752 |
| pathological | fast | spans | 2.4461 | 4.041 | 4.147 | 444988 | 30 | 991998 |
| pathological | reference | spans | 1.6749 | 5.556 | 7.751 | 304691 | 23 | 1090364 |
| synthetic_1024_sparse | production | pieces | 6.3156 | 0.870 | 0.973 | 815676 | 10 | 128444 |
| synthetic_1024_sparse | fast | pieces | 1.9935 | 2.756 | 3.219 | 257469 | 29 | 454637 |
| synthetic_1024_sparse | reference | pieces | 1.5245 | 3.606 | 4.181 | 196896 | 23 | 477356 |
| synthetic_1024_medium | production | pieces | 8.4854 | 0.657 | 0.708 | 782367 | 10 | 126402 |
| synthetic_1024_medium | fast | pieces | 2.5974 | 2.200 | 2.407 | 239484 | 29 | 392597 |
| synthetic_1024_medium | reference | pieces | 1.4655 | 3.605 | 4.896 | 135121 | 22 | 482980 |
| synthetic_1024_dense | production | pieces | 6.2979 | 0.914 | 0.934 | 572844 | 10 | 126870 |
| synthetic_1024_dense | fast | pieces | 2.7957 | 1.929 | 2.443 | 254289 | 29 | 390665 |
| synthetic_1024_dense | reference | pieces | 1.4859 | 3.683 | 4.179 | 135151 | 22 | 483652 |
| synthetic_1024_hierarchical | production | pieces | 7.4045 | 0.741 | 0.817 | 336747 | 10 | 126870 |
| synthetic_1024_hierarchical | fast | pieces | 2.9261 | 1.935 | 2.095 | 133074 | 29 | 357001 |
| synthetic_1024_hierarchical | reference | pieces | 1.3414 | 3.842 | 6.206 | 61005 | 22 | 469316 |
| synthetic_4096_sparse | production | pieces | 7.4818 | 2.780 | 4.631 | 687687 | 10 | 544620 |
| synthetic_4096_sparse | fast | pieces | 2.7059 | 7.869 | 11.355 | 248710 | 29 | 1615877 |
| synthetic_4096_sparse | reference | pieces | 1.4883 | 14.471 | 19.163 | 136796 | 22 | 1964696 |
| synthetic_4096_dense | production | pieces | 7.4631 | 2.873 | 3.542 | 676389 | 10 | 546534 |
| synthetic_4096_dense | fast | pieces | 2.7792 | 7.990 | 8.967 | 251882 | 29 | 1607177 |
| synthetic_4096_dense | reference | pieces | 1.4277 | 15.390 | 19.739 | 129397 | 22 | 1969104 |

## Interpretation rules

- Exact parity gate passed before any timing data was accepted.
- Individual workloads remain visible; geomean summaries never hide a per-workload regression.
- No performance claim is based on one run or generalized beyond the measured environment.
- Correctness evidence lives in #110/#114; this benchmark makes no correctness claims.
