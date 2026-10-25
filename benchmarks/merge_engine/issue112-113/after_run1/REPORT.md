# Merge-Engine Benchmark Report (Issue #111)

- Generated: 2026-10-25T02:20:54.019277+00:00
- Commit: `9614ca63f7239576321c2c3d443a0f6448bb42ff` (dirty=False)
- Platform: Windows 10 / 3.10.11 / 16 logical CPUs / 16.85 GB
- Policy: workers=1, warmup=3, repetitions=7, peak RSS 41.9 MB
- Parity gate: exact 3-way (production/fast/reference), both output paths, passed before timing.

## Geometric means over text workloads (pieces path, primary metric)

| Path | Text-workload geomean MB/s | Cells |
|---|---|---|
| production | 8.1191 | 8 |
| fast | 1.8513 | 8 |
| reference | 1.3595 | 8 |

## Fast vs production throughput ratio (pieces path, per workload)

| Workload | production MB/s | fast MB/s | speedup |
|---|---|---|---|
| cross_word_heavy | 11.9534 | 2.6209 | 4.561x |
| long | 16.0331 | 3.6966 | 4.337x |
| medium | 10.8182 | 1.3473 | 8.029x |
| multilingual | 14.3851 | 3.4100 | 4.218x |
| pathological | 12.6761 | 2.5004 | 5.070x |
| punctuation_heavy | 3.6699 | 0.7977 | 4.600x |
| short | 3.0435 | 1.3012 | 2.339x |
| source_code | 4.4716 | 1.1944 | 3.744x |
| synthetic_1024_dense | 8.4413 | 2.7963 | 3.019x |
| synthetic_1024_hierarchical | 5.6653 | 3.0898 | 1.834x |
| synthetic_1024_medium | 7.0538 | 2.9026 | 2.430x |
| synthetic_1024_sparse | 6.7202 | 2.7553 | 2.439x |
| synthetic_4096_dense | 8.4178 | 2.7936 | 3.013x |
| synthetic_4096_sparse | 7.9522 | 2.7682 | 2.873x |

## Individual workload records

| Workload | Path | Output | MB/s | p50 ms | p95 ms | tokens/s | allocs/call | alloc peak B |
|---|---|---|---|---|---|---|---|---|
| short | production | pieces | 3.0435 | 0.009 | 0.045 | 691700 | 12 | 3134 |
| short | fast | pieces | 1.3012 | 0.026 | 0.074 | 295733 | 24 | 7643 |
| short | reference | pieces | 0.7056 | 0.047 | 0.141 | 160367 | 21 | 7936 |
| short | production | spans | 3.7515 | 0.007 | 0.029 | 852619 | 10 | 1886 |
| short | fast | spans | 1.1579 | 0.029 | 0.067 | 263158 | 24 | 7827 |
| short | reference | spans | 0.9981 | 0.036 | 0.065 | 226831 | 21 | 8120 |
| medium | production | pieces | 10.8182 | 0.065 | 0.198 | 1665152 | 10 | 10192 |
| medium | fast | pieces | 1.3473 | 0.735 | 1.227 | 207382 | 29 | 72998 |
| medium | reference | pieces | 2.4411 | 0.412 | 0.448 | 375739 | 22 | 88256 |
| medium | production | spans | 12.8997 | 0.074 | 0.105 | 1985547 | 10 | 10192 |
| medium | fast | spans | 1.4568 | 0.668 | 0.855 | 224235 | 30 | 74486 |
| medium | reference | spans | 2.2590 | 0.447 | 0.507 | 347708 | 23 | 89744 |
| long | production | pieces | 16.0331 | 0.284 | 0.338 | 2240569 | 10 | 53160 |
| long | fast | pieces | 3.6966 | 1.265 | 1.382 | 516579 | 29 | 340442 |
| long | reference | pieces | 2.5270 | 1.765 | 2.382 | 353144 | 22 | 386100 |
| long | production | spans | 14.3788 | 0.325 | 0.371 | 2009380 | 10 | 53160 |
| long | fast | spans | 3.3755 | 1.398 | 1.479 | 471714 | 30 | 345930 |
| long | reference | spans | 2.1593 | 2.039 | 2.642 | 301754 | 23 | 391588 |
| multilingual | production | pieces | 14.3851 | 0.050 | 0.082 | 2197717 | 10 | 9904 |
| multilingual | fast | pieces | 3.4100 | 0.227 | 0.270 | 520974 | 29 | 56930 |
| multilingual | reference | pieces | 2.0892 | 0.393 | 0.476 | 319177 | 22 | 68472 |
| multilingual | production | spans | 11.2866 | 0.059 | 0.124 | 1724349 | 10 | 9904 |
| multilingual | fast | spans | 3.1054 | 0.248 | 0.310 | 474430 | 30 | 58066 |
| multilingual | reference | spans | 2.3661 | 0.326 | 0.403 | 361487 | 23 | 69608 |
| source_code | production | pieces | 4.4716 | 0.208 | 0.325 | 1873821 | 10 | 40152 |
| source_code | fast | pieces | 1.1944 | 0.867 | 0.966 | 500496 | 29 | 218317 |
| source_code | reference | pieces | 0.9056 | 1.160 | 1.222 | 379474 | 22 | 252884 |
| source_code | production | spans | 4.5601 | 0.223 | 0.287 | 1910907 | 10 | 40152 |
| source_code | fast | spans | 1.0564 | 0.932 | 1.248 | 442688 | 30 | 222077 |
| source_code | reference | spans | 0.7880 | 1.239 | 1.756 | 330213 | 23 | 256644 |
| punctuation_heavy | production | pieces | 3.6699 | 0.691 | 0.758 | 2295105 | 10 | 198152 |
| punctuation_heavy | fast | pieces | 0.7977 | 3.149 | 3.381 | 498903 | 29 | 864486 |
| punctuation_heavy | reference | pieces | 0.5742 | 4.362 | 4.874 | 359114 | 22 | 951932 |
| punctuation_heavy | production | spans | 2.9866 | 0.843 | 0.967 | 1867802 | 10 | 198152 |
| punctuation_heavy | fast | spans | 0.7335 | 3.504 | 3.659 | 458717 | 30 | 878902 |
| punctuation_heavy | reference | spans | 0.5431 | 4.711 | 4.823 | 339636 | 23 | 966348 |
| cross_word_heavy | production | pieces | 11.9534 | 0.314 | 0.350 | 2279996 | 10 | 92416 |
| cross_word_heavy | fast | pieces | 2.6209 | 1.431 | 1.524 | 499921 | 29 | 374294 |
| cross_word_heavy | reference | pieces | 1.2693 | 3.147 | 3.680 | 242113 | 22 | 422068 |
| cross_word_heavy | production | spans | 9.5492 | 0.360 | 0.563 | 1821430 | 10 | 92416 |
| cross_word_heavy | fast | spans | 2.4277 | 1.557 | 1.590 | 463057 | 30 | 380486 |
| cross_word_heavy | reference | spans | 1.6407 | 2.108 | 3.561 | 312955 | 23 | 428260 |
| pathological | production | pieces | 12.6761 | 0.741 | 0.991 | 2306018 | 10 | 199752 |
| pathological | fast | pieces | 2.5004 | 3.873 | 5.062 | 454864 | 29 | 975758 |
| pathological | reference | pieces | 1.9446 | 5.103 | 5.140 | 353763 | 22 | 1074124 |
| pathological | production | spans | 9.2195 | 0.919 | 2.018 | 1677199 | 10 | 199752 |
| pathological | fast | spans | 2.3676 | 4.081 | 4.791 | 430716 | 30 | 991998 |
| pathological | reference | spans | 1.7813 | 5.521 | 6.611 | 324049 | 23 | 1090364 |
| synthetic_1024_sparse | production | pieces | 6.7202 | 0.804 | 1.022 | 867926 | 10 | 128444 |
| synthetic_1024_sparse | fast | pieces | 2.7553 | 2.029 | 2.111 | 355849 | 29 | 454637 |
| synthetic_1024_sparse | reference | pieces | 1.6249 | 3.433 | 3.695 | 209857 | 23 | 477356 |
| synthetic_1024_medium | production | pieces | 7.0538 | 0.764 | 0.932 | 650364 | 10 | 126402 |
| synthetic_1024_medium | fast | pieces | 2.9026 | 1.924 | 2.073 | 267622 | 29 | 392597 |
| synthetic_1024_medium | reference | pieces | 1.4035 | 3.529 | 5.249 | 129406 | 22 | 482980 |
| synthetic_1024_dense | production | pieces | 8.4413 | 0.661 | 0.718 | 767797 | 10 | 126870 |
| synthetic_1024_dense | fast | pieces | 2.7963 | 1.958 | 2.439 | 254345 | 29 | 390665 |
| synthetic_1024_dense | reference | pieces | 1.5685 | 3.436 | 3.919 | 142670 | 22 | 483652 |
| synthetic_1024_hierarchical | production | pieces | 5.6653 | 0.802 | 1.913 | 257653 | 10 | 126870 |
| synthetic_1024_hierarchical | fast | pieces | 3.0898 | 1.786 | 2.068 | 140523 | 29 | 357001 |
| synthetic_1024_hierarchical | reference | pieces | 1.5244 | 3.667 | 3.893 | 69327 | 22 | 469316 |
| synthetic_4096_sparse | production | pieces | 7.9522 | 2.792 | 3.127 | 730926 | 10 | 544620 |
| synthetic_4096_sparse | fast | pieces | 2.7682 | 7.854 | 9.743 | 254442 | 29 | 1615877 |
| synthetic_4096_sparse | reference | pieces | 1.5204 | 14.072 | 19.473 | 139749 | 22 | 1964696 |
| synthetic_4096_dense | production | pieces | 8.4178 | 2.654 | 2.849 | 762918 | 10 | 546534 |
| synthetic_4096_dense | fast | pieces | 2.7936 | 7.861 | 9.387 | 253188 | 29 | 1607177 |
| synthetic_4096_dense | reference | pieces | 1.5308 | 13.860 | 19.959 | 138739 | 22 | 1969104 |

## Interpretation rules

- Exact parity gate passed before any timing data was accepted.
- Individual workloads remain visible; geomean summaries never hide a per-workload regression.
- No performance claim is based on one run or generalized beyond the measured environment.
- Correctness evidence lives in #110/#114; this benchmark makes no correctness claims.
