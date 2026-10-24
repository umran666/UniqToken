# Merge-Engine Benchmark Report (Issue #111)

- Generated: 2026-10-24T09:32:17.225618+00:00
- Commit: `6e6e4d33b969308a0f2770f447667ff9b9add0eb` (dirty=True)
- Platform: Windows 10 / 3.10.11 / 16 logical CPUs / 16.85 GB
- Policy: workers=1, warmup=3, repetitions=7, peak RSS 45.65 MB
- Parity gate: exact 3-way (production/fast/reference), both output paths, passed before timing.

## Geometric means over text workloads (pieces path, primary metric)

| Path | Text-workload geomean MB/s | Cells |
|---|---|---|
| production | 8.6531 | 8 |
| fast | 1.7693 | 8 |
| reference | 1.5135 | 8 |

## Fast vs production throughput ratio (pieces path, per workload)

| Workload | production MB/s | fast MB/s | speedup |
|---|---|---|---|
| cross_word_heavy | 11.6231 | 2.3233 | 5.003x |
| long | 13.3794 | 3.4094 | 3.924x |
| medium | 13.9562 | 3.0496 | 4.576x |
| multilingual | 13.9192 | 1.6621 | 8.374x |
| pathological | 12.5507 | 2.3028 | 5.450x |
| punctuation_heavy | 3.5424 | 0.7064 | 5.014x |
| short | 4.6596 | 1.3333 | 3.495x |
| source_code | 5.0226 | 1.1027 | 4.555x |
| synthetic_1024_dense | 8.3384 | 2.2438 | 3.716x |
| synthetic_1024_hierarchical | 7.3262 | 1.0517 | 6.966x |
| synthetic_1024_medium | 8.3326 | 2.2980 | 3.626x |
| synthetic_1024_sparse | 7.0077 | 2.1999 | 3.185x |
| synthetic_4096_dense | 3.7450 | 1.1437 | 3.274x |
| synthetic_4096_sparse | 3.8283 | 0.9995 | 3.830x |

## Individual workload records

| Workload | Path | Output | MB/s | p50 ms | p95 ms | tokens/s | allocs/call | alloc peak B |
|---|---|---|---|---|---|---|---|---|
| short | production | pieces | 4.6596 | 0.006 | 0.025 | 1059002 | 12 | 3134 |
| short | fast | pieces | 1.3333 | 0.027 | 0.064 | 303030 | 21 | 7072 |
| short | reference | pieces | 1.1698 | 0.032 | 0.064 | 265856 | 21 | 7936 |
| short | production | spans | 3.9691 | 0.007 | 0.025 | 902062 | 10 | 1886 |
| short | fast | spans | 1.2116 | 0.030 | 0.070 | 275374 | 21 | 7256 |
| short | reference | spans | 1.1257 | 0.032 | 0.074 | 255848 | 21 | 8120 |
| medium | production | pieces | 13.9562 | 0.067 | 0.102 | 2148163 | 10 | 10192 |
| medium | fast | pieces | 3.0496 | 0.319 | 0.408 | 469397 | 25 | 70960 |
| medium | reference | pieces | 2.3672 | 0.430 | 0.477 | 364366 | 22 | 88256 |
| medium | production | spans | 10.0990 | 0.094 | 0.142 | 1554455 | 10 | 10192 |
| medium | fast | spans | 2.9336 | 0.330 | 0.427 | 451539 | 26 | 72448 |
| medium | reference | spans | 2.0433 | 0.519 | 0.545 | 314503 | 23 | 89744 |
| long | production | pieces | 13.3794 | 0.352 | 0.410 | 1869722 | 10 | 53160 |
| long | fast | pieces | 3.4094 | 1.379 | 1.462 | 476450 | 25 | 333900 |
| long | reference | pieces | 2.5911 | 1.819 | 2.002 | 362090 | 22 | 386100 |
| long | production | spans | 10.0003 | 0.456 | 0.638 | 1397505 | 10 | 53160 |
| long | fast | spans | 3.1894 | 1.462 | 1.553 | 445705 | 26 | 339388 |
| long | reference | spans | 2.5394 | 1.863 | 1.959 | 354878 | 23 | 391588 |
| multilingual | production | pieces | 13.9192 | 0.053 | 0.080 | 2126538 | 10 | 9904 |
| multilingual | fast | pieces | 1.6621 | 0.478 | 0.504 | 253935 | 25 | 55280 |
| multilingual | reference | pieces | 2.5170 | 0.306 | 0.374 | 384546 | 22 | 68472 |
| multilingual | production | spans | 11.5332 | 0.058 | 0.107 | 1762014 | 10 | 9904 |
| multilingual | fast | spans | 2.9442 | 0.266 | 0.314 | 449814 | 26 | 56416 |
| multilingual | reference | spans | 2.3782 | 0.323 | 0.406 | 363332 | 23 | 69608 |
| source_code | production | pieces | 5.0226 | 0.201 | 0.244 | 2104688 | 10 | 40152 |
| source_code | fast | pieces | 1.1027 | 0.935 | 1.017 | 462067 | 25 | 213740 |
| source_code | reference | pieces | 0.8700 | 1.181 | 1.316 | 364566 | 22 | 252884 |
| source_code | production | spans | 4.6774 | 0.221 | 0.255 | 1960036 | 10 | 40152 |
| source_code | fast | spans | 1.0654 | 0.981 | 1.028 | 446474 | 26 | 217500 |
| source_code | reference | spans | 0.8297 | 1.255 | 1.319 | 347665 | 23 | 256644 |
| punctuation_heavy | production | pieces | 3.5424 | 0.706 | 0.787 | 2215391 | 10 | 198152 |
| punctuation_heavy | fast | pieces | 0.7064 | 3.564 | 3.945 | 441803 | 25 | 848076 |
| punctuation_heavy | reference | pieces | 0.5850 | 4.335 | 4.610 | 365878 | 22 | 951932 |
| punctuation_heavy | production | spans | 3.1932 | 0.789 | 0.881 | 1997006 | 10 | 198152 |
| punctuation_heavy | fast | spans | 0.6273 | 3.929 | 4.799 | 392336 | 26 | 862492 |
| punctuation_heavy | reference | spans | 0.5523 | 4.646 | 4.772 | 345403 | 23 | 966348 |
| cross_word_heavy | production | pieces | 11.6231 | 0.316 | 0.374 | 2217000 | 10 | 92416 |
| cross_word_heavy | fast | pieces | 2.3233 | 1.584 | 1.905 | 443147 | 25 | 366988 |
| cross_word_heavy | reference | pieces | 1.8902 | 2.004 | 2.041 | 360536 | 22 | 422068 |
| cross_word_heavy | production | spans | 10.5385 | 0.351 | 0.393 | 2010116 | 10 | 92416 |
| cross_word_heavy | fast | spans | 1.9888 | 1.712 | 2.894 | 379342 | 26 | 373180 |
| cross_word_heavy | reference | spans | 1.7989 | 2.108 | 2.139 | 343116 | 23 | 428260 |
| pathological | production | pieces | 12.5507 | 0.767 | 0.850 | 2283215 | 10 | 199752 |
| pathological | fast | pieces | 2.3028 | 4.217 | 4.794 | 418915 | 25 | 957324 |
| pathological | reference | pieces | 1.5845 | 5.026 | 12.099 | 288252 | 22 | 1074124 |
| pathological | production | spans | 11.0241 | 0.892 | 0.932 | 2005504 | 10 | 199752 |
| pathological | fast | spans | 2.3634 | 4.150 | 4.386 | 429952 | 26 | 973564 |
| pathological | reference | spans | 1.4467 | 6.754 | 8.634 | 263178 | 23 | 1090364 |
| synthetic_1024_sparse | production | pieces | 7.0077 | 0.797 | 0.861 | 905065 | 10 | 128444 |
| synthetic_1024_sparse | fast | pieces | 2.1999 | 2.540 | 2.679 | 284124 | 25 | 474380 |
| synthetic_1024_sparse | reference | pieces | 1.1694 | 4.425 | 6.593 | 151030 | 23 | 504720 |
| synthetic_1024_medium | production | pieces | 8.3326 | 0.663 | 0.754 | 768271 | 10 | 126402 |
| synthetic_1024_medium | fast | pieces | 2.2980 | 2.459 | 2.536 | 211881 | 25 | 433330 |
| synthetic_1024_medium | reference | pieces | 1.4170 | 3.967 | 4.135 | 130653 | 22 | 529566 |
| synthetic_1024_dense | production | pieces | 8.3384 | 0.668 | 0.734 | 758438 | 10 | 126870 |
| synthetic_1024_dense | fast | pieces | 2.2438 | 2.482 | 2.654 | 204090 | 25 | 432062 |
| synthetic_1024_dense | reference | pieces | 1.4430 | 3.857 | 4.232 | 131249 | 22 | 530902 |
| synthetic_1024_hierarchical | production | pieces | 7.3262 | 0.761 | 0.839 | 333185 | 10 | 126870 |
| synthetic_1024_hierarchical | fast | pieces | 1.0517 | 5.211 | 6.299 | 47830 | 25 | 379364 |
| synthetic_1024_hierarchical | reference | pieces | 0.6219 | 9.137 | 9.459 | 28284 | 22 | 516566 |
| synthetic_4096_sparse | production | pieces | 3.8283 | 5.587 | 8.060 | 351874 | 10 | 544620 |
| synthetic_4096_sparse | fast | pieces | 0.9995 | 22.178 | 27.012 | 91870 | 25 | 1778916 |
| synthetic_4096_sparse | reference | pieces | 0.6446 | 33.311 | 41.555 | 59247 | 22 | 2150676 |
| synthetic_4096_dense | production | pieces | 3.7450 | 6.128 | 6.553 | 339412 | 10 | 546534 |
| synthetic_4096_dense | fast | pieces | 1.1437 | 20.849 | 21.559 | 103655 | 25 | 1772942 |
| synthetic_4096_dense | reference | pieces | 0.5847 | 36.424 | 50.564 | 52990 | 22 | 2157810 |

## Interpretation rules

- Exact parity gate passed before any timing data was accepted.
- Individual workloads remain visible; geomean summaries never hide a per-workload regression.
- No performance claim is based on one run or generalized beyond the measured environment.
- Correctness evidence lives in #110/#114; this benchmark makes no correctness claims.
