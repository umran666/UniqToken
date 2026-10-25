# Merge-Engine Benchmark Report (Issue #111)

- Generated: 2026-10-25T02:20:59.668167+00:00
- Commit: `cca0dcf899007b312fc2e50eeac82472c9149380` (dirty=False)
- Platform: Windows 10 / 3.10.11 / 16 logical CPUs / 16.85 GB
- Policy: workers=1, warmup=3, repetitions=7, peak RSS 42.37 MB
- Parity gate: exact 3-way (production/fast/reference), both output paths, passed before timing.

## Geometric means over text workloads (pieces path, primary metric)

| Path | Text-workload geomean MB/s | Cells |
|---|---|---|
| production | 8.2192 | 8 |
| fast | 1.8833 | 8 |
| reference | 1.4908 | 8 |

## Fast vs production throughput ratio (pieces path, per workload)

| Workload | production MB/s | fast MB/s | speedup |
|---|---|---|---|
| cross_word_heavy | 11.7699 | 2.3646 | 4.978x |
| long | 16.0246 | 3.4133 | 4.695x |
| medium | 14.1218 | 3.0689 | 4.602x |
| multilingual | 8.2146 | 2.8713 | 2.861x |
| pathological | 12.5423 | 2.2921 | 5.472x |
| punctuation_heavy | 3.4023 | 0.7199 | 4.726x |
| short | 4.2541 | 1.2638 | 3.366x |
| source_code | 5.2440 | 1.0669 | 4.915x |
| synthetic_1024_dense | 6.1018 | 2.2949 | 2.659x |
| synthetic_1024_hierarchical | 5.7184 | 2.1928 | 2.608x |
| synthetic_1024_medium | 6.7547 | 1.9619 | 3.443x |
| synthetic_1024_sparse | 7.0525 | 2.1801 | 3.235x |
| synthetic_4096_dense | 7.4335 | 2.1172 | 3.511x |
| synthetic_4096_sparse | 8.4452 | 2.0955 | 4.030x |

## Individual workload records

| Workload | Path | Output | MB/s | p50 ms | p95 ms | tokens/s | allocs/call | alloc peak B |
|---|---|---|---|---|---|---|---|---|
| short | production | pieces | 4.2541 | 0.007 | 0.030 | 966851 | 12 | 3134 |
| short | fast | pieces | 1.2638 | 0.028 | 0.075 | 287238 | 21 | 7072 |
| short | reference | pieces | 1.2495 | 0.030 | 0.062 | 283976 | 21 | 7936 |
| short | production | spans | 4.7458 | 0.007 | 0.022 | 1078582 | 10 | 1886 |
| short | fast | spans | 1.0834 | 0.029 | 0.095 | 246219 | 21 | 7256 |
| short | reference | spans | 1.1553 | 0.033 | 0.064 | 262566 | 21 | 8120 |
| medium | production | pieces | 14.1218 | 0.066 | 0.109 | 2173655 | 10 | 10192 |
| medium | fast | pieces | 3.0689 | 0.316 | 0.405 | 472363 | 25 | 70960 |
| medium | reference | pieces | 2.4393 | 0.406 | 0.471 | 375457 | 22 | 88256 |
| medium | production | spans | 13.1298 | 0.074 | 0.097 | 2020964 | 10 | 10192 |
| medium | fast | spans | 2.9500 | 0.329 | 0.425 | 454076 | 26 | 72448 |
| medium | reference | spans | 2.2537 | 0.438 | 0.504 | 346896 | 23 | 89744 |
| long | production | pieces | 16.0246 | 0.289 | 0.330 | 2239377 | 10 | 53160 |
| long | fast | pieces | 3.4133 | 1.392 | 1.447 | 476991 | 25 | 333900 |
| long | reference | pieces | 2.6731 | 1.789 | 1.813 | 373558 | 22 | 386100 |
| long | production | spans | 13.9799 | 0.329 | 0.383 | 1953640 | 10 | 53160 |
| long | fast | spans | 1.8741 | 2.176 | 4.180 | 261905 | 26 | 339388 |
| long | reference | spans | 2.1821 | 2.013 | 2.844 | 304944 | 23 | 391588 |
| multilingual | production | pieces | 8.2146 | 0.086 | 0.134 | 1255001 | 10 | 9904 |
| multilingual | fast | pieces | 2.8713 | 0.253 | 0.386 | 438678 | 25 | 55280 |
| multilingual | reference | pieces | 2.4887 | 0.304 | 0.378 | 380213 | 22 | 68472 |
| multilingual | production | spans | 12.5771 | 0.059 | 0.086 | 1921506 | 10 | 9904 |
| multilingual | fast | spans | 2.1999 | 0.276 | 0.585 | 336098 | 26 | 56416 |
| multilingual | reference | spans | 2.3254 | 0.330 | 0.425 | 355270 | 23 | 69608 |
| source_code | production | pieces | 5.2440 | 0.192 | 0.242 | 2197489 | 10 | 40152 |
| source_code | fast | pieces | 1.0669 | 0.953 | 1.065 | 447064 | 25 | 213740 |
| source_code | reference | pieces | 0.8351 | 1.183 | 1.460 | 349936 | 22 | 252884 |
| source_code | production | spans | 4.7410 | 0.213 | 0.255 | 1986712 | 10 | 40152 |
| source_code | fast | spans | 0.8014 | 1.002 | 2.607 | 335834 | 26 | 217500 |
| source_code | reference | spans | 0.8467 | 1.237 | 1.306 | 354826 | 23 | 256644 |
| punctuation_heavy | production | pieces | 3.4023 | 0.710 | 0.958 | 2127736 | 10 | 198152 |
| punctuation_heavy | fast | pieces | 0.7199 | 3.458 | 4.248 | 450248 | 25 | 848076 |
| punctuation_heavy | reference | pieces | 0.4133 | 5.708 | 9.362 | 258443 | 22 | 951932 |
| punctuation_heavy | production | spans | 2.7048 | 0.829 | 1.203 | 1691546 | 10 | 198152 |
| punctuation_heavy | fast | spans | 0.6315 | 3.851 | 5.126 | 394960 | 26 | 862492 |
| punctuation_heavy | reference | spans | 0.4996 | 4.680 | 7.559 | 312447 | 23 | 966348 |
| cross_word_heavy | production | pieces | 11.7699 | 0.313 | 0.366 | 2245007 | 10 | 92416 |
| cross_word_heavy | fast | pieces | 2.3646 | 1.608 | 1.696 | 451024 | 25 | 366988 |
| cross_word_heavy | reference | pieces | 1.8772 | 2.009 | 2.074 | 358068 | 22 | 422068 |
| cross_word_heavy | production | spans | 9.6573 | 0.369 | 0.477 | 1842038 | 10 | 92416 |
| cross_word_heavy | fast | spans | 1.6057 | 1.923 | 3.638 | 306267 | 26 | 373180 |
| cross_word_heavy | reference | spans | 1.8094 | 2.062 | 2.209 | 345129 | 23 | 428260 |
| pathological | production | pieces | 12.5423 | 0.780 | 0.817 | 2281686 | 10 | 199752 |
| pathological | fast | pieces | 2.2921 | 4.199 | 5.273 | 416980 | 25 | 957324 |
| pathological | reference | pieces | 1.8570 | 5.217 | 6.266 | 337833 | 22 | 1074124 |
| pathological | production | spans | 9.4077 | 1.036 | 1.202 | 1711443 | 10 | 199752 |
| pathological | fast | spans | 2.1825 | 4.439 | 5.130 | 397046 | 26 | 973564 |
| pathological | reference | spans | 1.7986 | 5.545 | 5.597 | 327206 | 23 | 1090364 |
| synthetic_1024_sparse | production | pieces | 7.0525 | 0.789 | 0.847 | 910848 | 10 | 128444 |
| synthetic_1024_sparse | fast | pieces | 2.1801 | 2.555 | 2.675 | 281563 | 25 | 474380 |
| synthetic_1024_sparse | reference | pieces | 1.4560 | 3.808 | 4.126 | 188050 | 23 | 504720 |
| synthetic_1024_medium | production | pieces | 6.7547 | 0.743 | 1.460 | 622793 | 10 | 126402 |
| synthetic_1024_medium | fast | pieces | 1.9619 | 2.954 | 3.457 | 180888 | 25 | 433330 |
| synthetic_1024_medium | reference | pieces | 1.4179 | 3.966 | 4.233 | 130733 | 22 | 529566 |
| synthetic_1024_dense | production | pieces | 6.1018 | 0.911 | 0.982 | 555005 | 10 | 126870 |
| synthetic_1024_dense | fast | pieces | 2.2949 | 2.446 | 2.530 | 208735 | 25 | 432062 |
| synthetic_1024_dense | reference | pieces | 1.2805 | 3.942 | 6.729 | 116474 | 22 | 530902 |
| synthetic_1024_hierarchical | production | pieces | 5.7184 | 0.817 | 1.398 | 260064 | 10 | 126870 |
| synthetic_1024_hierarchical | fast | pieces | 2.1928 | 2.565 | 2.725 | 99724 | 25 | 379364 |
| synthetic_1024_hierarchical | reference | pieces | 1.2871 | 4.339 | 4.620 | 58534 | 22 | 516566 |
| synthetic_4096_sparse | production | pieces | 8.4452 | 2.678 | 2.734 | 776241 | 10 | 544620 |
| synthetic_4096_sparse | fast | pieces | 2.0955 | 10.564 | 11.723 | 192604 | 25 | 1778916 |
| synthetic_4096_sparse | reference | pieces | 1.2917 | 16.546 | 22.873 | 118723 | 22 | 2150676 |
| synthetic_4096_dense | production | pieces | 7.4335 | 2.837 | 4.553 | 673706 | 10 | 546534 |
| synthetic_4096_dense | fast | pieces | 2.1172 | 10.668 | 11.128 | 191882 | 25 | 1772942 |
| synthetic_4096_dense | reference | pieces | 1.3460 | 16.167 | 21.638 | 121993 | 22 | 2157810 |

## Interpretation rules

- Exact parity gate passed before any timing data was accepted.
- Individual workloads remain visible; geomean summaries never hide a per-workload regression.
- No performance claim is based on one run or generalized beyond the measured environment.
- Correctness evidence lives in #110/#114; this benchmark makes no correctness claims.
