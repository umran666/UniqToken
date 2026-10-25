# Merge-Engine Benchmark Report (Issue #111)

- Generated: 2026-10-24T09:19:54.955928+00:00
- Commit: `cec462a039d44fb36b990efb5cffdf7f6f5c09af` (dirty=False)
- Platform: Windows 10 / 3.10.11 / 16 logical CPUs / 16.85 GB
- Policy: workers=1, warmup=3, repetitions=7, peak RSS 45.7 MB
- Parity gate: exact 3-way (production/fast/reference), both output paths, passed before timing.

## Geometric means over text workloads (pieces path, primary metric)

| Path | Text-workload geomean MB/s | Cells |
|---|---|---|
| production | 7.8398 | 8 |
| fast | 1.8163 | 8 |
| reference | 1.4214 | 8 |

## Fast vs production throughput ratio (pieces path, per workload)

| Workload | production MB/s | fast MB/s | speedup |
|---|---|---|---|
| cross_word_heavy | 11.8751 | 2.4439 | 4.859x |
| long | 10.2609 | 3.1922 | 3.214x |
| medium | 8.7920 | 2.3519 | 3.738x |
| multilingual | 13.5055 | 2.8993 | 4.658x |
| pathological | 12.7747 | 2.5537 | 5.002x |
| punctuation_heavy | 3.3733 | 0.7011 | 4.811x |
| short | 4.4573 | 1.1403 | 3.909x |
| source_code | 5.1352 | 1.0907 | 4.708x |
| synthetic_1024_dense | 7.9411 | 1.7934 | 4.428x |
| synthetic_1024_hierarchical | 7.0613 | 2.0317 | 3.476x |
| synthetic_1024_medium | 7.7416 | 2.1471 | 3.606x |
| synthetic_1024_sparse | 5.6020 | 2.0963 | 2.672x |
| synthetic_4096_dense | 2.5097 | 1.0809 | 2.322x |
| synthetic_4096_sparse | 3.6979 | 1.0266 | 3.602x |

## Individual workload records

| Workload | Path | Output | MB/s | p50 ms | p95 ms | tokens/s | allocs/call | alloc peak B |
|---|---|---|---|---|---|---|---|---|
| short | production | pieces | 4.4573 | 0.006 | 0.028 | 1013025 | 12 | 3134 |
| short | fast | pieces | 1.1403 | 0.035 | 0.075 | 259163 | 21 | 7072 |
| short | reference | pieces | 1.0732 | 0.034 | 0.073 | 243902 | 21 | 7936 |
| short | production | spans | 3.9186 | 0.007 | 0.034 | 890585 | 10 | 1886 |
| short | fast | spans | 1.2031 | 0.029 | 0.079 | 273437 | 21 | 7256 |
| short | reference | spans | 0.9837 | 0.034 | 0.079 | 223571 | 21 | 8120 |
| medium | production | pieces | 8.7920 | 0.078 | 0.253 | 1353282 | 10 | 10192 |
| medium | fast | pieces | 2.3519 | 0.356 | 0.699 | 362001 | 25 | 70960 |
| medium | reference | pieces | 2.4328 | 0.416 | 0.456 | 374459 | 22 | 88256 |
| medium | production | spans | 4.7300 | 0.210 | 0.364 | 728056 | 10 | 10192 |
| medium | fast | spans | 2.0015 | 0.384 | 0.844 | 308068 | 26 | 72448 |
| medium | reference | spans | 2.2268 | 0.434 | 0.554 | 342752 | 23 | 89744 |
| long | production | pieces | 10.2609 | 0.361 | 0.750 | 1433928 | 10 | 53160 |
| long | fast | pieces | 3.1922 | 1.490 | 1.592 | 446105 | 25 | 333900 |
| long | reference | pieces | 2.6827 | 1.748 | 1.829 | 374902 | 22 | 386100 |
| long | production | spans | 14.3938 | 0.321 | 0.373 | 2011477 | 10 | 53160 |
| long | fast | spans | 3.0369 | 1.475 | 2.165 | 424398 | 26 | 339388 |
| long | reference | spans | 2.4513 | 1.899 | 2.094 | 342561 | 23 | 391588 |
| multilingual | production | pieces | 13.5055 | 0.051 | 0.085 | 2063337 | 10 | 9904 |
| multilingual | fast | pieces | 2.8993 | 0.251 | 0.353 | 442945 | 25 | 55280 |
| multilingual | reference | pieces | 2.5228 | 0.302 | 0.389 | 385420 | 22 | 68472 |
| multilingual | production | spans | 9.7657 | 0.077 | 0.108 | 1491985 | 10 | 9904 |
| multilingual | fast | spans | 2.0784 | 0.319 | 0.647 | 317538 | 26 | 56416 |
| multilingual | reference | spans | 2.2808 | 0.329 | 0.409 | 348459 | 23 | 69608 |
| source_code | production | pieces | 5.1352 | 0.190 | 0.250 | 2151890 | 10 | 40152 |
| source_code | fast | pieces | 1.0907 | 0.940 | 1.086 | 457034 | 25 | 213740 |
| source_code | reference | pieces | 0.7756 | 1.356 | 1.440 | 325001 | 22 | 252884 |
| source_code | production | spans | 4.5843 | 0.225 | 0.267 | 1921038 | 10 | 40152 |
| source_code | fast | spans | 1.0479 | 0.986 | 1.117 | 439134 | 26 | 217500 |
| source_code | reference | spans | 0.8484 | 1.231 | 1.313 | 355506 | 23 | 256644 |
| punctuation_heavy | production | pieces | 3.3733 | 0.711 | 0.913 | 2109632 | 10 | 198152 |
| punctuation_heavy | fast | pieces | 0.7011 | 3.499 | 4.229 | 438467 | 25 | 848076 |
| punctuation_heavy | reference | pieces | 0.5646 | 4.569 | 4.944 | 353095 | 22 | 951932 |
| punctuation_heavy | production | spans | 3.1586 | 0.794 | 0.876 | 1975359 | 10 | 198152 |
| punctuation_heavy | fast | spans | 0.6871 | 3.609 | 4.001 | 429726 | 26 | 862492 |
| punctuation_heavy | reference | spans | 0.4933 | 4.712 | 6.755 | 308494 | 23 | 966348 |
| cross_word_heavy | production | pieces | 11.8751 | 0.309 | 0.364 | 2265057 | 10 | 92416 |
| cross_word_heavy | fast | pieces | 2.4439 | 1.519 | 1.641 | 466158 | 25 | 366988 |
| cross_word_heavy | reference | pieces | 1.1189 | 2.840 | 7.678 | 213412 | 22 | 422068 |
| cross_word_heavy | production | spans | 10.2332 | 0.368 | 0.400 | 1951889 | 10 | 92416 |
| cross_word_heavy | fast | spans | 1.9007 | 1.874 | 2.665 | 362541 | 26 | 373180 |
| cross_word_heavy | reference | spans | 1.8463 | 2.043 | 2.109 | 352169 | 23 | 428260 |
| pathological | production | pieces | 12.7747 | 0.764 | 0.831 | 2323957 | 10 | 199752 |
| pathological | fast | pieces | 2.5537 | 3.840 | 4.067 | 464569 | 25 | 957324 |
| pathological | reference | pieces | 1.9243 | 5.110 | 5.340 | 350065 | 22 | 1074124 |
| pathological | production | spans | 9.1681 | 0.988 | 1.256 | 1667857 | 10 | 199752 |
| pathological | fast | spans | 2.1894 | 4.446 | 5.372 | 398298 | 26 | 973564 |
| pathological | reference | spans | 1.7614 | 5.288 | 7.415 | 320429 | 23 | 1090364 |
| synthetic_1024_sparse | production | pieces | 5.6020 | 0.932 | 1.424 | 723517 | 10 | 128444 |
| synthetic_1024_sparse | fast | pieces | 2.0963 | 2.660 | 3.102 | 270743 | 25 | 474380 |
| synthetic_1024_sparse | reference | pieces | 1.2935 | 4.227 | 5.188 | 167055 | 23 | 504720 |
| synthetic_1024_medium | production | pieces | 7.7416 | 0.665 | 1.022 | 713781 | 10 | 126402 |
| synthetic_1024_medium | fast | pieces | 2.1471 | 2.572 | 3.071 | 197964 | 25 | 433330 |
| synthetic_1024_medium | reference | pieces | 1.2765 | 3.979 | 6.411 | 117694 | 22 | 529566 |
| synthetic_1024_dense | production | pieces | 7.9411 | 0.681 | 0.882 | 722304 | 10 | 126870 |
| synthetic_1024_dense | fast | pieces | 1.7934 | 2.586 | 5.209 | 163120 | 25 | 432062 |
| synthetic_1024_dense | reference | pieces | 1.4414 | 3.867 | 4.095 | 131109 | 22 | 530902 |
| synthetic_1024_hierarchical | production | pieces | 7.0613 | 0.771 | 0.901 | 321141 | 10 | 126870 |
| synthetic_1024_hierarchical | fast | pieces | 2.0317 | 2.667 | 3.206 | 92398 | 25 | 379364 |
| synthetic_1024_hierarchical | reference | pieces | 0.6150 | 9.591 | 10.280 | 27967 | 22 | 516566 |
| synthetic_4096_sparse | production | pieces | 3.6979 | 5.816 | 7.575 | 339893 | 10 | 544620 |
| synthetic_4096_sparse | fast | pieces | 1.0266 | 21.137 | 27.840 | 94360 | 25 | 1778916 |
| synthetic_4096_sparse | reference | pieces | 0.6344 | 35.551 | 40.379 | 58312 | 22 | 2150676 |
| synthetic_4096_dense | production | pieces | 2.5097 | 8.347 | 14.891 | 227458 | 10 | 546534 |
| synthetic_4096_dense | fast | pieces | 1.0809 | 21.077 | 23.187 | 97962 | 25 | 1772942 |
| synthetic_4096_dense | reference | pieces | 0.6778 | 32.715 | 38.580 | 61427 | 22 | 2157810 |

## Interpretation rules

- Exact parity gate passed before any timing data was accepted.
- Individual workloads remain visible; geomean summaries never hide a per-workload regression.
- No performance claim is based on one run or generalized beyond the measured environment.
- Correctness evidence lives in #110/#114; this benchmark makes no correctness claims.
