# Merge-Engine Benchmark Report (Issue #111)

- Generated: 2026-10-24T09:30:54.618351+00:00
- Commit: `6e6e4d33b969308a0f2770f447667ff9b9add0eb` (dirty=True)
- Platform: Windows 10 / 3.10.11 / 16 logical CPUs / 16.85 GB
- Policy: workers=1, warmup=3, repetitions=7, peak RSS 45.51 MB
- Parity gate: exact 3-way (production/fast/reference), both output paths, passed before timing.

## Geometric means over text workloads (pieces path, primary metric)

| Path | Text-workload geomean MB/s | Cells |
|---|---|---|
| production | 7.9612 | 8 |
| fast | 1.7698 | 8 |
| reference | 1.3202 | 8 |

## Fast vs production throughput ratio (pieces path, per workload)

| Workload | production MB/s | fast MB/s | speedup |
|---|---|---|---|
| cross_word_heavy | 11.7407 | 2.2432 | 5.234x |
| long | 13.5475 | 2.9902 | 4.531x |
| medium | 14.4505 | 3.0675 | 4.711x |
| multilingual | 14.9233 | 2.5742 | 5.797x |
| pathological | 9.0468 | 2.4201 | 3.738x |
| punctuation_heavy | 3.5049 | 0.5838 | 6.004x |
| short | 4.3441 | 1.1920 | 3.645x |
| source_code | 3.4154 | 1.0790 | 3.165x |
| synthetic_1024_dense | 3.9737 | 1.0819 | 3.673x |
| synthetic_1024_hierarchical | 3.2312 | 1.0891 | 2.967x |
| synthetic_1024_medium | 4.1166 | 1.0008 | 4.113x |
| synthetic_1024_sparse | 5.5354 | 1.8549 | 2.984x |
| synthetic_4096_dense | 3.8921 | 1.0825 | 3.595x |
| synthetic_4096_sparse | 3.3440 | 1.0917 | 3.063x |

## Individual workload records

| Workload | Path | Output | MB/s | p50 ms | p95 ms | tokens/s | allocs/call | alloc peak B |
|---|---|---|---|---|---|---|---|---|
| short | production | pieces | 4.3441 | 0.006 | 0.030 | 987306 | 12 | 3134 |
| short | fast | pieces | 1.1920 | 0.028 | 0.075 | 270898 | 21 | 7072 |
| short | reference | pieces | 1.1729 | 0.031 | 0.073 | 266565 | 21 | 7936 |
| short | production | spans | 4.0633 | 0.007 | 0.032 | 923483 | 10 | 1886 |
| short | fast | spans | 1.1920 | 0.030 | 0.074 | 270898 | 21 | 7256 |
| short | reference | spans | 1.1119 | 0.033 | 0.078 | 252708 | 21 | 8120 |
| medium | production | pieces | 14.4505 | 0.065 | 0.102 | 2224246 | 10 | 10192 |
| medium | fast | pieces | 3.0675 | 0.315 | 0.406 | 472160 | 25 | 70960 |
| medium | reference | pieces | 2.0513 | 0.509 | 0.550 | 315741 | 22 | 88256 |
| medium | production | spans | 12.0486 | 0.074 | 0.128 | 1854539 | 10 | 10192 |
| medium | fast | spans | 2.8768 | 0.341 | 0.433 | 442806 | 26 | 72448 |
| medium | reference | spans | 2.2056 | 0.442 | 0.522 | 339491 | 23 | 89744 |
| long | production | pieces | 13.5475 | 0.346 | 0.372 | 1893208 | 10 | 53160 |
| long | fast | pieces | 2.9902 | 1.523 | 1.966 | 417867 | 25 | 333900 |
| long | reference | pieces | 2.3483 | 1.795 | 2.620 | 328161 | 22 | 386100 |
| long | production | spans | 14.3713 | 0.323 | 0.370 | 2008334 | 10 | 53160 |
| long | fast | spans | 2.9909 | 1.561 | 1.959 | 417965 | 26 | 339388 |
| long | reference | spans | 2.1750 | 2.098 | 2.589 | 303950 | 23 | 391588 |
| multilingual | production | pieces | 14.9233 | 0.049 | 0.075 | 2279946 | 10 | 9904 |
| multilingual | fast | pieces | 2.5742 | 0.285 | 0.487 | 393277 | 25 | 55280 |
| multilingual | reference | pieces | 2.5101 | 0.306 | 0.377 | 383483 | 22 | 68472 |
| multilingual | production | spans | 9.5258 | 0.059 | 0.161 | 1455326 | 10 | 9904 |
| multilingual | fast | spans | 1.4187 | 0.535 | 0.700 | 216746 | 26 | 56416 |
| multilingual | reference | spans | 1.3223 | 0.642 | 0.785 | 202018 | 23 | 69608 |
| source_code | production | pieces | 3.4154 | 0.320 | 0.334 | 1431227 | 10 | 40152 |
| source_code | fast | pieces | 1.0790 | 0.947 | 1.083 | 452157 | 25 | 213740 |
| source_code | reference | pieces | 0.8620 | 1.188 | 1.435 | 361214 | 22 | 252884 |
| source_code | production | spans | 4.6869 | 0.216 | 0.268 | 1964035 | 10 | 40152 |
| source_code | fast | spans | 0.7794 | 1.185 | 1.801 | 326621 | 26 | 217500 |
| source_code | reference | spans | 0.5750 | 1.808 | 2.248 | 240939 | 23 | 256644 |
| punctuation_heavy | production | pieces | 3.5049 | 0.712 | 0.856 | 2191907 | 10 | 198152 |
| punctuation_heavy | fast | pieces | 0.5838 | 4.424 | 4.935 | 365076 | 25 | 848076 |
| punctuation_heavy | reference | pieces | 0.3491 | 7.208 | 9.826 | 218332 | 22 | 951932 |
| punctuation_heavy | production | spans | 1.9762 | 1.375 | 1.603 | 1235898 | 10 | 198152 |
| punctuation_heavy | fast | spans | 0.6342 | 3.862 | 4.754 | 396625 | 26 | 862492 |
| punctuation_heavy | reference | spans | 0.5452 | 4.628 | 4.889 | 340956 | 23 | 966348 |
| cross_word_heavy | production | pieces | 11.7407 | 0.314 | 0.365 | 2239428 | 10 | 92416 |
| cross_word_heavy | fast | pieces | 2.2432 | 1.700 | 1.855 | 427864 | 25 | 366988 |
| cross_word_heavy | reference | pieces | 1.5368 | 2.331 | 3.187 | 293125 | 22 | 422068 |
| cross_word_heavy | production | spans | 10.1950 | 0.357 | 0.451 | 1944594 | 10 | 92416 |
| cross_word_heavy | fast | spans | 2.0561 | 1.780 | 2.209 | 392174 | 26 | 373180 |
| cross_word_heavy | reference | spans | 1.5837 | 2.099 | 3.245 | 302067 | 23 | 428260 |
| pathological | production | pieces | 9.0468 | 0.978 | 1.470 | 1645779 | 10 | 199752 |
| pathological | fast | pieces | 2.4201 | 4.019 | 4.391 | 440256 | 25 | 957324 |
| pathological | reference | pieces | 1.4072 | 6.572 | 10.026 | 255995 | 22 | 1074124 |
| pathological | production | spans | 11.0496 | 0.875 | 0.957 | 2010141 | 10 | 199752 |
| pathological | fast | spans | 2.0457 | 4.614 | 6.252 | 372149 | 26 | 973564 |
| pathological | reference | spans | 1.7864 | 5.387 | 6.435 | 324977 | 23 | 1090364 |
| synthetic_1024_sparse | production | pieces | 5.5354 | 0.829 | 1.418 | 714908 | 10 | 128444 |
| synthetic_1024_sparse | fast | pieces | 1.8549 | 2.688 | 3.783 | 239570 | 25 | 474380 |
| synthetic_1024_sparse | reference | pieces | 1.2484 | 4.011 | 6.104 | 161236 | 23 | 504720 |
| synthetic_1024_medium | production | pieces | 4.1166 | 1.357 | 1.735 | 379556 | 10 | 126402 |
| synthetic_1024_medium | fast | pieces | 1.0008 | 5.314 | 6.769 | 92278 | 25 | 433330 |
| synthetic_1024_medium | reference | pieces | 0.6755 | 8.563 | 8.754 | 62284 | 22 | 529566 |
| synthetic_1024_dense | production | pieces | 3.9737 | 1.404 | 1.489 | 361440 | 10 | 126870 |
| synthetic_1024_dense | fast | pieces | 1.0819 | 5.193 | 5.633 | 98407 | 25 | 432062 |
| synthetic_1024_dense | reference | pieces | 0.7320 | 7.706 | 8.230 | 66577 | 22 | 530902 |
| synthetic_1024_hierarchical | production | pieces | 3.2312 | 1.718 | 2.048 | 146953 | 10 | 126870 |
| synthetic_1024_hierarchical | fast | pieces | 1.0891 | 5.060 | 5.818 | 49533 | 25 | 379364 |
| synthetic_1024_hierarchical | reference | pieces | 0.5958 | 9.036 | 12.087 | 27095 | 22 | 516566 |
| synthetic_4096_sparse | production | pieces | 3.3440 | 6.583 | 8.520 | 307359 | 10 | 544620 |
| synthetic_4096_sparse | fast | pieces | 1.0917 | 20.607 | 21.613 | 100340 | 25 | 1778916 |
| synthetic_4096_sparse | reference | pieces | 0.6292 | 36.457 | 43.151 | 57835 | 22 | 2150676 |
| synthetic_4096_dense | production | pieces | 3.8921 | 5.753 | 6.224 | 352751 | 10 | 546534 |
| synthetic_4096_dense | fast | pieces | 1.0825 | 20.955 | 21.882 | 98112 | 25 | 1772942 |
| synthetic_4096_dense | reference | pieces | 0.6974 | 31.480 | 38.952 | 63202 | 22 | 2157810 |

## Interpretation rules

- Exact parity gate passed before any timing data was accepted.
- Individual workloads remain visible; geomean summaries never hide a per-workload regression.
- No performance claim is based on one run or generalized beyond the measured environment.
- Correctness evidence lives in #110/#114; this benchmark makes no correctness claims.
