# Merge-Engine Benchmark Report (Issue #111)

- Generated: 2026-10-25T02:21:14.555549+00:00
- Commit: `9614ca63f7239576321c2c3d443a0f6448bb42ff` (dirty=False)
- Platform: Windows 10 / 3.10.11 / 16 logical CPUs / 16.85 GB
- Policy: workers=1, warmup=3, repetitions=7, peak RSS 42.0 MB
- Parity gate: exact 3-way (production/fast/reference), both output paths, passed before timing.

## Geometric means over text workloads (pieces path, primary metric)

| Path | Text-workload geomean MB/s | Cells |
|---|---|---|
| production | 8.3316 | 8 |
| fast | 1.8714 | 8 |
| reference | 1.5016 | 8 |

## Fast vs production throughput ratio (pieces path, per workload)

| Workload | production MB/s | fast MB/s | speedup |
|---|---|---|---|
| cross_word_heavy | 11.7862 | 2.2924 | 5.141x |
| long | 16.1907 | 2.7920 | 5.799x |
| medium | 13.8533 | 3.1613 | 4.382x |
| multilingual | 14.6822 | 2.9023 | 5.059x |
| pathological | 9.8049 | 2.2814 | 4.298x |
| punctuation_heavy | 3.6451 | 0.7117 | 5.122x |
| short | 3.1884 | 1.3671 | 2.332x |
| source_code | 5.2493 | 1.1540 | 4.549x |
| synthetic_1024_dense | 8.5030 | 2.8845 | 2.948x |
| synthetic_1024_hierarchical | 7.2281 | 2.8206 | 2.563x |
| synthetic_1024_medium | 8.4019 | 2.9805 | 2.819x |
| synthetic_1024_sparse | 6.2846 | 2.1120 | 2.976x |
| synthetic_4096_dense | 8.4340 | 2.4094 | 3.500x |
| synthetic_4096_sparse | 6.2470 | 2.7685 | 2.256x |

## Individual workload records

| Workload | Path | Output | MB/s | p50 ms | p95 ms | tokens/s | allocs/call | alloc peak B |
|---|---|---|---|---|---|---|---|---|
| short | production | pieces | 3.1884 | 0.007 | 0.046 | 724638 | 12 | 3134 |
| short | fast | pieces | 1.3671 | 0.026 | 0.065 | 310697 | 24 | 7643 |
| short | reference | pieces | 1.1742 | 0.033 | 0.066 | 266870 | 21 | 7936 |
| short | production | spans | 4.8889 | 0.007 | 0.019 | 1111111 | 10 | 1886 |
| short | fast | spans | 1.3592 | 0.028 | 0.055 | 308914 | 24 | 7827 |
| short | reference | spans | 1.1324 | 0.033 | 0.068 | 257353 | 21 | 8120 |
| medium | production | pieces | 13.8533 | 0.065 | 0.117 | 2132324 | 10 | 10192 |
| medium | fast | pieces | 3.1613 | 0.305 | 0.434 | 486585 | 29 | 72998 |
| medium | reference | pieces | 2.4082 | 0.421 | 0.455 | 370670 | 22 | 88256 |
| medium | production | spans | 13.2640 | 0.073 | 0.098 | 2041612 | 10 | 10192 |
| medium | fast | spans | 3.0972 | 0.315 | 0.392 | 476728 | 30 | 74486 |
| medium | reference | spans | 2.3065 | 0.434 | 0.485 | 355020 | 23 | 89744 |
| long | production | pieces | 16.1907 | 0.290 | 0.316 | 2262592 | 10 | 53160 |
| long | fast | pieces | 2.7920 | 1.657 | 1.992 | 390178 | 29 | 340442 |
| long | reference | pieces | 2.6156 | 1.764 | 1.997 | 365514 | 22 | 386100 |
| long | production | spans | 14.3358 | 0.321 | 0.389 | 2003377 | 10 | 53160 |
| long | fast | spans | 3.3882 | 1.383 | 1.509 | 473491 | 30 | 345930 |
| long | reference | spans | 2.5076 | 1.878 | 1.971 | 350435 | 23 | 391588 |
| multilingual | production | pieces | 14.6822 | 0.050 | 0.075 | 2243114 | 10 | 9904 |
| multilingual | fast | pieces | 2.9023 | 0.276 | 0.339 | 443409 | 29 | 56930 |
| multilingual | reference | pieces | 2.5025 | 0.304 | 0.372 | 382324 | 22 | 68472 |
| multilingual | production | spans | 12.7331 | 0.057 | 0.093 | 1945338 | 10 | 9904 |
| multilingual | fast | spans | 2.3262 | 0.320 | 0.416 | 355390 | 30 | 58066 |
| multilingual | reference | spans | 2.3778 | 0.322 | 0.396 | 363270 | 23 | 69608 |
| source_code | production | pieces | 5.2493 | 0.191 | 0.247 | 2199686 | 10 | 40152 |
| source_code | fast | pieces | 1.1540 | 0.885 | 0.987 | 483600 | 29 | 218317 |
| source_code | reference | pieces | 0.7887 | 1.197 | 1.890 | 330504 | 22 | 252884 |
| source_code | production | spans | 3.2814 | 0.298 | 0.405 | 1375061 | 10 | 40152 |
| source_code | fast | spans | 1.1203 | 0.927 | 1.006 | 469469 | 30 | 222077 |
| source_code | reference | spans | 0.8140 | 1.247 | 1.390 | 341108 | 23 | 256644 |
| punctuation_heavy | production | pieces | 3.6451 | 0.698 | 0.757 | 2279606 | 10 | 198152 |
| punctuation_heavy | fast | pieces | 0.7117 | 3.356 | 5.208 | 445093 | 29 | 864486 |
| punctuation_heavy | reference | pieces | 0.5720 | 4.410 | 4.837 | 357728 | 22 | 951932 |
| punctuation_heavy | production | spans | 2.8229 | 0.815 | 1.139 | 1765438 | 10 | 198152 |
| punctuation_heavy | fast | spans | 0.7186 | 3.518 | 3.700 | 449420 | 30 | 878902 |
| punctuation_heavy | reference | spans | 0.5238 | 4.719 | 5.849 | 327609 | 23 | 966348 |
| cross_word_heavy | production | pieces | 11.7862 | 0.313 | 0.368 | 2248107 | 10 | 92416 |
| cross_word_heavy | fast | pieces | 2.2924 | 1.684 | 1.860 | 437254 | 29 | 374294 |
| cross_word_heavy | reference | pieces | 1.6263 | 2.094 | 2.989 | 310203 | 22 | 422068 |
| cross_word_heavy | production | spans | 10.4346 | 0.359 | 0.399 | 1990299 | 10 | 92416 |
| cross_word_heavy | fast | spans | 2.3861 | 1.590 | 1.639 | 455136 | 30 | 380486 |
| cross_word_heavy | reference | spans | 1.8037 | 2.088 | 2.262 | 344047 | 23 | 428260 |
| pathological | production | pieces | 9.8049 | 1.000 | 1.306 | 1783698 | 10 | 199752 |
| pathological | fast | pieces | 2.2814 | 4.479 | 4.799 | 415028 | 29 | 975758 |
| pathological | reference | pieces | 1.9039 | 5.151 | 5.806 | 346362 | 22 | 1074124 |
| pathological | production | spans | 11.4796 | 0.848 | 0.951 | 2088358 | 10 | 199752 |
| pathological | fast | spans | 2.3818 | 3.896 | 5.248 | 433290 | 30 | 991998 |
| pathological | reference | spans | 0.9395 | 10.789 | 11.069 | 170920 | 23 | 1090364 |
| synthetic_1024_sparse | production | pieces | 6.2846 | 0.905 | 0.959 | 811669 | 10 | 128444 |
| synthetic_1024_sparse | fast | pieces | 2.1120 | 2.567 | 3.493 | 272765 | 29 | 454637 |
| synthetic_1024_sparse | reference | pieces | 1.5980 | 3.518 | 3.772 | 206387 | 23 | 477356 |
| synthetic_1024_medium | production | pieces | 8.4019 | 0.662 | 0.718 | 774660 | 10 | 126402 |
| synthetic_1024_medium | fast | pieces | 2.9805 | 1.873 | 2.004 | 274805 | 29 | 392597 |
| synthetic_1024_medium | reference | pieces | 1.6357 | 3.471 | 3.486 | 150814 | 22 | 482980 |
| synthetic_1024_dense | production | pieces | 8.5030 | 0.653 | 0.704 | 773414 | 10 | 126870 |
| synthetic_1024_dense | fast | pieces | 2.8845 | 1.930 | 2.059 | 262370 | 29 | 390665 |
| synthetic_1024_dense | reference | pieces | 1.4943 | 3.659 | 4.423 | 135921 | 22 | 483652 |
| synthetic_1024_hierarchical | production | pieces | 7.2281 | 0.773 | 0.921 | 328723 | 10 | 126870 |
| synthetic_1024_hierarchical | fast | pieces | 2.8206 | 1.882 | 2.418 | 128276 | 29 | 357001 |
| synthetic_1024_hierarchical | reference | pieces | 1.4471 | 3.755 | 4.311 | 65814 | 22 | 469316 |
| synthetic_4096_sparse | production | pieces | 6.2470 | 3.000 | 5.928 | 574197 | 10 | 544620 |
| synthetic_4096_sparse | fast | pieces | 2.7685 | 8.134 | 8.440 | 254469 | 29 | 1615877 |
| synthetic_4096_sparse | reference | pieces | 1.3894 | 14.759 | 21.508 | 127710 | 22 | 1964696 |
| synthetic_4096_dense | production | pieces | 8.4340 | 2.655 | 2.839 | 764387 | 10 | 546534 |
| synthetic_4096_dense | fast | pieces | 2.4094 | 8.314 | 11.768 | 218369 | 29 | 1607177 |
| synthetic_4096_dense | reference | pieces | 1.4670 | 14.565 | 20.241 | 132959 | 22 | 1969104 |

## Interpretation rules

- Exact parity gate passed before any timing data was accepted.
- Individual workloads remain visible; geomean summaries never hide a per-workload regression.
- No performance claim is based on one run or generalized beyond the measured environment.
- Correctness evidence lives in #110/#114; this benchmark makes no correctness claims.
