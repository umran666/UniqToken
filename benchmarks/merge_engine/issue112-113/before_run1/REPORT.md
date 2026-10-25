# Merge-Engine Benchmark Report (Issue #111)

- Generated: 2026-10-25T02:20:48.476208+00:00
- Commit: `cca0dcf899007b312fc2e50eeac82472c9149380` (dirty=False)
- Platform: Windows 10 / 3.10.11 / 16 logical CPUs / 16.85 GB
- Policy: workers=1, warmup=3, repetitions=7, peak RSS 42.53 MB
- Parity gate: exact 3-way (production/fast/reference), both output paths, passed before timing.

## Geometric means over text workloads (pieces path, primary metric)

| Path | Text-workload geomean MB/s | Cells |
|---|---|---|
| production | 8.5601 | 8 |
| fast | 1.9055 | 8 |
| reference | 1.4095 | 8 |

## Fast vs production throughput ratio (pieces path, per workload)

| Workload | production MB/s | fast MB/s | speedup |
|---|---|---|---|
| cross_word_heavy | 12.0756 | 2.3996 | 5.032x |
| long | 16.1939 | 3.3445 | 4.842x |
| medium | 14.7704 | 3.1254 | 4.726x |
| multilingual | 14.5933 | 3.0050 | 4.856x |
| pathological | 12.9698 | 2.1789 | 5.953x |
| punctuation_heavy | 2.1869 | 0.7117 | 3.073x |
| short | 4.5765 | 1.3333 | 3.432x |
| source_code | 5.2692 | 1.1153 | 4.724x |
| synthetic_1024_dense | 7.5947 | 1.9102 | 3.976x |
| synthetic_1024_hierarchical | 7.3657 | 2.1926 | 3.359x |
| synthetic_1024_medium | 8.5012 | 2.1223 | 4.006x |
| synthetic_1024_sparse | 6.8227 | 2.0955 | 3.256x |
| synthetic_4096_dense | 4.4554 | 2.0087 | 2.218x |
| synthetic_4096_sparse | 7.5941 | 2.1412 | 3.547x |

## Individual workload records

| Workload | Path | Output | MB/s | p50 ms | p95 ms | tokens/s | allocs/call | alloc peak B |
|---|---|---|---|---|---|---|---|---|
| short | production | pieces | 4.5765 | 0.006 | 0.028 | 1040119 | 12 | 3134 |
| short | fast | pieces | 1.3333 | 0.026 | 0.067 | 303030 | 21 | 7072 |
| short | reference | pieces | 1.1386 | 0.033 | 0.071 | 258780 | 21 | 7936 |
| short | production | spans | 4.2957 | 0.007 | 0.030 | 976290 | 10 | 1886 |
| short | fast | spans | 1.3018 | 0.028 | 0.062 | 295858 | 21 | 7256 |
| short | reference | spans | 1.1378 | 0.034 | 0.067 | 258589 | 21 | 8120 |
| medium | production | pieces | 14.7704 | 0.066 | 0.091 | 2273480 | 10 | 10192 |
| medium | fast | pieces | 3.1254 | 0.316 | 0.395 | 481068 | 25 | 70960 |
| medium | reference | pieces | 2.3613 | 0.422 | 0.479 | 363462 | 22 | 88256 |
| medium | production | spans | 13.1081 | 0.073 | 0.103 | 2017624 | 10 | 10192 |
| medium | fast | spans | 2.7808 | 0.352 | 0.411 | 428026 | 26 | 72448 |
| medium | reference | spans | 2.2522 | 0.442 | 0.512 | 346666 | 23 | 89744 |
| long | production | pieces | 16.1939 | 0.288 | 0.323 | 2263034 | 10 | 53160 |
| long | fast | pieces | 3.3445 | 1.399 | 1.506 | 467378 | 25 | 333900 |
| long | reference | pieces | 2.3471 | 1.937 | 2.426 | 327996 | 22 | 386100 |
| long | production | spans | 11.9860 | 0.343 | 0.510 | 1674993 | 10 | 53160 |
| long | fast | spans | 2.8020 | 1.546 | 2.202 | 391571 | 26 | 339388 |
| long | reference | spans | 2.4154 | 1.925 | 2.182 | 337540 | 23 | 391588 |
| multilingual | production | pieces | 14.5933 | 0.050 | 0.078 | 2229534 | 10 | 9904 |
| multilingual | fast | pieces | 3.0050 | 0.244 | 0.380 | 459103 | 25 | 55280 |
| multilingual | reference | pieces | 1.4816 | 0.449 | 1.005 | 226356 | 22 | 68472 |
| multilingual | production | spans | 12.5914 | 0.056 | 0.089 | 1923688 | 10 | 9904 |
| multilingual | fast | spans | 2.9324 | 0.262 | 0.310 | 448006 | 26 | 56416 |
| multilingual | reference | spans | 2.3532 | 0.326 | 0.402 | 359523 | 23 | 69608 |
| source_code | production | pieces | 5.2692 | 0.188 | 0.267 | 2208044 | 10 | 40152 |
| source_code | fast | pieces | 1.1153 | 0.926 | 1.005 | 467382 | 25 | 213740 |
| source_code | reference | pieces | 0.8757 | 1.194 | 1.281 | 366968 | 22 | 252884 |
| source_code | production | spans | 4.7337 | 0.214 | 0.259 | 1983641 | 10 | 40152 |
| source_code | fast | spans | 0.9433 | 1.020 | 1.647 | 395303 | 26 | 217500 |
| source_code | reference | spans | 0.8288 | 1.250 | 1.327 | 347320 | 23 | 256644 |
| punctuation_heavy | production | pieces | 2.1869 | 0.737 | 2.768 | 1367658 | 10 | 198152 |
| punctuation_heavy | fast | pieces | 0.7117 | 3.567 | 4.002 | 445114 | 25 | 848076 |
| punctuation_heavy | reference | pieces | 0.5351 | 4.741 | 5.130 | 334633 | 22 | 951932 |
| punctuation_heavy | production | spans | 2.5387 | 0.995 | 1.263 | 1587709 | 10 | 198152 |
| punctuation_heavy | fast | spans | 0.6736 | 3.771 | 4.027 | 421260 | 26 | 862492 |
| punctuation_heavy | reference | spans | 0.5207 | 4.785 | 5.385 | 325641 | 23 | 966348 |
| cross_word_heavy | production | pieces | 12.0756 | 0.306 | 0.345 | 2303304 | 10 | 92416 |
| cross_word_heavy | fast | pieces | 2.3996 | 1.577 | 1.671 | 457695 | 25 | 366988 |
| cross_word_heavy | reference | pieces | 1.8941 | 2.009 | 2.039 | 361285 | 22 | 422068 |
| cross_word_heavy | production | spans | 10.4408 | 0.361 | 0.401 | 1991477 | 10 | 92416 |
| cross_word_heavy | fast | spans | 2.3173 | 1.619 | 1.700 | 441994 | 26 | 373180 |
| cross_word_heavy | reference | spans | 1.6761 | 2.150 | 2.763 | 319697 | 23 | 428260 |
| pathological | production | pieces | 12.9698 | 0.749 | 0.833 | 2359448 | 10 | 199752 |
| pathological | fast | pieces | 2.1789 | 4.306 | 5.675 | 396375 | 25 | 957324 |
| pathological | reference | pieces | 1.8771 | 5.184 | 5.529 | 341480 | 22 | 1074124 |
| pathological | production | spans | 11.2839 | 0.863 | 0.927 | 2052756 | 10 | 199752 |
| pathological | fast | spans | 2.2268 | 4.452 | 4.676 | 405104 | 26 | 973564 |
| pathological | reference | spans | 1.7673 | 5.615 | 5.859 | 321504 | 23 | 1090364 |
| synthetic_1024_sparse | production | pieces | 6.8227 | 0.819 | 0.874 | 881166 | 10 | 128444 |
| synthetic_1024_sparse | fast | pieces | 2.0955 | 2.668 | 2.907 | 270637 | 25 | 474380 |
| synthetic_1024_sparse | reference | pieces | 1.4670 | 3.828 | 4.079 | 189470 | 23 | 504720 |
| synthetic_1024_medium | production | pieces | 8.5012 | 0.653 | 0.724 | 783819 | 10 | 126402 |
| synthetic_1024_medium | fast | pieces | 2.1223 | 2.514 | 3.542 | 195679 | 25 | 433330 |
| synthetic_1024_medium | reference | pieces | 1.3794 | 4.034 | 4.359 | 127183 | 22 | 529566 |
| synthetic_1024_dense | production | pieces | 7.5947 | 0.715 | 0.849 | 690798 | 10 | 126870 |
| synthetic_1024_dense | fast | pieces | 1.9102 | 2.768 | 3.874 | 173749 | 25 | 432062 |
| synthetic_1024_dense | reference | pieces | 1.3755 | 4.040 | 4.427 | 125113 | 22 | 530902 |
| synthetic_1024_hierarchical | production | pieces | 7.3657 | 0.737 | 0.938 | 334985 | 10 | 126870 |
| synthetic_1024_hierarchical | fast | pieces | 2.1926 | 2.551 | 2.713 | 99717 | 25 | 379364 |
| synthetic_1024_hierarchical | reference | pieces | 1.2175 | 4.522 | 5.270 | 55370 | 22 | 516566 |
| synthetic_4096_sparse | production | pieces | 7.5941 | 2.740 | 4.489 | 698010 | 10 | 544620 |
| synthetic_4096_sparse | fast | pieces | 2.1412 | 10.636 | 11.189 | 196812 | 25 | 1778916 |
| synthetic_4096_sparse | reference | pieces | 1.2641 | 16.460 | 26.686 | 116192 | 22 | 2150676 |
| synthetic_4096_dense | production | pieces | 4.4554 | 5.295 | 5.425 | 403799 | 10 | 546534 |
| synthetic_4096_dense | fast | pieces | 2.0087 | 10.829 | 12.833 | 182052 | 25 | 1772942 |
| synthetic_4096_dense | reference | pieces | 1.2969 | 16.298 | 22.953 | 117542 | 22 | 2157810 |

## Interpretation rules

- Exact parity gate passed before any timing data was accepted.
- Individual workloads remain visible; geomean summaries never hide a per-workload regression.
- No performance claim is based on one run or generalized beyond the measured environment.
- Correctness evidence lives in #110/#114; this benchmark makes no correctness claims.
