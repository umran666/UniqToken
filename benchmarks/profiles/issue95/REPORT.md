# UniqToken hot-path profile

Commit: `a0fe7a8eab04a930bf659c157ec901fa76bc60f8`; build: `release`; Rayon threads: `1`.
Warmup: 2; repetitions: 7; iterations per repetition: 10.

Exact parity gate passed before timings. Input MB/s uses normalized UTF-8 bytes (decimal MB).

| Workload | Normalized bytes | Encode MB/s | Encode ms | Decode ms | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| short_single | 60 | 2.25 | 0.027 | 0.019 | 375516 |
| short_batch | 2080 | 3.16 | 0.658 | 0.662 | 632421 |
| medium_single | 1260 | 1.90 | 0.663 | 0.248 | 236763 |
| medium_batch | 40480 | 1.84 | 22.018 | 8.339 | 232538 |
| long_single | 5940 | 2.68 | 2.216 | 1.178 | 298350 |
| long_batch | 190240 | 2.56 | 74.449 | 39.281 | 285403 |
| multilingual_single | 948 | 1.23 | 0.771 | 0.147 | 156850 |
| multilingual_batch | 30496 | 1.26 | 24.277 | 4.824 | 163448 |
| source_code_single | 1310 | 1.36 | 0.962 | 0.370 | 457176 |
| source_code_batch | 42080 | 1.41 | 29.846 | 12.442 | 474971 |

Diagnostic stage elapsed time (ms, summed across rows):

| Workload | Gate | Normalize | Regex | Grapheme + chunks | Cached segmentation | Output copy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short_single | 0.002 | 0.002 | 0.005 | 0.001 | 0.002 | 0.001 |
| short_batch | 0.050 | 0.063 | 0.187 | 0.025 | 0.060 | 0.051 |
| medium_single | 0.020 | 0.024 | 0.502 | 0.009 | 0.024 | 0.016 |
| medium_batch | 0.606 | 0.723 | 15.871 | 0.316 | 0.797 | 0.510 |
| long_single | 0.081 | 0.095 | 1.496 | 0.035 | 0.109 | 0.060 |
| long_batch | 2.720 | 3.312 | 49.587 | 1.389 | 3.376 | 2.312 |
| multilingual_single | 0.013 | 0.016 | 0.601 | 0.036 | 0.020 | 0.014 |
| multilingual_batch | 0.384 | 0.482 | 18.533 | 1.096 | 0.635 | 0.415 |
| source_code_single | 0.020 | 0.024 | 0.719 | 0.024 | 0.070 | 0.042 |
| source_code_batch | 0.660 | 0.774 | 23.193 | 0.761 | 2.307 | 1.353 |

Largest measured stage per workload (median and observed min/max, ms):

| Workload | Stage | Median | Observed range |
| --- | --- | ---: | ---: |
| short_single | pretokenization_regex | 0.005 | 0.005-0.006 |
| short_batch | pretokenization_regex | 0.187 | 0.182-0.202 |
| medium_single | pretokenization_regex | 0.502 | 0.483-0.601 |
| medium_batch | pretokenization_regex | 15.871 | 15.653-16.641 |
| long_single | pretokenization_regex | 1.496 | 1.426-2.613 |
| long_batch | pretokenization_regex | 49.587 | 47.536-52.488 |
| multilingual_single | pretokenization_regex | 0.601 | 0.576-0.798 |
| multilingual_batch | pretokenization_regex | 18.533 | 18.024-19.209 |
| source_code_single | pretokenization_regex | 0.719 | 0.694-0.946 |
| source_code_batch | pretokenization_regex | 23.193 | 22.254-23.705 |

All stage ranges and raw wall samples are in `results.json`.
`python_callstack.txt` is a cProfile call-stack report; native Rust frames are opaque there.
The native diagnostic replays the exact normalization, regex, grapheme, cached segmentation, and token-copy calls.
Cached segmentation combines trie lookup, Viterbi/lattice work, and byte fallback. Merge application is inactive for the fused path.
Python/Rust conversion and allocation cannot be subtracted reliably from parallel wall time; no separate percentage is claimed.
