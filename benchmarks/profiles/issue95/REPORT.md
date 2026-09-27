# UniqToken hot-path profile

Commit: `88a386e3f50a2829fa2e105350b05a660afb130f`; build: `release`; Rayon threads: `1`.
Warmup: 2; repetitions: 7; iterations per repetition: 10.

Exact parity gate passed before timings. Input MB/s uses normalized UTF-8 bytes (decimal MB).

| Workload | Normalized bytes | Encode MB/s | Encode ms | Decode ms | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| short_single | 60 | 1.15 | 0.052 | 0.065 | 192493 |
| short_batch | 2080 | 1.44 | 1.445 | 1.520 | 287877 |
| medium_single | 1260 | 0.70 | 1.812 | 0.602 | 86663 |
| medium_batch | 40480 | 0.70 | 57.653 | 21.069 | 88808 |
| long_single | 5940 | 1.76 | 3.384 | 1.306 | 195308 |
| long_batch | 190240 | 1.07 | 177.475 | 84.731 | 119724 |
| multilingual_single | 948 | 0.50 | 1.896 | 0.308 | 63833 |
| multilingual_batch | 30496 | 0.51 | 59.659 | 10.297 | 66511 |
| source_code_single | 1310 | 0.54 | 2.439 | 0.819 | 180424 |
| source_code_batch | 42080 | 0.50 | 83.930 | 30.293 | 168903 |

Diagnostic stage CPU time (ms, summed across rows):

| Workload | Gate | Normalize | Regex | Grapheme + chunks | Cached segmentation | Output copy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short_single | 0.003 | 0.004 | 0.014 | 0.001 | 0.004 | 0.003 |
| short_batch | 0.097 | 0.123 | 0.491 | 0.057 | 0.138 | 0.113 |
| medium_single | 0.037 | 0.047 | 1.138 | 0.016 | 0.046 | 0.031 |
| medium_batch | 1.248 | 1.543 | 40.594 | 0.659 | 1.597 | 1.116 |
| long_single | 0.094 | 0.114 | 1.714 | 0.042 | 0.132 | 0.070 |
| long_batch | 5.578 | 7.180 | 110.184 | 2.801 | 6.771 | 4.991 |
| multilingual_single | 0.022 | 0.028 | 1.340 | 0.060 | 0.035 | 0.024 |
| multilingual_batch | 0.833 | 1.027 | 47.961 | 2.160 | 1.302 | 0.903 |
| source_code_single | 0.039 | 0.048 | 1.706 | 0.046 | 0.133 | 0.082 |
| source_code_batch | 1.505 | 1.773 | 65.025 | 1.721 | 4.886 | 3.238 |

Largest measured stage per workload (median and observed 95% range, ms):

| Workload | Stage | Median | Observed range |
| --- | --- | ---: | ---: |
| short_single | pretokenization_regex | 0.014 | 0.011-0.015 |
| short_batch | pretokenization_regex | 0.491 | 0.439-0.536 |
| medium_single | pretokenization_regex | 1.138 | 1.083-1.269 |
| medium_batch | pretokenization_regex | 40.594 | 27.656-43.161 |
| long_single | pretokenization_regex | 1.714 | 1.512-2.456 |
| long_batch | pretokenization_regex | 110.184 | 107.611-111.692 |
| multilingual_single | pretokenization_regex | 1.340 | 1.269-1.430 |
| multilingual_batch | pretokenization_regex | 47.961 | 45.065-51.204 |
| source_code_single | pretokenization_regex | 1.706 | 1.505-1.779 |
| source_code_batch | pretokenization_regex | 65.025 | 61.839-67.625 |

All stage ranges and raw wall samples are in `results.json`.
`python_callstack.txt` is a cProfile call-stack report; native Rust frames are opaque there.
The native diagnostic replays the exact normalization, regex, grapheme, cached segmentation, and token-copy calls.
Cached segmentation combines trie lookup, Viterbi/lattice work, and byte fallback. Merge application is inactive for the fused path.
Python/Rust conversion and allocation cannot be subtracted reliably from parallel wall time; no separate percentage is claimed.
