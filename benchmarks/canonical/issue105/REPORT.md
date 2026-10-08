# Canonical End-to-End Tokenizer Benchmark Report (Issue #105)

## 1. Hardware, Environment & Worker Fingerprint

- **OS / Platform**: `Windows-11-10.0.26200-SP0`
- **Processor**: `Intel64 Family 6 Model 154 Stepping 3, GenuineIntel` (AMD64)
- **Cores**: 8 physical / 12 logical
- **Total RAM**: 7.65 GB
- **Python Version**: `3.13.1 (tags/v3.13.1:0671451, Dec  3 2024, 19:06:28) [MSC v.1942 64 bit (AMD64)]`
- **Git Commit**: `a0fd66636051dbf03b83525a8557da335949c39b` (dirty: `False`)
- **Worker Count**: 1 (strictly single-worker controlled execution)
- **Primary Metric**: `throughput_mb_s (decimal MB/s: 10^6 bytes/s)`
- **Warmup / Repetitions / Iterations**: 3 warmup, 7 reps, 5 iterations/rep

## 2. Geometric Mean Throughput Summary

Geometric mean throughput (decimal MB/s) across the fixed matrix. Displayed alongside, and not substituting, individual workload records.

| Tokenizer | Overall Geomean (MB/s) | Batch 1 (MB/s) | Batch 8 (MB/s) | Batch 32 (MB/s) | Batch 128 (MB/s) | Batch Scaling (128 vs 1) | Implementation Engine |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **uniqtoken** | **0.489** | 0.634 | 0.540 | 0.512 | 0.326 | 0.51x | Python (native core uncompiled on Windows MSVC without link.exe) |
| **sentencepiece** | **14.710** | 19.794 | 6.094 | 15.346 | 25.295 | 1.28x | SentencePiece C++ runtime |
| **tokenizers** | **6.970** | 3.855 | 5.802 | 8.771 | 12.035 | 3.12x | Tokenizers Rust runtime |

## 3. Workload Comparison Table (Normalized MB/s Primary)

Primary column order prioritizes normalized input MB/s. Tokens/s is marked as segmentation-dependent.

| Workload | Batch Size | Tokenizer | **Input MB/s [95% Range]** | Tokens/s (seg-dep) | p50 Lat (ms) | p95 Lat (ms) | Dec p50 (ms) | Peak RSS (MB) | CPU % |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `short` | 1 | **uniqtoken** | **0.439** [0.30, 0.45] | 73,099* | 0.137 | 0.198 | 0.007 | 31.4 | 0.0% |
| `short` | 1 | **sentencepiece** | **10.233** [3.54, 10.48] | 8,139,535* | 0.004 | 0.012 | 0.005 | 32.8 | 0.0% |
| `short` | 1 | **tokenizers** | **2.638** [1.40, 2.67] | 599,520* | 0.017 | 0.031 | 0.001 | 32.8 | 0.0% |
| `short` | 8 | **uniqtoken** | **0.349** [0.29, 0.42] | 69,743* | 1.491 | 1.772 | 0.068 | 31.3 | 58.3% |
| `short` | 8 | **sentencepiece** | **0.518** [0.36, 0.60] | 419,160* | 0.725 | 1.050 | 0.728 | 33.1 | 111.0% |
| `short` | 8 | **tokenizers** | **2.922** [1.85, 3.52] | 769,231* | 0.129 | 0.203 | 0.039 | 34.3 | 0.0% |
| `short` | 32 | **uniqtoken** | **0.449** [0.40, 0.46] | 89,745* | 4.635 | 5.185 | 0.265 | 31.4 | 94.3% |
| `short` | 32 | **sentencepiece** | **1.793** [1.38, 2.20] | 1,450,001* | 0.839 | 1.089 | 0.644 | 33.5 | 367.5% |
| `short` | 32 | **tokenizers** | **5.290** [5.12, 5.95] | 1,361,238* | 0.284 | 0.294 | 0.096 | 35.0 | 1437.9% |
| `short` | 128 | **uniqtoken** | **0.204** [0.20, 0.23] | 41,343* | 40.877 | 42.769 | 3.306 | 32.1 | 103.6% |
| `short` | 128 | **sentencepiece** | **7.583** [6.84, 8.68] | 6,138,018* | 0.797 | 0.884 | 0.772 | 33.8 | 284.3% |
| `short` | 128 | **tokenizers** | **7.468** [7.23, 7.72] | 1,897,936* | 0.809 | 0.836 | 0.251 | 35.9 | 385.0% |
| `medium` | 1 | **uniqtoken** | **1.418** [0.92, 1.73] | 176,690* | 0.889 | 1.367 | 0.105 | 31.7 | 92.9% |
| `medium` | 1 | **sentencepiece** | **20.833** [16.50, 20.93] | 8,088,235* | 0.049 | 0.062 | 0.044 | 32.8 | 0.0% |
| `medium` | 1 | **tokenizers** | **5.236** [3.97, 5.32] | 805,955* | 0.195 | 0.257 | 0.012 | 33.5 | 0.0% |
| `medium` | 8 | **uniqtoken** | **1.520** [1.36, 1.63] | 192,289* | 6.657 | 7.423 | 0.848 | 31.7 | 105.8% |
| `medium` | 8 | **sentencepiece** | **8.529** [6.65, 13.01] | 3,326,733* | 0.960 | 1.230 | 0.979 | 34.9 | 189.9% |
| `medium` | 8 | **tokenizers** | **6.916** [5.35, 10.01] | 1,077,495* | 1.183 | 1.531 | 0.261 | 39.3 | 445.3% |
| `medium` | 32 | **uniqtoken** | **1.569** [1.45, 1.62] | 198,502* | 25.793 | 27.889 | 3.785 | 31.6 | 90.8% |
| `medium` | 32 | **sentencepiece** | **23.335** [20.66, 29.82] | 9,101,148* | 1.403 | 1.585 | 1.195 | 37.4 | 516.6% |
| `medium` | 32 | **tokenizers** | **9.936** [8.30, 11.73] | 1,545,284* | 3.295 | 3.945 | 0.749 | 41.8 | 549.7% |
| `medium` | 128 | **uniqtoken** | **0.835** [0.68, 1.18] | 105,677* | 194.063 | 239.096 | 32.024 | 33.1 | 109.7% |
| `medium` | 128 | **sentencepiece** | **33.229** [29.49, 40.80] | 12,964,607* | 3.942 | 4.441 | 2.199 | 43.5 | 227.8% |
| `medium` | 128 | **tokenizers** | **14.906** [13.69, 15.71] | 2,316,323* | 8.786 | 9.569 | 2.450 | 48.6 | 822.7% |
| `long` | 1 | **uniqtoken** | **1.725** [1.52, 1.82] | 191,916* | 3.444 | 3.901 | 0.441 | 32.5 | 102.0% |
| `long` | 1 | **sentencepiece** | **17.941** [12.66, 18.37] | 14,811,865* | 0.264 | 0.374 | 0.358 | 33.0 | 0.0% |
| `long` | 1 | **tokenizers** | **4.432** [2.58, 4.98] | 619,424* | 1.067 | 1.833 | 0.049 | 34.2 | 77.2% |
| `long` | 8 | **uniqtoken** | **1.580** [1.44, 1.77] | 176,418* | 30.110 | 32.933 | 3.725 | 33.2 | 93.3% |
| `long` | 8 | **sentencepiece** | **20.234** [18.45, 22.50] | 16,707,281* | 1.871 | 2.053 | 1.926 | 37.0 | 528.7% |
| `long` | 8 | **tokenizers** | **8.518** [6.87, 10.57] | 1,193,824* | 4.445 | 5.514 | 0.771 | 42.7 | 374.9% |
| `long` | 32 | **uniqtoken** | **0.970** [0.87, 1.49] | 108,354* | 196.098 | 218.046 | 19.148 | 33.3 | 96.9% |
| `long` | 32 | **sentencepiece** | **24.840** [18.62, 26.43] | 20,509,860* | 6.097 | 8.132 | 4.752 | 45.8 | 281.2% |
| `long` | 32 | **tokenizers** | **13.940** [11.70, 17.77] | 1,952,975* | 10.865 | 12.941 | 2.513 | 51.5 | 709.1% |
| `long` | 128 | **uniqtoken** | **0.928** [0.87, 0.99] | 103,701* | 819.860 | 878.116 | 113.894 | 37.2 | 102.3% |
| `long` | 128 | **sentencepiece** | **26.036** [23.71, 29.35] | 21,498,277* | 23.269 | 25.553 | 15.363 | 75.5 | 305.1% |
| `long` | 128 | **tokenizers** | **18.611** [15.26, 19.02] | 2,606,915* | 32.553 | 39.711 | 9.502 | 73.1 | 717.9% |
| `multilingual` | 1 | **uniqtoken** | **0.087** [0.08, 0.09] | 11,118* | 10.883 | 11.358 | 0.066 | 31.9 | 97.9% |
| `multilingual` | 1 | **sentencepiece** | **35.740** [26.87, 36.20] | 3,790,614* | 0.022 | 0.029 | 0.013 | 32.7 | 0.0% |
| `multilingual` | 1 | **tokenizers** | **4.296** [3.82, 4.37] | 786,505* | 0.184 | 0.207 | 0.012 | 33.3 | 0.0% |
| `multilingual` | 8 | **uniqtoken** | **0.047** [0.04, 0.08] | 6,177* | 160.607 | 172.926 | 0.773 | 31.9 | 98.3% |
| `multilingual` | 8 | **sentencepiece** | **7.537** [5.31, 11.03] | 824,840* | 0.844 | 1.197 | 0.608 | 34.2 | 163.9% |
| `multilingual` | 8 | **tokenizers** | **6.043** [4.43, 7.64] | 1,120,318* | 1.052 | 1.437 | 0.223 | 38.6 | 413.6% |
| `multilingual` | 32 | **uniqtoken** | **0.045** [0.04, 0.05] | 5,830* | 680.579 | 717.573 | 4.276 | 32.2 | 97.7% |
| `multilingual` | 32 | **sentencepiece** | **33.205** [23.01, 35.84] | 3,633,706* | 0.766 | 1.105 | 0.688 | 35.0 | 108.0% |
| `multilingual` | 32 | **tokenizers** | **9.286** [8.45, 11.50] | 1,718,096* | 2.740 | 3.012 | 0.589 | 41.6 | 379.2% |
| `multilingual` | 128 | **uniqtoken** | **0.042** [0.04, 0.04] | 5,420* | 2933.599 | 2952.079 | 22.397 | 33.8 | 115.3% |
| `multilingual` | 128 | **sentencepiece** | **44.481** [37.35, 53.21] | 4,878,603* | 2.288 | 2.725 | 1.414 | 37.8 | 228.4% |
| `multilingual` | 128 | **tokenizers** | **12.609** [10.82, 13.37] | 2,330,905* | 8.072 | 9.404 | 2.612 | 49.0 | 829.8% |
| `source_code` | 1 | **uniqtoken** | **1.097** [0.98, 1.25] | 368,577* | 1.194 | 1.336 | 0.150 | 31.6 | 73.9% |
| `source_code` | 1 | **sentencepiece** | **22.227** [18.92, 22.32] | 9,525,826* | 0.047 | 0.056 | 0.046 | 32.7 | 0.0% |
| `source_code` | 1 | **tokenizers** | **3.236** [2.33, 3.29] | 1,140,286* | 0.324 | 0.452 | 0.026 | 33.8 | 124.7% |
| `source_code` | 8 | **uniqtoken** | **1.165** [1.13, 1.23] | 392,420* | 9.031 | 9.299 | 1.220 | 31.7 | 99.3% |
| `source_code` | 8 | **sentencepiece** | **12.469** [8.51, 12.95] | 5,363,962* | 0.676 | 0.990 | 0.663 | 34.6 | 183.3% |
| `source_code` | 8 | **tokenizers** | **6.319** [5.39, 6.79] | 2,234,574* | 1.333 | 1.562 | 0.388 | 39.1 | 422.0% |
| `source_code` | 32 | **uniqtoken** | **1.144** [1.07, 1.18] | 385,558* | 36.768 | 39.345 | 5.224 | 31.9 | 99.5% |
| `source_code` | 32 | **sentencepiece** | **24.663** [19.25, 28.63] | 10,609,831* | 1.366 | 1.751 | 1.162 | 38.0 | 309.4% |
| `source_code` | 32 | **tokenizers** | **7.628** [6.42, 8.49] | 2,695,318* | 4.418 | 5.250 | 1.271 | 42.8 | 863.7% |
| `source_code` | 128 | **uniqtoken** | **0.554** [0.53, 0.58] | 186,782* | 303.723 | 317.472 | 44.618 | 34.0 | 103.9% |
| `source_code` | 128 | **sentencepiece** | **35.490** [21.49, 38.58] | 15,271,782* | 3.799 | 6.274 | 2.540 | 42.6 | 232.6% |
| `source_code` | 128 | **tokenizers** | **9.665** [9.41, 9.88] | 3,413,735* | 13.948 | 14.331 | 5.080 | 52.7 | 713.6% |

\* *Note: tokens/s is segmentation-dependent; models with smaller vocabulary or higher subword fragmentation produce higher token counts for identical text.*

## 4. Upstream Optimization Evidence Ledger (#95 to #104)

Consolidated record of validated methodology and historical optimization decisions, preserving both retained and rejected findings:

| Issue | Area | Finding & Measured Evidence | Decision Status |
| :---: | :--- | :--- | :--- |
| **#95** | Canonical Rust tokenizer hot-path profiler | Regex pretokenization was identified as the largest native diagnostic stage across all fixtures. Fused Rust path established. | `retained` |
| **#96-99** | Viterbi scratch bounding & prefix allocations | Prefix-only allocation reduction caused repeatable batch throughput regressions on dense multilingual texts. Combined bounded scratch was retained while prefix-only isolation was rejected. | `partially_reverted_and_superseded` |
| **#100** | Parallel batch scheduling and CPU saturation | Oversubscribing beyond physical core capacity (16 workers on 8-core host) caused severe thread contention and negative speedup. Worker count capped at physical cores. | `retained_with_limits` |
| **#101** | Python native-entry guards & FFI overhead | Unconditional Python NFKC normalization added 15-30% overhead on clean text. Replaced with delimiter-triggered normalization while preserving surrogate check and security gates. | `retained` |
| **#103** | Decode and ID-to-text reconstruction | Repeated regex matching in byte fallback hot path reduced throughput by 3x. Replaced with precomputed O(1) lookup table; added escape prefix fast path in metaspace restoration. | `retained` |
| **#104** | Tokenizer memory benchmark | Established peak RSS, loaded vocabulary footprint, and linear O(L) long-document memory scaling with zero steady-state buffer retention across repeated runs. | `retained` |

## 5. Interpretation Rules & Metric Definitions

1. **Normalized Input MB/s (Primary Metric)**:
   Calculated as `normalized_input_bytes / (encode_time_seconds * 1_000_000)` using decimal megabytes. Measures raw text processing throughput and is the only valid cross-tokenizer comparison metric.

2. **Tokens per Second (Secondary & Segmentation-Dependent)**:
   Tokens/s varies widely across tokenizers trained with different vocabulary algorithms or piece budgets. A tokenizer that splits a word into 5 characters emits 5x more tokens/s than a tokenizer emitting 1 whole-word token, despite accomplishing identical tokenization work.

3. **Single vs Batch Scaling**:
   Batch size 1 isolates single-string invocation and Python boundary overhead. Batch sizes 8, 32, and 128 measure vectorization, amortized overhead, and memory efficiency under high-throughput processing.

4. **Memory Residency**:
   Process Peak RSS (Working Set) captures OS paging, buffer allocation peaks, and native C++/Rust heap memory.

5. **Environment Scoping**:
   All measurements are strictly confined to the recorded hardware and software environment. No generalization beyond this host is claimed.

## 6. Acceptance Criteria Status

- [x] **One documented CLI command**: `python -m benchmarks.canonical_benchmark --output benchmarks/canonical/issue105`.
- [x] **Machine-readable, versioned JSON output**: Emitted to `results.json` (`schema_version: 1.0`).
- [x] **Human-readable comparison table**: Provided in section 2 and section 3.
- [x] **Benchmark input/configuration/result artifact hashes**: Cryptographically verified in metadata and `manifest.json`.
- [x] **MB/s is the primary cross-tokenizer field and display order**: Placed first in comparison tables.
- [x] **Parity and completeness validation run before publication**: Strict parity gate verified on all fixtures.
- [x] **Interpretation rules distinguish implementation speed, token-count differences, latency, and memory**: Documented in section 5.
- [x] **No result is presented beyond the measured environment**: Strictly scoped in report and metadata.
