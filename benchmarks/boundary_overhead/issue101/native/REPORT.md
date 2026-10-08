# Native boundary profiling and token output ownership (#101)

The fused token APIs now retain owned segmentation Arcs until PyO3 constructs ordinary Python lists of strings. This removes intermediate Rust token-string clones. The Rust `Vec<String>` entry points remain available. IDs, offsets, decode, tokenizer decisions, and public signatures are unchanged.

## Evidence and scope

- [Issue #95 profile](../../../profiles/issue95/REPORT.md) measured 2.312 ms of native output copying for long batches and 1.353 ms for source-code batches on its historical build. These are Rust output-copy timings, not percentages of FFI cost.
- The [earlier Python guard study](../REPORT.md) remains intact. This follow-up isolates actual native boundary operations on the implementation after #96/#99 and #136.
- Only five embedded synthetic engineering fixtures and a deterministic 400-token model are used. No held-out data was accessed by these measurements. Phase C was not launched; frozen research and released artifacts are unchanged.

## Method

- Comprehensive source: `d22bc9ad1cc45fa53db7ff4017f6bc544a64dad0`; baseline: `a0fd66636051dbf03b83525a8557da335949c39b`. Native source is identical to the implementation commit `49acc0e0f6d0f29d73fc58aeed6447ccf79c26f9` for the later harness-only commits.
- Environment: Windows-10-10.0.26300-SP0; Python 3.10.11; rustc 1.98.0 (88d9e12ae 2026-08-18). Cargo release settings and `abi3-py39` are unchanged; Rayon has one thread.
- Default release binaries are measured in four separate processes, before/after/after/before. Each of 50 API/workload cells has three warmups and 22 individual timed calls per variant. p95 is the nearest-rank quantile; output destruction is outside the timer.
- A second confirmation alternates before/after within each of 21 trials in one process, with three warmups, for the ten token workloads and their IDs controls. The paired ratio is the median of trial ratios; it can differ from the ratio of independent p50s.
- Call counts instrument named `rust_*` function entries in an untimed invocation. They exclude C property getters such as `ViterbiSpan` attribute access. Iterative batch encoding has 32 functional native entries; fused batches have one. Decode has none.
- MB/s uses normalized input UTF-8 bytes and decimal MB. Decode uses the corresponding normalized input size, rather than token-ID storage size.
- Native probes replay input extraction, the token pipeline, and result materialization as separate timed stages. IDs conversion is a control built from the same segmentation, not an isolated replay of the production IDs engine.
- Native probe timing uses the opt-in `allocation-profile` binary with request counters inactive; its allocator still tracks live Rust bytes. This instrumentation can inflate allocation-heavy reference timings. Never subtract these timings from default-build latency or report them as FFI percentages.
- Allocation counters run separately and report Rust requests/requested bytes/peak live bytes/before live bytes/after live bytes. They exclude CPython allocations. Tracemalloc reports Python retained/peak bytes; process RSS is a lifetime peak, not a per-cell allocation measurement.
- Input release and a root-list INCREF/DECREF cycle are measured separately. The latter excludes element destruction and can be below the timer resolution. Per-element creation/reference work remains inside materialization.

## Paired default-build confirmation

| Workload | API | Before MB/s | After MB/s | Before p50/p95 ms | After p50/p95 ms | Paired ratio [within-run bootstrap interval] |
| --- | --- | ---: | ---: | --- | --- | --- |
| short_single | tokens | 0.691 | 0.712 | 0.0868/0.1282 | 0.0843/0.1120 | 1.030 [1.011, 1.066] |
| short_single | ids | 0.720 | 0.726 | 0.0833/0.1435 | 0.0826/0.1198 | 1.008 [0.970, 1.054] |
| short_batch | tokens | 1.262 | 1.343 | 1.6488/1.7820 | 1.5486/1.7161 | 1.061 [1.015, 1.097] |
| short_batch | ids | 1.477 | 1.437 | 1.4080/1.5664 | 1.4470/1.5989 | 0.990 [0.944, 1.035] |
| medium_single | tokens | 0.580 | 0.588 | 2.1741/3.0261 | 2.1414/4.7840 | 1.015 [1.002, 1.039] |
| medium_single | ids | 0.603 | 0.605 | 2.0889/2.4378 | 2.0826/2.3189 | 1.001 [0.967, 1.014] |
| medium_batch | tokens | 0.602 | 0.604 | 67.2340/88.3191 | 67.0593/81.8183 | 1.015 [0.986, 1.045] |
| medium_batch | ids | 0.617 | 0.606 | 65.5877/75.3461 | 66.7717/78.8123 | 0.989 [0.967, 1.033] |
| long_single | tokens | 0.908 | 0.929 | 6.5428/9.2221 | 6.3910/7.4704 | 1.002 [0.988, 1.057] |
| long_single | ids | 0.972 | 0.955 | 6.1114/6.6295 | 6.2190/7.0684 | 0.975 [0.934, 1.008] |
| long_batch | tokens | 0.949 | 0.964 | 200.5668/242.5163 | 197.2917/240.8889 | 1.035 [1.010, 1.126] |
| long_batch | ids | 0.974 | 1.000 | 195.3439/241.2482 | 190.2357/220.3447 | 1.060 [0.958, 1.129] |
| multilingual_single | tokens | 0.329 | 0.310 | 2.8843/3.4164 | 3.0558/6.2679 | 0.955 [0.901, 0.983] |
| multilingual_single | ids | 0.371 | 0.368 | 2.5529/3.1424 | 2.5735/3.2409 | 1.010 [0.983, 1.025] |
| multilingual_batch | tokens | 0.377 | 0.377 | 80.9555/90.4570 | 80.9743/94.4440 | 1.004 [0.946, 1.019] |
| multilingual_batch | ids | 0.615 | 0.546 | 49.5909/84.6624 | 55.8566/83.3299 | 0.988 [0.954, 0.999] |
| source_code_single | tokens | 0.355 | 0.360 | 3.6899/4.2195 | 3.6345/4.1666 | 0.999 [0.981, 1.053] |
| source_code_single | ids | 0.376 | 0.379 | 3.4840/3.6248 | 3.4552/3.9268 | 1.013 [0.966, 1.044] |
| source_code_batch | tokens | 0.433 | 0.435 | 97.0765/138.5001 | 96.8025/126.1023 | 1.017 [1.009, 1.030] |
| source_code_batch | ids | 0.436 | 0.429 | 96.4549/123.0539 | 98.1044/122.7676 | 0.990 [0.902, 1.017] |

Ratios above one favor the candidate. Token ratios range from 0.955 to 1.061 in this paired run; multilingual single-text encoding is slower. Unchanged IDs controls also vary. These are conditional measurements on one environment, not a universal speedup or cross-machine confidence statement. The allocation reduction is the stronger result.

## Direct native token materialization

| Workload | Reference/candidate p50 ms | Rust requests before/after | Requested bytes before/after | Extra peak Rust bytes before/after | Intermediate token bytes copied before/after |
| --- | --- | --- | --- | --- | --- |
| short_single | 0.0063/0.0037 | 12/0 | 324/0 | 324/0 | 60/0 |
| short_batch | 0.2034/0.1083 | 449/0 | 12972/0 | 12972/0 | 2220/0 |
| medium_single | 0.0754/0.0416 | 159/0 | 5052/0 | 5052/0 | 1260/0 |
| medium_batch | 2.4808/1.2520 | 5153/0 | 164268/0 | 164268/0 | 40620/0 |
| long_single | 0.3736/0.2239 | 663/0 | 21828/0 | 21828/0 | 5940/0 |
| long_batch | 10.8212/6.0277 | 21281/0 | 701100/0 | 701100/0 | 190380/0 |
| multilingual_single | 0.0663/0.0377 | 123/0 | 3876/0 | 3876/0 | 948/0 |
| multilingual_batch | 1.8812/0.9682 | 4001/0 | 126636/0 | 126636/0 | 30636/0 |
| source_code_single | 0.1573/0.0644 | 442/0 | 11894/0 | 11894/0 | 1310/0 |
| source_code_batch | 4.8346/1.6088 | 14209/0 | 383212/0 | 383212/0 | 42220/0 |

For the long batch, the controlled materialization replay removes 21,281 Rust heap requests, 701,100 requested bytes and 190,380 token-string bytes copied. Python output strings/lists still require construction. The replay first owns segmentation Arcs in both modes; it isolates materialization and is not a byte-for-byte reconstruction of the old complete pipeline.

## Input access and remaining costs

| Workload | Candidate input p50 microseconds | Rust input requests / bytes | Rust input bytes copied |
| --- | ---: | --- | ---: |
| short_single | 0.50 | 1 / 44 | 44 |
| short_batch | 6.30 | 1 / 768 | 0 |
| medium_single | 1.30 | 1 / 1020 | 1020 |
| medium_batch | 19.90 | 1 / 768 | 0 |
| long_single | 4.70 | 1 / 4730 | 4730 |
| long_batch | 53.70 | 1 / 768 | 0 |
| multilingual_single | 7.60 | 1 / 792 | 792 |
| multilingual_batch | 112.80 | 1 / 768 | 0 |
| source_code_single | 1.40 | 1 / 1050 | 1050 |
| source_code_batch | 23.00 | 1 / 768 | 0 |

On the pinned abi3-py39 build, single `&str` extraction uses PyO3's owned `Cow<str>` fallback; batches retain `PyBackedStr` handles without intermediate Rust strings. This does not imply zero-copy CPython input handling: PyO3 can allocate UTF-8 Python bytes buffers. Their exact copy volume is unmeasured and remains null in the receipts. No ABI change or buffer API is proposed.

## Complete default-build matrix

The separate-process measurements below are retained in full, including control drift and regressions. They motivated the paired confirmation above and should not be used alone to attribute gains to the patch.

| Workload | API | Before MB/s | After MB/s | Before p50/p95 ms | After p50/p95 ms | p50 ratio |
| --- | --- | ---: | ---: | --- | --- | ---: |
| short_single | tokens | 0.913 | 1.341 | 0.0658/0.0953 | 0.0447/0.0691 | 1.469 |
| short_single | ids | 0.846 | 1.359 | 0.0709/0.2298 | 0.0442/0.0467 | 1.607 |
| short_single | offsets | 0.096 | 0.182 | 0.6250/1.1843 | 0.3299/0.3533 | 1.895 |
| short_single | decode | 1.021 | 1.529 | 0.0587/0.0636 | 0.0393/0.0417 | 1.497 |
| short_batch | tokens | 1.225 | 1.640 | 1.6984/1.8805 | 1.2685/2.5322 | 1.339 |
| short_batch | ids | 1.263 | 2.051 | 1.6471/1.8493 | 1.0140/1.4310 | 1.624 |
| short_batch | offsets | 0.112 | 0.127 | 18.4984/20.5225 | 16.4202/20.7985 | 1.127 |
| short_batch | decode | 0.981 | 1.084 | 2.1195/2.5365 | 1.9181/2.3290 | 1.105 |
| short_batch | iterative_tokens | 0.947 | 1.102 | 2.1963/3.3187 | 1.8881/2.3650 | 1.163 |
| short_batch | iterative_ids | 1.028 | 0.924 | 2.0234/2.3168 | 2.2503/2.4739 | 0.899 |
| medium_single | tokens | 0.658 | 0.660 | 1.9137/2.2615 | 1.9095/4.3586 | 1.002 |
| medium_single | ids | 0.613 | 0.639 | 2.0570/2.1545 | 1.9712/3.1517 | 1.043 |
| medium_single | offsets | 0.157 | 0.173 | 8.0045/9.5149 | 7.2899/8.8680 | 1.098 |
| medium_single | decode | 1.528 | 2.403 | 0.8245/0.9517 | 0.5244/0.8100 | 1.572 |
| medium_batch | tokens | 0.643 | 0.695 | 62.9839/68.3647 | 58.2677/65.8953 | 1.081 |
| medium_batch | ids | 0.655 | 0.703 | 61.8321/69.0497 | 57.5785/65.7287 | 1.074 |
| medium_batch | offsets | 0.140 | 0.144 | 288.7982/396.1433 | 281.9035/393.9700 | 1.024 |
| medium_batch | decode | 1.555 | 1.639 | 26.0250/27.8953 | 24.7028/29.0763 | 1.054 |
| medium_batch | iterative_tokens | 0.592 | 0.619 | 68.3242/71.6132 | 65.3780/72.2466 | 1.045 |
| medium_batch | iterative_ids | 0.632 | 0.625 | 64.0893/77.1369 | 64.8026/71.9334 | 0.989 |
| long_single | tokens | 0.959 | 1.007 | 6.1913/6.6118 | 5.8959/6.1558 | 1.050 |
| long_single | ids | 0.939 | 1.039 | 6.3227/6.7341 | 5.7169/5.9900 | 1.106 |
| long_single | offsets | 0.149 | 0.145 | 39.8890/42.2865 | 40.8814/43.7484 | 0.976 |
| long_single | decode | 1.645 | 1.590 | 3.6103/3.9081 | 3.7369/4.3844 | 0.966 |
| long_batch | tokens | 0.996 | 1.005 | 191.0033/205.1714 | 189.2448/198.3978 | 1.009 |
| long_batch | ids | 1.009 | 1.098 | 188.5129/205.4859 | 173.2634/197.8925 | 1.088 |
| long_batch | offsets | 0.145 | 0.135 | 1309.1282/1520.8346 | 1407.7298/1472.3234 | 0.930 |
| long_batch | decode | 1.662 | 1.668 | 114.4629/139.7730 | 114.0217/122.0640 | 1.004 |
| long_batch | iterative_tokens | 0.954 | 0.961 | 199.4558/210.9481 | 197.9246/211.6881 | 1.008 |
| long_batch | iterative_ids | 1.046 | 0.981 | 181.7925/224.6202 | 194.0185/202.0155 | 0.937 |
| multilingual_single | tokens | 0.370 | 0.346 | 2.5595/3.1061 | 2.7416/2.9790 | 0.934 |
| multilingual_single | ids | 0.363 | 0.339 | 2.6083/3.0863 | 2.8003/3.0966 | 0.931 |
| multilingual_single | offsets | 0.022 | 0.021 | 43.6025/51.5146 | 44.7154/48.0571 | 0.975 |
| multilingual_single | decode | 2.128 | 2.052 | 0.4455/0.4796 | 0.4621/0.4891 | 0.964 |
| multilingual_batch | tokens | 0.401 | 0.381 | 76.0630/83.7072 | 80.0729/87.9237 | 0.950 |
| multilingual_batch | ids | 0.387 | 0.383 | 78.7746/90.7741 | 79.7136/88.4830 | 0.988 |
| multilingual_batch | offsets | 0.021 | 0.021 | 1447.0912/1627.3078 | 1477.7045/1556.3806 | 0.979 |
| multilingual_batch | decode | 2.025 | 1.924 | 15.0616/16.7087 | 15.8486/17.2834 | 0.950 |
| multilingual_batch | iterative_tokens | 0.375 | 0.369 | 81.2942/89.9357 | 82.7280/87.8682 | 0.983 |
| multilingual_batch | iterative_ids | 0.384 | 0.369 | 79.3919/88.2861 | 82.5592/92.6887 | 0.962 |
| source_code_single | tokens | 0.375 | 0.417 | 3.4954/4.4347 | 3.1407/3.4900 | 1.113 |
| source_code_single | ids | 0.414 | 0.416 | 3.1669/4.7067 | 3.1482/3.9728 | 1.006 |
| source_code_single | offsets | 0.091 | 0.086 | 14.3264/18.6255 | 15.2030/16.6693 | 0.942 |
| source_code_single | decode | 1.071 | 1.072 | 1.2231/1.3317 | 1.2221/1.4365 | 1.001 |
| source_code_batch | tokens | 0.471 | 0.440 | 89.3126/97.8891 | 95.7298/100.4321 | 0.933 |
| source_code_batch | ids | 0.486 | 0.444 | 86.6356/101.9156 | 94.8298/102.1219 | 0.914 |
| source_code_batch | offsets | 0.083 | 0.073 | 509.0309/618.3539 | 576.3584/627.3881 | 0.883 |
| source_code_batch | decode | 1.110 | 1.079 | 37.9166/43.6810 | 38.9937/40.1808 | 0.972 |
| source_code_batch | iterative_tokens | 0.442 | 0.425 | 95.2367/113.4137 | 99.0181/113.3775 | 0.962 |
| source_code_batch | iterative_ids | 0.473 | 0.433 | 89.0330/100.8595 | 97.1380/110.0165 | 0.917 |

## Memory, parity, and validation

- The raw public receipts retain both per-process Python retained/peak measurements and RSS peaks for every cell. Overall process RSS peaks range from 220.94 to 251.19 MiB; they include imports, caches, warmups and previous workloads and cannot establish a per-call memory saving.
- Exact public snapshots agree across both default binaries and the instrumented binary for 29 cases: IDs, token text, raw offsets, batch outputs, decode, invalid IDs, security policies, compatibility markers, private-use escapes and surrogates. Ordinary spans also match an independent alignment oracle. All 50 matrix cells and 20 paired cells verify output hashes and named crossing counts.
- Public native signature text agrees across all three builds. Returned containers remain normal mutable lists of immutable strings; mutating a result and clearing the model cache leaves retained strings valid.
- Local validation: 27 default Rust tests and 30 allocation-profile Rust tests; Clippy with warnings denied in both configurations; 69 focused Python tests plus five subtests after publication; Ruff lint/format; Mypy on 37 source files; isolated combined-wheel install/uninstall/reinstall and CLI verification. No full local Python-suite green claim is made.
- GitHub full CI passed the implementation commit `49acc0e` and reporting-test commit `d22bc9a`: nine Python OS/version jobs, Rust, three native wheels and source distribution. Final PR-head status must be checked separately.

## Reproduction and receipts

Build the baseline commit and the implementation in separate checkouts with `python -m maturin build --release`, extracting each native extension from its wheel. Build a third implementation extension with `--features allocation-profile`. Preserve the default binaries separately; allocation-instrumented binaries are rejected for public timing.

```sh
python -m benchmarks.profile_native_boundary --before-native <baseline-extension> --after-native <default-candidate-extension> --profile-native <allocation-profile-extension> --baseline-commit a0fd66636051dbf03b83525a8557da335949c39b --output <new-directory> --raw-workers <new-local-file> --repetitions 11 --warmup 3 --threads 1
python -m benchmarks.profile_native_boundary --before-native <baseline-extension> --after-native <default-candidate-extension> --paired-output <new-paired-directory> --repetitions 21 --warmup 3 --threads 1
```

- [Metadata](metadata.json): commits, model/fixture/source/native hashes and environment.
- [Public API raw samples](public_api.jsonl): all 50 cells, crossing counts, outputs and Python/process memory.
- [Native raw stages](native_stages.jsonl): input/computation/materialization/release/reference timings, Rust allocation counts and copy volumes.
- [Paired raw samples](paired/results.jsonl) and [paired provenance](paired/metadata.json).
- [Checksums](manifest.json); binaries and development caches are deliberately untracked.
