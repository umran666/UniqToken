# Hot-path profiler (issue #95)

From the repository root, build the native module in release mode and run the fixed profiler:

```powershell
maturin develop --release --skip-install
python -m benchmarks.profile_hot_paths --output benchmarks/profiles/issue95
```

The one profiling command is `python -m benchmarks.profile_hot_paths --output benchmarks/profiles/issue95` after the release build. On Windows with the GNU Rust toolchain, `dlltool.exe` must be on `PATH` for the build. The runner defaults to one Rayon thread, two warmups, seven repetitions, and ten iterations per repetition. Overrides are recorded in `results.json`.

The profiler trains a deterministic 400-piece toy vocabulary from its five embedded fixtures. The fixed matrix contains short, medium, long, multilingual, and source-code text, each as a single input and a 32-row batch. Batch rows get distinct numbered suffixes, and the fixture hash covers the complete matrix. It does not open research data or invoke Phase A/B/C programs. The runner checks public single and batch token strings and IDs, exact offset spans against an independent token-stream alignment oracle, decode parity, and diagnostic Rust replay before timing or writing performance artifacts. The oracle shares the public normalizer's character-to-source alignment but derives token boundaries without either tokenizer offset API. A failed gate raises and writes no new result files.

Public encode and decode timings are wall time; throughput is normalized UTF-8 input bytes divided by encode wall time in decimal MB/s. Tokens/s is secondary. The diagnostic Rust replay times security gating, normalization, regex matching, grapheme snapping plus chunk copies, cached segmentation, and token/ID copying. Its per-row elapsed times are summed; they must not be added to or subtracted from batch wall time. Cached segmentation includes cache lookup, trie traversal, Viterbi/lattice work, and byte fallback. The fused path excludes SuperBPE merge application. The FFI boundary, allocation, and copying overlap these operations and cannot be independently isolated by this instrumentation. The call-stack artifact covers Python dispatch and treats Rust as an opaque native call. Ranges are the observed minimum and maximum across seven repeated samples, not confidence intervals.

`results.json` contains every sample, observed 2.5/97.5 percentile range, exact runtime and build fingerprint, fixture and vocabulary hashes, and the parity result. `REPORT.md` provides the full stage table and scoped bottleneck summary. `python_callstack.txt` is the cProfile call-stack artifact. `manifest.json` gives SHA-256 hashes for those three files. These are measurements of this fixed fixture vocabulary and hardware, not evidence that a proposed optimization improves other vocabularies or workloads.
