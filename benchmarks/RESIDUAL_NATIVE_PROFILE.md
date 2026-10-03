# Residual native profiling protocol

The #95 warm-cache profile identifies regex work as the largest measured stage
and combines prefix lookup, segmentation and fallback in one cache stage. It
does not justify a trie layout, allocation or Viterbi rewrite. This extension
measures current cold segmentation and ordinary complete batch APIs before
retaining any optimization for #96-#99.

The segmentation matrix fixes 8K/16K/32K receipted baseline vocabularies,
32/256/4096-code-point inputs, and Latin, CJK, Indic, Arabic and repeated-ASCII
fixtures. The full pipeline fixes the 8K baseline, batch sizes 1/8/32/128,
short/medium/long inputs, string and ID outputs, and warm/cold cache scenarios.
Long inputs are single words above the existing 1024-byte cache cutoff, so
even the warm scenario exercises segmentation. Short/medium fixtures include
ordinary words, whitespace and punctuation. Input sequences and model hashes
are fixed; no train, validation or held-out test corpus is opened by this tool.

Primary timings use identical default release builds without allocation
instrumentation, one Rayon worker, Python hash seed zero, three warmups and
eleven repetitions. Iterations per repetition are fixed by input length and
batch size. Per-call wall time includes result materialization and disposal;
cold cache clearing is excluded. CPU utilization covers the entire repetition,
including cache clearing, and is expressed relative to one CPU core. Report
every raw sample, median and nearest-rank p95, normalized input decimal MB/s
and parity-identical tokens/s. Pair before/after records only if model, fixture,
token/ID/offset output and error hashes agree. Repeat before/after rounds on an
otherwise idle host; do not retain a change whose advantage is within noise.

A separate, untimed release build with the `allocation-profile` feature wraps
the ordinary System allocator. It records Rust requests and full requested
bytes (a successful realloc is one request for its full new size), Rust live
requested bytes before/after, and peak live requested bytes during the call.
Python and external-library heaps are excluded from these counts. Feature
instrumentation is absent from default builds. Measurements require isolated
fresh workers, since unrelated concurrent Rust allocation would contaminate
the global count. Nested or overlapping profiling rejects instead of blocking
across Python GIL release.

Each condition also records operating-system process high-water RSS in a fresh
worker, with pre-measurement high-water shown separately. It includes imports,
model/trie construction and runtime state; subtracting high-water marks does
not isolate live working-set allocation. Rust live-byte counters separately
describe temporary allocation and retained cache/output memory. No new pooling
or cache is introduced by the profiler.

The baseline diagnostic replay reports owned prefix collection time, DP time,
edges and states. Its per-position timers have overhead and exclude edge
assembly/backtracking, so these are stage observations, not exhaustive
production wall-time fractions. A new candidate must be justified by the
current allocation/profile evidence and measured independently.

Run from a clean committed checkout. Record the compiler, build flags, build
source commit, source/runtime identity, native binary hashes, hardware and
every request/response hash. Outputs must be new. Build ordinary and instrumented
extensions from the same commit and preserve each binary in a separate directory.

```powershell
maturin build --release --interpreter C:/Users/shaik/AppData/Local/Programs/Python/Python310/python.exe --out path/to/default-wheels
maturin build --release --features allocation-profile --interpreter C:/Users/shaik/AppData/Local/Programs/Python/Python310/python.exe --out path/to/allocation-wheels
python -m benchmarks.profile_residual_native --native path/to/default/uniqtoken_core.pyd --allocation-native path/to/instrumented/uniqtoken_core.pyd --models benchmarks/byte_fallback/issue86 --build-commit FULL_SHA --output artifacts/residual-native
```
