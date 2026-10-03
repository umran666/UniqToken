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

The follow-up `benchmarks.compare_residual_native` protocol compares baseline,
prefix-only and compact variants in the same fresh worker, with separate tries.
It preserves the 93-cell matrix and adds four synthetic dense-prefix lengths
(32, 256, 4096 and 16384 characters). The stress vocabulary has all 16 prefixes
of repeated `a`, four controls and 256 fallback bytes; it is not a trained model.
Two independent worker rounds each have three warmups and eleven repetitions.
Calibration chooses a common iteration count targeting 50 ms for the fastest
variant, bounded by 80000 iterations. Ordering alternates forward/reverse across
repetitions and rounds. All variants use the same compiler and release flags.

Report median paired throughput ratios and fixed-seed 95 percent percentile
bootstrap intervals in each round. These describe local timing variation, not
corpus or hardware population uncertainty. Retain the prefix change only where
both rounds show a gain beyond noise in its targeted cold segmentation cells;
likewise require a repeatable compact-path gain. Reject a retained final variant
with repeatable statistically detectable batch-size-one degradation (both rounds'
upper ratio bounds below one); inspect isolated regressions instead of hiding
them in a pooled average. Warm short/medium batches primarily measure unchanged
cache and output paths. Long chunks exceed the existing cache limit in both
warm and cold scenarios. No additional cache or pooling is introduced.

CPU utilization pools CPU time and wall time over all eleven blocks rather
than averaging short, quantized Windows CPU samples. It includes cache clearing
and loop overhead and remains approximate. The two variants' allocation and
RSS observations run in separate fresh processes for every cell, outside primary
timing. Shared timing-worker RSS cannot attribute memory to a variant and is not
used for that purpose. Hash and verify every stream, error and selected path
score before timing; all raw requests, responses and build identities are retained.

The compact path visits at most `L` trie characters at each of `n` source
positions: worst-case time is O(n L), scratch space O(n), and selected output
space O(n) (at most four fallback tokens per source character). With an unbounded
maximum token length, L can equal n and time becomes quadratic; scratch remains
linear. The former lattice held O(n L) owned edges whose copied prefix strings
can occupy O(n L squared) bytes on dense prefixes. Pruned decoding retains its
existing lattice and pruning behavior. The trie representation, EM expectation
path, segmentation cache policy and public prefix-list API remain unchanged.
