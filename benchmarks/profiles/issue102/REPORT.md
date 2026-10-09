# Issue #102: already-NFKC ASCII fast path

The release-build comparison passes the defined parity domain and shows an
end-to-end improvement on the ASCII subset of already-NFKC input. Non-ASCII
input retains full NFKC normalization. This is not a general Unicode fast
path or a cross-tokenizer performance claim.

## Safety and scope

[SAFETY.md](SAFETY.md) specifies the invariant written before implementation
and reviewed against the token-only normalizer, aligned normalizer, legacy
native normalizer and native security gate. ASCII NFKC is the identity;
borrowing that intermediate cannot change downstream text or alignment.
All configured transforms and the canonical security check still execute.
Public APIs, Python code, Unicode/grapheme algorithms and release settings
are unchanged. The invariant is provided for maintainer review in the PR.

The exhaustive Rust domain passed **5,576,833 comparisons**: all ASCII strings
of length 0–2 plus all 1,112,064 Unicode scalar values in five fixed contexts.
The before/after binaries had **zero misses** across 20 synthetic texts and
all 64 native normalizer configurations: plain/aligned normalization,
custom metaspace, pretokenization, tokens, IDs, source offsets, single/batch
paths and decoding. Separate security-policy, invalid-input, reserved-space
and native-signature comparisons also passed. The receipt contains the
domain and output hashes. No held-out data was accessed; no Phase C was run;
existing research and release artifacts were not modified.

## Method

Baseline: `a0fd66636051dbf03b83525a8557da335949c39b` (main). Measured source:
`8253f15e4ec5aa435eac425421c202716a8041ed`. Later changes only format the raw
numeric arrays compactly and add verification/documentation; production
Rust and the measurement method are unchanged. [results.json](results.json)
records the committed source hashes, native binary hashes, model hash,
fixtures, every timing sample and stage medians.

Windows build 26300; Intel Core i5-12500H (12 cores / 16 logical processors);
Python 3.10.11 / Unicode 13.0.0; Rust/Cargo 1.98.0; release LTO; default
features; **no allocation profiler**; Rayon threads = 1. Both models have the
same 400-entry vocabulary trained only on the five embedded #95 fixtures.

There are 41 paired trials per variant/workload, alternating before/after
order, with three public encode calls averaged per trial and five warmup
trials. Python alias binding is outside the timer; the production ASCII
detection, security gate, normalization, segmentation, FFI and output
creation are inside it. MB/s uses normalized UTF-8 bytes and decimal MB.
P95 refers to trial-average latency, not individual-call tail latency.
Intervals are deterministic bootstrap intervals for the paired median
within this run; they do not measure variation between machines or runs.

## Throughput

`_1` is a single input; `_32` is a 32-row batch. Ratios are optimized/baseline
latency, so below 1 is faster. MB/s is computed from each variant's median
latency; the paired median ratio is a different summary of the same samples.

| Workload | Before MB/s | After MB/s | Paired ratio | 95% interval |
| --- | ---: | ---: | ---: | --- |
| short_1 | 0.785 | 0.862 | 0.901 | 0.875–0.943 |
| short_32 | 1.336 | 1.573 | 0.853 | 0.838–0.880 |
| medium_1 | 0.587 | 0.624 | 0.943 | 0.907–0.956 |
| medium_32 | 0.604 | 0.636 | 0.947 | 0.931–0.961 |
| long_1 | 0.940 | 1.025 | 0.920 | 0.897–0.943 |
| long_32 | 0.948 | 1.038 | 0.919 | 0.905–0.923 |
| source_code_1 | 0.418 | 0.428 | 0.968 | 0.956–0.990 |
| source_code_32 | 0.430 | 0.446 | 0.970 | 0.946–0.983 |
| normalized_multilingual_1 | 0.365 | 0.367 | 0.999 | 0.968–1.017 |
| normalized_multilingual_32 | 0.381 | 0.381 | 0.999 | 0.985–1.017 |
| non_normalized_multilingual_1 | 0.366 | 0.371 | 0.997 | 0.973–1.044 |
| non_normalized_multilingual_32 | 0.379 | 0.381 | 0.999 | 0.987–1.014 |
| compatibility_start_1 | 0.771 | 0.793 | 0.995 | 0.974–1.016 |
| compatibility_start_32 | 0.788 | 0.791 | 1.004 | 0.987–1.019 |
| compatibility_end_1 | 0.776 | 0.781 | 0.996 | 0.968–1.010 |
| compatibility_end_32 | 0.792 | 0.792 | 1.004 | 0.988–1.016 |
| combining_end_1 | 0.778 | 0.793 | 0.985 | 0.975–1.011 |
| combining_end_32 | 0.795 | 0.792 | 0.996 | 0.990–1.010 |
| normalized_combining_1 | 0.440 | 0.439 | 1.004 | 0.966–1.024 |
| normalized_combining_32 | 0.500 | 0.498 | 0.999 | 0.984–1.010 |

ASCII paired median latency improves 3.0–14.7%; all eight intervals exclude
1. Ratio-of-median throughput improves 2.3–17.7%. Every non-ASCII interval
includes 1: no end-to-end regression was detected on these normalized and
non-normalized controls, including a late compatibility character. This
bounded result is not a guarantee for all workloads or hardware.

The #95 report already identified material normalization work on long ASCII
batches. Current equivalent diagnostic replays measure normalization at
**10.262 → 2.461 ms** and the canonical security gate at **8.050 → 0.125 ms**
for `long_32`, whose public median latency is **200.726 → 183.311 ms**.
These are stage medians of summed per-row timers over seven untimed replays;
they are not additive wall-time percentages. Regex remains the largest
measured stage. Historical #95 timings are not directly compared with this
run's absolute throughput.

## Reproduce

Build main's default release extension into `before.pyd` (or the platform's
`.so`), then build the PR's extension into `after.pyd` with the same toolchain
and features. Use separate checkouts/build directories and install the PR
checkout with `python -m maturin develop --release` before running:

```sh
python -m benchmarks.profile_nfkc_fast_path --before /path/to/before.pyd --after /path/to/after.pyd --baseline-commit a0fd66636051dbf03b83525a8557da335949c39b --output /path/to/new-results.json
```

The harness requires clean committed source and a new output file, verifies
native execution/signatures/models and exact parity before recording timings,
and rejects allocation-profile binaries. Do not run CPU-heavy checks alongside
the measurements. The supplied baseline binary was independently built from
current main during #101 validation; its SHA-256 is pinned in the receipt.

Local validation: 100 focused Python tests plus 572 subtests; Rust default
20 unit + 8 C ABI tests; allocation-profile 23 unit + 8 C ABI tests; Clippy
with warnings denied for both feature sets; repository Ruff lint/format.
The fresh combined wheel passed isolated public/native imports, encode/decode,
CLI help, uninstall and reinstall checks. Full Python CI is reported on the
PR separately; these focused local checks are not a full-suite claim.
