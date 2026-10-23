# Experimental native merge prototype

This is the first Rust prototype for [#109](https://github.com/umran666/UniqToken/issues/109),
implementing the existing [SuperBpePassV1 contract](MERGE_ENGINE_DESIGN.md).
It is an internal research interface, not a replacement for the default merge
path or the existing Python `FastMergeEngine`. Neither the `"fast"` selector nor
the tokenizer's defaults select this prototype.

## Execution and provenance

The implementation uses call-local Rust nodes, adjacency indices, checked
generations and a priority queue ordered by eligible pass and original left
leaf. It does not prioritize CEM scores or learned ranks. Newly merged operands
become available next pass; untouched pairs can still run in the current pass.
Stale candidates do not draw dropout decisions. A dropped boundary is reconsidered
only after a constituent changes.

The implementation was written from this repository's execution contract and
oracle. No FastBPE source was copied, translated or vendored. General arena,
heap and generation-counter data structures are not evidence of faster execution.

## Supported capabilities

| Input | Prototype support |
| --- | --- |
| Profile | `SuperBpePassV1` only |
| Pieces | Exact UTF-8-representable Python strings, including empty pieces |
| Table | Exact `MembershipMergeTable` with plain `dict` membership and rank metadata |
| IDs | Existing nonnegative integer IDs through `2**64 - 1`; sparse IDs allowed |
| Ranks | Valid nonnegative integer metadata; ignored for scheduling |
| Cuts | Plain `frozenset` of original interior leaf boundaries |
| Extra legality | Absent or unmodified exact `AlwaysAllowLegality` |
| Dropout | Absent or unmodified exact `BooleanTapeDecisions` over a plain Boolean list/tuple |
| Output | Original leaf ranges, pieces, merged IDs and merge/decision counts |

Custom table, legality, decision or container implementations and overridden
methods are rejected before dispatch. Live Python RNG cursors, surrogate
strings and wider IDs are explicitly unsupported. Invalid configuration,
unsupported capabilities, exhausted tapes and internal invariant failures use
the existing internal error classifications; none silently invokes a Python
engine. Tape cursors advance by actual draws, including on execution errors.

The Rust call detaches from Python using owned snapshots and retains no Python
objects or scratch buffers afterward. Compiling/copying these snapshots is part
of an adapter call; this is not a zero-copy or allocation-optimized implementation.
Concurrent mutation of a table, constraints or decision tape during a call is
unsupported.

Source payloads remain caller-owned. Existing `plan_to_tokens` projection keeps
singleton tokens and uses first/last raw-character endpoints for merged groups,
including overlapping fallback spans. Leaf indices are not character or byte
offsets. This prototype adds no public byte-span field or stronger round-trip
guarantee.

## Build and verify

Use the project's existing Python and Rust development prerequisites:

```sh
maturin build --release --out dist --interpreter python
python -m pip install --force-reinstall dist/uniqtoken_core-*.whl
python -m unittest discover -s tests -p 'test_native_merge*.py' -v
cargo test --manifest-path crates/uniqtoken_core/Cargo.toml --all-targets
cargo test --manifest-path crates/uniqtoken_core/Cargo.toml --release --all-targets
```

When running from a source checkout, use the existing editable-install workflow
so its native extension is rebuilt too. A stale extension is an explicit error;
the native tests do not skip or fall back when the private binding is absent.
The binding is deliberately excluded from package exports and public stubs.

The tests compare actual compiled execution with `ReferenceMergeEngine`, including
400 trials with seed `1092026`, pass barriers, overlaps, changed-constituent dropout,
partial/exhausted tapes, Unicode-derived pieces, fallback/special spellings,
hard cuts, sparse IDs, source-span projection and unsupported capabilities.
This is initial prototype evidence, not the full artifact suite in #110 or
complete SuperBPE/CEM coverage in #114.

An explicit exploratory entry runs the existing synthetic matrix with a full
plan-equality gate before timing:

```sh
python -m benchmarks.fast_merge_engine_benchmark --engine native --out .venv/native-merge-smoke.json
```

Use an existing ignored output directory. Native runs require `--out` to protect
the historical benchmark output. The unchanged default invocation still selects
the Python engine. These timings compare against the reference engine; they do
not satisfy #111's controlled production comparison or justify #112/#113
optimization or #116 adoption. No speed claim or default change follows from
this prototype.
