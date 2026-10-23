# Native Merge Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the experimental native prototype requested by issue #109 without changing tokenizer defaults or claiming improved performance.

**Architecture:** Follow the merged `SuperBpePassV1` execution contract. A private Rust node arena and candidate heap order work by eligible pass and original left-leaf index, never by learned rank. An internal Python adapter constructs immutable inputs, checks capabilities, and projects the returned leaf groups using the existing output adapters.

**Tech Stack:** Existing Rust crate, standard-library collections, existing PyO3 0.29, Python unittest; no new dependencies.

**Spec:** [MergeEngine execution contract](../../MERGE_ENGINE_DESIGN.md), particularly scheduling, randomness, errors and borrowing; [issue #109](https://github.com/umran666/UniqToken/issues/109).

## Global Constraints

- Preserve `SuperBpePassV1`, vocabulary membership, supplied IDs and leftmost non-overlapping pass order.
- New merge results become operands only in the next pass. Stale candidates consume no dropout decisions.
- Dropout blocks persist until a constituent changes; position changes alone do not release them.
- No public exports, default-path changes, selector changes, model-format changes, scoring changes or fallback to Python execution.
- Preserve caller-owned source spans and payloads through original leaf ranges; native code does not infer text or byte offsets.
- No research dataset access, Phase C launch, frozen-artifact edits, release changes or performance conclusions.
- The first supported native capabilities are exact UTF-8 membership tables, nonnegative IDs fitting `u64`, hard cuts, absent/always-allow additional legality and absent/Boolean-tape decisions.
- Recognize exact supported table/legality/tape implementations; reject subclasses or instance overrides whose behavior would be replaced by compilation. Reject unsupported profiles, arbitrary callbacks, live Python RNG cursors, surrogate strings and wider IDs before dispatch. These are explicit prototype capability limits, not a redefinition of supported semantics or production readiness.
- Classify invalid identities, cuts, negative/noninteger IDs and malformed tape cursors as `InvalidConfiguration`; unsupported capabilities as `UnsupportedSemantics`; tape exhaustion as `DecisionError`; corrupted adjacency/partition as `InvariantViolation`.
- No copied, translated or vendored FastBPE implementation. Use only repository contracts and ordinary data structures.

## Review Focus

- Overlapping candidates and changed adjacency must not alter pass visitation or consume extra decisions.
- Empty pieces, repeated pieces and sparse IDs must preserve group provenance and terminate.
- Hard cuts refer to original leaves; byte fallback may have overlapping raw spans.
- Unsupported requests must fail before any decision is consumed; exhausted tapes must preserve the consumed cursor and return no partial plan.
- A missing/stale native extension must fail the native-specific tests rather than exercise a Python fallback.

## Task 1: Native execution and oracle parity

**Files:**
- Create `crates/uniqtoken_core/src/merge_engine.rs`: private execution types, queue, native binding and Rust unit tests.
- Modify `crates/uniqtoken_core/src/lib.rs`: private module declaration and underscored experimental binding registration only.
- Create `uniqtoken/native_merge_engine.py`: private research adapter; do not re-export it or change `get_merge_engine`.
- Create `tests/test_native_merge_engine.py`: actual compiled execution against `ReferenceMergeEngine` and literal hand-checked vectors.

**Interfaces:**
- Adapter: `NativeFastMergeEngine.apply(input, table, constraints, decisions=None) -> MergePlan`, using the existing internal contract types.
- Private binding: `_experimental_merge_plan(pieces, eligible, hard_cuts, tape)` accepts owned UTF-8 strings, a string-to-`u64` membership map, validated cuts and an optional finite Boolean tape.
- Binding result carries ordered `(start, end, piece, merged_id)` groups, applied merges, consumed decisions and an optional classified error. Dispatch only the tape suffix after its existing cursor; return actual consumed count on every execution error as well as success. The adapter advances the original cursor accordingly, preserves exhaustion-message accounting against the full original tape, and never returns a partial plan.
- Pure Rust execution returns a result containing a plan or a classified error with consumed-decision count. It holds no Python objects or reusable input buffers after the call.

- [ ] Write tests asserting a native entry exists and executes literal pass-barrier and leftmost-overlap cases; confirm failure before implementation.
- [ ] Add Rust unit tests for leftmost selection, next-pass availability, stale adjacency, unchanged-constituent dropout blocking, changed-constituent unblocking, empty pieces, hard cuts and exhausted decisions.
- [ ] Implement call-local arena nodes with leaf ranges, piece identity, merged ID, neighbors, liveness and checked generation counters. Queue keys contain eligible pass, left-leaf position, both node identities and both generations.
- [ ] Discard stale/dead/nonadjacent, obsolete-pass and duplicate-boundary entries before legality or RNG. Resolve current pieces, check cuts, then consume at most one tape decision. After a merge in pass `p`, enqueue both changed neighboring boundaries for `p+1`, invalidate old generations and preserve untouched candidates for `p`. A dropped left pair still permits its right constituent to participate later in `p`; an unchanged dropped boundary is not requeued.
- [ ] Check that outputs partition the input and that `applied_merges == input_length - output_length`, including empty input. Bound counter arithmetic explicitly.
- [ ] Implement the adapter's capability checks before dispatch, including model identity, cut range, UTF-8 representability, exact table/legality implementation, integer width and Boolean tape type/cursor. Reject overridden `resolve`, `allows` and `drop_next` behavior explicitly. Translate native errors to existing internal classifications and advance tape cursors by actual consumed draws on success and every execution error.
- [ ] Compare groups, IDs, pieces, merge counts and decision counts against the reference with fixed seeded random cases and adversarial Unicode/special/fallback/repeated inputs. Use independent identical tapes; test rank metadata does not change scheduling.
- [ ] Assert original raw-span projection, including overlapping fallback spans, is unchanged and inputs are not mutated.
- [ ] Assert unsupported requests consume no decisions, including overridden supported-class subclasses and instance methods. Test partially consumed and already exhausted tapes, plus failure after one or more consumed decisions. Patch the reference path to raise if invoked during a native request.
- [ ] Run `python -m unittest discover -s tests -p test_native_merge_engine.py -v` against a freshly built extension; require no native-unavailable skip.
- [ ] Run Rust tests in debug and release, default and no-default-feature configurations. Record actual local toolchain failures rather than using the September extension as evidence.

## Task 2: Research entry, validation and PR

**Files:**
- Modify `benchmarks/fast_merge_engine_benchmark.py`: explicit native-engine selection with exact pre-timing parity, leaving the existing default invocation unchanged.
- Create `tests/test_native_merge_benchmark.py`: isolated benchmark-entry and evidence-output protection tests.
- Create `docs/NATIVE_MERGE_PROTOTYPE.md`: capability limits, execution provenance, fresh-build commands and validation evidence, with no performance/adoption claim.

**Interfaces:**
- Benchmark opt-in flag selects `NativeFastMergeEngine`; unavailable native support is a hard error. Timing remains exploratory and cannot satisfy issue #111's production comparison contract.
- No frozen benchmark receipt or methodology is rewritten. Full differential artifacts (#110), semantic coverage (#114), controlled production comparisons (#111) and adoption (#116) remain separate dependencies.

- [ ] Add a focused entry-point test proving explicit native selection performs parity checking and cannot silently select the Python engine.
- [ ] Run the synthetic benchmark entry as a smoke check only; label outputs exploratory and keep generated outputs untracked.
- [ ] Run `cargo test --manifest-path crates/uniqtoken_core/Cargo.toml --all-targets`, release tests and the repository's Clippy checks.
- [ ] Run `python -m unittest discover -s tests -p "test_*.py" -v` and repository Ruff/Mypy commands with the freshly built extension. Report every failure; establish baseline reproduction for unrelated failures.
- [ ] Build the combined wheel and sdist, verify unchanged public imports/default tokenizer behavior, and run the existing release-verification tests.
- [ ] Obtain an independent whole-diff review focused on the five Review Focus cases.
- [ ] Check `git diff --check`, changed paths, and that research/release artifacts and held-out data were untouched.
- [ ] Create a ready PR for #109 with exact supported/unsupported capabilities and local validation limits. Require the existing Linux/macOS/Windows CI matrix on the exact head before claiming cross-platform build success. Do not merge or claim downstream issues complete from this prototype.

## Status

Plan approved by the user on 2026-10-23 for implementation in this session. Native execution and oracle tests are in progress. PR #142's independent corpus-identity repair is already published and is not part of this plan.
