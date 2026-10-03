# Native encode and segmentation comparison

Two independent fresh-worker rounds, eleven paired repetitions per round, one Rayon worker.
Primary throughput is normalized decimal input MB/s. All token/ID/offset/error/decode and selected-path score hashes agree.

## Repeatability gates

- prefix, segmentation: {'repeatable_gain': 49, 'repeatable_degradation': 0, 'inconclusive_or_mixed': 0}
- prefix, batch: {'repeatable_gain': 19, 'repeatable_degradation': 2, 'inconclusive_or_mixed': 27}
- compact, segmentation: {'repeatable_gain': 49, 'repeatable_degradation': 0, 'inconclusive_or_mixed': 0}
- compact, batch: {'repeatable_gain': 22, 'repeatable_degradation': 0, 'inconclusive_or_mixed': 26}

Batch-size-one rejection gate: PASS; repeatable degraded cells: 0.

## Every Cell

`metrics.csv` contains all 97 cells for each variant, with p50/p95 latency, MB/s, parity-identical tokens/s, both CPU observation windows, and both round intervals. Raw requests/responses and hashes are in the source bundle.

A repeatable gain requires both round interval lower bounds above 1. A repeatable degradation requires both upper bounds below 1. Other cells are inconclusive or mixed; they are never silently discarded. The interval envelope shown in CSV is the union of the two round intervals, not a new confidence interval.

## Memory and Complexity

Every batch size has isolated allocation/requested-byte/live-peak and process high-water RSS observations. Raw segmentation counters cover the uncached Rust decoder before Python materialization; its timing API includes the ordinary memoization wrapper with cold clears outside call wall. Full-batch counters wrap the ordinary fused Rust API before Python materialization. Rust counts exclude Python and C++ heaps. Process RSS includes imports, model construction and validation; its high-water subtraction does not isolate live scratch. Prefix-only allocation and memory attribution is measured separately for all 49 segmentation cells; its trie layout and vocabulary are identical.

The compact unpruned algorithm takes O(n L) time and O(n) scratch plus O(n) selected output. At most four fallback tokens are emitted per source character. With no maximum subword length, L can equal n, yielding quadratic time but linear scratch. The old dense lattice stores O(n L) owned edges and up to O(n L squared) copied prefix bytes. Pruned decoding retains the old lattice, edge sorting and tie behavior. No additional cache, buffer pooling, unbounded retention, trie representation change or SIMD is introduced.

## Limits

CPU uses pooled process CPU/wall over eleven blocks, including loop overhead and cache clears; Windows accounting remains quantized. These are fixed local fixtures and two process rounds, not a general hardware/corpus claim. Warm short/medium batches exercise the unchanged cache/output path. Long words exceed its 1024-byte cache cutoff in both cache scenarios. No corpus, held-out test or downstream LM is used.

## Build Sources

- baseline: `658033b3479be40d402e1af880d0891417c6085a`
- prefix: `9a1c3e1e6097d5d4c378e166821a63ef3052b3cc`
- compact: `9e84dc25307ccc53575b0fa38c0b8b1e80f50c5b`

Compiler: `rustc 1.98.0 (88d9e12ae 2026-08-18)`

Release opt-level 3, LTO enabled, one codegen unit; allocation instrumentation disabled for all primary timings.
