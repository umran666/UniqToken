# Merge-Engine Adoption Decision (Issue #116)

**Decision: Outcome 2 — keep FastMergeEngine experimental and opt-in.**
The production merge core remains the inlined SuperBPE pass loop in
`CustomTokenizer._apply_cross_word_merges` / `_with_spans`. FastMergeEngine
stays selectable via `merge_engine="fast"` (or the
`UNIQTOKEN_MERGE_ENGINE=fast` environment variable) for evaluation and
prototyping. This decision is evidence-backed and revisable; the revisit
trigger is the native prototype (#109), not the passage of time.

Status: accepted, 2026-10-24.

## Evidence base

| Area | Artifact | Result |
|---|---|---|
| Correctness oracle | #108 (ReferenceMergeEngine) | readable pass-loop oracle matching production semantics |
| Differential verification | #110 (PR #147), 11,377 cases over 10 categories | zero mismatches, incl. dropout draw-order equality |
| Full SuperBPE/CEM semantics | #114 (PR #148), end-to-end on a CEM-trained tokenizer | zero drift across `encode`/IDs/offsets/`sample`/batch/decode |
| Production vs engines benchmark | #111 (PR #149), receipts in `benchmarks/merge_engine/issue111/` | production 7.84 MB/s geomean; fast 1.82 (4.3× slower); reference 1.42 |
| Engine optimizations | #112/#113 (PR #150), receipts in `benchmarks/merge_engine/issue112-113/` | fast engine +11–19% (best case: no workload regressed; hierarchical +127%); still ~3–4× slower than production in pure Python |
| Native prototype | #109 (PR #146, Rust arena + generation-guarded heap) | demonstrates the ceiling; out of scope for this decision |

All performance numbers are environment-scoped (single Windows 11 laptop,
noise floor documented in the receipts); the decision rests on the
*relative* production-vs-engine gap, which is far larger than that noise
floor on every workload.

## Trade-off table

| Dimension | Production inlined loop | FastMergeEngine (after #112/#113) | Assessment |
|---|---|---|---|
| Throughput (text workloads, pure Python) | 7.84 MB/s geomean | ~2.2–3.4 MB/s per cell (was 1.8–3.2) | production ~3–4× faster; decisive |
| Latency p50 | lowest on every cell | improved by hybrid worklist but still behind | production wins |
| Memory | per-pass list + remap dict | flat arrays + frontier; stateless per call | comparable; neither retains memory |
| Worst-case time | O(n·P) rescans | O(n + merges) hybrid (dense phases rescan-equivalent) | fast engine's asymptotics are better, but constant factors dominate at realistic n |
| Complexity / maintenance | ~80 lines, trivially auditable | ~150 lines with hybrid scheduling + exactness reasoning | production simpler to maintain |
| Compatibility (SuperBpePassV1) | reference semantics | proven identical (11,377 differential cases + full-semantics suite) | parity is not a blocker |
| Rollout / rollback | default path | opt-in env/arg, zero default-path changes | opt-in keeps blast radius zero |
| Native path | skipped for SuperBPE models | n/a (Python) | #109 prototype is the credible route to a real speedup |

## Reasoning

1. **Correctness is no longer the blocker** — #110/#114 are closed with
   zero-mismatch evidence, so a future adoption would not need new
   correctness work.
2. **Performance is the blocker, and Python cannot close it.** Even after
   the #112/#113 work, the engine pays dataclass/plan plumbing and
   interpreter overhead the inlined loop avoids. The measured gap (~3–4×)
   exceeds anything plausible from further Python micro-optimization.
3. **The inlined loop is also the simpler code.** Adopting a slower, more
   complex engine purely for architectural cleanliness would regress every
   user with no compensating capability; the generic `MergeEngine`
   protocol already provides the seam for experiments without adoption.
4. **The native prototype is the credible successor.** PR #146's Rust
   arena + generation-guarded heap ports the same scheduling ideas with
   native constant factors. If it lands with parity and a measured win,
   the adoption question should be reopened with fresh #111-methodology
   receipts — that is the revisit trigger for this decision.

## Consequences

- `FastMergeEngine` remains importable and opt-in; its selection surface
  (`get_merge_engine`, `merge_engine=` kwarg, env var) is stable API.
- The #110/#114 suites and the #111 benchmark run against both engines,
  so any future change that reopens adoption has its evidence harness
  already in place.
- #116 is closed with Outcome 2. #109/#146 continue as the performance
  track; #112/#113 are closed as delivered optimizations within the
  experimental engine.
