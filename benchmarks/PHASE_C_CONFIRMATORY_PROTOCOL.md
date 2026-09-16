# Phase C Confirmatory Protocol

## Status

This document freezes the design for a future confirmatory follow-up before any
Phase C LM result exists.

- Phase B is exploratory screening.
- The 16K byte-matched cohort is a post-screen exploratory candidate for
  confirmation. Its selection was not independent of Phase B outcomes.
- Phase C is a fresh three-seed experiment with materially larger training
  exposure, an independent validation partition, and a frozen analysis plan.
- This document does not authorize execution or test access.

The official Phase B screening ledger is bound by SHA-256
`0c145550a2b10b266f83d35e4cc5edc335eb401e0e133ebd09d908402539fb33`.
The official Phase B report must be referenced by its full-file SHA-256 from the
commit that adds this protocol. Neither artifact may be replaced in place.

## Research question

Does the lower BPB observed for UT-SuperBPE in the one-seed 16K byte-matched
Phase B screen persist across three new paired LM seeds after materially more
training, relative to both SPM-Unigram and Boundary-BPE?

The answer is limited to the frozen 16K tokenizer artifacts, model architecture,
corpus, normalization, and training regime below. It cannot establish general
UniqToken superiority.

## Fixed cohort

Only these Phase A tokenizer artifacts are eligible:

| Tokenizer | Vocabulary | Phase A artifact |
|---|---:|---|
| SPM-Unigram | 16,384 | `sp_unigram-16384` |
| Boundary-BPE | 16,384 | `boundary_bpe-16384` |
| UT-SuperBPE | 16,384 | `uniq_superbpe-16384` |

Artifact hashes, tokenizer configurations, normalization, special-token IDs,
and byte-fallback configuration must match the Phase B selection and Phase A
ledger exactly. Tokenizers are not retrained. SPM-BPE and UT-Unigram remain
outside this narrow follow-up; their omission is not evidence of inferiority.

## Training exposure

Phase C uses exactly **300,000,000 normalized UTF-8 bytes** selected from the
already frozen, untruncated 500,000,000-byte Phase A training source.

The exposure freezer must:

1. Exclude all 349 documents in the Phase B 1,000,000-byte exposure.
2. Preserve the existing normalization, source revisions, licenses, deduplication
   decisions, and 30 language/domain strata.
3. Allocate the 300,000,000-byte total proportionally to the source manifest's
   frozen per-stratum normalized-byte allocations using deterministic
   largest-remainder integer allocation.
4. Select whole upstream documents only and solve every per-stratum quota exactly.
5. Use deterministic source order and deterministic exact-packing tie breaking.
6. Inspect no tokenizer output, token count, FLOP count, validation value, or LM
   result while selecting documents.
7. Publish ordered document IDs, source and normalized byte totals, per-stratum
   accounting, source manifest SHA-256, and exposure manifest SHA-256.
8. Fail closed if any exact quota is infeasible, any Phase B exposure document is
   included, or any training/evaluation overlap is detected.

The resulting exposure manifest is an execution blocker until generated,
independently verified, committed, and bound into a launch receipt. The
300,000,000-byte target may not be silently reduced after feasibility is known.

## Validation and test isolation

Phase C uses only the confirmation half of frozen FLORES-200 dev, assigned by the
existing normalized-document SHA-256 parity rule. Phase B used the disjoint
screening half. The confirmation assignment hash and bytes must be frozen before
training.

FLORES-200 devtest remains untouched final-test data. Phase C training,
checkpointing, stopping, selection, and this report's success decision may not
read devtest. A later final-test evaluation requires a separate authorization
after the complete Phase C validation ledger is immutable; it must evaluate all
nine trained conditions, not only a validation winner.

## Model and training configuration

Phase C intentionally retains the Phase B screening architecture so the
follow-up isolates increased training exposure rather than confounding the
comparison with model scale:

- decoder-only causal Transformer
- 2 layers, `d_model=128`, 4 heads, `d_ff=512`
- untied input embeddings and output head
- learned positional embeddings, context 128
- ReLU, dropout 0.1
- AdamW, learning rate 0.001, weight decay 0.01
- betas `(0.9, 0.999)`, epsilon `1e-8`
- batch size 1 and float32, matching Phase B
- fixed one-pass traversal of the exact ordered 300 MB exposure
- no validation-driven early stopping or checkpoint selection

This design confirms only the small-model Phase B interaction. It is not a claim
about a larger production-scale LM.

## Seeds and conditions

Use paired LM seeds **1, 2, and 3**, all distinct from Phase B seed 0. Each seed
uses the same initialization and data-order seed across tokenizers.

The complete grid contains nine conditions:

`3 tokenizers x 1 vocabulary x 1 byte-matched regime x 3 seeds`

No FLOP-matched condition advances because none passed the post-screen
exploratory rule and the Phase B FLOP exposure was scientifically inadequate for
normal-training claims.

## Endpoints and analysis

The primary endpoint is confirmation-validation BPB computed from total causal
NLL and total normalized UTF-8 bytes. Lower is better. Total NLL is retained as
the additive audit quantity. Token perplexity may be reported within a tokenizer
but is not a cross-tokenizer ranking metric.

For each seed, calculate paired BPB deltas:

- `UT-SuperBPE - SPM-Unigram`
- `UT-SuperBPE - Boundary-BPE`

For each comparison, publish all three seed-level deltas, their arithmetic mean,
sample standard deviation, and two-sided 95% Student-t confidence interval with
two degrees of freedom. With only three seeds, confidence intervals are
descriptive and no asymptotic normality claim is made.

The exploratory candidate is considered confirmed only if both conditions hold:

1. UT-SuperBPE has lower validation BPB than each baseline in all three paired
   seeds.
2. UT-SuperBPE's mean relative BPB reduction versus the stronger baseline, where
   the stronger baseline is the one with lower three-seed mean BPB, is at least
   2.0%.

Otherwise the Phase B candidate is not confirmed. No alternative endpoint,
subgroup, seed deletion, or threshold may replace this rule after results exist.
All failures, non-finite values, interrupted conditions, and reruns must be
published with provenance.

## Integrity and publication

Before launch, the runner must bind and verify:

- this protocol's commit and full-file SHA-256;
- the Phase B ledger and report SHA-256 values;
- Phase A tokenizer artifact hashes;
- source and new 300 MB exposure manifest hashes;
- independent confirmation-validation assignment hash;
- exact model/training configuration and seeds;
- Git commit, Rust extension hash, Python/package/CUDA fingerprint;
- persistent checkpoint/resume controls and hard time/spending limits.

Each condition is written atomically. Resume skips only a complete condition
whose full provenance and artifact hashes revalidate. A final Phase C ledger is
published only after all 9/9 records validate. Partial execution remains
checkpoint evidence, never a complete result ledger.

## Remaining launch blockers

Phase C is not launchable until all of the following exist:

1. Exact tokenizer-blind 300 MB exposure manifest and independent verification.
2. Frozen confirmation-validation assignment receipt.
3. A Phase C runner reviewed against the current Phase B ledger schema; the older
   generic Phase C CLI is explicitly not authorized.
4. Analytical compute, wall-time, and cost feasibility receipt for all nine
   conditions without changing this protocol based on anticipated results.
5. Clean implementation commit, matching Linux runtime fingerprint, durable
   checkpoint probe, and immutable launch authorization.

No Phase C training and no test access may occur while any blocker remains.
