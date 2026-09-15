# Phase B FLOP Coverage-Policy Amendment Review

Status: approved at the methodological level by the user. Implementation and provenance gates remain launch-blocking. The budget remains exactly 1,000,000 normalized bytes; the quota-truncated exposure is rejected and must be regenerated.

## Existing protocol

- FLOP-matched target: `100,000,000,000` analytical FLOPs, accepted within 1%.
- Frozen exposure: the exact 353-document, 1,000,000 normalized-byte exposure.
- Existing coverage gate: at least 30 fully predicted documents at the FLOP stop.
- Existing scheduler: stops once the FLOP target is reached and may stop inside a document window.

The v6 tokenizer-only preflight established that the 1 MB exposure contains ample one-pass compute for every condition. The five rejections arise because the fixed target is reached before 30 documents, not because the exposure cannot supply the target.

## Proposed amendment

Keep the scientific compute budget unchanged and remove the incompatible competitive coverage threshold:

> A FLOP-matched condition is eligible when it reaches the declared FLOP interval and fully predicts at least **one exposure document** before the stop. `fully_predicted_documents` is a diagnostic reported for every condition, not a ranking or exclusion criterion.

The one-document minimum is a basic non-empty-run validity check, specified independently of tokenizer results. It prevents a no-op schedule while avoiding a condition-dependent threshold. The amendment must be committed and recorded before any LM result is opened or used for selection.

The same FLOP interval and one-document validity check apply to all tokenizers, vocabulary budgets, seeds, and evaluation regimes that use this FLOP-matched protocol. A condition that fails the FLOP interval or predicts zero complete documents remains blocked; no budget shortening, substitution, or partial-condition publication is permitted.

## Terminal-document semantics

The scheduler must distinguish document completion from byte-accounting prefixes:

- A document is `fully_predicted` when all of its text targets and its terminal EOS target have been consumed.
- If the FLOP stop lands exactly after those targets, that document counts toward the coverage minimum even if the current complete-document byte counter excludes it from the prefix total.
- A document with any unconsumed text or EOS target does not count as fully predicted.
- The record must retain both `completed_training_documents` (the existing byte-prefix counter) and `fully_predicted_documents` (the coverage-policy counter).
- No document membership, ordering, exposure bytes, or tokenizer input may change because of this accounting distinction.

For UT-SuperBPE 32K, this makes the existing terminal-document observation explicit: 29 byte-prefix documents plus a terminal document fully predicted at the stop gives 30 fully predicted documents, while the existing byte-prefix counter remains 29.

## Required record changes

Every FLOP-matched condition record must include:

- `flop_target`
- `flop_tolerance`
- `flop_interval`
- `coverage_policy_version`
- `minimum_fully_predicted_documents: 1`
- `fully_predicted_documents`
- `completed_training_documents`
- `terminal_document.fully_predicted`
- `terminal_document.target_tokens_consumed_including_eos`
- `coverage_gate`

The ledger must reject records missing these fields or reporting a fully predicted-document count below 1. Byte-matched records remain governed by the exact 1,000,000 normalized-byte requirement and are not converted into FLOP-matched records.

## Why this is not a post-hoc relaxation

This amendment does not change the target after observing LM results, and it does not choose a tokenizer or regime winner. It is a tokenizer-only feasibility correction made before Phase B training. The rule is a general non-empty-run validity check, with no validation/test metrics or LM outputs consulted.

The amendment should be rejected rather than activated if independent review finds that the one-document validity check is applied inconsistently, or that any coverage value was computed from mutable data or a changed tokenizer implementation.

## Independent blockers remain

This amendment does not resolve:

1. Reproduction of the pinned Linux tokenizer/runtime and extension hash on the GPU environment.
2. A clean commit containing the exact preflight/harness implementation, recorded before launch.
3. Provenance review of the source-manifest entry marked `truncated_to_quota`, including why it is compatible with the whole-document frozen exposure.

No Phase B launch is authorized until those blockers and this amendment have passed review and been committed as the active protocol.
