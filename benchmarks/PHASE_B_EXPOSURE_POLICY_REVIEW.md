# Phase B representative training exposure: review proposal

Status: DRAFT FOR REVIEW ONLY. Not approved, implemented, sampled, or executed.
Review revision: 2 (explicit per-stratum accounting and sampling/preflight separation).
No tokenizer training, LM training, evaluation, or new scientific results accompany
this document. The active protocol and frozen selection remain unchanged.

## Fixed boundaries

This proposal changes only the choice and order of capped LM training documents.
It does not change the tokenizer cohort, tokenizers, vocabulary budgets, dataset,
normalization, special tokens, model, optimizer, LM seed, matching regimes, compute
estimator, validation partition, scoring rule, or final-test isolation.

- Harness commit inspected: `3d5e38d59d8bb628a0c51d6ca38807f2c5bb2307`.
- Frozen selection: `artifacts/phase-b-selection-v1.json`.
- Selection full-file SHA-256:
  `d3449786b636216057e01e1372b4805c1181c481ca521a74c9e29e07676a0ea4`.
- Dataset manifest SHA-256:
  `2ad27746d9c139c8d7e814e89c977c9bd4410033d5c1b97b673ee1b893ece7ff`.
- Parent Phase A screening training assignment:
  `43632315ee494ebec3d1bbc60e10ca34f3b44a549b993b4de20b44d3f1387810`.
- Screening validation assignment:
  `b772aa53702656009b4952b150df947043edd1b5e92a1201a842c21240f099e8`.
- Conditions: `sp_unigram`, `boundary_bpe`, `uniq_superbpe`, each at
  16384 / 32768 / 65536 entries; seed 0; both matching regimes, 18 LM runs.
- Existing ceilings stay at 1,000,000 normalized UTF-8 training bytes for byte
  matching and 1e11 analytical training FLOPs for FLOP matching. Neither is increased.
- Higher bytes per token means better compression. No BPT or other tokenizer/LM
  score is an input to exposure sampling or ordering.

The 999,467-byte English prefix was a proposed command budget, not an executed
result or a budget fixed in the tokenizer selection. Preserve its readiness report
as evidence; do not edit it to imply multilingual coverage. This proposal replaces
that prefix with a newly declared, exact whole-document budget beneath the same cap.

## Eligible source pool

Use only the existing 10,931-document, 74,998,005-normalized-byte Phase A screening
training assignment reconstructed from the frozen local train file. Do not enlarge
the pool to the full 500 MB corpus, download data, or change split membership.

The original manifest, train-file hash, deduplication status, per-document IDs,
normalized hashes, and parent training-assignment hash must verify before sampling.
Exposure policy determines which documents enter the fixed training exposure and
their order. It must not inspect tokenizer outputs, token counts, FLOPs, validation
metrics, or Phase B results. Sampling reads training text and immutable metadata
only, including normalized/source byte lengths. It must not load tokenizer models,
call an encoder or FLOP estimator, or accept scheduling-check results as inputs.
It must not read any validation/test text or LM result files. Validation
provenance can be checked against its frozen hash without using its text to sample.
The existing evaluation loader/protocol remains a separate, unchanged responsibility.

Keep every selected document whole and retain its original boundary. No truncation,
sentence extraction, synthetic text, concatenation into new training documents,
special-token insertion by the sampler, duplicated fillers, or quality/length
filters beyond the explicit packing rules below. Existing BOS/text/EOS processing
still belongs to the LM runner. Source and normalized UTF-8 byte counts stay distinct.

## Explicit per-stratum allocation and report

The following 30 rows, not four broad averages, are the allocation and acceptance
units. Their weights retain the proposed parent mixture without silently changing
it. Broad totals (320,000 / 320,000 / 160,000 / 200,000 bytes) are reconciliation
checks only: one language cannot compensate for another's shortfall. In particular,
Indic, CJK and Arabic-family languages must never pass by a combined average.
This is a stratified feasibility sample, not a universal language distribution.

Quota percentages below use the fixed 1,000,000-normalized-byte ceiling as their
denominator. They are allocated shares, NOT measured achieved shares. The future
report must separately give `100 * actual_normalized_bytes / B_actual` for every
stratum; summing achieved shares over all 30 rows must give 100%.

Each row requires at least one whole coverage document AND the integer minimum
bytes shown (`ceil(0.90 * quota)`). The global 950,000-byte minimum is additional.
`Pending` is not zero and is not a measured count: no sample has been generated.
`E` means early exhaustion is permitted only under the explicit rule below;
`R` refers to the fully specified deterministic rank/tie/order rule below.

| Parent domain / language | Normalized-byte quota | Allocated % | Minimum: documents / bytes | Selected whole documents | Early exhaustion | Order |
| --- | ---: | ---: | ---: | --- | --- | --- |
| latin_english / en (English) | 320000 | 32.0000% | 1 / 288000 | Pending | E | R |
| indic_cjk_arabic / ar (Arabic) | 22858 | 2.2858% | 1 / 20573 | Pending | E | R |
| indic_cjk_arabic / bn (Bengali) | 22858 | 2.2858% | 1 / 20573 | Pending | E | R |
| indic_cjk_arabic / fa (Persian) | 22857 | 2.2857% | 1 / 20572 | Pending | E | R |
| indic_cjk_arabic / gu (Gujarati) | 22857 | 2.2857% | 1 / 20572 | Pending | E | R |
| indic_cjk_arabic / hi (Hindi) | 22857 | 2.2857% | 1 / 20572 | Pending | E | R |
| indic_cjk_arabic / ja (Japanese) | 22857 | 2.2857% | 1 / 20572 | Pending | E | R |
| indic_cjk_arabic / kn (Kannada) | 22857 | 2.2857% | 1 / 20572 | Pending | E | R |
| indic_cjk_arabic / ko (Korean) | 22857 | 2.2857% | 1 / 20572 | Pending | E | R |
| indic_cjk_arabic / ml (Malayalam) | 22857 | 2.2857% | 1 / 20572 | Pending | E | R |
| indic_cjk_arabic / mr (Marathi) | 22857 | 2.2857% | 1 / 20572 | Pending | E | R |
| indic_cjk_arabic / ta (Tamil) | 22857 | 2.2857% | 1 / 20572 | Pending | E | R |
| indic_cjk_arabic / te (Telugu) | 22857 | 2.2857% | 1 / 20572 | Pending | E | R |
| indic_cjk_arabic / ur (Urdu) | 22857 | 2.2857% | 1 / 20572 | Pending | E | R |
| indic_cjk_arabic / zh (Chinese) | 22857 | 2.2857% | 1 / 20572 | Pending | E | R |
| cyrillic_african / am (Amharic) | 26667 | 2.6667% | 1 / 24001 | Pending | E | R |
| cyrillic_african / bg (Bulgarian) | 26667 | 2.6667% | 1 / 24001 | Pending | E | R |
| cyrillic_african / ru (Russian) | 26667 | 2.6667% | 1 / 24001 | Pending | E | R |
| cyrillic_african / sw (Swahili) | 26667 | 2.6667% | 1 / 24001 | Pending | E | R |
| cyrillic_african / uk (Ukrainian) | 26666 | 2.6666% | 1 / 24000 | Pending | E | R |
| cyrillic_african / yo (Yoruba) | 26666 | 2.6666% | 1 / 24000 | Pending | E | R |
| code / c | 22223 | 2.2223% | 1 / 20001 | Pending | E | R |
| code / cpp | 22223 | 2.2223% | 1 / 20001 | Pending | E | R |
| code / go | 22222 | 2.2222% | 1 / 20000 | Pending | E | R |
| code / java | 22222 | 2.2222% | 1 / 20000 | Pending | E | R |
| code / javascript | 22222 | 2.2222% | 1 / 20000 | Pending | E | R |
| code / python | 22222 | 2.2222% | 1 / 20000 | Pending | E | R |
| code / rust | 22222 | 2.2222% | 1 / 20000 | Pending | E | R |
| code / sql | 22222 | 2.2222% | 1 / 20000 | Pending | E | R |
| code / typescript | 22222 | 2.2222% | 1 / 20000 | Pending | E | R |

The integer quotas use equal language shares within the original parent domains,
with division remainders assigned one byte at a time by ascending language ID.
The table is normative; parent-domain grouping does not replace row-level checks.
No unlisted strata, silent language substitutions or quota transfers are allowed.

### Early exhaustion rule E

Yes, any stratum can stop contributing before the global 1 MB ceiling. Distinguish
these cases in every row of the eventual report:

- `quota_filled`: its assigned quota is exactly filled; normal local completion.
- `no_whole_document_fits`: candidates remain but none fits the unused quota;
  report the unused bytes and count of oversized/skipped candidates.
- `eligible_pool_exhausted`: every eligible unique candidate has been consumed;
  report eligible versus selected document counts and byte totals.

Use that order of precedence when more than one description applies. Underfilling
is acceptable ONLY if that row still passes its coverage/minimum-byte gates and
the global minimum passes. Otherwise fail the sample. Do not borrow another row's
quota, repeat documents, truncate, or grow the cap. Separately report the final
global position of each stratum's selected queue; exhaustion of that queue during
playback is not evidence that the entire eligible parent pool was exhausted.

### Required generated report and exact hash

After separate approval to generate a sample, report ALL 30 rows, including any
failure rows. In addition to the quotas and minimums above, each row must contain
actual normalized bytes, achieved percentage of `B_actual`, original source bytes,
eligible/selected whole-document counts, unused quota, exhaustion reason, and its
ordered selected document IDs. Record the rule R/version, coverage-document ID,
each selected document's rank, tie-break keys and global order position, plus a
per-stratum hash of its ordered document-ID/normalized-text-hash/byte-count list.
The report must show explicit per-stratum PASS/FAIL, not just four domain totals.

The resulting complete exposure manifest does not exist yet. Its exact full-file
SHA-256 and selected document counts are therefore PENDING, not invented values.
Generation must publish the real 64-hex SHA-256 in a separate receipt/report,
linking all 30 rows to the same manifest; the per-stratum ordered-list hashes are
additional checks, not substitutes for that full manifest hash. Do not place a
manifest's full-file hash inside its own hashed contents. Preserve the existing
canonical-content-hash convention separately. A missing hash or pending count is
a hard blocker to preflight/launch, not an acceptable completed exposure report.

## Deterministic selection and order

This section is rule R, shared by all 30 explicitly identified strata above.
Proposed policy identifier: `phase_b_stratified_whole_document_exposure_v1`.
It is distinct from the unchanged LM seed. There is no sampler-seed search.

1. Define a candidate rank using the existing canonical JSON SHA-256 convention
   (`research.digest`) on an object containing `policy`, `purpose="candidate"`,
   `domain`, `language`, original document `id`, and `normalized_text_hash`.
   The text hash uses the parent's existing normalized-document hash convention.
   Sort ranks ascending, with document ID as the final tie-breaker. Duplicate IDs
   or normalized hashes fail validation rather than creating two candidates.
2. In every stratum, reserve one coverage document: the shortest whole document
   that fits that stratum's byte quota. Break length ties by candidate rank, then
   ID. Fail if no document fits. These 30 coverage documents are explicitly
   length-biased; they are not an unbiased estimate of the corpus distribution.
3. Visit the remaining documents of each stratum once in candidate-rank order.
   Include a document if its normalized byte length fits the unfilled quota;
   otherwise skip it. Do not backtrack, reroll the rank salt, optimize a subset
   for tokenizer scores, or redistribute a shortfall to another stratum.
4. Order coverage documents first. Within each macro stratum, order its language
   queue by the canonical digest of `{policy, purpose="coverage_order", domain,
   language}`, with language ID as tie-breaker. Visit macro strata cyclically in
   ascending domain-ID order, taking one queued coverage document at a time and
   skipping exhausted queues. The first four documents therefore cover all four
   macro strata, and the first 30 cover every language/domain stratum.
5. Order the remaining selected documents by weighted byte service. Start each
   stratum's served-byte counter with its coverage-document length. Repeatedly
   choose a nonempty stratum minimizing `served_normalized_bytes / quota_bytes`;
   compare ratios by integer cross multiplication. Break ties by `(domain,
   language)` ascending. Emit its next selected document in candidate-rank order,
   update served bytes, and continue until all selected documents are emitted.
6. Let `B_actual` be the sum of normalized UTF-8 lengths of this ordered list.
   Freeze its IDs, order, hashes, byte counts and `B_actual` BEFORE any LM run.
   Use `B_actual` as the exact byte-matched budget for every condition, not a
   rounded 1 MB or a tokenizer-specific achieved count. Whole documents mean
   `B_actual` may be below the ceiling; this is declared before execution, not
   accepted retroactively as an under-budget result.

Proposed acceptance thresholds, to approve before generating a sample:

- Every one of the 30 strata contributes at least one unique whole document.
- No stratum exceeds its quota; each uses at least 90% of its quota.
- Total normalized exposure is between 950,000 and 1,000,000 bytes inclusive.
- Every document belongs to the verified parent assignment, with no repeated ID
  or normalized hash in the sampled list.
- Repeating construction with the same frozen inputs produces byte-identical
  exposure-plan content, regardless of directory enumeration or worker scheduling.

These are proposed representativeness/packing gates, not measured feasibility.
If the source pool cannot satisfy them with whole documents, publish the failure
and stop for review. Do not silently lower thresholds, expand the pool, duplicate
documents, alter the cap, or try different rank salts until one passes.

## Applying the same exposure to both regimes

Both regimes consume the identical frozen sampled list in the identical order.
All nine tokenizers see the same document source; no tokenizer selects its own
language mix, crop, rank, or start offset.

For byte matching, process the entire sampled list exactly once. The shared
normalized budget is `B_actual`; the shared secondary exposure is the sum of
original source bytes for precisely those documents. This gives identical text,
document boundaries and source/normalized byte exposure across all nine conditions.

For FLOP matching, preserve the existing stopping rule and estimator, including
vocabulary-dependent output projection and the one-percent tolerance. Stop at the
budget in the same ordered stream; retain the existing repeat-from-start behavior
if a condition completes the list before reaching its compute budget. Do not
equalize actual bytes by changing the compute budget or the stream per tokenizer.
Different tokenizations can reach different stopping points. FLOP matching must
not be described as identical byte exposure or identical achieved language shares.

The coverage-first order reduces the English-prefix failure mode but does not
guarantee coverage under 1e11 FLOPs. Exposure construction and its acceptance gates
finish WITHOUT tokenizer outputs, token counts or FLOPs. Only after the exposure
manifest and the all-30-strata report are frozen and hashed may a separately
authorized, read-only TOKENIZER-ONLY preflight encode these training documents
with the nine saved artifacts and simulate the existing window/FLOP stopping rule.
It must perform no LM initialization, training, NLL evaluation, or test access.
The preflight is not part of the sampler and cannot publish a replacement exposure
manifest. Verify exposure and selection hashes before and after the check; write
its feasibility findings to a separate report linked to those unchanged hashes.

Proposed launch gate: each of the nine FLOP-matched schedules must finish all
30 coverage documents within its budget. Report predicted completed documents,
per-stratum source/normalized bytes, and the partial-final-document boundary.
Actual execution must later verify the schedule accounting. Reaching one short
document per language is only a coverage floor, not proportional exposure or
evidence of adequate learning. Report achieved shares and imbalance separately.

The scheduling check is a PASS/FAIL feasibility gate only: it cannot change the
frozen document list/order, select tokenizers, or choose a budget after inspecting
token counts. If any condition fails coverage, stop the entire launch for an
explicit policy review. Do not drop that tokenizer, relax its gate, or resample
to make it pass. Preserve the failed plan/check as evidence.

Keep the existing byte accounting scope for FLOP matching: complete-document
source/normalized byte totals exclude a partial final document, while token/FLOP
totals include all executed windows. Mark the partial document's ID and token
boundary as audit metadata without inventing a source-byte fraction. No change to
validation NLL, BPB, CE, perplexity, contexts, or test isolation is proposed.

## Additive provenance, not a rewritten selection

If approved, create a NEW immutable exposure-plan artifact, separately versioned
and hashed. Link it to the original selection SHA and original dataset/assignment
identities; never replace them with the subset hash. Preserve the frozen selection
bytes and the old readiness/prefix evidence. Proposed exposure-plan fields are:

- Policy identifier/version and exact allocation, rank, order and acceptance rules.
- Original selection SHA, dataset manifest/train-file hashes, source revisions,
  parent screening training-assignment hash and unchanged validation identity.
- Ordered document IDs, domains, languages, normalized text hashes, original
  source provenance references and source/normalized byte lengths.
- Coverage-document flags, per-stratum quotas/actual totals/document counts,
  allocated and achieved percentages, integer minimums, exhaustion reasons,
  deterministic rank/order evidence, per-stratum ordered-list hashes, ceiling
  and `B_actual`, aggregate source bytes and ordered exposure hash.
- Original tokenizer artifact/configuration hashes, complete condition list,
  approved policy/sampler implementation hashes and generating Git commit.
- Explicit `tokenizer_outputs_used_for_sampling=false`,
  `token_counts_used_for_sampling=false`, `flops_used_for_sampling=false`,
  `selection_metrics_used=false`, `lm_results_used=false`, and
  `validation_or_test_text_used_for_sampling=false` declarations.
- Canonical content SHA-256 plus an externally pinned full-file SHA-256. Keep
  timestamped approval/execution events in a separate receipt so deterministic
  sample content does not depend on generation time.

The eventual execution plan must lock BOTH selection and exposure hashes before
the first LM starts. Results distinguish parent corpus assignment from effective
LM exposure assignment; the subset must not masquerade as a new parent dataset
or as the whole 75 MB having been presented to the LM. Save per-stratum exposure
audits for each regime, preserving original normalized/source byte definitions.
Changing the exposure after any LM result exists requires a separately labeled
experiment, not an in-place update or a resume with a different sample.

### Current implementation compatibility

This policy cannot be activated by changing a CLI byte count alone. The current
runner passes the parent order directly to the LM, and the frozen selection pins
the current executable implementation digest. Changing the sampler/runner would
therefore fail the current selection validation even if all tokenizer conditions
and the selection file's SHA remain unchanged.

A later implementation needs an explicitly reviewed ADDITIVE exposure authorization
and an exact code-change approval linking the unchanged original selection to the
new exposure-plan integration. It must continue verifying the historical tokenizer,
dataset and evaluation code; it must not simply ignore the frozen implementation
pin, forge an identity, or rewrite the selection. That compatibility decision is
not implemented or authorized by this draft. Preserving both the original runner
unchanged and a newly rebalanced exposure is not currently supported.

## Required checks before activation

Future regression coverage should prove deterministic selection/order, all 30
strata and quota gates, exact whole-document byte budgets, source-byte auditing,
unchanged validation/test access behavior, rejection of unapproved parent documents,
stale selection/dataset/artifact/exposure hashes, missing languages, duplicates,
and changes after execution starts. Reordering the input enumeration must not
change the sample after parent assignment verification. A sample-construction
failure and a FLOP-coverage failure must launch zero LM conditions.

Changing tokenizer outputs, token counts, FLOP estimates or metrics, or supplying
LM result files must have no effect on the sampler (which must not accept those
inputs). Test that a failing preflight leaves manifest bytes, all selected IDs/order,
and all 30 accounting rows unchanged. Test quota and percentage arithmetic for each
row independently, with no aggregate compensation for a failed row; require the
real manifest hash before preflight. Regression tests must separately
show identical exposure under byte matching and truthful, potentially different
completed-document exposure under FLOP matching. Existing causal-LM accounting and
evaluation regression gates remain required; no new evaluation metric is proposed.

After review approval, the sequence would be: implement the reviewed sampling and
additive authorization, run synthetic regressions, construct and freeze ONE exposure
plan, perform the separately authorized tokenizer-only scheduling check, review
its coverage report, then configure persistence/cost limits and seek launch approval.
No stage in that sequence is executed by this review document.

## Interpretation limits and decisions for review

Approval would cover the quota-preserving whole-document policy, the explicit
short-document coverage bias, the proposed 90%/95% packing thresholds, and the
all-30-strata FLOP launch gate. None is claimed feasible until checked. Refusing
to launch because the cap is too small is a valid outcome.

One MB remains a very small sample. Quotas preserve coarse language/domain coverage,
not topic, document-length, dialect, code-project or licensing-distribution balance.
Coverage-first ordering particularly biases very short FLOP-matched prefixes.
Report these limitations and actual exposure shares; do not call this confirmatory
or infer multilingual LM superiority merely because every stratum appears.

This proposal keeps the nine selected tokenizer conditions and screening decision
fixed. Its only intended effect is to replace accidental corpus-file-order exposure
with an explicitly declared, reproducible training exposure before LM results exist.
