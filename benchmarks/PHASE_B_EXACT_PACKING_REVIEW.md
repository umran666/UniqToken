# Phase B exact whole-document packing: revision for review

Status: DESIGN ONLY. Not implemented, approved, searched, sampled, or executed.
This is an additive proposal, not an amendment to existing evidence. The earlier
review document and the 992,457-byte rejected candidate must remain byte-unchanged.
No tokenizer/FLOP feasibility preflight or GPU run is authorized by this document.

## Scope and fixed identities

Replace only the one-pass greedy packing step of policy
`phase_b_stratified_whole_document_exposure_v1`. Proposed packing policy ID:
`phase_b_stratified_whole_document_exact_subset_v2`.

Preserve the original parent pool (10,931 documents, 74,998,005 normalized bytes),
all 30 integer quotas from `PHASE_B_EXPOSURE_POLICY_REVIEW.md`, source provenance,
normalization, whole-document boundaries, coverage requirements, nine tokenizer
conditions, LM seed, both matching regimes, and all evaluation rules.

- Frozen selection SHA-256:
  `d3449786b636216057e01e1372b4805c1181c481ca521a74c9e29e07676a0ea4`.
- Frozen source manifest SHA-256:
  `2ad27746d9c139c8d7e814e89c977c9bd4410033d5c1b97b673ee1b893ece7ff`.
- Original review document SHA-256:
  `fa24447df1a6cdebb5f7f727f83f993009df84031c2d9df5deab6320971735a8`.
- Rejected candidate SHA-256:
  `3bcee5fac2046b12f880ffa6dfffc1d5804eba17d5215c4eb4273cb83b04ccee`.

The rejected candidate remains rejected, with 317 documents and a 7,543-byte gap.
It is a diagnostic input to this policy review, not an incumbent to be patched,
scored, or optimized. The v2 solver constructs a separate candidate from the same
eligible parent documents. It does not take the rejected candidate's IDs as input.

## Exactness and allocation tolerances

For stratum s, let Q_s be its existing integer quota and B_s its selected normalized
UTF-8 byte count. Existing constraints are `ceil(0.9 * Q_s) <= B_s <= Q_s`, at least
one coverage document, and `sum(Q_s) = 1,000,000`.

Consequently `sum(B_s) = 1,000,000` is possible under these same limits only when
**every B_s equals Q_s exactly**. No positive quota transfer or final stratum
overshoot is allowed. The old 90% minimum remains checked but exact filling is
now the stronger acceptance gate. It is not enough to compensate one language's
shortfall with another language's excess. Allocated and achieved normalized-byte
percentages will coincide on success, without changing the quota table.

The proposed search explores alternative inclusions and exclusions within each
stratum, including reversing greedy inclusions that caused undershoot. Because
every document has a positive byte length, states above that stratum's remaining
quota can be discarded without losing an exact solution. An overshooting candidate
is rejected during search, not included and later cropped. This supplies a bounded
exact search without introducing a new overshoot tolerance or a cross-stratum
balancing step.

Exact filling is not guaranteed to exist. For example, even-length eligible
documents cannot fill an odd residual. A complete search can prove infeasibility
under these constraints; a time/memory interruption cannot.

## Deterministic exact search

The solver accepts only verified parent document metadata, normalized byte lengths,
the fixed quota table, and the fixed ranking/coverage rules. It must not accept or
load tokenizer artifacts, outputs, token counts, FLOPs, validation metrics, or LM
results. Document identities and source-byte counts are retained for auditing;
neither source-byte length nor downstream results are a packing objective.

1. Verify the frozen source/parent assignment and unique IDs/normalized hashes.
   All eligible documents remain in their original `(domain, language)` stratum.
2. Preserve the original hash-ranking inputs exactly. In particular, retain the
   v1 namespace `phase_b_stratified_whole_document_exposure_v1` for BOTH
   `purpose="candidate"` and `purpose="coverage_order"`. The v2 packing ID is
   recorded separately, not used to reroll the rank salt. Candidate order remains
   ascending `(candidate_rank, document_id)`.
3. Preserve the shortest-fitting mandatory coverage document for each stratum,
   tie-broken by candidate rank then ID. Let its normalized length be C and its
   quota Q. Fail if none fits. Set residual target T = Q - C. Never replace this
   coverage document to make a later search succeed. If T is zero, select only
   this document for the stratum.
4. Remove the mandatory document from the search candidates. A candidate longer
   than T cannot participate and is excluded with an auditable reason. Keep all
   other candidates in their original rank order; no random subset, rank window,
   document-count target, or undocumented candidate cap. Equal-length documents
   remain distinct documents and can each be selected at most once.
5. Solve the following finite zero-one subset-sum problem exactly. Write candidate
   lengths as w[0..n-1]. Define suffix reachable sums R[i] over the integers 0..T:

   ```text
   R[n] = {0}
   for i = n-1 down to 0:
       R[i] = R[i+1] union {u + w[i] : u in R[i+1], u + w[i] <= T}
   ```

   A Boolean bitset implements this recurrence exactly with integer shifts,
   union and a mask. Each transition uses the NEXT suffix, not an in-place
   unbounded-knapsack update. Thus no document can be selected twice.
6. If T is absent from R[0], report `INFEASIBLE_UNDER_FIXED_COVERAGE_AND_QUOTA`.
   Do not return a nearest sum as a usable exposure. Preserve the search report.
   This conclusion concerns the fixed mandatory coverage document, not every
   possible policy that could choose a different coverage document.
7. If T is reachable, reconstruct a UNIQUE witness using an include-first rule.
   Start `remaining = T`. Visit candidates in ascending rank order. Include
   candidate i if `w[i] <= remaining` and `remaining - w[i]` is reachable in
   R[i+1]; then subtract w[i]. Otherwise exclude it, requiring `remaining` to
   remain reachable in R[i+1]. Stop at residual zero and exclude all later
   candidates. Require zero residual at the end. This chooses the
   lexicographically inclusion-preferred feasible subset in the frozen order,
   not the fewest documents or a best-scoring subset.
8. Repeat independently for all 30 strata. Process strata in ascending
   `(domain, language)` order. Report every stratum, including failed ones. Any
   incomplete or infeasible stratum blocks final exposure publication.

Example for a unit test, not a corpus result: quota 12, mandatory coverage length
2, remaining candidate order [6, 5, 5]. Greedy packing stalls at 8 total bytes;
the exact solver excludes 6 and includes both distinct 5-byte documents to reach
12. It does not alter document lengths or the mandatory coverage choice.

## Search bounds and failure behavior

The search space is fixed by the verified parent inventory and quotas: at most
10,931 eligible documents in total and residual T <= 320,000 for any stratum.
It visits at most n candidate transitions per stratum, each over T+1 Boolean sum
states. With bitsets this is O(n * ceil((T+1)/word_bits)) bitset work. Keeping all
suffix snapshots for deterministic reconstruction uses O((n+1)*(T+1)) bits per
stratum, excluding object/metadata overhead; process strata sequentially to avoid
retaining 30 snapshot collections at once. The total candidate/state dimensions
are declared and recorded before solving, not chosen using tokenizer information.

An implementation must calculate the snapshot storage requirement and declare
its host memory/time safety limits before the search. These are operational
abort limits, not heuristic search windows or acceptance criteria. If insufficient
resources, timeouts or errors prevent completing the recurrence, report
`SEARCH_INCOMPLETE_RESOURCE_LIMIT` or `SEARCH_ERROR`, NEVER infeasibility or a
successful partial exposure. No fallback to greedy packing, random retries,
shortened candidate lists, alternative rank salts or alternate coverage choices.

Subset-sum arithmetic is integer-only; no floating-point tolerances. Determinism
must depend on frozen inputs and the specified tie rules, not thread scheduling,
solver-version heuristics, hash-map iteration, or which worker finishes first.
Library/code changes that preserve the recurrence must reproduce the same witness.

## Ordering and independent validation

After the 30 subsets succeed, preserve rule R's existing coverage-first global
order and weighted byte-service order for the remaining documents. The coverage
documents and coverage-order ranks remain unchanged from v1. Each stratum's
remaining queue uses the original candidate rank order. Compare served/quota
ratios with integer cross multiplication; ties use `(domain, language)` ascending.
Changing selected membership can change subsequent global positions, but cannot
change the ordering rule. Never order by token counts, speed or FLOPs.

A verifier independent of the subset solver must check the emitted witness:

- Exact parent membership, IDs, normalized text hashes, byte lengths, source
  provenance and source-byte counts; no missing/duplicate documents.
- The mandatory coverage document in every stratum; exact original Q_s for
  every stratum; exactly 1,000,000 normalized bytes globally.
- All 30 selected-document counts and source/normalized totals; allocated and
  achieved shares; zero unused quota; coverage and exact-quota PASS per row.
- Correct rank/tie evidence and exact global order under unchanged rule R.
- Reconstructed solution matches the deterministic include-first witness, not
  merely any feasible combination. This can be checked by a second run of the
  fixed solver during construction verification, never by preflight resampling.

No document is truncated, duplicated as filler or synthesized. No training
language is substituted and no quota is increased. The existing cap is exactly
met, not rounded or renamed after generation.

## New evidence and immutable preflight boundary

Do not edit or replace any v1 file. If this proposal is approved and implemented,
publish a separate v2 policy snapshot, generation receipt and exposure manifest.
Record the original frozen selection/source/parent hashes, the exact v2 policy
SHA, unchanged v1 rank namespaces, sampler code SHA and commit, per-stratum search
dimensions/status and deterministic witness, selected IDs/order, source/normalized
byte counts, all 30 audit rows, ordered-list hashes and full manifest SHA-256.
Hash the final manifest externally; never put its own full-file hash inside it.

Only all-30-strata success and independent witness validation may create a
usable final exposure manifest. A failed search may publish separately labeled
diagnostic evidence but no usable `exposure.json`. The exact final exposure SHA
is unknown until construction succeeds; this proposal invents neither a hash nor
a document count. Preserve the 992,457-byte attempt and its receipt as rejected.

After a successful manifest is frozen, a separately authorized tokenizer-only
FLOP feasibility check may READ that exact sample and produce a linked PASS/FAIL
report. It must verify selection/exposure hashes before and after, cannot call
the sampler/solver or regenerate/reorder documents, and cannot adjust quotas,
coverage, tokenizer conditions or budgets. A failed check blocks launch and does
not trigger automatic sampling changes. No LM training or evaluation is part of
packing or feasibility verification.

The earlier implementation-pin compatibility blocker still applies: a new sampler
or exposure-plan integration cannot bypass the original frozen LM implementation
pin. This design review does not implement that integration or authorize a GPU
launch, Modal deployment, or budget changes.

## Regression gates before any v2 sample

Required future tests: the greedy-trap example; exact-fill success; an unreachable
residual; no fitting mandatory document; a solution requiring two distinct
equal-length documents; duplicate-document rejection; no reuse of a single
document; zero residual; deterministic tied ranks and witness reconstruction;
input enumeration independence; exactness for each of all 30 quotas; no transfer
across strata; rejection of altered IDs/bytes/order/provenance; identical canonical
output on repeated construction; and resource interruption without success or a
false infeasibility claim.

Verify the solver and emitted witness against exhaustive enumeration on tiny
synthetic cases, including the include-first tie rule. Test the largest declared
state dimensions without using research results to choose solver parameters.
Keep tokenizer, token-count, FLOP, validation and LM-result APIs unavailable to
the sampler. Test that preflight failure leaves every exposure byte unchanged.
These tests and the real exact-packing search are future actions, not work executed
while writing this document.

## Decision requested

Approve or revise exact per-stratum zero-one subset-sum packing, retaining the
mandatory coverage documents, v1 rank namespaces, and final ordering rules. This
is a deliberate review of the greedy algorithm, not a silent relaxation of the
experimental byte budget. Even a complete exact solver may correctly report that
the fixed constraints cannot be satisfied. No feasibility claim is made here.
