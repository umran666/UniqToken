# UniqToken v2 candidate: constrained atomic scalar admission

Status: proposal only. No production objective is changed. Evidence is retained
and locally validated in the linked PRs; their required external reviews and
merges remain an integration gate. Issue #94 must not close until the blocking
evidence is accepted. Adoption additionally requires missing English/code
validation and the future experiment below. This document does not assert that
the candidate works generally or that any tokenizer is globally best.

## Evidence and mechanism

The completed multilingual diagnostics in [#121](https://github.com/umran666/UniqToken/pull/121)
retain heavy fallback and fragmentation in some observed Indic strata. Those
measurements describe tokenizer behavior, not downstream quality. The atomic
recovery study in [#124](https://github.com/umran666/UniqToken/pull/124) provides
nine exact-budget models and complete train/validation receipts. Unweighted
whole-scalar admission reduces aggregate fallback from 18.35% to 9.95% at 8K,
8.80% to 3.82% at 16K, and 5.85% to 2.28% at 32K. Every observed validation
stratum passes its already declared 1% BpT loss gate. Additional fallback
weighting is slightly worse in aggregate at 8K/16K and identical at 32K.
These are observed effects on the inspected corpus, not independent confirmation.

The component ablations in [#127](https://github.com/umran666/UniqToken/pull/127)
find that ordinary SuperBPE eligibility excludes byte candidates, making its
fallback-cost component constant. Weighting a constant cannot prioritize scalar
repair. Document-frequency-only and boundary-only selection each lose more than
1% BpT in three observed strata. A boundary proxy is not evidence of improved
linguistic segmentation. The dynamically trained 8K reference reproduces #124's
baseline artifact byte for byte.

The fixed-corpus scaling in [#128](https://github.com/umran666/UniqToken/pull/128)
contains fourteen exact-budget models and an explicit 128K SentencePiece failure
receipt. The five-axis analysis in [#129](https://github.com/umran666/UniqToken/pull/129)
shows scope-dependent trade-offs. Current UniqToken-r64 remains on the aggregate
five-axis frontier only at 16K; several other UniqToken budgets are dominated on
those named axes in this bounded matrix. That result selects neither a global
winner nor an arbitrary combined score. It motivates preserving competitive
baselines and all vocabulary budgets in the future matrix.

Hypothesis: adding a bounded eligible class of missing complete Unicode scalars
can reduce fallback byte emission where ordinary pair eligibility cannot. It
can also displace useful ordinary additions and alter Viterbi choices. The
experiment must measure those losses rather than assuming that admission helps.

## Formal objective

For total budget V, train the named current Unigram/CEM seed at V - 64 using
the existing `byte_fallback_analysis.train_base` configuration. Preserve its
IDs, four controls, 256 canonical byte leaves and maximum subword length 16.

An atomic candidate c is exactly one complete Unicode scalar absent from the
seed vocabulary, encoded by that seed entirely as its two, three or four UTF-8
byte leaves. Require at least two observed training occurrences. Reject byte
notation, partial UTF-8 prefixes, controls, surrogates and existing vocabulary
entries. Candidate counts and probabilities use training only.

Let f(c) count these recovery events, N_e count all training token emissions
(including document EOS), and b_1...b_k be c's canonical byte leaves. Rank atomic
candidates by the existing unweighted recovery utility, ascending:

```text
J_A(c) = f(c) * (sum_j log P(b_j) - log(f(c) / N_e))
```

Its unit is occurrence-weighted natural-log score difference. It is a training
proxy, not downstream loss or measured bytes saved after re-encoding. Break
ties by descending f(c), then scalar text. Admit at most 16 candidates with
empirical log probability log(f(c)/N_e), append IDs, and normalize the complete
distribution using the existing fixed-order `fsum` path. No fallback weighting
is used in the candidate objective; the measured extra weight did not support it.

Spend the remaining 64 - admitted_atomic_count additions through unchanged
dynamic ordinary SuperBPE. At each iteration its current score is:

```text
J_C(a,b) = f(a,b) * (log P(a) + log P(b) - log(f(a,b) / N_pairs))
```

N_pairs is the current algorithm's counted adjacent-pair total, including the
existing stream-boundary policy. Preserve its negative-score eligibility,
minimum pair frequency two, control/byte exclusions, cross-word rule, tie order
and probability/ID behavior. The two stage denominators are explicitly distinct;
there is no claim that their scores form one calibrated weighted loss. The
16-of-64 allocation is a fixed hypothesis, not a tuned optimal ratio.

Exact V is mandatory. Abort or emit an explicit failed condition if the seed
or available additions cannot reach V. Never pad, relabel or substitute.
Production runtime byte semantics remain unchanged; admission stores the actual
scalar text, never a pseudo-subword containing byte notation.

## Frozen exploratory matrix

The machine-readable plan is
[superbpe_v2_candidate_v1.json](../benchmarks/protocols/superbpe_v2_candidate_v1.json).
It fixes the existing manifest SHA-256, ordered selection rule, normalization,
budgets, controls, cohort, deterministic configuration and gates. It introduces
no held-out test access and changes no frozen Phase A/B/C protocol or artifact.

Use the first 32 frozen documents per domain/language and first 2,048 normalized
code points from each, identically across all conditions. Training and validation
stay disjoint by source path, document IDs and exact normalized excerpts.
Normalization is `NFKC_unicode_spaces_v1` before the existing metaspace encoding.
Budgets are 8,192, 16,384, 32,768, 65,536 and 131,072, including all controls
and bytes. Use one Rayon worker, Python hash seed zero and deterministic trainers;
there is no invented seed variation.

At every budget compare current `uniq_superbpe_r64`, the unweighted constrained
atomic candidate, weight-five atomic recovery as a named negative control,
SPM-Unigram and Boundary-BPE. The current frozen 128K SentencePiece configuration
has a failure receipt; keep it unavailable rather than substituting another
trainer or smaller vocabulary. Run the existing recovery and scaling drivers
into new directories, preserve their identities and exact hashes, and join only
records with identical assignments and normalization counts.

This validation corpus has already been inspected. The future matrix is a
predeclared exploratory extension/reproduction, not an independent confirmatory
test. English and code validation are missing. Freeze an independent, disjoint
validation extension and its hashes before any adoption/confirmation decision;
never use the held-out test set to fill that gap. The current manifest cannot
certify those missing strata.

## Evaluation and rejection gates

Report aggregate and every language/domain/stratum: normalized bytes/token,
fallback leaf emissions and their source-byte fraction, fallback percentage of
tokens, token density per code point and UTF-8 byte, p50/p95/max contiguous
fallback-byte span, exact token-length histograms, vocabulary utilization,
unobserved additions in training/validation, and whitespace/punctuation split
proxies. Name all denominators. Fallback percentage of tokens can increase when
ordinary compression removes tokens even if fallback byte emissions are fixed;
it is not a substitute for the source-byte fraction.

Every comparison uses `uniq_superbpe_r64` at the same total budget, validation
stratum, source assignments and normalization. Freeze its model SHA-256 and
measured counts before evaluating the candidate; do not substitute a later run.
Pool raw document counts and span histograms within each reported scope before
computing fractions or percentiles, rather than averaging document-level ratios.

Define `fallback_source_byte_fraction` as canonical byte-leaf emissions (one
byte per leaf) divided by `normalized_utf8_bytes`. The per-stratum fallback
ratio is the candidate fraction divided by the frozen baseline fraction;
the cap of 1.0 means no increase, not an absolute source-byte fraction cap.
Aggregate relative reduction is `1 - candidate_fraction / baseline_fraction`;
the 0.1 threshold means a 10% relative reduction, not ten percentage points.
The p95 fallback-span ratio likewise divides candidate p95 by baseline p95.
The zero-fallback and empty-histogram rules below apply before these divisions.

For each whitespace and punctuation category separately, the split-run proxy
is `100 * split_runs / runs` on normalized text. Compare candidate percent minus
frozen baseline percent in each stratum; a baseline of 20.0% permits at most
20.5%, not a relative 0.5% increase. Zero runs gives a null proxy and an explicitly
not-applicable category with its run counts; missing counts block certification.
Never replace a null proxy with zero.

All following gates are required at every feasible candidate budget:

1. Exact vocabulary/ID/control/byte accounting, finite normalized probabilities,
   complete UTF-8 scalar admission, normalized roundtrip, reload and offset parity.
2. Each observed validation stratum retains at least 99% of current-r64 BpT.
   Missing or undefined required strata block certification rather than passing.
3. No observed stratum increases fallback bytes per normalized source byte;
   no stratum increases its p95 contiguous fallback-byte span.
4. Whitespace/punctuation split-run proxies increase by at most 0.5 percentage
   points per observed stratum. Undefined baseline categories remain explicit.
5. Aggregate fallback bytes per normalized source byte decrease by at least 10%
   in at least two of the five feasible UniqToken budgets, while gates 1-4 hold.
6. Before adoption, the independent English/code extension satisfies the same
   gates, including exact source/assignment hashes fixed before measurement.
7. Report fresh-process wall/CPU training cost, OS peak RSS and five-axis Pareto
   classification against every named baseline. No quality advantage is inferred
   from these dimensions. One timing observation has no confidence interval;
   repeat costs before any efficiency/adoption claim.

A zero fallback baseline requires zero candidate fallback emissions. An empty
span histogram is a known absence of fallback, with percentile reported as null;
it passes only if the candidate also has no fallback. This is distinct from
missing measurements. English (`en`) and every currently missing code language
(`c`, `cpp`, `go`, `java`, `javascript`, `python`, `rust`, `sql`, `typescript`)
must have disjoint validation coverage before adoption.

A correctness, per-stratum or source-integrity failure rejects that candidate
budget and blocks production adoption of this policy. No meaningful fallback
reduction rejects the repair hypothesis. Missing required validation or failed
conditions remain blocked and visible. Do not tune thresholds, cap, weights,
candidate pool, input selection or budgets in response; a changed policy needs
a new versioned plan and experiment. Retain current production SuperBPE, all
failure receipts and negative results. No downstream LM claim is licensed.

The already observed <=32K recovery results are descriptive evidence for this
hypothesis. They do not retroactively pass these new gates, supply missing
English/code validation, or demonstrate the five-budget candidate works.
