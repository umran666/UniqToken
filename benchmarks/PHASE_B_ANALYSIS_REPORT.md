# Phase B Screening Analysis

## Status and scope

This report is a deterministic analysis of the completed Phase B screening ledger.
It is screening evidence, not confirmatory evidence, and makes no general tokenizer
superiority claim.

- Ledger: `artifacts/phase-b-screen-771d62a-complete/ledger.json`
- Ledger SHA-256: `0c145550a2b10b266f83d35e4cc5edc335eb401e0e133ebd09d908402539fb33`
- Harness commit: `771d62a81d4010186a676fab15acfc3f8e616167`
- Selection SHA-256: `d3449786b636216057e01e1372b4805c1181c481ca521a74c9e29e07676a0ea4`
- Source manifest SHA-256: `af11dc30fc96c683f92434d27c79fc45dd649dfb3787c6d9b02f81075aedd0c5`
- Exposure SHA-256: `68ce58aaf90754d8e5bc8b52a8de249fe3cdd4f5876148dd832cd45d61675e2e`
- Conditions: 18/18 complete, one seed (`0`)
- Evaluation: screening validation only; the held-out test remained forbidden and unopened
- Result label: `SCREENING`

The ledger was independently revalidated against the locked plan, Phase A
tokenizer artifacts, exact 349-document exposure, byte assignments, parameter and
FLOP accounting, and held-out NLL calculations before this report was produced.

## Metric definitions

Lower validation bits per normalized UTF-8 byte (BPB) is better:

`BPB = total_validation_NLL_nats / (normalized_validation_bytes * ln(2))`

All conditions use the same screening-validation documents and normalized byte
denominator. NLL and BPB therefore have identical rankings within this report.
Token perplexity is present in the ledger but is not used for cross-tokenizer
ranking because token units differ between tokenizers.

`Delta vs best` is the condition BPB minus the lowest BPB at the same vocabulary
and budget regime. `Relative delta` is `(condition / best - 1) * 100%`.

## FLOP-matched validation

| Vocab | Tokenizer | BPB | NLL (nats) | Delta vs best | Relative delta |
|---:|---|---:|---:|---:|---:|
| 16K | SPM-Unigram | **3.278959** | 5,753,391.703 | +0.000000 | +0.00% |
| 16K | Boundary-BPE | 3.701121 | 6,494,133.306 | +0.422162 | +12.87% |
| 16K | UT-SuperBPE | 3.520241 | 6,176,753.966 | +0.241282 | +7.36% |
| 32K | SPM-Unigram | **2.907290** | 5,101,247.007 | +0.000000 | +0.00% |
| 32K | Boundary-BPE | 3.032921 | 5,321,682.994 | +0.125630 | +4.32% |
| 32K | UT-SuperBPE | 3.072791 | 5,391,641.026 | +0.165501 | +5.69% |
| 64K | SPM-Unigram | **2.683315** | 4,708,251.024 | +0.000000 | +0.00% |
| 64K | Boundary-BPE | 2.801852 | 4,916,241.549 | +0.118537 | +4.42% |
| 64K | UT-SuperBPE | 2.878272 | 5,050,330.006 | +0.194957 | +7.27% |

SPM-Unigram has the lowest single-seed validation BPB at every vocabulary in
this regime. This does not establish a normally trained LM advantage because the
fixed analytical budget permits extremely little training and increasingly
consists of vocabulary-dependent output-projection FLOPs.

## Byte-matched validation

| Vocab | Tokenizer | BPB | NLL (nats) | Delta vs best | Relative delta |
|---:|---|---:|---:|---:|---:|
| 16K | SPM-Unigram | 3.111264 | 5,459,147.214 | +0.172152 | +5.86% |
| 16K | Boundary-BPE | 3.281505 | 5,757,858.682 | +0.342393 | +11.65% |
| 16K | UT-SuperBPE | **2.939112** | 5,157,081.829 | +0.000000 | +0.00% |
| 32K | SPM-Unigram | **2.836720** | 4,977,422.170 | +0.000000 | +0.00% |
| 32K | Boundary-BPE | 2.878525 | 5,050,773.868 | +0.041804 | +1.47% |
| 32K | UT-SuperBPE | 2.866409 | 5,029,514.365 | +0.029688 | +1.05% |
| 64K | SPM-Unigram | **2.632774** | 4,619,569.480 | +0.000000 | +0.00% |
| 64K | Boundary-BPE | 2.660996 | 4,669,089.044 | +0.028222 | +1.07% |
| 64K | UT-SuperBPE | 2.809894 | 4,930,351.168 | +0.177120 | +6.73% |

UT-SuperBPE has the lowest single-seed BPB at 16K. This makes it a post-screen
exploratory candidate for confirmation, not an independently selected
confirmatory condition. At 32K it is 1.05% above
SPM-Unigram and at 64K it is 6.73% above SPM-Unigram. This is a candidate
interaction for confirmation, not evidence of a general UniqToken advantage.

## Regime deltas

Negative values mean the byte-matched condition has lower BPB than the
corresponding FLOP-matched condition. These are not controlled estimates of a
regime causal effect: the regimes expose models to different token and byte
amounts.

| Vocab | Tokenizer | FLOP BPB | Byte BPB | Byte minus FLOP | Relative change |
|---:|---|---:|---:|---:|---:|
| 16K | SPM-Unigram | 3.278959 | 3.111264 | -0.167695 | -5.11% |
| 16K | Boundary-BPE | 3.701121 | 3.281505 | -0.419616 | -11.34% |
| 16K | UT-SuperBPE | 3.520241 | 2.939112 | -0.581129 | -16.51% |
| 32K | SPM-Unigram | 2.907290 | 2.836720 | -0.070570 | -2.43% |
| 32K | Boundary-BPE | 3.032921 | 2.878525 | -0.154396 | -5.09% |
| 32K | UT-SuperBPE | 3.072791 | 2.866409 | -0.206382 | -6.72% |
| 64K | SPM-Unigram | 2.683315 | 2.632774 | -0.050541 | -1.88% |
| 64K | Boundary-BPE | 2.801852 | 2.660996 | -0.140857 | -5.03% |
| 64K | UT-SuperBPE | 2.878272 | 2.809894 | -0.068378 | -2.38% |

## Training exposure

The FLOP-matched conditions predict only 1,869 to 6,536 training targets and
consume 5,271 to 16,298 normalized bytes. The byte-matched conditions consume
all 1,000,000 normalized bytes and predict 253,119 to 410,089 targets.

| Vocab | Tokenizer | Regime | Training targets | Normalized bytes | Complete documents | Targets / parameter |
|---:|---|---|---:|---:|---:|---:|
| 16K | SPM-Unigram | FLOP | 6,517 | 16,298 | 31 | 0.001414 |
| 16K | Boundary-BPE | FLOP | 6,475 | 16,298 | 31 | 0.001405 |
| 16K | UT-SuperBPE | FLOP | 6,536 | 16,298 | 31 | 0.001419 |
| 16K | SPM-Unigram | byte | 340,331 | 1,000,000 | 349 | 0.073865 |
| 16K | Boundary-BPE | byte | 410,089 | 1,000,000 | 349 | 0.089005 |
| 16K | UT-SuperBPE | byte | 340,565 | 1,000,000 | 349 | 0.073916 |
| 32K | SPM-Unigram | FLOP | 3,576 | 13,393 | 30 | 0.000406 |
| 32K | Boundary-BPE | FLOP | 3,588 | 11,428 | 26 | 0.000408 |
| 32K | UT-SuperBPE | FLOP | 3,589 | 13,393 | 29 | 0.000408 |
| 32K | SPM-Unigram | byte | 285,046 | 1,000,000 | 349 | 0.032385 |
| 32K | Boundary-BPE | byte | 363,056 | 1,000,000 | 349 | 0.041248 |
| 32K | UT-SuperBPE | byte | 290,396 | 1,000,000 | 349 | 0.032993 |
| 64K | SPM-Unigram | FLOP | 1,869 | 8,064 | 20 | 0.000109 |
| 64K | Boundary-BPE | FLOP | 1,885 | 5,271 | 14 | 0.000110 |
| 64K | UT-SuperBPE | FLOP | 1,885 | 6,900 | 18 | 0.000110 |
| 64K | SPM-Unigram | byte | 253,119 | 1,000,000 | 349 | 0.014724 |
| 64K | Boundary-BPE | byte | 335,018 | 1,000,000 | 349 | 0.019489 |
| 64K | UT-SuperBPE | byte | 256,724 | 1,000,000 | 349 | 0.014934 |

Even the byte-matched screen supplies substantially fewer than one target per
parameter. The FLOP-matched ratios are three to four orders of magnitude below
one. Neither regime represents normal-scale LM training.

## Parameter and FLOP decomposition

Core parameters remain constant at 413,184. Input embeddings and the untied
output head each scale linearly with vocabulary. The analytical estimator includes
the vocabulary-dependent output projection; this term consumes 82.24%, 90.31%,
and 94.93% of the SPM-Unigram FLOP-matched totals at 16K, 32K, and 64K.

| Vocab | Core params | Input embedding | Output head | Total params | Example | Core FLOPs | Output FLOPs | Output share | Total FLOPs |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 16K | 413,184 | 2,097,152 | 2,097,152 | 4,607,488 | FLOP SPM-U | 17,706,845,184 | 82,002,837,504 | 82.24% | 99,709,682,688 |
| 16K | 413,184 | 2,097,152 | 2,097,152 | 4,607,488 | byte SPM-U | 933,764,803,584 | 4,282,355,023,872 | 82.10% | 5,216,119,827,456 |
| 32K | 413,184 | 4,194,304 | 4,194,304 | 8,801,792 | FLOP SPM-U | 9,652,592,640 | 89,992,986,624 | 90.31% | 99,645,579,264 |
| 32K | 413,184 | 4,194,304 | 4,194,304 | 8,801,792 | byte SPM-U | 781,545,394,176 | 7,173,417,467,904 | 90.18% | 7,954,962,862,080 |
| 64K | 413,184 | 8,388,608 | 8,388,608 | 17,190,400 | FLOP SPM-U | 5,024,922,624 | 94,069,850,112 | 94.93% | 99,094,772,736 |
| 64K | 413,184 | 8,388,608 | 8,388,608 | 17,190,400 | byte SPM-U | 693,745,990,656 | 12,739,896,410,112 | 94.84% | 13,433,642,400,768 |

The table uses SPM-Unigram as a compact example. Parameter counts are identical
across tokenizers at a vocabulary size. Exact FLOPs differ with predicted target
counts and sequence packing and remain available in every ledger record.

## Scientific interpretation

Phase B supports only these conclusions:

1. The three tokenizers produce measurably different one-seed validation outcomes
   under the two declared screening regimes.
2. UT-SuperBPE's 16K byte-matched result is a candidate worth confirmation.
3. SPM-Unigram has the lowest FLOP-matched BPB at all three vocabulary sizes in
   this screen.
4. Vocabulary-dependent output-head compute dominates the fixed-FLOP regime,
   especially at 64K.

Phase B does not support a claim that UniqToken generally wins, improves
multilingual LM efficiency, beats SentencePiece generally, or improves normally
trained language models. It uses one seed, a small model, and severely limited
training exposure. SPM-BPE and UT-Unigram were excluded from Phase B by explicit
user declaration, not because Phase A established that they were inferior.

## Post-screen exploratory advancement rule

This rule was created **after inspecting Phase B screening results and before any
three-seed follow-up result**. It is a deterministic post-screen exploratory
advancement rule. It was not preregistered, was not independent of the observed
Phase B outcomes, and cannot convert Phase B into confirmatory evidence.

A vocabulary/regime cohort advances only when all of the following hold:

1. The decision uses screening-validation BPB only; test data remains unopened.
2. UT-SuperBPE has the lowest BPB among UT-SuperBPE, SPM-Unigram, and
   Boundary-BPE at the same vocabulary and regime.
3. UT-SuperBPE improves on the strongest baseline by at least 2.0% relative BPB.
4. The entire three-tokenizer cohort advances together so the proposed method is
   never confirmed without both predeclared baselines.
5. Confirmatory training must use three seeds and a separately frozen, materially
   larger training exposure/budget. Phase B's 1 MB exposure and `1e11` FLOP screen
   are not reused as evidence of meaningful training.
6. The untouched test remains closed until the confirmatory protocol, exposure,
   budgets, seeds, stopping rule, aggregation, and statistical analysis are frozen.

Applying this rule identifies exactly one post-screen exploratory candidate
cohort: **16K byte-matched SPM-Unigram, Boundary-BPE, and UT-SuperBPE**. Within
that one-seed 16K byte-matched screening comparison, UT-SuperBPE is 5.53% lower
in BPB than the strongest baseline, SPM-Unigram (`2.939112` versus `3.111264`).
This scoped difference is not evidence of a general UniqToken advantage. No
FLOP-matched, 32K byte-matched, or 64K byte-matched cohort advances.

Selection under this rule authorizes protocol design and feasibility work only.
It does not authorize test access or establish a confirmatory result.
