# Atomic fallback recovery diagnostic

Tokenizer-only descriptive experiment; no held-out test access or LM result.
All conditions share seed vocabulary, frozen training excerpts, validation excerpts, normalization and exact final budget.
Recovery admits whole missing Unicode scalars before the remaining ordinary SuperBPE merges. It is a separate admission policy, not pairwise byte-prefix merging.
The regression gate checks every observed validation stratum at the predeclared 1% BpT loss threshold.
Missing validation strata cannot be certified. Unobserved validation additions are not proof of intrinsically dead tokens.

## Vocabulary budget 8192

| Condition | BpT | Fallback % | Recovery slots | Training-unobserved | Validation-unobserved | Worst BpT regression % | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 3.13504 | 18.34763 | 0 | 1 | 44 | 0.0000 | True |
| atomic_recovery | 3.32952 | 9.95246 | 16 | 1 | 31 | 0.9419 | True |
| fallback_weighted | 3.32501 | 10.13264 | 16 | 1 | 31 | 0.9419 | True |

atomic_recovery: 7/20 strata emit fewer fallback bytes; 0 have a longer fallback-span p95.

fallback_weighted: 7/20 strata emit fewer fallback bytes; 0 have a longer fallback-span p95.

8192: weighted and unweighted validation metrics are different; inspect all strata and gates, without selecting a winner on aggregate alone.

## Vocabulary budget 16384

| Condition | BpT | Fallback % | Recovery slots | Training-unobserved | Validation-unobserved | Worst BpT regression % | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 4.03990 | 8.79696 | 0 | 1 | 56 | 0.0000 | True |
| atomic_recovery | 4.17347 | 3.81861 | 16 | 1 | 42 | 0.0000 | True |
| fallback_weighted | 4.17260 | 3.87269 | 16 | 1 | 42 | 0.0000 | True |

atomic_recovery: 7/20 strata emit fewer fallback bytes; 0 have a longer fallback-span p95.

fallback_weighted: 7/20 strata emit fewer fallback bytes; 0 have a longer fallback-span p95.

16384: weighted and unweighted validation metrics are different; inspect all strata and gates, without selecting a winner on aggregate alone.

## Vocabulary budget 32768

| Condition | BpT | Fallback % | Recovery slots | Training-unobserved | Validation-unobserved | Worst BpT regression % | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 4.53950 | 5.84506 | 0 | 1 | 57 | 0.0000 | True |
| atomic_recovery | 4.64637 | 2.28423 | 16 | 1 | 42 | 0.0000 | True |
| fallback_weighted | 4.64637 | 2.28423 | 16 | 1 | 42 | 0.0000 | True |

atomic_recovery: 10/20 strata emit fewer fallback bytes; 0 have a longer fallback-span p95.

fallback_weighted: 10/20 strata emit fewer fallback bytes; 0 have a longer fallback-span p95.

32768: weighted and unweighted validation metrics are identical (no demonstrated weighting benefit).


Per-stratum fallback frequency and contiguous byte-span p50/p95/max are in fallback_metrics.csv; exact histograms, merge records and token utilization audits are in results.json.
The following training domain/language pairs are absent from validation, which uses distinct FLORES domain labels: code:c, code:cpp, code:go, code:java, code:javascript, code:python, code:rust, code:sql, code:typescript, cyrillic_african:am, cyrillic_african:bg, cyrillic_african:ru, cyrillic_african:sw, cyrillic_african:uk, cyrillic_african:yo, indic_cjk_arabic:ar, indic_cjk_arabic:bn, indic_cjk_arabic:fa, indic_cjk_arabic:gu, indic_cjk_arabic:hi, indic_cjk_arabic:ja, indic_cjk_arabic:kn, indic_cjk_arabic:ko, indic_cjk_arabic:ml, indic_cjk_arabic:mr, indic_cjk_arabic:ta, indic_cjk_arabic:te, indic_cjk_arabic:ur, indic_cjk_arabic:zh, latin_english:en
The original PR's approximate-budget numbers and hard-coded positive conclusions are withdrawn. These data do not establish a general multilingual advantage or a downstream-quality improvement.

Languages without any validation coverage: c, cpp, en, go, java, javascript, python, rust, sql, typescript.
