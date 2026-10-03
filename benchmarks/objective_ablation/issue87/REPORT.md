# SuperBPE component ablations

Explanatory tokenizer-only measurements. No final objective, downstream claim or held-out test access.
All fixed-pool ablations see identical candidates. Current SuperBPE is the dynamic reference; pool_ce controls for freezing its candidate pool and scoring state.
Constant components cannot be identified by this pool: fallback_cost

| Condition | BpT | Fallback % | Tokens/character | Whitespace split % | Punctuation split % |
| --- | ---: | ---: | ---: | ---: | ---: |
| current_superbpe | 3.13504 | 18.34763 | 0.69757 | 0.61005 | 4.38787 |
| pool_ce | 3.13757 | 18.36241 | 0.69701 | 0.61005 | 4.38787 |
| compression | 3.14759 | 18.42105 | 0.69479 | 0.61005 | 4.38787 |
| frequency | 3.13123 | 18.32533 | 0.69842 | 0.61005 | 4.38787 |
| boundary | 3.11431 | 18.22632 | 0.70221 | 0.61005 | 4.38787 |
| fallback | 3.14759 | 18.42105 | 0.69479 | 0.61005 | 4.38787 |
| fragmentation | 3.14207 | 18.38879 | 0.69601 | 0.61005 | 4.38787 |
| compression_boundary_fragmentation | 3.14207 | 18.38879 | 0.69601 | 0.61005 | 4.38787 |
| ce_compression | 3.14189 | 18.38771 | 0.69605 | 0.61005 | 4.38787 |

## Per-stratum trade-offs

Counts compare observed validation strata with current SuperBPE. A BpT loss above 1% is reported as a regression, not hidden by the aggregate.

| Condition | Strata with >1% BpT loss | Strata with more fallback | Strata with more punctuation splitting |
| --- | ---: | ---: | ---: |
| pool_ce | 0 | 4 | 0 |
| compression | 0 | 12 | 0 |
| frequency | 3 | 8 | 0 |
| boundary | 3 | 0 | 0 |
| fallback | 0 | 12 | 0 |
| fragmentation | 0 | 6 | 0 |
| compression_boundary_fragmentation | 0 | 6 | 0 |
| ce_compression | 0 | 5 | 0 |

The CSV/JSON retain every language/domain/stratum, exact histograms and normalization counts. Missing validation domains are not inferred from training observations.
Component values are training proxies. Re-encoding and greedy interactions can differ from their first-order predictions. No aggregate result selects a production objective.
