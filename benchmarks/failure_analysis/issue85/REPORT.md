# Multilingual tokenizer failure diagnostics

Descriptive tokenizer-only evidence. These associations do not establish causes or a globally best tokenizer. No LM was initialized or trained; held-out test data was not opened or hashed. The exploratory 16K byte-matched Phase B result is not confirmation.

## Scope and provenance

- Analysis source commit: `e8e33a07b7e660eecc292b106a7c06e0d8733552`; source hash: `9bb536eaffbf5726e4e8106d3a5054f08ee5070b583b141b26f57ed8d522a3e9`.
- Frozen dataset manifest SHA-256: `2ad27746d9c139c8d7e814e89c977c9bd4410033d5c1b97b673ee1b893ece7ff`.
- Frozen Phase A ledger SHA-256: `a1a113cad3d67d07ea772a193b3cdf5de57bf716dad35c37d73115fa29d925ff`.
- Training diagnostics: ordered whole-document prefixes, cap 32 per stratum (0 means all), across 30 frozen training strata.
- Validation: complete original screening half, using the frozen normalized-document SHA-256 parity assignment. Confirmation validation is excluded from scoring.
- Missing screening-validation languages: code/c, code/cpp, code/go, code/java, code/javascript, code/python, code/rust, code/sql, code/typescript, latin_english/en. These have training diagnostics only; no substitute validation corpus was created.
- Validation domain labels remain FLORES source labels; they are not relabeled as training web/code domains.

| Split | Documents | Normalized bytes | Unicode characters | Strata |
| --- | ---: | ---: | ---: | ---: |
| train | 960 | 10150281 | 7080929 | 30 |
| validation | 9918 | 2531410 | 1142632 | 20 |

## Aggregate observations

Aggregate ratios pool counts; languages and domains remain separate in the CSVs and plots.

| Split | Vocab | Tokenizer | Bytes/token | Tokens/character | Byte fallback % | Observed vocab % | Rare % | Cross-word merge % |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 16384 | SPM-Unigram | 2.8317 | 0.5062 | 1.5469 | 82.2222 | 0.1886 | 0.0000 |
| validation | 16384 | SPM-Unigram | 3.7785 | 0.5863 | 0.0405 | 50.2564 | 0.4576 | 0.0000 |
| train | 32768 | SPM-Unigram | 3.3911 | 0.4227 | 1.8525 | 85.6489 | 0.4883 | 0.0000 |
| validation | 32768 | SPM-Unigram | 4.7375 | 0.4676 | 0.0507 | 50.1373 | 1.3731 | 0.0000 |
| train | 65536 | SPM-Unigram | 3.7979 | 0.3774 | 2.0748 | 80.2402 | 1.8449 | 0.0000 |
| validation | 65536 | SPM-Unigram | 5.6173 | 0.3944 | 0.0601 | 45.3626 | 7.1196 | 0.0000 |
| train | 16384 | Boundary-BPE | 2.5191 | 0.5690 | 0.0000 | 82.0269 | 0.1613 | 0.0000 |
| validation | 16384 | Boundary-BPE | 3.3339 | 0.6645 | 0.0357 | 48.5226 | 0.3747 | 0.0000 |
| train | 32768 | Boundary-BPE | 2.9035 | 0.4937 | 0.0000 | 85.6702 | 0.4318 | 0.0000 |
| validation | 32768 | Boundary-BPE | 3.9196 | 0.5652 | 0.0420 | 46.5694 | 0.9061 | 0.0000 |
| train | 65536 | Boundary-BPE | 3.2069 | 0.4470 | 0.0000 | 80.1532 | 1.3828 | 0.0000 |
| validation | 65536 | Boundary-BPE | 4.4216 | 0.5010 | 0.0473 | 42.4144 | 4.5005 | 0.0000 |
| train | 16384 | UT-SuperBPE | 2.9476 | 0.4863 | 4.6846 | 81.4530 | 0.1783 | 12.5991 |
| validation | 16384 | UT-SuperBPE | 3.6708 | 0.6035 | 14.2018 | 54.9939 | 0.3998 | 1.8678 |
| train | 32768 | UT-SuperBPE | 3.5533 | 0.4034 | 1.5217 | 87.5992 | 0.4621 | 16.0524 |
| validation | 32768 | UT-SuperBPE | 4.7485 | 0.4666 | 5.3495 | 61.3631 | 1.7152 | 1.9386 |
| train | 65536 | UT-SuperBPE | 4.0059 | 0.3578 | 0.5107 | 84.7204 | 2.3843 | 17.6572 |
| validation | 65536 | UT-SuperBPE | 5.4611 | 0.4057 | 2.1133 | 55.1013 | 9.0026 | 2.2794 |

## Largest observed failure signals

The following are measured on matching screening-validation documents and vocabulary budgets. They describe this cohort and assignment only.

| Vocab | Language | UT-SuperBPE byte fallback % | SPM-Unigram byte fallback % | Difference (percentage points) |
| ---: | --- | ---: | ---: | ---: |
| 16384 | gu | 33.4904 | 0.0057 | +33.4847 |
| 16384 | bn | 33.1618 | 0.0054 | +33.1564 |
| 16384 | ml | 33.0348 | 0.0000 | +33.0348 |
| 16384 | hi | 32.2259 | 0.0060 | +32.2199 |
| 16384 | mr | 29.9313 | 0.0054 | +29.9259 |
| 16384 | te | 22.6501 | 0.0052 | +22.6449 |
| 16384 | kn | 22.1588 | 0.0051 | +22.1537 |
| 32768 | gu | 17.8195 | 0.0074 | +17.8121 |

| Vocab | Tokenizer | Language | Punctuation tokens/run | Whitespace tokens/run |
| ---: | --- | --- | ---: | ---: |
| 16384 | UT-SuperBPE | te | 1.1454 | 1.0131 |
| 32768 | UT-SuperBPE | te | 1.1454 | 1.0131 |
| 65536 | UT-SuperBPE | te | 1.1454 | 1.0131 |
| 16384 | UT-SuperBPE | mr | 1.1357 | 1.0040 |
| 32768 | UT-SuperBPE | mr | 1.1357 | 1.0040 |
| 65536 | UT-SuperBPE | mr | 1.1357 | 1.0040 |
| 16384 | SPM-Unigram | te | 1.1354 | 1.0135 |
| 32768 | SPM-Unigram | te | 1.1285 | 1.0135 |

## Plausible hypotheses, not causal conclusions

- Where fallback percentages differ, investigate byte-fallback edge scores and vocabulary coverage in a controlled matched-candidate ablation (#86). This diagnostic does not isolate those mechanisms.
- Where language-specific vocabulary utilization and rare-token emissions differ, investigate training allocation and merge distribution (#88). Sample size, scripts and source-domain differences are alternative explanations.
- Where punctuation or whitespace runs touch many tokens, inspect boundary rules with fixed normalization and vocabulary (#89). Multi-byte fallback can itself split one punctuation character; token/run ratios do not prove a boundary-rule defect.

## Metric definitions and limitations

- `bytes_per_token`: normalized UTF-8 bytes / emitted tokens; higher is more compression.
- `tokens_per_unicode_character`: emitted tokens / normalized Python Unicode characters.
- `byte_fallback_percent`: 100 * canonical byte-token emissions / all emissions.
- `token_length_bytes`: exact decoded source-byte contribution per token; fallback contributes one byte, not its spelling length.
- `vocabulary_utilization_percent`: 100 * distinct observed IDs / (vocabulary size - four controls); byte IDs remain eligible.
- `rare_token_percent`: 100 * emissions with pooled diagnostic-training count <= rare_threshold / emissions; not full tokenizer-training rarity when capped.
- `cross_word_merge_rate_percent`: 100 * observed SuperBPE pair applications / initial adjacent token boundaries, summed within documents; other engines have zero cross-word application events.
- `cross_word_token_percent`: 100 * emitted tokens intersecting at least two whitespace-delimited non-whitespace fields / emissions; not morphological word segmentation.
- `fragmentation`: maximal whitespace (str.isspace) or Unicode category P runs; tokens_per_run is token-byte-span intersections / runs; split_run_percent is 100 * runs touched by >1 token / runs.
- `aggregation`: pool counts, vocabulary-ID unions and length histograms before ratios/quantiles; never average stratum rates.
- `nulls`: undefined zero-denominator ratios and absent validation coverage are null, never zero evidence.
- Rare threshold: <= 5 occurrences in the pooled diagnostic-training sample. Counts of zero are included and reported separately. With a nonzero document cap this is sample rarity, not the learned model's complete training frequency.
- Vocabulary utilization and tail frequencies depend on sample exposure. Training observations are in-sample; they cannot establish held-out code behavior. Missing validation coverage remains an evidence gap for #94.
- A cross-word event is an application in the observed SuperBPE pass; emitted cross-field tokens are reported separately. These are not estimates of a causal compression benefit.
- UTF-8 token source contributions exactly reconstruct each normalized document. Byte fallback contributes one byte even when multiple bytes represent one Unicode character. No raw-offset non-overlap assumption is imposed on public Token spans.
- Plots and tables use the same recorded counts. No significance tests, downstream advantage, algorithm change, or Phase A/B/C confirmation is claimed.

See `results.json` for configuration, definitions, model hashes, assignments and all records; `strata.csv`, `languages.csv`, `domains.csv`, `aggregate.csv`, `coverage.csv`, and `token_lengths.csv` for separate tables. PNGs show aggregate and per-language diagnostics.
