# Vocabulary budget scaling

Tokenizer-only descriptive matrix on one frozen diagnostic train/validation selection.
No held-out test access, LM run, or modification of frozen Phase A/B/C artifacts.
UT-SuperBPE uses 64 reserved merge slots at every budget (uniq_superbpe_r64); this is explicitly a diagnostic variant, not a replay of the Phase A training configuration.
All budgets include four controls and 256 byte leaves. Underfilled conditions have failure receipts and no favorable metric imputation.

| Tokenizer | Budget | Status | Training s | Peak RSS MiB |
| --- | ---: | --- | ---: | ---: |
| sp_unigram | 8192 | complete | 37.8343 | 289.60 |
| boundary_bpe | 8192 | complete | 53.9211 | 1082.11 |
| uniq_superbpe_r64 | 8192 | complete | 157.9821 | 591.64 |
| sp_unigram | 16384 | complete | 35.1503 | 289.50 |
| boundary_bpe | 16384 | complete | 81.8000 | 1130.75 |
| uniq_superbpe_r64 | 16384 | complete | 164.3453 | 612.60 |
| sp_unigram | 32768 | complete | 23.0301 | 289.27 |
| boundary_bpe | 32768 | complete | 111.0091 | 1153.58 |
| uniq_superbpe_r64 | 32768 | complete | 177.0882 | 645.91 |
| sp_unigram | 65536 | complete | 12.8390 | 288.99 |
| boundary_bpe | 65536 | complete | 110.4681 | 1174.98 |
| uniq_superbpe_r64 | 65536 | complete | 164.4856 | 675.39 |
| sp_unigram | 131072 | budget_not_reached | NA | NA |
| boundary_bpe | 131072 | complete | 146.1055 | 1194.29 |
| uniq_superbpe_r64 | 131072 | complete | 204.2959 | 791.28 |

Failure receipt for sp_unigram at 131072: Internal: D:\a\sentencepiece\sentencepiece\src\trainer_interface.cc(664) [(trainer_spec_.vocab_size()) == (model_proto->pieces_size())] Vocabulary size too high (131072). Please set it to a value <= 79552.


## Measurement scope

one wall/CPU observation per fresh worker, training plus serialization; excludes imports and evaluation.
fresh-process high-water RSS through training and artifact serialization; includes imports and normalized excerpt inputs. The pre-training process high-water RSS is recorded separately.
Each condition runs in a new process, with one Rayon worker, Python hash seed zero and deterministic trainer configuration. There is one timing observation per condition; no statistical confidence interval is claimed.
The driver runs conditions sequentially. Model validation and all train/validation diagnostics precede a complete receipt.
Density uses normalized Unicode code points and UTF-8 bytes. Token lengths are exact normalized source-byte contributions. Metrics, histograms and utilization retain aggregate/domain/language/stratum detail.
The plots leave gaps for failed conditions. Logs retain the original failure. Models and output hashes are included in the final manifest.
Validation has FLORES domain labels. Training domains and English/code languages without validation cannot be certified from this matrix.
Scaling curves describe this bounded corpus and trainer configuration. They do not establish a universal scaling law or a global winner.
