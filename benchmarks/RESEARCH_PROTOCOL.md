# Final experiment harness

This is an execution protocol, not a result report. Tokenizer stages run from the
repository root using `python -m benchmarks.run_phase_a`. LM stages remain in
`python -m benchmarks.run_research_experiments`. The older
`run_matched_budget_eval.py` remains a train/validation diagnostic, not the final
research runner. Historical ledgers are unchanged and cannot enter this protocol.

## Primary cohort and stages

All vocabulary budgets include exactly four control tokens and 256 byte tokens:
`<|unk|>=0`, `<|pad|>=1`, `<|bos|>=2`, `<|eos|>=3`. Remaining IDs are contiguous.
The harness aligns control IDs in fresh UniqToken artifacts before any LM is
initialized; it does not change tokenizer pieces or scores to fill budgets.

| ID | Primary tokenizer |
| --- | --- |
| `sp_unigram` | SentencePiece Unigram |
| `sp_bpe` | SentencePiece BPE |
| `boundary_bpe` | Repository BPE trained/applied within whitespace boundaries |
| `uniq_unigram` | UniqToken Unigram |
| `uniq_superbpe` | UniqToken Unigram plus cross-word CEM |

Boundary-BPE is a primary baseline, not an optional fallback. Historical 64K
comparisons motivated retaining it; those invalidated experiments do not establish
current superiority. This runner preserves whitespace in Boundary-BPE input and
output. Do not interpret differences from historical numbers as improvements.

Phase A is an explicit two-stage gate. `A-SCREEN` trains all five tokenizers at
16,384, 32,768 and 65,536 total entries on one deterministic, stratified frozen
subset targeting 75,000,000 normalized bytes. The subset uses whole documents and
proportional language/domain quotas; it must remain between 50 and 100 decimal MB.
Every row and ledger is labeled `SCREENING`. Screening reports held-out validation
bytes per token, tokens per Unicode character, byte-fallback percentage, achieved
vocabulary, training wall-clock time, and normalized input MB/s. Screening is for
feasibility and condition selection only and is never confirmatory evidence.

The frozen FLORES dev validation split is deterministically divided by normalized
document SHA-256 parity. `A-SCREEN` sees only the screening half. `A-CONFIRM` trains
only deterministically selected tokenizer/vocabulary conditions on the complete
500 MB frozen training corpus and evaluates only the complementary confirmation
validation half. FLORES devtest remains declared by hash but is never opened,
tokenized, scored, or included in either Phase A ledger.

Selection chooses exactly one tokenizer per vocabulary budget by ascending
screening-validation tokens per Unicode character, then validation byte-fallback
percentage, then fixed cohort order. Training time and test data cannot affect the
choice. The selection artifact is bound to the complete screening ledger SHA-256,
and confirmation recomputes the selection before doing work. This policy is a
feasibility gate, not a universal superiority criterion.

Both stages check normalized roundtrips and save models.
Any vocabulary shortfall, zero-merge SuperBPE, or failed condition aborts completion.
There is no vocabulary padding, smaller-budget retry, or baseline substitution.
An interrupted Phase A stage may be continued with `--resume`. The runner validates
`plan.json`, every numbered condition record, current commit and extension,
dataset assignment, tokenizer configuration, vocabulary target, metrics, and
artifact hashes before skipping a condition. Any mismatch aborts the resume.
Condition files and the final ledger are atomically published. An A-SCREEN
`ledger.json` appears only after all 15 conditions validate; an A-CONFIRM ledger
appears only after every selected condition validates. Resume is unavailable for
LM Phases B and C. A resumable condition has `status=condition_complete`; this
describes the condition, not completion of its stage.
SuperBPE reserves `min(V // 10, 4000)` entries for CEM; if the existing trainer
cannot fill that reserve, report the failure. CEM receives EOS between documents.
SentencePiece receives every normalized document as ordered, contiguous chunks of
at most 1,024 Unicode characters. Concatenating the chunks exactly reconstructs
the normalized documents, so no normalized UTF-8 bytes are added, removed, or
reordered. This bounds SentencePiece's internal training units and avoids its
document-length-dependent numerical failure. It uses identity normalization after
the shared preprocessing, coverage 1.0, one thread, no dummy prefix, and no
whitespace collapse. Each Phase A row records this representation in
`training_input`; other tokenizers record `unit=normalized_documents`.
UniqToken uses the existing training defaults except four specials, byte fallback,
minimum frequency 1, and explicit Python EM (`min_edge_log_prob=-inf`). Seed mining
and encoding may use installed native operations. No tokenizer algorithm changes
are introduced by this protocol.

Phase B loads the exact Phase A artifacts and screens the full cohort/grid under
both matching regimes with one paired LM seed. It evaluates validation NLL only.
LM test loss is not computed or available for condition selection. Caps per
condition are 100 billion estimated FLOPs and 1,000,000 normalized training bytes;
these are safety limits, not claims of scientifically sufficient training.

Phase C requires a complete Phase B ledger and an explicit selection JSON bound
to its SHA-256 hash. Only selected tokenizer/vocabulary/regime triples run, each
with three distinct new paired LM seeds. Validation and test NLL are evaluated
once at the end of the fixed training budget; there is no test-driven checkpoint
selection. Publish the selection rationale and all successes/failures. Selection
is not evidence that an omitted baseline lost. Tokenizers are frozen from A;
the three seeds measure LM training variation, not tokenizer retraining variation.

## Data and normalization

Use `python -m benchmarks.freeze_phase_a_dataset` to prepare the Phase A corpus.
It accepts only immutable 40-character dataset commit hashes and writes no final
directory unless every gate succeeds. It freezes exactly 400,000,000 normalized
UTF-8 bytes from `allenai/MADLAD-400` `data-v1p5/clean_docs_v2` (160 MB
Latin/English, 160 MB Indic+CJK+Arabic, 80 MB Cyrillic+African) and exactly
100,000,000 normalized UTF-8 bytes of permissively licensed Python,
JavaScript/TypeScript, Java, SQL, C/C++, Rust, and Go from `bigcode/the-stack`
release `v1.3`; code records are admitted only when every dataset license identifier
is in the freezer's explicit permissive SPDX allowlist. It uses disjoint,
deterministically assigned halves of FLORES-200 dev for screening and confirmation
validation. FLORES devtest remains final test-only; neither dev nor devtest can
enter training.

The exact FLORES-200 repository, immutable revision, and license must be supplied
and approved before freezing, because public mirrors and upstream access differ.
The freezer downloads selected shards once into `sources/`, SHA-256 hashes them,
then all later phases consume only those local files. It rejects exact and
near-duplicate training documents and exact or near overlap with either FLORES
split.

Each frozen UTF-8 JSONL document has this shape:

```json
{"id":"globally-unique-document-id","text":"the complete document","language":"en","domain":"latin_english","raw_utf8_bytes":21,"normalized_utf8_bytes":21,"source":{"dataset":"allenai/MADLAD-400","revision":"immutable-commit","url":"pinned-source-url","local_path":"sources/...","source_file_sha256":"actual SHA-256","license":"ODC-By-1.0"},"dedup":{"status":"accepted_after_exact_and_near_eval_check"}}
```

Provide a manifest (paths relative to the manifest):

```json
{
  "schema_version": 2,
  "dataset_id": "your-versioned-corpus-id",
  "source": "documented source and revision",
  "license": "applicable license",
  "deduplication": "document the exact and near-duplicate procedure",
  "normalization": "NFKC_unicode_spaces_v1",
  "freeze": {
    "immutable": true,
    "source_files": [{"local_path": "sources/...", "sha256": "actual SHA-256", "file_bytes": 123, "dataset": "pinned dataset", "revision": "immutable commit", "release_variant": "release or variant", "url": "pinned source URL", "license": "applicable license"}],
    "source_revisions": {"dataset": "immutable commit"},
    "selection": {"byte_unit": "MB_decimal", "groups": [{"split": "train", "dataset": "pinned dataset", "release_variant": "release or variant", "language": "en", "domain": "latin_english", "documents": 1, "raw_utf8_bytes": 21, "normalized_utf8_bytes": 21}]}
  },
  "splits": {
    "train": {"path": "train.jsonl", "sha256": "actual file SHA-256"},
    "validation": {"path": "validation.jsonl", "sha256": "actual file SHA-256"},
    "test": {"path": "test.jsonl", "sha256": "actual file SHA-256"}
  }
}
```

The runner verifies source-file hashes, nonempty splits, globally unique IDs,
document byte counts, source provenance, and no duplicate normalized documents
within or across splits. Ordering is fingerprinted. It accepts only immutable,
local source inventories and recomputes every per-language/domain selection group;
no experiment-time network fetching occurs. Split before
tokenizer training and freeze language/domain assignments externally.

Each split and ordered document entry records both `source_utf8_bytes` (the
original JSONL document's decoded `text` string encoded as UTF-8, before
normalization) and `normalized_utf8_bytes` (after shared normalization). Source
counts exclude JSON syntax, record separators, and escape serialization overhead.
They include any whitespace/newlines inside the text. Both counts exclude BOS/EOS.
The assignment fingerprint includes source and normalized document hashes.

Shared preprocessing applies NFKC and the existing Unicode-space-to-ASCII-space
mapping, preserving case, punctuation, repeated spaces, tabs, and newlines. BPB
denominators refer to this normalized UTF-8 text, not the original pre-normalized
files. Corpus control-token syntax (`<|`) and reserved metaspace/private-use escape
characters are rejected rather than silently escaped or dropped. Unsupported
roundtrips fail. Investigate such failures before changing the protocol or corpus.

With normalization enabled the contract is
`decode(encode(x)) == normalize(x)`, not unconditional raw-byte reconstruction.
Here `normalize` means the externally visible normalized text, not the internal
metaspace representation returned by the library's `Normalizer.normalize` method.
Security sanitization is an additional transformation in the general tokenizer
API; the experiment excludes its reserved inputs. NFKC is not byte-for-byte lossless.

## Causal architecture

The existing `CausalMiniTransformer` is used as a decoder-only causal LM, despite
PyTorch naming its masked stack `TransformerEncoder`. It has learned positional
embeddings, pre-layer normalization, ReLU FFNs, dropout 0.1, a final layer norm,
and an **untied**, bias-free vocabulary head. It predicts next tokens causally.
Training uses FP32, AdamW (betas 0.9/0.999, epsilon 1e-8, weight decay 0.01), batch
size 1, no LR schedule, and deterministic PyTorch algorithms. Unsupported device
or computation failures abort; there is no substitute statistical language model.

| Stage | Layers / width / FFN / heads / context | LR | Total parameters at 16K / 32K / 64K |
| --- | --- | --- | --- |
| B | 2 / 128 / 512 / 4 / 128 | 0.001 | 4,607,488 / 8,801,792 / 17,190,400 |
| C | 12 / 768 / 3072 / 12 / 1024 | 0.0003 | 111,008,256 / 136,174,080 / 186,505,728 |

These are architecture counts, not experimental measurements. Every LM row records:

- `core_params`: `L*(4*d*d + 2*d*f + f + 9*d) + C*d + 2*d`, where L is layers,
  d is width, f is FFN width and C is context capacity. This vocabulary-independent
  group includes learned positional embeddings, Transformer biases/norms, and the
  final norm. `non_embedding_params` is an equal compatibility alias, not a claim
  that positional embeddings are excluded. It is 413,184 in B and 85,842,432 in C.
- `input_embedding_params`: `V*d`, the input vocabulary lookup table.
- `output_head_params`: `V*d`, the separate, untied, bias-free output projection.
- `total_params`: `core_params + input_embedding_params + output_head_params`.

The runner checks the total and components against the instantiated model and
rejects tied input/head weights. The architecture and 16K/32K/64K vocabularies are
unchanged. It is not named "125M"; tests check both architectures at all capacities.

## Matching and evaluation

Both regimes use the same ordered training documents; they answer different
questions and neither is universally preferable. Tokenizer training always uses
the full training split. LM batches never cross document boundaries.

- **FLOP-matched:** a shared requested analytical budget. For a length-S training
  window the estimator is `6*L*S*(4*d*d + 2*d*f) + 12*L*S*S*d + 6*S*d*V`.
  This counts dense matmul forward/backward work, including the vocabulary head.
  It excludes embedding lookups, normalization, activation, optimizer, and evaluation
  costs; it is not a hardware profiler or a wall-clock matching claim. A shortened
  final training window is allowed. Actual work must be within 1% below budget,
  never above. The matched quantity is total analytical FLOPs, not
  vocabulary-independent compute.
- **Byte-matched:** the same exact ordered document prefix, cycling training data
  if needed. The budget must end at a document boundary, with no byte rounding or
  mid-character truncation. A multiple of the total training bytes always qualifies.
  Numbers of tokens, windows, and optimizer updates can differ across tokenizers.

The unchanged FLOP estimator is version `dense_matmul_forward_backward_v1`.
Each LM row records it as `flop_estimator_version` (also `model_config.flop_estimator`)
and separates the accumulated work across executed windows:

- `core_analytical_flops`: sum of `6*L*S*(4*d*d + 2*d*f) + 12*L*S*S*d`.
  For identical window lengths this component is independent of vocabulary size.
- `output_projection_flops`: sum of `6*S*d*V`, explicitly vocabulary-dependent.
- `actual_analytical_flops`: the sum of those two fields, used for FLOP matching.

`training_target_tokens` stores sum(S), including EOS targets;
`training_sequence_length_squared_sum` stores sum(S*S), including shortened final
windows. The loader recomputes both FLOP components from these totals, vocabulary,
and the fixed architecture, and verifies their sum and estimator version. Lookup,
optimizer, normalization, activation, and evaluation work remain excluded. This
does not remove output-projection cost or silently switch to tied embeddings.

The explicit byte-budget policy is `byte_budget_field=normalized_utf8_bytes`,
with `byte_audit_field=source_utf8_bytes` as the secondary audit measure. Phase A
records both full-split tokenizer-training totals in `training_bytes`. LM rows
record both totals for the same ordered completed training-document prefix in
`training_bytes`, its count in `completed_training_documents`, and
`training_byte_scope=complete_document_prefix`. Repeated passes count both byte
types again. Document selection never depends on tokenizer segmentation.
The freezer rejects NUL and tokenizer-reserved text before quota selection because
SentencePiece can otherwise skip an entire input unit while appearing to continue.

For byte-matched LM conditions, `training_bytes.normalized_utf8_bytes` equals the
requested budget exactly; source exposure is recomputed from those same documents,
not used to select a different prefix. These are LM training-exposure counts, not
the CPU work of fitting tokenizers or pre-encoding the full training split. For
FLOP-matched conditions they are complete-prefix audit totals only: a final partly
accounted document is excluded, not assigned a fabricated source-byte fraction.
Token/FLOP totals still include every executed window. `completed_document_bytes`
is a compatibility alias for the normalized complete-prefix total.

Validation/test tokenizer and LM metrics also record both byte fields. Their old
`utf8_bytes` field remains an alias for normalized bytes. BPB and bytes-per-token
continue to use normalized bytes; source bytes are audit-only. For example, a
source `\uFB01` character has three UTF-8 bytes but normalizes to two-byte `fi`.

Each document is scored as BOS -> text tokens -> EOS, including the first text
token and every final short window. Nonoverlapping windows reset learned positions
and use at most the declared context; the previous target becomes the next input.
EOS contributes NLL but no text bytes. Validation and test totals are separate:

`test_BPB = test_total_NLL_nats / (test_total_normalized_UTF8_bytes * ln(2))`.

Token cross-entropy is total NLL divided by predicted tokens (including EOS).
Token perplexity is its exponential, with explicit overflow status if necessary;
it is not comparable across tokenizers as though they shared a prediction alphabet.
BPB is the causal token-sequence codelength of normalized documents, including EOS,
under the stated finite-context evaluation rule, not a marginal over segmentations.
Phase A's character density includes whitespace and is valid for CJK; it is not
word fertility or evidence of morphological accuracy.

## Provenance and commands

LM research ledgers require shared ledger schema 3 and research schema 5. Phase A
stage ledgers require stage schema 1. All require complete
expected conditions, tokenizer/model identity, exact vocabularies, dataset manifest
and assignment hashes, seeds, full model configuration, matching regime/budget,
Git commit, source hash, dependency versions, and installed extension binary hash
(or explicit unavailable status). The hash identifies installed bytes, not proof
that a binary was rebuilt from the recorded Rust source. Verify that separately.
The loader compares all provenance with the current environment and dataset.
Accounting-incomplete research-schema-1 ledgers are rejected, not migrated or
rewritten. Historical results remain untouched.

Start from a reviewed, committed, clean worktree, with a native extension rebuilt
from that commit and its binary SHA-256 recorded. Run outputs under ignored
`artifacts/` (or outside the repository). Existing output directories are rejected.
An interrupted Phase A stage retains its plan and atomically completed condition
files, but no complete ledger is written. `--resume` consumes only conditions that
match the current plan, code/build identity, dataset, tokenizer configuration,
budget, metrics, and artifacts. Any mismatch aborts.

PowerShell commands, from the repository root, after choosing immutable source
commits and a licensed FLORES-200 source:

```powershell
python -m benchmarks.freeze_phase_a_dataset --output artifacts/data --madlad-revision <40-char-MADLAD-commit> --stack-revision <40-char-The-Stack-commit> --stack-release v1.3 --flores-repo <approved-FLORES-200-repo> --flores-revision <40-char-FLORES-commit> --flores-license <license>
python -m benchmarks.run_phase_a screen --dataset artifacts/data/manifest.json --output artifacts/phase-a-screen
python -m benchmarks.run_phase_a select --dataset artifacts/data/manifest.json --screening artifacts/phase-a-screen/ledger.json --output artifacts/phase-a-selection.json
python -m benchmarks.run_phase_a confirm --dataset artifacts/data/manifest.json --screening artifacts/phase-a-screen/ledger.json --selection artifacts/phase-a-selection.json --output artifacts/phase-a-confirm
```

These are commands for future experiments, not runs performed during harness
development. The first command runs only A-SCREEN; it cannot start A-CONFIRM or an
LM stage. The selection and confirmation commands are separate, explicit actions.
Do not pass the new staged ledgers into the older LM runner until that later
interface is explicitly reviewed; this redesign does not run or alter LM phases.

Before C, author `artifacts/selection.json` with this structure, replacing the
screening hash and choosing actual conditions from B based on validation:

```json
{
  "schema_version": 1,
  "screening_sha256": "actual SHA-256 of final-b/ledger.json",
  "rationale": "pre-registered validation-based selection rule and its application",
  "conditions": [["boundary_bpe", 65536, "bytes"], ["uniq_superbpe", 65536, "bytes"]]
}
```

The example selection is not a recommendation or a claim that other conditions
lost. Preserve primary baselines relevant to every confirmatory comparison. A new
dataset, source tree, dependency/extension build, or altered tokenizer requires a
fresh A/B chain. Before Phase A, supply licensed frozen data, complete deduplication,
freeze training settings, verify the runtime/build, and commit the reviewed harness.
Exact 64K feasibility across all five trainers remains to be established by A.
