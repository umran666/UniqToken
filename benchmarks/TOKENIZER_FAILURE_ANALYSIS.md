# Multilingual tokenizer failure analysis

`python -m benchmarks.analyze_tokenizer_failures` implements the tokenizer-only
diagnostic in [issue #85](https://github.com/umran666/UniqToken/issues/85).
It loads the existing SPM-Unigram, Boundary-BPE and UT-SuperBPE artifacts from a
complete A-SCREEN ledger at 16,384, 32,768 and 65,536 entries. It does not train
tokenizers or initialize/train an LM. Frozen Phase A/B/C artifacts are read-only.

## Run

Use the original frozen Phase A dataset manifest and completed ledger together.
The loader verifies their relationship, exact training and screening-validation
assignments, all model artifact hashes, controls, budgets and historical
condition records. A replacement dataset with the same name is not equivalent.
Commit the reviewed code first; a dirty checkout is rejected.
Install the `bench` extra for the analysis and plots (`pip install -e ".[bench]"`).

```bash
python -m benchmarks.analyze_tokenizer_failures \
  --dataset artifacts/data/phase-a-madlad-stack-flores-v1/manifest.json \
  --phase-a artifacts/phase-a-screen-modal-f57b93d/screen-migrated-f57b93d/ledger.json \
  --output artifacts/failure-analysis-issue85 \
  --max-training-documents-per-stratum 32 \
  --rare-threshold 5
```

The frozen corpus and model artifacts are external inputs, not bundled example
data. Absolute input paths are accepted, allowing a separate worktree to analyze
the original artifacts. The output must be a new directory outside both input
directories. Existing evidence is never overwritten. A complete `manifest.json`
with output hashes is published only after all conditions, legacy validation
checks, tables and plots succeed.

The default training diagnostic selects the first 32 **whole documents per
stratum from the original frozen tokenizer-training assignment**, retaining
original order. It records selected document-ID and normalized-text assignment
hashes and exposure counts. This is a bounded diagnostic sample, not tokenizer
retraining, a new source assignment, or the complete training-frequency census.
Set the cap to `0` for all documents in that assignment. Validation always uses
the entire original screening half; there is no evaluation sampling option.
`--vocab-budget` can select existing budgets explicitly; duplicate budgets fail.
`--no-plots` is available for tables-only checks; normal runs produce plots.

## Coverage and integrity

The current frozen training inventory has 30 language/domain strata (including
nine programming-language strata), while screening validation has 20 FLORES
languages. Therefore all 30 strata can receive **training** diagnostics, but
held-out code metrics and some natural-language metrics are unavailable. The
report and `coverage.csv` identify that gap explicitly. Validation retains its
`flores200` domain; it is not relabeled as web or code. Training, validation,
domain, language and aggregate tables remain distinct. Do not use in-sample
training observations as evidence of held-out code performance.

The entry point uses `run_phase_a.load_stage_source`, which reads only train and
validation. It checks declared split paths for aliases/escape before opening
either split. It rejects normalized-text or document-ID train/validation overlap.
Only the declared test hash is copied as metadata; the test file is never opened,
hashed, tokenized or scored. Source inventories are not rescanned or deduplicated
against held-out test. Confirmation validation is excluded from metrics.

Every loaded tokenizer must reproduce its original aggregate A-SCREEN validation
token count, fallback count/percentage, bytes per token and tokens per character.
A mismatch fails loudly rather than reinterpreting historical results. The run
also verifies source/model hashes and source-code/native-extension identity
before publishing. No tokenizer, merge, normalization, or default API is changed.

## Metric definitions

All denominators use shared **normalized** text, excluding BOS/EOS. Original
source byte counts remain available separately. Empty denominators are `null`
in JSON (blank in scalar CSV columns), not a measured zero.

| Metric | Definition |
| --- | --- |
| Bytes per token | Total normalized UTF-8 bytes / emitted tokens. Higher means more compression on this assignment. |
| Tokens per Unicode character | Emitted tokens / Python Unicode characters in normalized text, not graphemes or guessed words. |
| Byte-fallback percentage | 100 * emitted canonical `<0xXX>` byte tokens / all emitted tokens. |
| Token-length distribution | Histogram of each token's exact decoded source-byte contribution, plus pooled nearest-rank p50/p95, mean and maximum. A byte token contributes one byte, not its six-character piece spelling. |
| Vocabulary utilization | Observed distinct IDs / (total vocabulary minus four controls), as a percentage. The denominator retains 256 byte entries; aggregate uses the ID union, not mean per-stratum utilization. |
| Rare-token frequency | Percentage of emissions whose pooled diagnostic-training ID count is at most the recorded threshold (default 5), including count zero. Unseen-in-diagnostic-training emissions are also reported separately. With a training cap this is sample rarity, not full-model-training rarity. |
| Cross-word merge rate | Accepted applications in the existing SuperBPE pass / initial adjacent token boundaries, summed within documents. A scoped observer records before/after counts without changing the pass. SPM and Boundary-BPE have zero SuperBPE execution events. |
| Cross-word token percentage | Emitted tokens touching at least two non-whitespace fields separated by whitespace / emitted tokens. This is separate from merge applications and is not a morphological-word metric. |
| Whitespace fragmentation | Maximal `str.isspace()` runs: Unicode character count, token-byte-span intersections, intersections per run, split-run percentage, excess fragments, and isolated/mixed tokens. |
| Punctuation fragmentation | Same counts for maximal Unicode general-category `P*` runs. Symbols such as currency signs are not reclassified as punctuation. Multiple byte fallback tokens can split one Unicode punctuation character. |

Token byte contributions are reconstructed from each model's emitted pieces and
checked to concatenate exactly to the normalized source UTF-8 bytes. This avoids
confusing normalized character offsets, piece spellings and raw byte spans.
Public fallback tokens may share character offsets; the analyzer does not impose
a disjointness requirement on those offsets. Unsupported reconstruction fails.

The metrics module pools sufficient counts, ID frequency counters and length
histograms before calculating language, domain and aggregate values. It never
averages per-stratum ratios or quantiles. All plots use the same recorded tables.

For Boundary-BPE the analyzer validates the byte maps once and temporarily uses
immutable copies of vocabulary/ID/rank lookup tables for each condition. Within
that read-only scope it memoizes repeated byte-map validation; the production
heap, whitespace chunking, pieces, IDs and round-trip checks are unchanged.
Original object fields/methods are restored even on failure. This reader is
recorded as `validated_immutable_maps_v1` and has production-ID parity tests.
It makes no throughput claim and changes no library code or frozen artifact.

## Outputs

- `results.json`: schema, deterministic configuration, metric definitions,
  code/runtime identity, frozen model/input hashes, assignments, coverage and
  all metric rows. It contains no corpus text or token-piece examples.
- `strata.csv`, `languages.csv`, `domains.csv`, `aggregate.csv`: separate scopes,
  including both explicitly labeled diagnostic splits and all requested metrics.
- `coverage.csv`: all 30 training strata and screening-validation availability.
- `token_lengths.csv`: long-form source-byte length histogram by split/scope/model.
- `aggregate_train.png`, `aggregate_validation.png`: nine metric panels across
  the three fixed vocabulary budgets.
- `languages_<split>_<budget>.png`: corresponding per-language metric panels.
- `REPORT.md`: measured failure signals and separate plausible hypotheses for
  #86, #88 and #89, limitations and the missing-validation gap for #94.
- `manifest.json`: complete-run marker and SHA-256 for every published artifact.

The retained [issue-85 report](failure_analysis/issue85/REPORT.md) and results
under `benchmarks/failure_analysis/issue85/` use the
configuration and source commit recorded in their receipt. They are descriptive
observations scoped to these frozen models and samples. They do not establish a
globally best tokenizer or a causal mechanism. Phase B's 16K byte-matched
UT-SuperBPE result remains exploratory; this task supplies no LM confirmation.

## Regression checks

```bash
python -m unittest tests.test_tokenizer_failure_analysis -v
python -m ruff check benchmarks/analyze_tokenizer_failures.py benchmarks/tokenizer_failure_metrics.py tests/test_tokenizer_failure_analysis.py
```

CI discovers the tests through the existing `unittest` runner. The fixed fixture
requires byte-identical JSON, CSV, Markdown, PNG and output hash receipts across
two runs on one runtime. Independent tests cover multi-byte punctuation fallback,
weighted aggregate accounting, vocabulary unions, quantiles, training-only rarity,
source reconstruction errors, the production SuperBPE observer, explicit sampling,
split aliases, a poison test path, and legacy validation metric drift. Cross-host
PNG byte identity is not promised; software versions are recorded and plots use
the DejaVu Sans font family.
