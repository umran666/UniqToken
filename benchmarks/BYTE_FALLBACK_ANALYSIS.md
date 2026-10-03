# Byte fallback pressure: atomic recovery protocol

This experiment addresses #86 with complete Unicode scalar recovery. It is an
offline research candidate; the library's canonical one-byte fallback format and
existing multilingual CEM behavior are unchanged.

## Review corrections

The original PR's results used approximately 1.6K vocabulary entries while
labeling them 8K, 16K and 32K. Those artifacts and positive conclusions are
withdrawn. Incomplete UTF-8 prefix entries were unreachable as source subwords
and could change literal-notation decoding. The revised experiment never creates
them. Byte spans count source bytes, not escaped spellings or merged IDs.

## Predeclared hypotheses

- H1: admitting missing whole characters selected from training fallback runs
  reduces fallback emissions and contiguous byte-span lengths on validation.
  Report all per-stratum changes, including any increased p95 or maximum.
- H2: selecting recovery slots using an added fallback-byte utility term changes
  the chosen characters and improves fallback relative to unweighted recovery.
  Identical selections or validation outcomes provide no evidence of benefit.
- H3: recovery can use 16 of 64 reserved vocabulary slots while keeping normalized
  bytes/token loss at or below 1% in EVERY observed validation stratum. A failed
  stratum rejects this policy for that budget. Missing validation coverage is
  explicitly uncertified. No aggregate winner overrides a failed stratum.
- Every addition is audited for training/validation emissions. Zero validation
  emissions alone do not establish that a token can never occur. Any incomplete
  byte-prefix addition is an implementation failure.

These hypotheses are registered in source before the revised measurements.
The former Latin-starvation hypothesis is not tested by this sequential policy:
it does not put ordinary pairs and character recoveries in one competing queue.

## Fixed matrix and units

All three conditions use 8,192 / 16,384 / 32,768 total vocabulary entries,
including four controls and 256 byte leaves. Each budget shares one Unigram seed
of target minus 64 entries. Baseline spends all 64 entries on ordinary SuperBPE.
Recovery conditions admit up to 16 missing Unicode scalars first, then spend the
remaining slots on the same SuperBPE implementation. Underfilled seeds or final
models are errors, never padded or relabeled.

For candidate character c, f is its count in training fallback runs, N is all
training token emissions, and B(c) is its sequence of canonical byte leaves:

    CE(c) = f * (sum(log P(b) for b in B(c)) - log(f/N))
    score(c) = CE(c) - lambda * f * len(UTF8(c))

Lower scores rank first, followed by higher frequency and lexical character
order. The CE-like admission term uses nats times occurrences. Lambda has units
nats per fallback byte and is fixed at 0 or 5; it is not tuned on validation.
Only complete two-, three-, or four-byte scalars occurring at least twice are
eligible. This is a fixed-candidate admission calculation, not an exact corpus
likelihood difference. Existing IDs are retained, then probabilities normalized.

Training takes the first 32 documents per domain/language stratum from the frozen
training split, each truncated to its first 2,048 normalized Unicode characters.
Validation independently applies that same rule to the frozen validation split.
Every selected ID and excerpt hash is retained. Exact normalized excerpt or ID
overlap aborts the run. This is a bounded diagnostic subset, not full Phase A
training or its published baseline. No random sampling, held-out test access,
language-model run, or change to any frozen Phase A/B/C artifact occurs.
The test hash is copied from manifest metadata; its file is never opened.

Bytes/token uses normalized UTF-8 source bytes divided by emitted tokens.
Fallback percentage is canonical fallback emissions / all emissions * 100.
Span histograms use contiguous fallback bytes; quantiles use nearest rank.
A stratum without spans has null span statistics. Ratios and regression gates
use unrounded counts. Major-stratum regression includes every observed
validation stratum, including tail languages.

## Reproduce

From a clean checkout with the native extension installed:

```powershell
$env:RAYON_NUM_THREADS = "1"
$env:PYTHONHASHSEED = "0"
python -m benchmarks.byte_fallback_analysis --dataset artifacts/data/phase-a-madlad-stack-flores-v1/manifest.json --output artifacts/byte-fallback-reviewed --budgets 8192 16384 32768
python -m unittest tests.test_byte_fallback_analysis -v
```

The output must be new and outside the frozen source directory. Results include
source/runtime identity, fixed configuration, assignment receipts, model hashes,
all learned additions, utilization, per-stratum fallback histograms, CSV and a
data-derived report. A final manifest hashes the complete output. Model files are
retained locally; their hashes identify the artifacts used for each result.
