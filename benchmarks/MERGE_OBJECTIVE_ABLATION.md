# Offline SuperBPE objective ablations

This is the explanatory experiment for #87. It does not install a new library
objective or tune against Phase B results, test data, or an LM.

## Frozen protocol

The first 32 documents per domain/language and first 2,048 normalized characters
per document are selected independently from the existing frozen train and
validation assignments. The #86 loader verifies source hashes, rejects path
aliases and exact ID/excerpt overlap, and never opens the held-out test file.
Every condition shares this selection, NFKC/Unicode-space normalization, four
control tokens, 256 byte leaves, an 8,128-entry Unigram seed and a 64-entry merge
reserve: exactly 8,192 total entries. No underfilled or padded condition counts.
The deterministic trainers use no random sampling or seed variation.

Candidate generation reproduces current SuperBPE's initial 200-token stream
breaks, byte/control exclusion, length limit, cross-word eligibility, frequency
of at least two, and negative CE threshold. Document EOS controls forbid
cross-document merges. This one initial pool is frozen across component
ablations. All candidate features and every selected token are retained.

The dynamic current SuperBPE implementation is an additional reference. It
updates counts and discovers recursive merges. The pool_ce control isolates the
effect of freezing the candidate pool/state from changing its objective; a
comparison with the dynamic reference alone would conflate both.

## Components

For candidate (a,b), f counts initial adjacent occurrences, N counts all initial
adjacent pairs, and g counts nonoverlapping replacements of this candidate alone.

| Component | Definition | Units | Direction |
| --- | --- | --- | --- |
| cross_entropy | f * (log P(a) + log P(b) - log(f/N)) | nats * occurrences | minimize |
| compression_gain | g | first-order tokens saved | maximize |
| document_frequency | number of training documents containing (a,b) | documents | maximize |
| boundary_cost | g if the join crosses whitespace/nonwhitespace, otherwise 0 | transitions covered | minimize |
| fallback_cost | 0 under the current non-byte candidate eligibility | fallback emissions added | minimize |
| fragmentation_penalty | -g if the join is inside one whitespace or Unicode-P punctuation run, otherwise 0 | run intersections removed, negated | minimize |

Metaspace represents whitespace for component accounting. Boundary cost is an
orthographic proxy, not linguistic correctness. Compression gain is a
single-candidate replacement prediction; actual Viterbi re-encoding, probability
renormalization and interactions may differ. A constant fallback component is a
measured limitation of this candidate pool, not evidence that fallback costs are
irrelevant. Atomic character recovery remains a separate #86 experiment.

Each component is divided by its maximum absolute value over the shared training
pool (use one when all values are zero). Minimize the component with the stated
sign. Predeclared combinations use equal coefficients:

- maximize compression, minimize boundary and fragmentation penalties;
- minimize CE and maximize compression.

Ties use higher adjacent frequency, then lexical left/right token order.
Duplicate concatenations consume only one slot. Constant-component selections
therefore reflect the tie-breaker, not a learned effect. No validation-driven
weight search occurs. Every admission uses the same initial empirical log(f/N)
and the same probability normalization.

## Measurements

Record normalized bytes/token, fallback percent, tokens per Unicode character,
exact source-byte length distributions, utilization and whitespace/punctuation
fragmentation at aggregate, domain, language and stratum levels. Report each
stratum with >1% BpT regression, increased fallback, or increased punctuation
splitting relative to current SuperBPE. Missing validation coverage remains
missing. Negative outcomes and constant components remain in the report.
These measurements neither select a final objective nor establish LM quality.

```powershell
$env:RAYON_NUM_THREADS = "1"
$env:PYTHONHASHSEED = "0"
python -m benchmarks.merge_objective_ablation --dataset artifacts/data/phase-a-madlad-stack-flores-v1/manifest.json --output artifacts/objective-ablation
python -m unittest tests.test_merge_objective_ablation -v
```

Run from a clean committed checkout with the release native extension installed.
The output must be new. Source/runtime, assignment, pool, model and output hashes
make each condition auditable. No frozen Phase A/B/C artifact is rewritten.

The committed `objective_ablation/issue87/evidence.zip` preserves every original
model, result and manifest byte. Reports and CSV remain directly readable;
`archive.json` binds the ZIP and previews. Tests verify and extract the archive
before checking the complete matrix. To inspect the original layout:

```powershell
python -m benchmarks.receipt_archive unpack --source benchmarks/objective_ablation/issue87 --output artifacts/issue87-original
```
