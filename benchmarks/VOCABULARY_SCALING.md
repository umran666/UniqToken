# Frozen tokenizer vocabulary scaling protocol

This tokenizer-only #92 matrix uses exact total budgets 8,192, 16,384, 32,768,
65,536 and 131,072. The cohort is SPM-Unigram, Boundary-BPE, and
`uniq_superbpe_r64`: current UniqToken Unigram/CEM with exactly 64 reserved
SuperBPE additions at every budget. This diagnostic variant is named explicitly;
it is not the historical Phase A training configuration.

All conditions use four controls and 256 canonical byte leaves. They share the
first 32 frozen training documents per domain/language and first 2,048 normalized
Unicode characters per document. Validation applies the same selection rule to
the disjoint frozen validation split. The source loader verifies frozen
train/validation hashes and rejects split path aliases or exact excerpt/ID
overlap before training. Selected IDs, excerpt hashes and source revisions are
retained. Test file content is never opened or hashed.

Candidate generation, normalization and sampling are fixed across budgets.
SPM uses identity normalization, preserved whitespace, deterministic input order
and one thread. Boundary-BPE does not merge across whitespace. UniqToken uses
minimum frequency one and the same seed/EM settings at each budget, with 64
ordinary SuperBPE slots and document EOS boundaries. No trainer uses controlled
random sampling; seed variation is therefore not invented.

## Resource measurements

The driver executes 15 independent workers sequentially. Each worker loads the
same normalized excerpt request and checks source/native-extension identity.
Training wall time and process CPU time cover model construction and artifact
serialization; imports, input loading and diagnostic evaluation are excluded.
There is one observation per condition, not a confidence interval.

Peak memory is operating-system process high-water RSS through training and
serialization: Windows GetProcessMemoryInfo or Unix getrusage. It includes Python,
Rust, trainer libraries, imports and input excerpts. The pre-training process
high-water is recorded separately; it is not subtracted from the peak because
high-water differences do not isolate live model allocation.

One Rayon thread and Python hash seed zero are fixed. Run on an otherwise idle
host; the hardware/software/native binary fingerprint is recorded. A
predeclared 900-second worker cap produces a resource_limit receipt. It does not
prove that the vocabulary budget is mathematically infeasible. Budget mismatch,
training failure, validation failure and worker failure have distinct receipts.
Failed/blocked conditions emit no compression metrics. No padding, corpus change,
budget relabeling or fallback tokenizer substitution is permitted.

## Output and interpretation

Every complete model is checked for exact vocabulary size, contiguous IDs, four
controls, 256 byte leaves, normalized roundtrip and byte reconstruction.
Diagnostics report normalized bytes/token, fallback percent, tokens per Unicode
code point and UTF-8 byte, exact byte-length histograms, vocabulary utilization,
merge counts and allocation provenance. Aggregate/domain/language/stratum rows
pool counts before ratios and quantiles.

The manifest hashes models, results, logs, CSV, report and plots. The private
worker request contains corpus excerpts and is omitted from published output;
its SHA-256 and the ordered assignment receipts are retained. The original
source manifest plus the fixed selection rule reproduces it.

Plot gaps show failed conditions. English/code languages lacking frozen
validation remain uncertified; training evidence does not substitute for them.
The curves describe this bounded diagnostic corpus and configuration. They do
not establish a universal scaling law, global winner or downstream LM quality.
Frozen Phase A/B/C artifacts and protocols remain unchanged.

```powershell
python -m benchmarks.vocabulary_scaling --dataset artifacts/data/phase-a-madlad-stack-flores-v1/manifest.json --output artifacts/vocabulary-scaling
python -m unittest tests.test_vocabulary_scaling -v
```

Use a clean committed checkout with the release native extension installed.
The output must be new and outside frozen inputs.
