# Script-safe token density and length distributions

Issue #90 uses the existing exact source-byte diagnostics. Tokens per Unicode
character counts normalized code points, including combining marks, and is not a
grapheme or linguistic-word measure. Tokens per normalized UTF-8 byte uses source
bytes before metaspace encoding. Bytes per token is its reciprocal when defined.
No whitespace-word denominator is used.

Token length is the exact decoded normalized source-byte contribution. A byte
fallback leaf contributes one byte even when several leaves share a character.
Histograms count token emissions; p50/p95/p99 use nearest rank. Empty ratios and
percentiles are null. Aggregate, domain and language results pool counts and
histograms from strata before calculating metrics.

## Reproduce

From a clean checkout, export the retained, hash-verified #85 tokenizer-only
diagnostic without reopening data, loading models, training an LM, or touching
the held-out test set:

```powershell
python -m benchmarks.token_density --analysis benchmarks/failure_analysis/issue85/results.json --output artifacts/token-density
python -m unittest tests.test_token_density -v
```

For new frozen-model diagnostics, first run
`python -m benchmarks.analyze_tokenizer_failures --help` and supply its required
dataset manifest, completed A-SCREEN ledger and a new output directory; pass
that run's receipted results.json to the density exporter.

Python callers can use `analyze_condition(tokenizer, assignments)` from
`benchmarks.tokenizer_failure_metrics` and project rows with
`benchmarks.token_density.project_record`. The existing observation path checks
that decoded byte contributions reconstruct the normalized source exactly.

The density ledger validates integer denominators, exact histogram totals,
derived ratios, pooled aggregates, complete language/domain coverage, duplicate
rows, and unsupported test splits. Both the general ledger and density schema
reject ambiguous fertility fields even inside nested metric objects.
The export retains the original measurement identity, model hashes, frozen
assignment receipts, and a distinct export commit. Re-exporting old counts does
not remeasure the tokenizers or imply new model results.

The retained export in [token_density/issue90](token_density/issue90) contains
963 rows from the nine frozen #85 models, with 18 aggregate rows, alongside
stratum/language/domain detail. Its manifest hashes every JSON/CSV artifact.
