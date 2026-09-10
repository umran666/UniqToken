# Archived Legacy Benchmark Ledgers & Figures

This directory contains archived benchmark code, ledgers, and figures. These artifacts are preserved for provenance only and are not evidence for the current implementation.

## Archived Artifacts
- `phase_fourteen_confirmatory_records.json`
- `phase_fourteen_confirmatory.png`
- `phase_fifteen_final_paper_records.json`
- `phase_fifteen_final_paper_figure.png`
- `matched_budget_eval_records_pre_integrity.json`
- `matched_budget_tradeoffs_pre_integrity.png`
- `run_phase_fourteen_confirmatory.py`
- `run_final_paper_audit.py`

## Invalidation Notice
These files have been superseded by the active harness in [`benchmarks/run_matched_budget_eval.py`](../run_matched_budget_eval.py). They must not be copied into current documentation or loaded as current results.

The archived Phase 14/15 workflow used data and result contracts that are no longer accepted. The pre-integrity matched-budget ledger also contains rows whose actual vocabulary size differs from the requested budget. Historical numbers are intentionally left unchanged; this notice, the filenames, and the directory location provide the invalidation context.

Current matched-budget ledgers require schema version 3, experiment version `research-integrity-heldout-v3`, a full Git commit hash and working-tree dirty status, a document-disjoint split, explicit `model_kind`, exact requested/actual vocabulary equality, and a complete condition grid. The loader in `benchmarks.ledger` rejects older schemas and ambiguous cross-script fertility fields. Historical artifact contents remain unchanged.
