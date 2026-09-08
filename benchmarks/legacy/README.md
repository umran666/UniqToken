# Archived Legacy Benchmark Ledgers & Figures

This directory contains archived experimental ledgers and figures from earlier Phase 14 and Phase 15 exploratory runs.

## Archived Artifacts
- `phase_fourteen_confirmatory_records.json`
- `phase_fourteen_confirmatory.png`
- `phase_fifteen_final_paper_records.json`
- `phase_fifteen_final_paper_figure.png`

## Invalidation Notice
Per **Issue #50** ("Rebuild matched-budget (8k-128k) benchmark harness and invalidate legacy ledgers"), these historical files have been superseded by the standardized, peer-review-grade benchmark suite located in [`benchmarks/run_matched_budget_eval.py`](../run_matched_budget_eval.py).

The current benchmark harness implements:
- Strictly matched analytical FLOP budgets on CUDA.
- Balanced multilingual evaluation across 8 domains (English, Hindi, Telugu, Arabic, Chinese, Russian, Code, Finnish).
- Standard metrics: Bytes per Token (BpT), Tokens per Byte (TpB), True Information Density / Bits per Byte (TID-BPB), downstream Cross-Entropy loss on matched small Transformer LMs (2L-128d, 4L-256d, 8L-512d), microsecond latency, peak RSS, and VRAM.
- Standard output ledgers in `benchmarks/matched_budget_eval_records.json`.
