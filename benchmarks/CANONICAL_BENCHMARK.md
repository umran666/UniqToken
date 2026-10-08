# Canonical End-to-End Tokenizer Benchmark Guide (Issue #105)

Consolidates validated measurement methodology and historical performance evidence from issues #95 through #104 into a unified, reproducible benchmark suite.

## Execution

From the repository root, run the fixed canonical benchmark:

```powershell
python -m benchmarks.canonical_benchmark --output benchmarks/canonical/issue105
```

The canonical command is `python -m benchmarks.canonical_benchmark --output benchmarks/canonical/issue105`. Default parameters are:
- `--warmup 3`: 3 warmup iterations before timing.
- `--repetitions 7`: 7 repeated trials per cell.
- `--iterations 5`: 5 internal iterations per repetition.
- `--workers 1`: strictly single-worker execution (aligned with #95 and #100 core capacity findings).
- `--build-mode release`: recorded in metadata.

## Exact Parity Gate

Before benchmark performance numbers are collected or emitted, the runner executes a strict parity gate across all five canonical fixtures:
1. Deterministic token strings and token IDs between single and batch executions.
2. Token boundary tiling check against an independent character alignment oracle.
3. Decode round-trip equality: `tok.decode(ids) == tok.decode_tokens(tokens)`.
4. Batch decode parity: `tok.decode_batch(batch_ids) == [tok.decode(seq) for seq in batch_ids]`.

Any failure raises an `AssertionError` and aborts the benchmark before writing results.

## Canonical Metrics

- **Primary Cross-Tokenizer Metric**:
  **Normalized input MB/s** (decimal megabytes processed per second: $10^6\text{ bytes/s}$). Evaluates raw text ingestion rate and is the only fair metric across different subword vocabulary representations.
- **Secondary Metrics**:
  - **Tokens per Second**: Clearly designated as *segmentation-dependent*. Models with smaller piece budgets or higher character fallback rates emit more tokens per character, artificially inflating tokens/s without performing more tokenization work.
  - **Latency (p50 and p95 ms)**: Median and 95th percentile wall time per call.
  - **Peak Memory (MB)**: Process Working Set / Peak RSS tracked via `psutil`.
  - **CPU Utilization (%)**: CPU percentage across worker runs.
  - **Batch Scaling**: Ratio of throughput at batch size 128 compared to batch size 1.

## Evaluated Tokenizers & Identity

Trained deterministically on the identical 400-piece fixture corpus (`list(FIXTURES.values()) * 3`):
1. **UniqToken**: Pure Python Unigram engine with Metaspace & Byte Fallback (noting that `uniqtoken_core` native Rust binary was uncompiled in this Windows host environment due to missing MSVC `link.exe`).
2. **SentencePiece**: Prebuilt SentencePiece C++ runtime (`0.2.2`), Unigram model with Byte Fallback (`spm.SentencePieceProcessor`).
3. **HuggingFace Tokenizers**: Prebuilt Tokenizers Rust runtime (`0.23.2`), Byte-Level BPE model (`tokenizers.Tokenizer`).

## Published Receipts

Located in `benchmarks/canonical/issue105/`:
- `results.json`: Versioned machine-readable results (`schema_version: "1.0"`) containing full metadata, hardware fingerprint, baseline availability checks, historical optimization ledger, geometric mean summaries, and individual workload records.
- `REPORT.md`: Comprehensive analytical Markdown report with geometric mean table, complete workload matrix, historical optimization ledger, and interpretation rules.
- `manifest.json`: Cryptographic SHA-256 hashes of `results.json` and `REPORT.md`.
- `artifacts/`: Serialized model artifacts (`uniqtoken_model/`, `spm.model`, `hf_tokenizer.json`) and training corpus.
