# Contributing to UniqToken

Thank you for your interest in contributing to **UniqToken**. It is a Python tokenizer research toolkit with trainable Unigram and BPE models and an optional Rust acceleration extension.

---

## Quickstart Development Setup

### 1. Prerequisites
- Python 3.10+ (`python --version`)
- A current stable Rust toolchain (`cargo --version`)
- `maturin` (for native PyO3 wheel compilation)

### 2. Clone and Setup Environment

```bash
git clone https://github.com/umran666/UniqToken.git
cd UniqToken

python -m venv .venv
source .venv/bin/activate

python -m pip install -e ".[test]"
python -m pip install maturin
```

### 3. Build the Native Rust Core

```bash
maturin develop --manifest-path crates/uniqtoken_core/Cargo.toml --release
```

---

## Testing & Code Quality

Before opening a pull request, ensure all tests, linters, and type checkers pass cleanly:

```bash
# 1. Rust Clippy & Check
cargo check --manifest-path crates/uniqtoken_core/Cargo.toml --all-targets
cargo clippy --manifest-path crates/uniqtoken_core/Cargo.toml --all-targets -- -D warnings

# 2. Python Formatting & Linting
python -m ruff format .
python -m ruff check .

# 3. Python Type Checking
python -m mypy uniqtoken

# 4. Run Full Test Suite
python -m pytest

# 5. Run Benchmark Suite
python benchmarks/benchmark_suite.py
```

---

## Codebase Architecture Tour

- `crates/uniqtoken_core/`: Native Rust acceleration core with PyO3 bindings, character prefix trie, dynamic programming Viterbi lattice, and Rayon parallel batch encoder.
- `uniqtoken/tokenizer.py`: `CustomTokenizer`, encoding, alignment, batching, and serialization facade.
- `uniqtoken/pre_tokenizer.py`: normalization, character alignment, and ordered regex boundaries.
- `uniqtoken/unigram_trainer.py`: EM-based Unigram vocabulary trainer with convergence checks.
- `uniqtoken/bpe_trainer.py`: BPE vocabulary training; cross-word CEM/SuperBPE extension lives in `uniqtoken/cem_merger.py`.
- `uniqtoken/hf_adapter.py`: native `PreTrainedTokenizerFast` adapter.
- `uniqtoken/integrations/`: serving integrations such as the vLLM adapter.
- `benchmarks/`: held-out diagnostics plus the fail-closed Phase A/B/C research harnesses and protocols.
- `tests/`: unit, differential, fuzz, native-parity, and research-integrity regression suites.

---

## Good First Issues & Roadmap

Use the current [GitHub issue tracker](https://github.com/umran666/UniqToken/issues) rather than this document as the source of open work. Confirm that an issue is still open and unassigned before starting implementation.
