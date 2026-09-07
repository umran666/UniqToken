# Contributing to UniqToken

Thank you for your interest in contributing to **UniqToken**! UniqToken is an ultra-fast, high-precision, zero-fallback Byte-Fallback Unigram Tokenizer engineered in Rust and Python.

---

## Quickstart Development Setup

### 1. Prerequisites
- Python 3.9+ (`python --version`)
- Rust & Cargo 1.70+ (`cargo --version`)
- `maturin` (for native PyO3 wheel compilation)

### 2. Clone and Setup Environment

```bash
git clone https://github.com/umran666/UniqToken.git
cd UniqToken

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install maturin ruff mypy pytest
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
python -m mypy .

# 4. Run Full Unit Test Suite (150+ tests)
python -m unittest discover -p "test_*.py" -v

# 5. Run Benchmark Suite
python benchmarks/benchmark_suite.py
```

---

## Codebase Architecture Tour

- `crates/uniqtoken_core/`: Native Rust acceleration core with PyO3 bindings, character prefix trie, dynamic programming Viterbi lattice, and Rayon parallel batch encoder.
- `tokenizer.py`: Top-level `CustomTokenizer` and `TokenSpan` APIs with zero-copy slice tracking.
- `pre_tokenizer.py`: `RegexPreTokenizer` with compiled regex LRU pattern caching.
- `normalizer.py`: Unicode normalization, whitespace collapsing, and exact character offset mapping.
- `unigram_trainer.py`: EM-based Unigram vocabulary trainer with early stopping and log-likelihood convergence.
- `seed_builder.py`: Multi-character PMI and frequency candidate miner.
- `bpe_trainer.py`: SuperBPE cross-word merge engine.
- `vocab_adapter.py`: Dynamic online vocabulary expansion and lossless ID compaction adapter.
- `hf_exporter.py`: HuggingFace `PreTrainedTokenizerFast` schema exporter.
- `hf_adapter.py`: Native `PreTrainedTokenizerFast` subclass (`UniqTokenizerFast`) that round-trips a trained tokenizer through the HuggingFace ecosystem.
- `streaming_decoder.py`: Incremental token streaming decoder with UTF-8 byte accumulation.
- `benchmarks/`: Empirical evaluation suite across diverse scripts and languages.

---

## Good First Issues & Roadmap

1. **Add Language Corpora**: Add evaluation texts for underrepresented languages (e.g. Swahili, Yoruba, Amharic, Vietnamese) to `benchmarks/benchmark_suite.py`.
2. **WebAssembly Target (`wasm32-unknown-unknown`)**: Enable `wasm-bindgen` in `crates/uniqtoken_core` for in-browser client tokenization.
3. **GGUF Export Support**: Add `.gguf` metadata exporter for direct consumption in `llama.cpp`.
4. **Interactive Colab / Playground**: Build a Gradio app or Colab notebook demonstrating tokenization comparisons.
5. **C-API Header Export**: Expose a clean `uniqtoken.h` C shared library for embedding in C/C++/Go applications.
