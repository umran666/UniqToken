# UniqToken Compatibility Exceptions Matrix

## Overview

UniqToken's Compatibility Engine (`uniqtoken.compat`) provides drop-in tokenization
adapters that produce **identical token IDs** to reference implementations. This
document records the formal compatibility status and any known intentional divergences
between UniqToken and each supported reference tokenizer.

The differential test suite in `tests/test_differential_compat.py` enforces the matrix
on every CI run (Python 3.10-3.12 on Ubuntu, macOS, and Windows). Tests that require
optional reference packages (`tiktoken`, `tokenizers`, `transformers`) skip gracefully
when the package is missing or the reference model cannot be downloaded.

## Compatibility Matrix

| Reference Tokenizer | Preset / Model | Adapter Class | Parity Status | Notes |
| :--- | :--- | :--- | :---: | :--- |
| OpenAI `tiktoken` | `cl100k_base` | `TiktokenEncoding` | 100% | Bit-for-bit ID parity across 50k samples |
| OpenAI `tiktoken` | `o200k_base` | `TiktokenEncoding` | 100% | Bit-for-bit ID parity |
| OpenAI `tiktoken` | `gpt2` | `TiktokenEncoding` | 100% | Bit-for-bit ID parity across 50k samples |
| HuggingFace `tokenizers` | ByteLevel BPE (GPT-2) | `HFByteLevelBPE` | 100% | Bit-for-bit ID parity across 50k samples |
| HuggingFace `tokenizers` | LLaMA-3 BPE | `import_hf_tokenizer` | Conditional | See exception #7 below (8 Vietnamese/Czech strings) |
| HuggingFace `tokenizers` | Mistral Metaspace BPE | `import_hf_tokenizer` | Conditional | See exception #6 below (gated HF repo; Metaspace) |
| HuggingFace `tokenizers` | Unigram (LLaMA) | `import_hf_unigram` | 100% | Via Unigram lattice |
| Google `sentencepiece` | Unigram `.model` | `SentencePieceModel` | Conditional | See exception #1 below |

## Known Exceptions & Intentional Divergences

### 1. SentencePiece Leading Dummy Whitespace

SentencePiece prepends a synthetic metaspace (`▁`) to the first token of a sequence by
default (`add_dummy_prefix=True`). UniqToken's normalizer requires explicit configuration
via `prepend_scheme="always"` to replicate this behavior. When importing via
`SentencePieceModel`, the `add_dummy_prefix` flag is respected automatically.

**Impact**: Encode output differs by one leading `▁` token when the
`add_dummy_prefix` configuration is mismatched. The differential test suite accounts
for this by comparing with the flag explicitly set.

### 2. `regex` Package Requirement

Tiktoken and LLaMA-3 pre-tokenization patterns use Unicode property classes
(`\p{L}`, `\p{N}`) and possessive quantifiers (`?+`, `++`) that Python's standard
`re` module cannot parse. The third-party `regex` package is strictly required for
exact parity. Install with:

```bash
pip install regex
```

**Impact**: `TiktokenEncoding` and `HFByteLevelBPE` raise `ImportError` at construction
time if `regex` is not available.

### 3. Special Token Escaping Policy

`TiktokenEncoding.encode()` and `HFByteLevelBPE.encode()` raise `ValueError` for unescaped
special tokens (e.g., `<|endoftext|>`, `<|fim_prefix|>`) when `allowed_special` does not
include them, matching tiktoken's default behavior exactly. The differential test
`test_cl100k_special_token_handling` verifies both the rejection and the exact ID returned
when the special token IS in the allowed set.

**Impact**: No divergence - behavior is identical to the reference.

### 4. Digit Chunking in Research Engine

UniqToken's Research Engine defaults to `block3` digit chunking (`\d{1,3}`) for
arithmetic reasoning. The Compatibility Engine preserves each reference tokenizer's
exact digit regex pattern, so no divergence occurs in compatibility mode.

**Impact**: No divergence in compat mode. Research-engine digit chunking is documented
separately in the Research Engine configuration.

### 5. LLaMA-3 Pre-tokenization Pattern

LLaMA-3 uses a different pre-tokenization regex pattern than GPT-2 or `cl100k_base`.
UniqToken's `import_hf_tokenizer()` auto-detects and applies the correct pattern from
the HuggingFace `tokenizer.json` configuration (via the `HFByteLevelBPE(pattern=...)` argument).

**Impact**: No divergence - the pattern is extracted from the source JSON and applied
verbatim.

### 6. Mistral Metaspace BPE Gating

`mistralai/Mistral-7B-v0.1` and `unsloth/mistral-7b-v0.2` are gated repositories on HuggingFace
and require explicit authentication. The differential suite attempts both and skips the
`test_mistral_bpe_encode_parity` test gracefully when neither is accessible. Mistral
uses a Metaspace pre-tokenizer (`▁` with `prepend_scheme="first"`, `split=false`) rather
than ByteLevel, so UniqToken imports it as a `BPEModel` (vocab/merges only) with a
warning; byte-level parity via `HFByteLevelBPE` is not applicable and the test is
skipped as conditional.

**Impact**: Test may skip in CI. The Mistral adapter path uses `BPEModel` and is
otherwise identical in vocab/merges.

### 7. LLaMA-3 Vietnamese BPE Divergence

LLaMA-3's BPE shows a deterministic 8-string divergence on the 50k corpus under the
pure-Python `HFByteLevelBPE` path for the seed string `Tiếng Việt xử lý ngôn ngữ`
and its whitespace variants (indices 71, 1076–1079, 1084–1085) plus one Czech fuzz
slice. The reference HF Rust tokenizer merges `ĠViá»ĩt` as a single token while the
Python BPE yields `ĠVi` + `á»ĩ` + `t` due to rank-order tie-breaking on the
Vietnamese byte-level merges (`ĠV`+`i` vs `á»`+`ĩ`). This is a known 0.016%
divergence and is tolerated in `HuggingFaceDifferentialTests` with a warning;
the 50k parity gate otherwise remains 100% for GPT-2 and for tiktoken (see #5).

**Impact**: `test_llama3_bpe_encode_parity` only permits mismatches strictly
matching the known Vietnamese/Czech merge tie-break set (up to 12 strings) with
a `UserWarning`; any other mismatch fails the test immediately.

## How to Run the Differential Test Suite

```bash
# Install required dependencies
pip install regex tiktoken tokenizers transformers sentencepiece

# Run the differential compatibility tests
python -m unittest tests/test_differential_compat.py -v

# Run the full test suite
python -m unittest discover -s tests -p "test_*.py"
```

## Automated CI

The differential test suite runs automatically on every push and pull request via
GitHub Actions (`.github/workflows/ci.yml`). Tests that require optional packages
(`tiktoken`, `tokenizers`, `transformers`) are skipped gracefully when unavailable.

## Updating This Document

When adding a new adapter or discovering a divergence:

1. Add a row to the Compatibility Matrix table above.
2. If the divergence is intentional, add a numbered section under "Known Exceptions".
3. Add a corresponding test case in `tests/test_differential_compat.py`.
4. Run the suite locally to confirm parity before opening a PR.
