# UniqToken

UniqToken is a Python tokenizer research toolkit with a bundled Rust extension. It supports training Unigram and BPE vocabularies, importing existing tokenizer formats, and tracing token spans back to input text. It is intended for experiments with vocabulary construction, multilingual text, and model integration; a newly trained vocabulary requires a language model trained or adapted for its token IDs.

![UniqToken banner](assets/banner.jpeg)

The current published release is **v1.0.0**: [GitHub release](https://github.com/umran666/UniqToken/releases/tag/v1.0.0), [PyPI package](https://pypi.org/project/uniqtoken-core/1.0.0/), and [release contents and limitations](RELEASE_v1.md). The `main` branch includes subsequent development and research diagnostics; installing v1.0.0 does not include those later changes.

## Features

Implemented in v1.0.0:

- Unigram training with expectation-maximization, deterministic Viterbi segmentation, and optional forward-filtering backward-sampling (FFBS) subword regularization.
- BPE training and ranked merge inference, plus CEM/SuperBPE post-training vocabulary extension with optional cross-word merging.
- Configurable seed ranking, script balancing, and boundary-entropy filters for vocabulary experiments.
- Configurable normalization and Unicode-aware pre-tokenization, complete UTF-8 byte fallback when all 256 byte tokens are present, and raw-text character spans.
- Token strings, integer IDs, batch encoding/decoding, incremental UTF-8 decoding, padding and attention masks, and tokenizer save/load.
- Compatibility importers for tiktoken ranks, Hugging Face Unigram/ByteLevel BPE JSON, and SentencePiece Unigram models, subject to the limits below.
- A CLI, Rust acceleration, Hugging Face integration/export, and GGUF export. Integration fidelity depends on the configuration and the tested downstream library versions.

## Installation

Requires **Python 3.10 or later**. The configured Python CI matrix covers 3.10, 3.11, and 3.12.

```bash
python -m pip install uniqtoken-core==1.0.0
```

The single distribution installs both `uniqtoken` (the public Python API) and `uniqtoken_core` (the native extension). Do not install a separate package named `uniqtoken`.

```python
import uniqtoken
import uniqtoken_core

print(uniqtoken.__version__)  # 1.0.0
```

The base dependency is `regex`. Published wheels bundle the native extension; a source installation requires a stable Rust toolchain and a platform linker. The build backend is Maturin.

Optional extras include `huggingface` for the Hugging Face adapter, `chat` for Jinja2 chat templates, `progress` for progress displays, `torch` for tensor output, and `tiktoken` for reference comparisons. For example:

```bash
python -m pip install "uniqtoken-core[huggingface]==1.0.0"
```

## Quick Start

This small corpus demonstrates the API; it does not produce a production-quality vocabulary.

```python
from uniqtoken import CustomTokenizer

corpus = [
    "hello world",
    "hello tokenizer",
    "a tokenizer encodes text",
    "text can include unseen characters",
]
tokenizer = CustomTokenizer.train_from_corpus(
    corpus,
    target_vocab_size=320,
    byte_fallback=True,
    verbose=False,
)

text = "hello world"
tokens = tokenizer.encode(text)       # List[str]: token strings
ids = tokenizer.encode_to_ids(text)   # List[int]: model-specific IDs
assert tokenizer.decode(ids) == text

print(tokens)
print(ids)
print([(token.text, token.raw_span)
       for token in tokenizer.encode_with_offsets(text)])

tokenizer.save("saved_model")
restored = CustomTokenizer.load("saved_model")
assert restored.encode_to_ids(text) == ids
```

The example's ASCII input survives the default normalization unchanged. For other inputs, decoded text reflects the configured normalization and special-token policy; raw byte preservation is not the default contract.

### CLI

Using the `saved_model` directory created above:

```bash
uniqtoken --help
uniqtoken train --help
uniqtoken encode --model saved_model --input "hello world" --to-ids --json --out token-ids.json
uniqtoken decode --model saved_model --input token-ids.json
```

The decode command prints `hello world`. `python -m uniqtoken.cli` is an alternative entry point. CLI training accepts UTF-8 corpus files through `--corpus`, trains Unigram by default, and can add SuperBPE merges with `--superbpe-merges`; BPE training is available through the Python API.

## Supported Tokenization

| Surface | Algorithm or format | Behavior |
| --- | --- | --- |
| `CustomTokenizer.train_from_corpus`, `UnigramTrainer` | Unigram | Train a new vocabulary; deterministic segmentation or optional sampling. |
| `BPETrainer`, `BPEModel` | BPE | Train on pre-tokenized chunks and apply ranked adjacent-symbol merges. |
| `CrossEntropyMerging`, `SuperBPE` | CEM/SuperBPE | Extend an existing Unigram vocabulary; SuperBPE enables cross-word merges. |
| `uniqtoken.compat.from_tiktoken` | tiktoken ranks | Preserve imported IDs; supply the matching regex pattern and special-token configuration. |
| `uniqtoken.compat.from_huggingface` | Unigram or BPE JSON | ByteLevel BPE has a dedicated adapter; other BPE configurations have limited fidelity. WordPiece is unsupported. |
| `uniqtoken.compat.from_sentencepiece` | SentencePiece Unigram `.model` | Import pieces, scores, and IDs; fidelity depends on representable preprocessing. |

Training classes are also available through `uniqtoken.train`. Compatibility loaders return wrappers that reject vocabulary mutation; use the training surface to create or extend vocabularies. Preserving imported IDs alone does not guarantee identical segmentation for every model.

For BPE, using `corpus` and `text` from the quick start:

```python
from uniqtoken import BPETrainer, Normalizer, RegexPreTokenizer

normalizer = Normalizer()
pre_tokenizer = RegexPreTokenizer()
chunks = [
    chunk
    for document in corpus
    for chunk in pre_tokenizer.pre_tokenize(normalizer.normalize(document))
]
bpe = BPETrainer(target_vocab_size=320, byte_fallback=True).train(chunks)
bpe_ids = bpe.encode_to_ids(normalizer.normalize(text))
assert bpe.decode(bpe_ids) == text
```

See the [compatibility exceptions](COMPATIBILITY_EXCEPTIONS.md) and [differential tests](tests/test_differential_compat.py) for tested formats and known divergences. SentencePiece imports respect the model's dummy-prefix flag; unsupported normalization details can still affect parity. Non-ByteLevel Hugging Face BPE imports retain vocabulary/merge data with warnings rather than promising complete tokenizer equivalence.

### Native Execution

The bundled PyO3 extension provides prefix-trie lookup, Viterbi segmentation, forward/backward expectations, n-gram mining, normalization, pre-tokenization, and Rayon batch execution. Supported paths dispatch automatically to Rust; unavailable operations or unsupported configurations use Python implementations.

There is no general `backend=` or `algorithm=` selector on `CustomTokenizer.train_from_corpus`. Choose Unigram or BPE through their respective training APIs. FFBS sampling and nonzero merge dropout use Python paths. Experimental `FastMergeEngine` work on `main` is outside the v1.0.0 release; see the [merge-engine design](docs/MERGE_ENGINE_DESIGN.md).

## Unicode, Byte Fallback, and Alignment

- **Normalization:** NFKC and Unicode-space mapping are enabled by default. Case folding, punctuation mapping, whitespace collapse, and stripping are configurable. Normalization can change characters and length; configure these options explicitly when raw-text reconstruction matters.
- **Byte fallback:** With `byte_fallback=True` and all 256 `<0x00>` through `<0xFF>` tokens configured, unseen valid UTF-8 text can be represented without replacing it with an unknown token. This does not guarantee useful linguistic segmentation. Invalid UTF-8 byte sequences are rejected during decoding.
- **Graphemes:** Pre-tokenization snaps boundaries to extended grapheme clusters, with tests for combining marks, Indic scripts, emoji sequences, and related cases. This is a pre-tokenization boundary rule, not a guarantee that each emitted subword is a whole grapheme or morpheme.
- **Offsets:** `encode_with_offsets` returns `Token` objects with `text`, `id`, and `raw_span=(start, end)`. These are **Python character-index offsets**, with an exclusive end, into the original input. Normalization and sanitization compose their source spans; expansions or byte fallback can give multiple tokens the same or overlapping raw span.
- **Special tokens:** Configured control-token strings can be allowed, escaped, or rejected. Escaping changes the decoded text. This policy is not a general defense against prompt injection.

## Performance

Throughput depends on the model, input, build, batching, cache state, and thread count. Cross-tokenizer comparisons should use normalized input bytes per second; token throughput is comparable only when the implementations emit the same token stream.

The retained [post-release native performance study](docs/NATIVE_PERFORMANCE_RESULTS.md) compares UniqToken implementations, not competing tokenizer libraries. Across 45 natural-script/vocabulary segmentation cells, the combined decoder (`9e84dc2`) measured **2.45-9.13x median paired speedups** over its internal baseline (`658033b`). Full-batch results had 22 repeatable gains and 26 inconclusive/mixed cells, with no repeatable degradation.

That study used fixed local fixtures on Windows x86-64 (Intel Family 6, Model 154; 16 logical CPUs), Python 3.10.11, Rust 1.98.0, release optimization level 3, LTO, one codegen unit, and one Rayon worker. It used two independent worker rounds with eleven paired repetitions per round; allocation instrumentation was disabled for primary timings. The [report](benchmarks/native_performance/issue96-99/summary/REPORT.md) and [metrics](benchmarks/native_performance/issue96-99/summary/metrics.csv) retain methods, per-cell uncertainty, and exact build identities.

These are segmentation measurements at the named post-release revisions, not v1.0.0 end-to-end speedups or evidence of superiority over SentencePiece/tiktoken. Separate [vocabulary-scaling diagnostics](benchmarks/scaling/issue92/REPORT.md) measured faster SentencePiece training at every completed shared budget in that bounded corpus/configuration; they do not establish encoding-throughput rankings. Benchmark the intended workload before choosing an implementation.

## Research

Engineering functionality and scientific evidence have different scopes:

| Stage | Status | Interpretation |
| --- | --- | --- |
| Phase A | Tokenizer screening completed | Controlled tokenizer diagnostics for feasibility and selection; token counts do not establish downstream model quality. |
| Phase B | All 18 exploratory LM screening conditions completed | One paired seed under FLOP-matched and byte-matched budgets; screening evidence, not confirmation. |
| Phase C | **Not executed** | The frozen confirmatory design exceeded the available free compute budget; no confirmatory result is claimed. |

In Phase B, UT-SuperBPE's 16K byte-matched result was a candidate for confirmation. SentencePiece Unigram had the lowest FLOP-matched bits per byte (BPB) at all three screened vocabulary sizes. The small model, one seed, and limited training exposure prevent general conclusions about multilingual LM efficiency or normally trained models. BPB comes from validation negative log-likelihood and normalized UTF-8 bytes, not vocabulary size alone.

The held-out FLORES-200 devtest set **remained unopened**. A frozen protocol and implemented runner do not authorize a Phase C launch.

- [Research protocol and provenance requirements](benchmarks/RESEARCH_PROTOCOL.md), including research schema 5 for the generic LM runner.
- [Phase B analysis and limitations](benchmarks/PHASE_B_ANALYSIS_REPORT.md).
- [Frozen Phase C protocol](benchmarks/PHASE_C_CONFIRMATORY_PROTOCOL.md) and [unexecuted status](benchmarks/PHASE_C_STATUS.md).
- Post-release [tokenizer failure analysis](benchmarks/TOKENIZER_FAILURE_ANALYSIS.md), [merge-objective ablations](benchmarks/MERGE_OBJECTIVE_ABLATION.md), and [scope-specific Pareto analysis](benchmarks/TOKENIZER_PARETO.md). These are descriptive diagnostics, not confirmatory LM evidence.
- [Manuscript draft](PAPER_DRAFT.md); versioned protocols and reports under `benchmarks/` remain the source of truth.
- [Archived legacy evidence](benchmarks/legacy/README.md), retained for provenance and excluded from current research claims.

The project makes no universal token-efficiency, language-quality, or API-cost reduction claim.

## Limitations

- A new or extended vocabulary is not interchangeable with an existing language model's tokenizer. `VocabularyAdapter` preserves existing IDs and assigns new IDs **above the maximum existing ID**; model embeddings and output layers still need resizing and adaptation.
- Achievable vocabulary size depends on the corpus and mandatory pieces. A small corpus may underfill a large requested budget; controlled research harnesses reject budget mismatches.
- Imported/exported tokenizer fidelity and Hugging Face integration are limited to representable configurations and tested library versions. Optional differential tests may skip when reference packages or models are unavailable.
- Image-tokenization components are experimental and require a trained or loaded visual codebook. No trained visual, audio, or neural codec checkpoint is bundled; audio tokenization is unsupported.
- The 2026-10-04 local Windows/Python 3.10 hygiene validation did **not** yield a fully green Python suite: a pre-existing tiktoken-adapter timing test and a Torchvision/Transformers environment failure remained. Passing package/import checks do not imply the full suite passed.

## Development

A source checkout requires Python 3.10+, a stable Rust toolchain, and a working platform linker. The following installs the checked-out branch, which can differ from the published release.

```bash
git clone https://github.com/umran666/UniqToken.git
cd UniqToken
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on POSIX shells or `.\.venv\Scripts\Activate.ps1` in PowerShell, then run:

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests -p "test_*.py" -v
python -m ruff check .
python -m ruff format --check .
python -m mypy uniqtoken
cargo test --manifest-path crates/uniqtoken_core/Cargo.toml --all-targets --locked
cargo clippy --manifest-path crates/uniqtoken_core/Cargo.toml --all-targets --locked -- -D warnings
```

The editable install builds the native extension through Maturin. For a distributable wheel:

```bash
python -m pip install "maturin>=1.15,<2.0"
maturin build --release --locked --out dist
```

The [CI workflow](.github/workflows/ci.yml) configures a **9-cell matrix** across Ubuntu, macOS, Windows, and Python 3.10-3.12, plus Rust checks, native-wheel verification, source-distribution builds, and tag-triggered PyPI publishing. Push triggers are restricted to the branch/tag patterns in the workflow; pull requests target `main` or `master`. This describes the configuration, not a claim that every current check is green.

Repository layout:

| Path | Purpose |
| --- | --- |
| `uniqtoken/` | Public Python API, training, compatibility, and integration modules. |
| `uniqtoken_core/` | Native extension package and type stubs. |
| `crates/uniqtoken_core/` | Rust core, C ABI, and optional WebAssembly bindings. |
| `tests/` | Core, compatibility, native-parity, packaging, and research-integrity tests. |
| `benchmarks/` | Benchmark code, frozen research contracts, reports, and provenance. |
| `docs/`, `examples/`, `notebooks/` | Design documents, integration examples, and tutorials. |

Generated builds, local caches, and `publish-dist/` are ignored. Preserved research/release artifacts are intentional repository contents.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and use the [issue tracker](https://github.com/umran666/UniqToken/issues) for current work; [ROADMAP.md](ROADMAP.md) provides architecture and historical context. Keep PRs focused, add tests appropriate to behavioral changes, and report actual validation results and skips.

Research changes must preserve frozen protocols, manifests, receipts, and held-out protections. Routine development does not authorize new research runs or test-set access.

## License

[MIT](LICENSE).
