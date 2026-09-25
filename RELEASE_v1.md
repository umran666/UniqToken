# UniqToken v1.0.0

## Release status

This document describes the proposed `v1.0.0` release baseline. Preparation of
the release does not publish a package, create a Git tag, or authorize any
additional research experiment.

## Included in v1

- Trainable Unigram and BPE tokenizers, including deterministic Viterbi
  segmentation and optional FFBS subword sampling.
- CEM/SuperBPE post-training vocabulary extension and cross-word merge support.
- Complete UTF-8 byte fallback when all 256 byte tokens are configured.
- NFKC-aware normalization, Unicode pre-tokenization, security filtering, and
  raw-text offset composition.
- Save/load support, streaming decoding, batching, CLI commands, and optional
  Rust acceleration through the separately built `uniqtoken-core` extension.
- Compatibility and export surfaces for the tested tiktoken, SentencePiece,
  Hugging Face, GGUF, and llama.cpp paths documented in the repository.
- Versioned research harnesses and the preserved Phase A and Phase B evidence.

## Evidence status

Phase A tokenizer screening is complete. Phase B completed all 18 exploratory
LM screening conditions, but it remains screening evidence rather than
confirmatory evidence. In particular, the 16K byte-matched UT-SuperBPE result
is not a confirmed general advantage.

The Phase C protocol was frozen before execution. Phase C was **not executed
because the required compute exceeded the available free budget**. The held-out
FLORES-200 devtest set remained unopened, and no Phase C scientific result or
failure is claimed.

## Known limitations

- With NFKC enabled, the round-trip contract is
  `decode(encode(x)) == normalize(x)`, not preservation of the original bytes.
- The Rust extension is optional. Unsupported native configurations use the
  Python implementation; the release makes no universal throughput claim.
- Imported tokenizer fidelity depends on representable normalization and
  pre-tokenization behavior. Unsupported details emit warnings as documented.
- The Hugging Face adapter is verified for the repository-tested versions and
  surfaces, not every downstream model or future Transformers release.
- Image tokenization is experimental and requires a trained or loaded visual
  codebook for meaningful token IDs. No trained visual, audio, or neural codec
  checkpoint is bundled; audio tokenization is unsupported.
- FastMergeCore/FastMergeEngine is not part of v1. Its design, verification,
  benchmarking, and possible adoption belong to the future v2 issue track.

## Baseline policy

Version 1.0.0 is the frozen implementation and evidence baseline for future v2
research. Future objective or merge-engine work must preserve v1 artifacts and
must not reinterpret Phase B as confirmation or modify the frozen Phase C
protocol.
