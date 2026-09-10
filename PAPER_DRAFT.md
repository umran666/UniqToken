# UniqToken: Implementation Description and Evaluation Protocol

**Status**: Research protocol; no comparative result claims
**Artifact repository**: `https://github.com/umran666/UniqToken`

## Abstract

UniqToken is a research implementation of trainable Unigram and BPE tokenizers with optional cross-word CEM/SuperBPE vocabulary extension. The implementation also includes byte fallback, Unicode-aware pre-tokenization, subword regularization, and raw-text offset tracking. This document describes the code and the experiments required to evaluate it. It does not claim that UniqToken improves language-model quality, cost, linguistic segmentation, throughput, or a Pareto frontier.

Earlier draft Tables 1-3, their ANOVA statistics, and derived Pareto claims were based on experimental artifacts that do not satisfy the current held-out-data and exact-vocabulary contracts. Those artifacts remain under `benchmarks/legacy/` for provenance. They must not be cited as results for the current implementation.

## 1. Implementation

### 1.1 Training

`CustomTokenizer.train_from_corpus` normalizes and pre-tokenizes training documents, constructs a seed vocabulary, and trains a Unigram model. Candidate ranking can use frequency, character savings, byte savings, or PMI. Optional settings can rebalance candidate selection across detected script families and filter candidates by an empirical boundary-entropy threshold.

The repository also contains a BPE trainer. CEM extends an existing vocabulary by appending selected merged tokens. SuperBPE is the CEM configuration with cross-word merging enabled. These are algorithmic mechanisms, not evidence that the learned tokens correspond to clitics, morphemes, roots, or any other linguistic gold standard.

### 1.2 Tokenization and decoding

The tokenizer applies sanitization, optional indentation compression, normalization, regex pre-tokenization, and model segmentation. Unknown text can be represented by UTF-8 byte tokens when byte fallback is enabled. Decoding reverses token serialization and optional indentation compression. With normalization enabled, the text contract is `decode(encode(x)) == normalize(x)`, subject to separately configured sanitization; NFKC is not byte-for-byte lossless. Here normalization denotes visible normalized text, not internal metaspace serialization. Offset APIs compose mappings through the preprocessing stages to return spans in the original input.

### 1.3 Compatibility and native execution

The compatibility namespace imports existing tiktoken, Hugging Face, and SentencePiece vocabularies while preserving their token IDs. The research namespace trains new vocabularies and therefore changes the token-ID space. A Rust extension accelerates selected trie, lattice, pre-tokenization, and batch operations; Python implementations remain available for supported paths. Performance depends on which path is active and must be recorded in any throughput experiment.

### 1.4 Scope exclusions

The supported multimodal surface includes text and the repository's visual patch/codebook path. The random-initialized audio RVQ and neural codec utilities are experimental internals, have no bundled trained checkpoint, and are not part of the supported public tokenizer API.

## 2. Current benchmark contract

The final research harness is `benchmarks/run_research_experiments.py`, with the executable methodology in `benchmarks/RESEARCH_PROTOCOL.md`. The older `benchmarks/run_matched_budget_eval.py` is a train/validation diagnostic. Small component checks also exist in `benchmarks/train_toy_transformer.py`, `benchmarks/downstream_eval.py`, and `benchmarks/benchmark_suite.py`.

Every current comparative run must satisfy all of the following:

1. Tokenizer and language-model training documents are disjoint from every document used for measurement.
2. The final runner requires frozen train/validation/test manifests before Phase A. Phase B uses validation LM NLL only for screening; Phase C requires an explicit screening-ledger-bound selection before computing test LM NLL. All data assignment and normalized-document fingerprints are shared across tokenizers. Exact-document duplication is rejected; near-duplicate removal remains an external corpus-preparation requirement.
3. Every trainable tokenizer reaches the exact requested vocabulary size. A shortfall is a failed condition, not a smaller-budget substitute.
4. A SuperBPE condition learns at least one cross-word merge. Zero-merge configurations are invalid.
5. Transformer rows are produced only by the declared Transformer implementation. Missing PyTorch, insufficient training tokens, or an unavailable requested device aborts the condition; no Laplace or unigram model is substituted.
6. Every persisted row records `model_kind`. Active JSON ledgers use shared schema version 3. The final runner additionally requires research schema 2, a clean Git commit, source and installed extension hashes, dataset manifest/assignment hashes, tokenizer artifact hashes, seeds, complete model configuration, and the matching regime/budget. Its loader validates these against the current runtime and rejects diagnostic or stale ledgers. An installed extension hash identifies bytes, not proof of a build from the current Rust source.
7. A matched comparison is complete only if every pre-registered tokenizer, vocabulary budget, model tier, and seed succeeds. Partial grids are diagnostic outputs, not matched comparative evidence.

## 3. Metrics

Tokenizer-only measurements may report bytes per token, tokens per byte, tokens per Unicode character, byte-fallback rate, latency, and input-byte throughput. `tokens_per_unicode_character` divides token count by raw Unicode code-point count, including whitespace. This applies consistently to CJK and mixed-script text; it does not measure morpheme or word-boundary accuracy. Ambiguous whitespace-based fertility fields are rejected by the current ledger loader.

Downstream causal language models may report token cross-entropy and byte-normalized negative log-likelihood:

$$
\operatorname{BPB} = \frac{\sum_{i=1}^{N} -\log p(x_i \mid x_{<i})}{B\log 2},
$$

where the numerator and byte count $B$ refer to the same held-out normalized UTF-8 documents. The final runner scores every text token plus EOS, starting from BOS, including short final windows; EOS adds NLL but no text bytes. Validation and test totals are separate. Perplexity is the exponential of mean token NLL and depends on each tokenizer's prediction alphabet. This is decoder-only causal next-token evaluation with fixed finite-context windows, not masked-model evaluation. A uniform vocabulary code length is not an LM bits-per-byte metric and is not reported as one.

Input bytes per second is the primary cross-tokenizer throughput unit. Token throughput depends on segmentation and is therefore not directly comparable across tokenizers that emit different token counts. Tokens per second may be used to compare implementation paths only when token streams are identical.

## 4. Required experimental design

### 4.1 Data

Use frozen, versioned corpora with documented source, license, language/domain composition, deduplication, and preprocessing. Split before all tokenizer training. Apply exact-document and near-duplicate checks across train, validation, and test. Synthetic corpora are acceptable for harness tests but not as the sole basis of general performance claims.

### 4.2 Tokenizer comparison

Phase A compares independently trained SentencePiece Unigram, SentencePiece BPE, Boundary-BPE, UniqToken Unigram, and UniqToken SuperBPE at exactly 16,384, 32,768, and 65,536 total entries. Boundary-BPE is primary because historical 64K comparisons motivate retaining it as a strong candidate, not because those invalidated artifacts establish current superiority. All budgets include the same four control tokens and 256 byte tokens. Freeze normalization, character coverage, maximum token length, and training-data allocation. Report failures rather than padding vocabularies or changing a baseline's requested budget.

### 4.3 Language-model comparison

Phase B screens all conditions with one paired seed using a 2-layer, width-128, FFN-512 causal LM. Phase C runs selected conditions with three new paired LM seeds using 12 layers, width 768, FFN 3072, 12 heads, and context 1024. The existing architecture has learned positions and untied input/output matrices. Its non-embedding count is 85,842,432; total counts at 16K/32K/64K are 111,008,256 / 136,174,080 / 186,505,728, verified against instantiated parameters. It is not labeled "125M". Tokenizers are frozen from Phase A, so paired seeds measure LM variation, not tokenizer-training variation.

Run both FLOP-matched and byte-matched regimes; neither is universally superior. The former uses a documented dense-matmul forward/backward estimator with at most 1% undershoot and no overshoot, not measured hardware FLOPs. The latter uses identical ordered normalized-document prefixes and rejects budgets ending inside documents. The ledger records requested/actual budgets, parameters, targets, optimizer updates, device, precision, context, and software provenance. Before efficiency claims, add separately controlled hardware profiling and end-to-end timing experiments; these are not inferred from analytical budgets.

### 4.4 Statistics

Use multiple paired seeds and publish every condition, including failed runs. Pre-register primary outcomes and statistical tests before the confirmatory run. Report uncertainty intervals and effect sizes. ANOVA or Pareto analysis is appropriate only after verifying independence, a complete factorial grid, correct repeated-measures structure, and robustness to alternative compute and memory constraints.

### 4.5 Linguistic evaluation

Claims about clitics, morphemes, or root preservation require annotated linguistic datasets and boundary-level precision, recall, and F1, evaluated by language. Compression and whitespace-based fertility cannot establish those claims.

## 5. Results status

No result table is current. The repository's historical Phase 14/15 records, figures, and pre-integrity matched-budget run are archived and invalid for current claims because of data leakage, incomplete exact-budget enforcement, or insufficient result provenance. Small smoke runs may demonstrate that the harness executes and enforces its contracts; they are not evidence of model superiority.

## 6. Open research questions

- Does SuperBPE improve held-out byte-normalized LM loss after exact vocabulary, data, and compute matching?
- Are any effects stable across languages, domains, vocabulary sizes, model capacities, and seeds?
- How sensitive are results to normalization, script balancing, entropy thresholds, and maximum token length?
- What is the tradeoff among sequence length, embedding/output parameters, attention compute, and measured end-to-end latency?
- Do learned boundaries align with annotated linguistic units, or do they only improve compression?
- Does the Rust path preserve exact output parity while improving input-byte throughput on controlled hardware?

## 7. Claim policy

Performance, cost, linguistic, capacity-coupling, and Pareto claims may be added only from a fresh, versioned ledger produced by the current harness, accompanied by the complete configuration and uncertainty analysis. Historical values must remain labeled as archival and must not be copied into the abstract, README, tables, figures, or conclusion as current evidence.
