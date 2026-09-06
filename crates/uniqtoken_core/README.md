# UniqToken Core (Rust Acceleration Engine)

High-performance native Rust crate for the UniqToken Tokenizer.

## Components
- `trie.rs`: Native PrefixTrie with fast AHashMap character branch indexing and common prefix search.
- `viterbi.rs`: Dynamic programming 1-best shortest-path Viterbi search and forward-backward EM posterior expectation aggregator in log-space.
- `lib.rs`: PyO3 C-extension binding exposing `uniqtoken_core` to Python.

## Building Native Extension
To compile the native extension into the local environment:

```bash
# Using maturin
pip install maturin
maturin develop --release

# Or build wheel
maturin build --release
```

The crate currently exposes an experimental Python extension API. The main Python package does not yet dispatch to it automatically; native packaging and Python/Rust parity tests must be completed before enabling that path in production.
