# UniqToken Core (Rust Acceleration Engine)

Native Rust acceleration crate for the UniqToken tokenizer.

## Components
- `trie.rs`: Native PrefixTrie with fast AHashMap character branch indexing and common prefix search.
- `viterbi.rs`: Dynamic programming 1-best shortest-path Viterbi search and forward-backward EM posterior expectation aggregator in log-space.
- `lib.rs`: PyO3 C-extension binding exposing `uniqtoken_core` to Python.

## Building Native Extension
To compile the native extension into the local environment:

```bash
# Run from the repository root so the wheel includes both Python and Rust.
pip install "maturin>=1.15,<2.0"
maturin develop --release

# Or build wheel
maturin build --release
```

The root mixed-project build installs both the public `uniqtoken` package and
the `uniqtoken_core` extension. The Python package dispatches supported
normalization, pre-tokenization, Viterbi, and batch operations to it. Unsupported
configurations use the Python implementation, while native computation errors
propagate instead of silently changing implementations. Python/Rust parity is
covered by the repository's native and differential test suites.
