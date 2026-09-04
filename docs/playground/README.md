# UniqToken Playground (issue #29)

Interactive in-browser tokenizer demo, powered entirely by `uniqtoken_core`
compiled to WebAssembly. Paste text, see colored token pills plus live metrics
(tokens, bytes/token, byte-fallback count, mean log-probability). The URL hash
carries the input as a shareable link. Nothing leaves the browser.

## Local development

Prerequisites: stable Rust with the `wasm32-unknown-unknown` target and
`wasm-pack`:

```bash
rustup target add wasm32-unknown-unknown
wasm-pack build crates/uniqtoken_core --target web \
  --out-dir ../../docs/playground/pkg --no-pack \
  -- --no-default-features --features wasm
```

Then serve this directory over HTTP (ES modules reject `file://`):

```bash
python -m http.server --directory docs/playground 8000
```

Useful checks (mirrored in `.github/workflows/wasm-playground.yml`):

```bash
cargo clippy --manifest-path crates/uniqtoken_core/Cargo.toml \
  --target wasm32-unknown-unknown --no-default-features --features wasm -- -D warnings
cargo test --manifest-path crates/uniqtoken_core/Cargo.toml --features wasm
```

## Demo vocabulary

`crates/uniqtoken_core/demo_vocab.json` embeds the playground vocabulary
(`[token, log_probability, token_id]` triples, IDs contiguous from 0,
including the unknown token and all 256 `<0xHH>` byte tokens so every input
tokenizes). Regenerate it from any trained tokenizer:

```python
from uniqtoken import CustomTokenizer
import json

tok = CustomTokenizer.train_from_corpus(corpus, target_vocab_size=1024, verbose=False)
triples = sorted(
    ((t, float(tok.model.vocab[t]), tok.model.token_to_id[t]) for t in tok.model.token_to_id),
    key=lambda item: item[2],
)
assert [i for _, _, i in triples] == list(range(len(triples)))
json.dump([[t, s, i] for t, s, i in triples], open("crates/uniqtoken_core/demo_vocab.json", "w"))
```

## Deployment

Pushes to `main` touching the crate, this directory, or the workflow build the
bundle, enforce the 1.5 MiB `.wasm` budget, and deploy via GitHub Pages
(`upload-pages-artifact` + `deploy-pages`). The compiled `pkg/` output is
never committed.
