//! WebAssembly bindings for the interactive tokenizer playground (issue #29).
//!
//! Exposes a self-contained demo tokenizer to JavaScript: the vocabulary is
//! embedded at compile time ([`DEMO_VOCAB_JSON`], regenerable by training any
//! UniqToken `CustomTokenizer` and dumping `[token, log_prob, id]` triples),
//! and byte fallback guarantees every input tokenizes.

use crate::error::{CoreError, CoreResult};
use crate::normalizer::normalize_inner;
use crate::pipeline::get_full_pretok_regex;
use crate::trie::RustPrefixTrie;
use crate::viterbi::decode_cached;
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;

/// Demo vocabulary: `[token, log_probability, token_id]` triples with IDs
/// contiguous from 0. Regenerate with a trained tokenizer, e.g.:
/// `CustomTokenizer.train_from_corpus(corpus, target_vocab_size=1024)`
/// then dump `[[tok, vocab[tok], token_to_id[tok]] for tok in token_to_id]`
/// sorted by ID into this file.
const DEMO_VOCAB_JSON: &str = include_str!("../demo_vocab.json");

const SPACE_CHAR: char = '\u{2581}';
const UNK_TOKEN: &str = "<|unk|>";
const DEFAULT_BYTE_LOG_P: f64 = -10.0;

/// `true` for `<0xHH>` byte-fallback tokens (mirrors
/// `ByteFallbackEngine.is_byte_token` on the Python side).
fn is_byte_token(token: &str) -> bool {
    let bytes = token.as_bytes();
    token.len() == 6
        && token.starts_with("<0x")
        && token.ends_with('>')
        && bytes[3].is_ascii_hexdigit()
        && bytes[4].is_ascii_hexdigit()
}

fn log_prob_of(trie: &RustPrefixTrie, token: &str) -> f64 {
    if is_byte_token(token) {
        return DEFAULT_BYTE_LOG_P;
    }
    trie.exact_metadata(token).map(|(_, log_p)| log_p).unwrap_or(DEFAULT_BYTE_LOG_P)
}

/// One encoded token with its ID, byte-fallback flag, and log probability.
#[derive(Clone, Debug)]
pub struct Piece {
    pub token: String,
    pub id: u32,
    pub byte_fallback: bool,
    pub log_p: f64,
}

/// Self-contained demo engine: embedded vocabulary plus byte fallback.
pub struct Engine {
    trie: RustPrefixTrie,
    unk_id: u32,
    vocab_size: usize,
}

impl Engine {
    /// Builds the engine from the embedded demo vocabulary.
    pub fn load() -> CoreResult<Self> {
        let entries: Vec<(String, f64, u32)> = serde_json::from_str(DEMO_VOCAB_JSON)
            .map_err(|err| CoreError(format!("demo vocabulary is corrupt: {err}")))?;
        if entries.is_empty() {
            return Err(CoreError("demo vocabulary is empty".to_string()));
        }
        let mut trie = RustPrefixTrie::default();
        let mut unk_id = 0u32;
        for (token, log_p, id) in &entries {
            trie.insert(token, *log_p, Some(*id))?;
            if token == UNK_TOKEN {
                unk_id = *id;
            }
        }
        let vocab_size = entries.len();
        Ok(Self { trie, unk_id, vocab_size })
    }

    /// Number of entries in the embedded demo vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Encodes text into token pieces (never fails on content: byte fallback
    /// covers every Unicode scalar value).
    pub fn pieces(&self, text: &str) -> CoreResult<Vec<Piece>> {
        let normalized = normalize_inner(text, SPACE_CHAR, true, true, false, false, false, false)?;
        let re = get_full_pretok_regex();
        let mut pieces = Vec::new();
        for m in re.find_iter(&normalized) {
            // Cached per chunk: repeat visits (e.g. live re-renders) skip the
            // trie walk and Viterbi DP entirely.
            let seg = decode_cached(m.as_str(), &self.trie, true).map_err(CoreError)?;
            for (token, token_id, _start, _end) in seg.iter() {
                let byte_fallback = is_byte_token(token);
                pieces.push(Piece {
                    id: token_id.unwrap_or(self.unk_id),
                    log_p: log_prob_of(&self.trie, token),
                    token: token.clone(),
                    byte_fallback,
                });
            }
        }
        Ok(pieces)
    }

    /// Mean token log probability (0.0 for empty input).
    pub fn avg_logprob(&self, pieces: &[Piece]) -> f64 {
        if pieces.is_empty() {
            return 0.0;
        }
        pieces.iter().map(|piece| piece.log_p).sum::<f64>() / pieces.len() as f64
    }
}

/// JavaScript-facing demo tokenizer for the playground page.
#[wasm_bindgen]
pub struct PlaygroundTokenizer {
    engine: Engine,
}

#[wasm_bindgen]
impl PlaygroundTokenizer {
    /// Builds the demo tokenizer from the embedded vocabulary.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<PlaygroundTokenizer, JsValue> {
        Engine::load()
            .map(|engine| Self { engine })
            .map_err(|err| JsValue::from_str(&err.to_string()))
    }

    /// Encodes the input once, returning `{tokens, ids, fallbackCount,
    /// bytesPerToken, avgLogprob}` so callers never re-run the
    /// normalize-to-Viterbi pipeline once per metric.
    pub fn encode(&self, text: &str) -> Result<JsValue, JsValue> {
        let pieces = self.encode_pieces(text)?;
        let tokens: Array = pieces
            .iter()
            .map(|piece| JsValue::from_str(&piece.token))
            .collect();
        let ids: Array = pieces.iter().map(|piece| JsValue::from(piece.id)).collect();
        let fallback_count = pieces.iter().filter(|piece| piece.byte_fallback).count() as u32;
        let bytes_per_token = if pieces.is_empty() {
            0.0
        } else {
            text.len() as f64 / pieces.len() as f64
        };
        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("tokens"), &tokens)?;
        Reflect::set(&result, &JsValue::from_str("ids"), &ids)?;
        Reflect::set(
            &result,
            &JsValue::from_str("fallbackCount"),
            &JsValue::from(fallback_count),
        )?;
        Reflect::set(
            &result,
            &JsValue::from_str("bytesPerToken"),
            &JsValue::from(bytes_per_token),
        )?;
        Reflect::set(
            &result,
            &JsValue::from_str("avgLogprob"),
            &JsValue::from(self.engine.avg_logprob(&pieces)),
        )?;
        Ok(result.into())
    }

    /// Number of entries in the embedded demo vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.engine.vocab_size()
    }

    fn encode_pieces(&self, text: &str) -> Result<Vec<Piece>, JsValue> {
        self.engine
            .pieces(text)
            .map_err(|err| JsValue::from_str(&err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> Engine {
        Engine::load().expect("demo vocabulary must parse and load")
    }

    #[test]
    fn demo_vocab_loads_with_full_byte_coverage() {
        let engine = engine();
        assert!(
            engine.vocab_size() >= 256 + 8,
            "demo vocab holds byte tokens plus words"
        );
        for byte in 0..=255u8 {
            let token = format!("<0x{byte:02X}>");
            assert!(engine.trie.exact_metadata(&token).is_some(), "missing byte token {token}");
        }
        assert!(engine.trie.exact_metadata(UNK_TOKEN).is_some());
    }

    #[test]
    fn english_text_roundtrips_through_pills() {
        let engine = engine();
        let pieces = engine.pieces("hello world").expect("encode must succeed");
        assert!(!pieces.is_empty());
        let decoded: String = pieces
            .iter()
            .map(|piece| piece.token.as_str())
            .collect::<Vec<_>>()
            .join("")
            .replace(SPACE_CHAR, " ");
        assert_eq!(decoded, "hello world");
        assert!(pieces.iter().all(|piece| piece.id < engine.vocab_size() as u32));
    }

    #[test]
    fn emoji_falls_back_to_bytes() {
        let engine = engine();
        let pieces = engine.pieces("hi 🚀").expect("encode must succeed");
        assert!(pieces.iter().any(|piece| piece.byte_fallback));
        assert!(pieces.iter().all(|piece| piece.id < engine.vocab_size() as u32));
        assert!(engine.avg_logprob(&pieces) < 0.0);
    }

    #[test]
    fn empty_input_yields_no_pieces_and_zero_metrics() {
        let engine = engine();
        let pieces = engine.pieces("").expect("encode must succeed");
        assert!(pieces.is_empty());
        assert_eq!(engine.avg_logprob(&pieces), 0.0);
    }

    #[test]
    fn byte_token_classifier_matches_python() {
        assert!(is_byte_token("<0x00>"));
        assert!(is_byte_token("<0xFF>"));
        assert!(!is_byte_token("<0xG0>"));
        assert!(!is_byte_token("<0x0>"));
        assert!(!is_byte_token("hello"));
        assert!(!is_byte_token("<|unk|>"));
    }
}
