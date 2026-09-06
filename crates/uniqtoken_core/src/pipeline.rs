//! High-performance native end-to-end normalization, pre-tokenization, and batch encoding pipeline.

#[cfg(feature = "python")]
use crate::error::{core_error, CoreError, CoreResult};
#[cfg(feature = "python")]
use crate::normalizer::normalize_inner;
#[cfg(feature = "python")]
use crate::trie::RustPrefixTrie;
#[cfg(feature = "python")]
use crate::viterbi::decode_cached;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use rayon::prelude::*;
use regex::Regex;
#[cfg(feature = "python")]
use std::collections::HashSet;
use std::sync::OnceLock;
#[cfg(feature = "python")]
use unicode_normalization::UnicodeNormalization;
#[cfg(feature = "python")]
use unicode_segmentation::UnicodeSegmentation;

#[cfg(feature = "python")]
static PRETOK_REGEX: OnceLock<Regex> = OnceLock::new();
static PRETOK_FULL_REGEX: OnceLock<Regex> = OnceLock::new();

#[cfg(feature = "python")]
fn get_pretok_regex() -> &'static Regex {
    PRETOK_REGEX.get_or_init(|| {
        // High-speed unicode-aware word/punctuation/whitespace splitting regex
        // Issue #43: keep the fast pre-tokenizer consistent with the full one
        // (1-3 digit number chunks instead of unbounded runs).
        Regex::new(r"<\|[^\s|]+\|>|\p{L}+(?:['’]\p{L}+)*|\p{N}{1,3}|[^\s\p{L}\p{N}\u{2581}]+|\u{2581}+|\s+").unwrap()
    })
}

pub(crate) fn get_full_pretok_regex() -> &'static Regex {
    PRETOK_FULL_REGEX.get_or_init(|| {
        let escaped_space = regex::escape("\u{2581}");
        let special_token = r"<\|[^\s|]+\|>";
        let url = r"https?://[a-zA-Z0-9][-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)";
        let email = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+";
        let hashtag = format!(r"{}?#[\p{{L}}\p{{N}}_]+", escaped_space);
        let mention = format!(r"{}?@[\p{{L}}\p{{N}}_]+", escaped_space);
        let emoji = r"(?:[\U0001F300-\U0001FAFF]|[\u2600-\u26FF]|[\u2700-\u27BF])(?:[\uFE0E\uFE0F])?(?:[\U0001F3FB-\U0001F3FF])?(?:\u200D(?:[\U0001F300-\U0001FAFF]|[\u2600-\u26FF]|[\u2700-\u27BF])(?:[\uFE0E\uFE0F])?(?:[\U0001F3FB-\U0001F3FF])?)*";
        let cjk = format!(r"{}?[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]+", escaped_space);
        let word = format!(r"{}?[\p{{L}}\p{{Nl}}\p{{No}}]+(?:['’][\p{{L}}\p{{Nl}}\p{{No}}]+)*", escaped_space);
        let hex_number = format!(r"{}?0[xX][0-9a-fA-F]+|{}?0[bB][01]+", escaped_space, escaped_space);
        // Issue #43: chunk numbers into 1-3 digit blocks (LLaMA-3/GPT-4 style).
        // \d is Unicode \p{Nd} in the regex crate, matching Python's \d exactly;
        // ASCII-only [0-9] would drop non-ASCII decimal digits entirely.
        let number = format!(r"{}?\d{{1,3}}", escaped_space);
        let space_marker = format!(r"{}+", escaped_space);
        let whitespace = r"[\s\x1c-\x1f]+";
        let punctuation = format!(r"{}?[^\p{{L}}\p{{N}}_\s\x1c-\x1f{}]|{}?_", escaped_space, escaped_space, escaped_space);
        let patterns = vec![
            special_token.to_string(),
            url.to_string(),
            email.to_string(),
            hashtag,
            mention,
            emoji.to_string(),
            cjk,
            word,
            hex_number,
            number,
            space_marker,
            whitespace.to_string(),
            punctuation,
        ];
        let combined = patterns.into_iter().map(|p| format!("(?:{})", p)).collect::<Vec<_>>().join("|");
        Regex::new(&combined).unwrap()
    })
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn rust_pre_tokenize(text: &str) -> Vec<String> {
    let re = get_full_pretok_regex();
    snapped_pretokens(text, re)
}

/// Issue #41: UAX #29 extended grapheme cluster snapping.
///
/// The pre-tokenizer regex matches on raw codepoints, so a boundary can fall
/// inside a grapheme (e.g. before a Devanagari virama U+094D or a Thai vowel
/// sign). Such a split emits orphan combining marks. Snap every match end
/// forward to the next grapheme boundary and merge overlapped matches so no
/// emitted chunk starts inside a cluster.
#[cfg(feature = "python")]
pub(crate) fn is_combining_mark(ch: char) -> bool {
    use unicode_general_category::{GeneralCategory, get_general_category};
    matches!(
        get_general_category(ch),
        GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
    )
}

#[cfg(feature = "python")]
fn snap_spans_to_graphemes(text: &str, spans: &[(usize, usize)]) -> Vec<(usize, usize)> {
    if spans.is_empty() {
        return Vec::new();
    }
    let mut boundaries: HashSet<usize> = HashSet::with_capacity(text.len() / 4 + 2);
    for (idx, _) in text.grapheme_indices(true) {
        boundaries.insert(idx);
    }
    boundaries.insert(text.len());
    let advance = |off: usize| -> usize {
        text[off..].chars().next().map_or(1, |c| c.len_utf8())
    };
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    let mut i = 0;
    while i < spans.len() {
        let (s, mut e) = spans[i];
        if !boundaries.contains(&s) {
            if let Some(last) = out.last_mut() {
                if e > last.1 {
                    last.1 = e;
                }
                while !boundaries.contains(&last.1) && last.1 < text.len() {
                    last.1 += advance(last.1);
                }
                while i + 1 < spans.len() && spans[i + 1].0 < last.1 {
                    i += 1;
                    if spans[i].1 > last.1 {
                        last.1 = spans[i].1;
                        while !boundaries.contains(&last.1) && last.1 < text.len() {
                            last.1 += advance(last.1);
                        }
                    }
                }
                i += 1;
                continue;
            }
        }
        while !boundaries.contains(&e) && e < text.len() {
            e += advance(e);
        }
        while i + 1 < spans.len() && spans[i + 1].0 < e {
            i += 1;
            if spans[i].1 > e {
                e = spans[i].1;
                while !boundaries.contains(&e) && e < text.len() {
                    e += advance(e);
                }
            }
        }
        // Degenerate leading orphan (text starts with \p{M}): fuse forward.
        if out.is_empty() && s < e {
            if let Some(first) = text[s..e].chars().next() {
                if is_combining_mark(first) && i + 1 < spans.len() {
                    e = spans[i + 1].1;
                    while !boundaries.contains(&e) && e < text.len() {
                        e += advance(e);
                    }
                    // Consume any further spans overlapped by the fusion.
                    let mut j = i + 1;
                    while j + 1 < spans.len() && spans[j + 1].0 < e {
                        j += 1;
                        if spans[j].1 > e {
                            e = spans[j].1;
                            while !boundaries.contains(&e) && e < text.len() {
                                e += advance(e);
                            }
                        }
                    }
                    i = j + 1;
                    out.push((s, e));
                    continue;
                }
            }
        }
        out.push((s, e));
        i += 1;
    }
    out
}

#[cfg(feature = "python")]
fn snapped_pretokens(text: &str, re: &Regex) -> Vec<String> {
    let spans: Vec<(usize, usize)> = re.find_iter(text).map(|m| (m.start(), m.end())).collect();
    snap_spans_to_graphemes(text, &spans)
        .into_iter()
        .map(|(s, e)| text[s..e].to_string())
        .collect()
}

/// Characters Python's `Normalizer` maps to whitespace replacements.
///
/// Mirrors `Normalizer.UNICODE_SPACES` plus the ASCII space; tabs, newlines,
/// and carriage returns are deliberately NOT mapped here so the pre-tokenizer
/// regex handles them exactly like the Python single-encode path.
#[cfg(feature = "python")]
fn is_python_unicode_space(ch: char) -> bool {
    matches!(
        ch,
        '\u{00A0}' | '\u{1680}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
    )
}

/// Normalizes a single string directly in native Rust.
///
/// Semantics mirror Python's `Normalizer.normalize` + metaspace substitution:
/// NFKC first, then only the configured unicode-space set (and plain space)
/// become the metaspace character. Other whitespace is left for the
/// pre-tokenizer's `\s` handling to keep Rust/Python parity on tabs/newlines.
#[cfg(feature = "python")]
pub fn normalize_string_native(text: &str, space_char: char) -> String {
    let mut normalized = String::with_capacity(text.len() + 8);
    let nfkc: String = text.nfkc().collect();
    for ch in nfkc.chars() {
        if ch == ' ' || is_python_unicode_space(ch) {
            normalized.push(space_char);
        } else {
            normalized.push(ch);
        }
    }
    normalized
}

/// Normalizes and pre-tokenizes a string into discrete subword chunks natively in Rust.
#[cfg(feature = "python")]
pub fn pre_tokenize_native(text: &str, space_char: char) -> Vec<String> {
    let normalized = normalize_string_native(text, space_char);
    let re = get_pretok_regex();
    snapped_pretokens(&normalized, re)
}

/// Native end-to-end pipeline: raw texts -> normalize -> regex pre-tokenize -> Viterbi DAG -> token IDs.
/// Executes completely in parallel across all CPU cores with the Python GIL released.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (texts, trie, byte_fallback=true, space_char=' '))]
pub fn rust_encode_text_batch(
    py: Python<'_>,
    texts: Vec<String>,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    space_char: char,
) -> CoreResult<Vec<Vec<u32>>> {
    if matches!(space_char, '\u{E000}' | '\u{E001}') {
        return core_error("space_char conflicts with reserved metaspace escape characters");
    }
    py.allow_threads(|| {
        texts
            .par_iter()
            .enumerate()
            .map(|(idx, raw_text)| {
                let chunks = pre_tokenize_native(raw_text, space_char);
                let mut sentence_ids: Vec<u32> = Vec::with_capacity(chunks.len() * 2);

                for chunk in chunks {
                    match decode_cached(&chunk, trie, byte_fallback) {
                        Ok(seg) => {
                            for (token, token_id, ..) in seg.iter() {
                                let id = token_id.ok_or_else(|| {
                                    CoreError(format!(
                                        "rust_encode_text_batch: decoded token {:?} has no integer ID",
                                        token
                                    ))
                                })?;
                                sentence_ids.push(id);
                            }
                        }
                        // Never silently turn a disconnected lattice into an
                        // empty or partial sequence — propagate with context.
                        Err(err) => {
                            return Err(CoreError(format!(
                                "rust_encode_text_batch: Viterbi decode failed for input #{} (chunk {:?}): {}",
                                idx, chunk, err
                            )));
                        }
                    }
                }
                Ok(sentence_ids)
            })
            .collect()
    })
}

/// Security gate for the fused native pipeline.
///
/// The Python pipeline runs `SecurityShield.sanitize` (NFKC + control-token
/// policy) before normalization. `sanitize` always checks the NFKC-canonical
/// form, independent of the Normalizer's `normalize_unicode` flag, so this
/// gate must do the same: when the canonicalized text cannot contain
/// control-token syntax, sanitize is provably the identity and the native
/// pipeline is exactly equivalent; otherwise bail out so Python handles
/// escaping/raising. Checking only the raw text would let
/// fullwidth-obfuscated syntax (e.g. U+FF1C/U+FF5C, issue #16) slip through
/// whenever normalization is disabled. Private-use metaspace escape characters
/// also belong to the Python Normalizer's escape dance.
#[cfg(feature = "python")]
fn native_security_gate(text: &str) -> CoreResult<()> {
    if text.contains('\u{E000}') || text.contains('\u{E001}') {
        return core_error("text contains private-use metaspace escape characters; use the Python pipeline");
    }
    // NFKC can synthesize '<' or '|' from fullwidth/compatibility chars
    // (e.g. '＜' U+FF1C -> '<', '｜' U+FF5C -> '|'), so the check must run
    // on the canonical form. NFKC is idempotent; the second pass inside
    // rust_normalize is negligible.
    let canonical: String = text.nfkc().collect();
    if canonical.contains("<|") {
        return core_error("text contains control-token syntax after NFKC; use the Python pipeline");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "python")]
fn encode_text_native_inner(
    text: &str,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    space_char: char,
    normalize_unicode: bool,
    normalize_unicode_spaces: bool,
    normalize_punctuation: bool,
    lowercase: bool,
    collapse_whitespaces: bool,
    strip_whitespace: bool,
) -> CoreResult<Vec<String>> {
    native_security_gate(text)?;
    let normalized = normalize_inner(
        text,
        space_char,
        normalize_unicode,
        normalize_unicode_spaces,
        normalize_punctuation,
        lowercase,
        collapse_whitespaces,
        strip_whitespace,
    )?;
    let re = get_full_pretok_regex();
    let mut tokens: Vec<String> = Vec::new();
    for chunk in snapped_pretokens(&normalized, re) {
        let seg = decode_cached(chunk.as_str(), trie, byte_fallback).map_err(CoreError)?;
        tokens.extend(seg.iter().map(|(token, ..)| token.clone()));
    }
    Ok(tokens)
}

/// Fused single-text encode: normalize + full pre-tokenizer regex + Viterbi in
/// ONE FFI crossing. Only equivalent to the Python pipeline when the caller
/// gates on the same config the Python path would use (see
/// `CustomTokenizer._native_pipeline_kwargs`); this function additionally
/// refuses texts that would need security-shield escaping.
#[cfg(feature = "python")]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (text, trie, byte_fallback=true, space_char='\u{2581}', normalize_unicode=true, normalize_unicode_spaces=true, normalize_punctuation=false, lowercase=false, collapse_whitespaces=false, strip_whitespace=false))]
pub fn rust_encode_text_native(
    text: &str,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    space_char: char,
    normalize_unicode: bool,
    normalize_unicode_spaces: bool,
    normalize_punctuation: bool,
    lowercase: bool,
    collapse_whitespaces: bool,
    strip_whitespace: bool,
) -> CoreResult<Vec<String>> {
    encode_text_native_inner(
        text,
        trie,
        byte_fallback,
        space_char,
        normalize_unicode,
        normalize_unicode_spaces,
        normalize_punctuation,
        lowercase,
        collapse_whitespaces,
        strip_whitespace,
    )
}

/// Fused batch encode: one FFI + Rayon across texts. On any per-text rejection
/// (e.g. control-token syntax) the whole batch errors so the caller can fall
/// back to the Python pipeline wholesale.
#[cfg(feature = "python")]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (texts, trie, byte_fallback=true, space_char='\u{2581}', normalize_unicode=true, normalize_unicode_spaces=true, normalize_punctuation=false, lowercase=false, collapse_whitespaces=false, strip_whitespace=false))]
pub fn rust_encode_text_native_batch(
    py: Python<'_>,
    texts: Vec<String>,
    trie: &RustPrefixTrie,
    byte_fallback: bool,
    space_char: char,
    normalize_unicode: bool,
    normalize_unicode_spaces: bool,
    normalize_punctuation: bool,
    lowercase: bool,
    collapse_whitespaces: bool,
    strip_whitespace: bool,
) -> CoreResult<Vec<Vec<String>>> {
    // ponytail: same sequential-below-32 rule as the raw batch functions.
    if texts.len() < 32 {
        return texts
            .iter()
            .map(|text| {
                encode_text_native_inner(
                    text,
                    trie,
                    byte_fallback,
                    space_char,
                    normalize_unicode,
                    normalize_unicode_spaces,
                    normalize_punctuation,
                    lowercase,
                    collapse_whitespaces,
                    strip_whitespace,
                )
            })
            .collect();
    }
    py.allow_threads(|| {
        texts
            .par_iter()
            .map(|text| {
                encode_text_native_inner(
                    text,
                    trie,
                    byte_fallback,
                    space_char,
                    normalize_unicode,
                    normalize_unicode_spaces,
                    normalize_punctuation,
                    lowercase,
                    collapse_whitespaces,
                    strip_whitespace,
                )
            })
            .collect()
    })
}

#[cfg(all(test, feature = "python"))]
mod tests {
    use super::*;

    #[test]
    fn security_gate_refuses_nfkc_synthesized_control_syntax() {
        // U+FF1C FULLWIDTH LESS-THAN SIGN and U+FF5C FULLWIDTH VERTICAL LINE
        // NFKC-map to '<' and '|'. The gate must refuse the obfuscated form
        // exactly like the literal one, mirroring SecurityShield.sanitize,
        // which always canonicalizes before matching (issue #16).
        assert!(native_security_gate("＜｜system｜＞").is_err());
        assert!(native_security_gate("<|system|>").is_err());
        assert!(native_security_gate("hello world").is_ok());
        assert!(native_security_gate("2 < 3 | 4").is_ok());
    }
}
