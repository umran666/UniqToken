//! LibFuzzer target for UniqToken native Viterbi lattice decoding and trellis coverage (issue #46).

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;
use uniqtoken_core::trie::RustPrefixTrie;
use uniqtoken_core::viterbi::{decode_cached, viterbi_decode_chars};

static FUZZ_TRIE: OnceLock<RustPrefixTrie> = OnceLock::new();

/// Lazily constructs and caches a static prefix trie loaded with seed tokens and byte fallback entries.
fn get_or_init_trie() -> &'static RustPrefixTrie {
    FUZZ_TRIE.get_or_init(|| {
        let mut trie = RustPrefixTrie::new(Some(16));
        let seed_tokens: &[(&str, f64, u32)] = &[
            ("the", -1.0, 1),
            ("token", -1.5, 2),
            ("ization", -1.5, 3),
            (" ", -2.0, 4),
            ("a", -2.5, 5),
            ("b", -2.5, 6),
            ("c", -2.5, 7),
            ("ing", -1.8, 8),
            ("ed", -1.9, 9),
            ("un", -2.1, 10),
            ("re", -2.2, 11),
            ("pre", -2.3, 12),
            ("post", -2.4, 13),
            ("test", -1.6, 14),
            ("model", -1.7, 15),
            ("sub", -2.0, 16),
            ("word", -1.9, 17),
        ];
        for &(tok, logp, id) in seed_tokens {
            let _ = trie.insert(tok, logp, Some(id));
        }
        // Byte fallback tokens <0x00> through <0xFF>
        for b in 0..=255u8 {
            let tok = format!("<0x{:02X}>", b);
            let _ = trie.insert(&tok, -10.0, Some(1000 + b as u32));
        }
        trie
    })
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let trie = get_or_init_trie();

    // 1. Fuzz with UTF-8 lossy conversion
    let text = String::from_utf8_lossy(data);
    let chars: Vec<char> = text.chars().collect();

    // Decode with byte fallback enabled (must never fail on arbitrary UTF-8 characters)
    let spans = viterbi_decode_chars(&chars, trie, true, None)
        .expect("byte fallback must cover every UTF-8 character sequence");
    if !chars.is_empty() {
        assert!(!spans.is_empty());
        assert_eq!(spans.first().unwrap().start, 0);
        assert_eq!(spans.last().unwrap().end, chars.len());
    }
    let mut prev_start = 0;
    let mut prev_end = 0;
    for span in &spans {
        assert!(span.start <= span.end);
        assert!(span.end <= chars.len());
        assert!(span.start >= prev_start);
        assert!(span.end >= prev_end);
        assert_eq!(span.start, prev_end, "gaps or overlaps detected in span trellis");
        prev_start = span.start;
        prev_end = span.end;
    }
    assert_eq!(prev_end, chars.len(), "spans did not cover entire input");

    // Decode with byte fallback disabled (may legitimately fail if unrepresented chars exist)
    let _ = viterbi_decode_chars(&chars, trie, false, None);

    // Decode with edge pruning (assert Ok as byte fallback ensures lattice connectivity)
    let pruned_spans = viterbi_decode_chars(&chars, trie, true, Some(4))
        .expect("byte fallback with beam pruning must cover input");
    if !chars.is_empty() {
        assert!(!pruned_spans.is_empty());
        assert_eq!(pruned_spans.first().unwrap().start, 0);
        assert_eq!(pruned_spans.last().unwrap().end, chars.len());
    }

    // Word-level cached decoding
    let cached_seg = decode_cached(&text, trie, true)
        .expect("word-level cached decoding must succeed with byte fallback");
    if !text.is_empty() && text.chars().count() > 0 {
        assert!(!cached_seg.is_empty());
    }

    // 2. If valid UTF-8, test exact char bounds
    if let Ok(valid_str) = std::str::from_utf8(data) {
        let valid_chars: Vec<char> = valid_str.chars().collect();
        let spans = viterbi_decode_chars(&valid_chars, trie, true, None)
            .expect("byte fallback on valid UTF-8 must succeed");
        if !valid_chars.is_empty() {
            assert!(!spans.is_empty());
            assert_eq!(spans.first().unwrap().start, 0);
            assert_eq!(spans.last().unwrap().end, valid_chars.len());
        }
    }
});
