#![no_main]

use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;
use uniqtoken_core::trie::RustPrefixTrie;
use uniqtoken_core::viterbi::{decode_cached, viterbi_decode_chars};

static FUZZ_TRIE: OnceLock<RustPrefixTrie> = OnceLock::new();

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

    // Decode with byte fallback enabled
    if let Ok(spans) = viterbi_decode_chars(&chars, trie, true, None) {
        let mut prev_start = 0;
        let mut prev_end = 0;
        for span in &spans {
            assert!(span.start <= span.end);
            assert!(span.end <= chars.len());
            assert!(span.start >= prev_start);
            assert!(span.end >= prev_end);
            prev_start = span.start;
            prev_end = span.end;
        }
    }

    // Decode with byte fallback disabled (may legitimately fail if unrepresented chars exist)
    let _ = viterbi_decode_chars(&chars, trie, false, None);

    // Decode with edge pruning
    let _ = viterbi_decode_chars(&chars, trie, true, Some(4));

    // Word-level cached decoding
    let _ = decode_cached(&text, trie, true);

    // 2. If valid UTF-8, test exact char bounds
    if let Ok(valid_str) = std::str::from_utf8(data) {
        let valid_chars: Vec<char> = valid_str.chars().collect();
        if let Ok(spans) = viterbi_decode_chars(&valid_chars, trie, true, None) {
            if !valid_chars.is_empty() {
                assert!(!spans.is_empty());
                assert_eq!(spans.first().unwrap().start, 0);
                assert_eq!(spans.last().unwrap().end, valid_chars.len());
            }
        }
    }
});
