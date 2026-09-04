#![cfg(feature = "python")]
//! Seed n-gram mining for vocabulary construction (Python bindings only).

use crate::error::CoreResult;
use crate::pipeline::is_combining_mark;
use ahash::AHashMap;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use unicode_segmentation::UnicodeSegmentation;

fn detect_script(token: &str) -> &'static str {
    for ch in token.chars() {
        let cp = ch as u32;
        if cp == 0x2581 || cp == 0x0020 || cp == 0x0009 || cp == 0x000A || cp == 0x000D || (0xE000..=0xF8FF).contains(&cp) {
            continue;
        }
        if (0x0041..=0x005A).contains(&cp) || (0x0061..=0x007A).contains(&cp) || (0x00C0..=0x024F).contains(&cp) {
            return "latin";
        } else if (0x0900..=0x097F).contains(&cp) {
            return "devanagari";
        } else if (0x0C00..=0x0C7F).contains(&cp) {
            return "telugu";
        } else if (0x0B80..=0x0BFF).contains(&cp) {
            return "tamil";
        } else if (0x0980..=0x09FF).contains(&cp) {
            return "bengali";
        } else if (0x0900..=0x0D7F).contains(&cp) {
            return "indic_other";
        } else if (0x4E00..=0x9FFF).contains(&cp)
            || (0x3400..=0x4DBF).contains(&cp)
            || (0x3040..=0x30FF).contains(&cp)
            || (0xAC00..=0xD7AF).contains(&cp)
        {
            return "cjk";
        } else if (0x0600..=0x06FF).contains(&cp) || (0x0750..=0x077F).contains(&cp) {
            return "arabic";
        } else if (0x0400..=0x04FF).contains(&cp) || (0x0500..=0x052F).contains(&cp) {
            return "cyrillic";
        } else if (0x0E00..=0x0E7F).contains(&cp) {
            return "thai";
        } else if ch.is_numeric() || token.starts_with("0x") || token.starts_with("SYS_") {
            return "numeric";
        }
    }
    "symbol"
}

fn max_ngram_for_chunk(chunk: &str, default_max: usize) -> usize {
    if detect_script(chunk) == "cjk" {
        std::cmp::min(default_max, 4)
    } else {
        default_max
    }
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (chunk_counts, max_ngram_length, special_tokens=None))]
pub fn rust_mine_ngrams(
    chunk_counts: HashMap<String, usize>,
    max_ngram_length: usize,
    special_tokens: Option<HashSet<String>>,
) -> CoreResult<HashMap<String, usize>> {
    let specials = special_tokens.unwrap_or_default();
    let mut ngram_counts: AHashMap<String, usize> = AHashMap::with_capacity(chunk_counts.len() * 8);
    for (chunk, freq) in chunk_counts {
        if specials.contains(&chunk) {
            continue;
        }
        if chunk.starts_with("<|") && chunk.ends_with("|>") {
            continue;
        }
        let clusters: Vec<&str> = chunk.graphemes(true).collect();
        let clen = clusters.len();
        let max_len = max_ngram_for_chunk(&chunk, max_ngram_length);
        for start in 0..clen {
            // Issue #41: skip n-grams starting with an orphan combining mark.
            if is_combining_mark(clusters[start].chars().next().unwrap()) {
                continue;
            }
            let mut end_limit = clen + 1;
            let ml = start + max_len + 1;
            if ml < end_limit {
                end_limit = ml;
            }
            for end in (start + 1)..end_limit {
                let piece: String = clusters[start..end].concat();
                *ngram_counts.entry(piece).or_insert(0) += freq;
            }
        }
    }
    Ok(ngram_counts.into_iter().collect())
}
