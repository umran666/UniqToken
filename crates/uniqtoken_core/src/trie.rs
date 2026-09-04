//! High-performance native Prefix Trie implementation for subword matching.

use crate::error::{core_error, CoreResult};
use ahash::AHashMap;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

/// Cached Viterbi segmentation of one chunk: (token, token_id, start, end) triples.
pub(crate) type CachedSegmentation = Arc<Vec<(String, Option<u32>, usize, usize)>>;

/// ponytail: cap is clear-all, not LRU; upgrade if eviction churn shows up.
const SEG_CACHE_CAP: usize = 100_000;


#[derive(Default, Clone)]
pub struct TrieNode {
    pub children: AHashMap<char, TrieNode>,
    pub token: Option<String>,
    pub token_id: Option<u32>,
    pub log_p: f64,
    pub is_terminal: bool,
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(Default, Clone)]
pub struct RustPrefixTrie {
    root: TrieNode,
    pub max_subword_len: Option<usize>,
    /// Word-level segmentation memoization, shared across Rayon workers.
    /// Lives on the trie itself so it is invalidated automatically whenever
    /// the vocabulary changes (Python builds a fresh RustPrefixTrie per vocab).
    seg_cache: Arc<Mutex<AHashMap<(bool, String), CachedSegmentation>>>,
}

/// Shared insert logic behind both the plain (all configurations) and the
/// Python-bound `insert` methods, which are mutually exclusive by feature and
/// therefore can never collide.
pub(crate) fn insert_token(
    trie: &mut RustPrefixTrie,
    token: &str,
    log_p: f64,
    token_id: Option<u32>,
) -> CoreResult<()> {
    if token.is_empty() {
        return core_error("token must not be empty");
    }
    if !log_p.is_finite() {
        return core_error("log_p must be finite");
    }

    let mut curr = &mut trie.root;
    for ch in token.chars() {
        curr = curr.children.entry(ch).or_default();
    }
    curr.is_terminal = true;
    curr.token = Some(token.to_string());
    curr.token_id = token_id;
    curr.log_p = log_p;
    Ok(())
}

#[cfg(not(feature = "python"))]
impl RustPrefixTrie {
    /// Plain insert for non-Python bindings (e.g. WebAssembly vocab loading).
    pub fn insert(&mut self, token: &str, log_p: f64, token_id: Option<u32>) -> CoreResult<()> {
        insert_token(self, token, log_p, token_id)
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl RustPrefixTrie {
    #[new]
    #[pyo3(signature = (max_subword_len=None))]
    pub fn new(max_subword_len: Option<usize>) -> Self {
        Self {
            root: TrieNode::default(),
            max_subword_len,
            seg_cache: Arc::new(Mutex::new(AHashMap::with_capacity(8192))),
        }
    }

    #[getter]
    pub fn max_subword_len(&self) -> Option<usize> {
        self.max_subword_len
    }

    #[setter]
    pub fn set_max_subword_len(&mut self, val: Option<usize>) {
        self.max_subword_len = val;
    }

    /// Inserts a non-empty subword with a finite log probability.
    pub fn insert(&mut self, token: &str, log_p: f64, token_id: Option<u32>) -> CoreResult<()> {
        insert_token(self, token, log_p, token_id)
    }

    /// Finds all matching prefixes for a slice of text starting at position 0.
    /// Returns tuples of (token, token_id, log_p, char_length).
    pub fn common_prefix_search(&self, text: &str) -> Vec<(String, Option<u32>, f64, usize)> {
        let mut results = Vec::with_capacity(8);
        let mut curr = &self.root;
        let mut char_count = 0;
        let max_len = self.max_subword_len.unwrap_or(usize::MAX);

        for ch in text.chars() {
            if char_count >= max_len {
                break;
            }
            if let Some(next_node) = curr.children.get(&ch) {
                curr = next_node;
                char_count += 1;
                if curr.is_terminal {
                    if let Some(ref tok) = curr.token {
                        results.push((tok.clone(), curr.token_id, curr.log_p, char_count));
                    }
                }
            } else {
                break;
            }
        }

        results
    }

    /// Checks whether the exact token exists in the Trie.
    pub fn contains(&self, token: &str) -> bool {
        self.exact_metadata(token).is_some()
    }

    /// Clears the shared word-level segmentation memoization cache.
    pub fn clear_seg_cache(&self) {
        if let Ok(mut cache) = self.seg_cache.lock() {
            cache.clear();
        }
    }

    /// Returns the number of cached segmentations.
    pub fn seg_cache_len(&self) -> usize {
        self.seg_cache.lock().map(|c| c.len()).unwrap_or(0)
    }
}

impl RustPrefixTrie {
    pub(crate) fn common_prefix_search_chars(
        &self,
        chars: &[char],
        start: usize,
    ) -> Vec<(String, Option<u32>, f64, usize)> {
        let mut results = Vec::with_capacity(8);
        let mut current = &self.root;
        let max_len = self.max_subword_len.unwrap_or(usize::MAX);
        for (offset, ch) in chars[start..].iter().enumerate() {
            if offset >= max_len {
                break;
            }
            let Some(next) = current.children.get(ch) else {
                break;
            };
            current = next;
            if current.is_terminal {
                if let Some(token) = &current.token {
                    results.push((token.clone(), current.token_id, current.log_p, offset + 1));
                }
            }
        }
        results
    }

    pub(crate) fn exact_metadata(&self, token: &str) -> Option<(Option<u32>, f64)> {
        let mut current = &self.root;
        for ch in token.chars() {
            current = current.children.get(&ch)?;
        }
        current.is_terminal.then_some((current.token_id, current.log_p))
    }

    /// Looks up a cached segmentation. Keyed by (byte_fallback, chunk) since
    /// both change the segmentation for a fixed vocabulary.
    pub(crate) fn seg_cache_get(&self, byte_fallback: bool, chunk: &str) -> Option<CachedSegmentation> {
        let key = (byte_fallback, chunk.to_string());
        self.seg_cache.lock().ok()?.get(&key).cloned()
    }

    /// Stores a segmentation. The key is re-allocated on insert; lookups on
    /// hot repeated chunks amortize this many times over.
    pub(crate) fn seg_cache_put(&self, byte_fallback: bool, chunk: &str, seg: CachedSegmentation) {
        if let Ok(mut cache) = self.seg_cache.lock() {
            let key = (byte_fallback, chunk.to_string());
            if !cache.contains_key(&key) && cache.len() >= SEG_CACHE_CAP {
                cache.clear();
            }
            cache.insert(key, seg);
        }
    }
}
