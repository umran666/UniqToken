//! UniqToken Core: High-performance native Rust acceleration module for UniqToken.

#[cfg(feature = "c_abi")]
pub mod c_abi;
pub mod error;
pub mod normalizer;
pub mod pipeline;
pub mod rust_tokenizer;
pub mod seed;
pub mod trie;
pub mod viterbi;
#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "python")]
use normalizer::{rust_normalize, rust_normalize_with_alignment};
#[cfg(feature = "python")]
use pipeline::{rust_encode_text_batch, rust_encode_text_native, rust_encode_text_native_batch, rust_pre_tokenize};
#[cfg(feature = "python")]
use rust_tokenizer::{rust_diagnostic_batch, RustTokenizer};
#[cfg(feature = "python")]
use seed::rust_mine_ngrams;
#[cfg(feature = "python")]
use trie::RustPrefixTrie;
#[cfg(feature = "python")]
use viterbi::{
    rust_diagnostic_viterbi, rust_encode_ids_batch, rust_encode_tokens_batch, rust_forward_backward_expectations,
    rust_viterbi_decode, rust_viterbi_decode_batch, ViterbiSpan,
};
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Native UniqToken Core Python Module
#[cfg(feature = "python")]
#[pymodule]
fn uniqtoken_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustPrefixTrie>()?;
    m.add_class::<RustTokenizer>()?;
    m.add_class::<ViterbiSpan>()?;
    m.add_function(wrap_pyfunction!(rust_viterbi_decode, m)?)?;
    m.add_function(wrap_pyfunction!(rust_diagnostic_viterbi, m)?)?;
    m.add_function(wrap_pyfunction!(rust_viterbi_decode_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rust_encode_tokens_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rust_encode_ids_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rust_encode_text_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rust_encode_text_native, m)?)?;
    m.add_function(wrap_pyfunction!(rust_encode_text_native_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rust_forward_backward_expectations, m)?)?;
    m.add_function(wrap_pyfunction!(rust_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(rust_normalize_with_alignment, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mine_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(rust_pre_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(rust_diagnostic_batch, m)?)?;
    Ok(())
}
