//! Shared error type for the Python and WebAssembly bindings.
//!
//! Core tokenizer logic reports failures through [`CoreError`] so neither the
//! PyO3 nor the `wasm-bindgen` shell owns the error vocabulary. The Python
//! binding converts it into `ValueError` via the `From` impl below.

use std::fmt;

/// Failure reported by core tokenizer operations (invalid configs,
/// disconnected lattices, tokens without integer IDs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreError(pub String);

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CoreError {}

/// Result alias used by every binding-facing core function.
pub type CoreResult<T> = Result<T, CoreError>;

/// Convenience constructor that keeps fallible call sites terse.
pub fn core_error<T>(msg: impl Into<String>) -> CoreResult<T> {
    Err(CoreError(msg.into()))
}

#[cfg(feature = "python")]
impl From<CoreError> for pyo3::PyErr {
    fn from(err: CoreError) -> Self {
        pyo3::exceptions::PyValueError::new_err(err.0)
    }
}
