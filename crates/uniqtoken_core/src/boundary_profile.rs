//! Opt-in sequential boundary stages. Never compiled into ordinary wheels.

use crate::allocation_profile::{measure, Counts};
use crate::error::{CoreError, CoreResult};
use crate::pipeline::{
    encode_native_segments_batch, encode_text_native_segments, extract_borrowed_strings,
};
use crate::trie::RustPrefixTrie;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyString, PyStringMethods};
use std::borrow::Cow;
use std::time::Instant;

fn stage<T>(allocations: bool, call: impl FnOnce() -> T) -> CoreResult<(T, u64, Counts)> {
    let start = Instant::now();
    let (value, counts) = if allocations {
        measure(call)?
    } else {
        (call(), (0, 0, 0, 0, 0))
    };
    Ok((value, start.elapsed().as_nanos() as u64, counts))
}

type BoundaryRow = (Py<PyAny>, [u64; 5], [Counts; 5], usize, usize, usize);

enum Input<'a> {
    Single(Cow<'a, str>),
    Batch(Vec<PyBackedStr>),
}

/// Input extraction, native computation, then Rust-to-Python materialization.
/// Reference materialization recreates owned Rust strings, not an old engine.
/// IDs are an unchanged conversion control. Counters cover Rust's allocator,
/// not CPython's allocator; collect timings with allocations=false separately.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (texts, trie, reference=false, output_ids=false, allocations=false, single=false, byte_fallback=true, space_char='\u{2581}', normalize_unicode=true, normalize_unicode_spaces=true, normalize_punctuation=false, lowercase=false, collapse_whitespaces=false, strip_whitespace=false))]
pub(crate) fn rust_profile_boundary<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    trie: &RustPrefixTrie,
    reference: bool,
    output_ids: bool,
    allocations: bool,
    single: bool,
    byte_fallback: bool,
    space_char: char,
    normalize_unicode: bool,
    normalize_unicode_spaces: bool,
    normalize_punctuation: bool,
    lowercase: bool,
    collapse_whitespaces: bool,
    strip_whitespace: bool,
) -> PyResult<BoundaryRow> {
    let (input, input_ns, input_counts) = stage(allocations, || -> PyResult<Input<'_>> {
        if single {
            // Matches PyO3's &str argument holder on the pinned abi3-py39 build.
            Ok(Input::Single(texts.cast::<PyString>()?.to_cow()?))
        } else {
            Ok(Input::Batch(extract_borrowed_strings(texts)?))
        }
    })?;
    let input = input?;
    let (input_bytes, rust_input_copied_bytes) = match &input {
        Input::Single(text) => (
            text.len(),
            if matches!(text, Cow::Owned(_)) {
                text.len()
            } else {
                0
            },
        ),
        Input::Batch(borrowed) => (borrowed.iter().map(|text| text.len()).sum(), 0),
    };
    let (rows, compute_ns, compute_counts) = stage(allocations, || match &input {
        Input::Single(text) => encode_text_native_segments(
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
        .map(|row| vec![row]),
        Input::Batch(borrowed) => encode_native_segments_batch(
            py,
            borrowed,
            trie,
            byte_fallback,
            space_char,
            normalize_unicode,
            normalize_unicode_spaces,
            normalize_punctuation,
            lowercase,
            collapse_whitespaces,
            strip_whitespace,
        ),
    })?;
    let rows = rows?;
    let cloned_bytes = if reference && !output_ids {
        rows.iter().flat_map(|row| row.iter()).map(str::len).sum()
    } else {
        0
    };
    let (output, output_ns, output_counts) =
        stage(allocations, || -> PyResult<Bound<'py, PyAny>> {
            if output_ids {
                let ids: CoreResult<Vec<Vec<u32>>> = rows
                    .iter()
                    .map(|row| {
                        row.0
                            .iter()
                            .flat_map(|seg| seg.iter())
                            .map(|(token, id, ..)| {
                                id.ok_or_else(|| {
                                    CoreError(format!("decoded token {token:?} has no integer ID"))
                                })
                            })
                            .collect()
                    })
                    .collect();
                let mut ids = ids?;
                let output = if single {
                    ids.remove(0).into_pyobject(py)
                } else {
                    ids.into_pyobject(py)
                };
                drop(rows);
                output
            } else if reference {
                let mut owned: Vec<Vec<String>> = rows
                    .iter()
                    .map(|row| row.iter().map(str::to_owned).collect())
                    .collect();
                let output = if single {
                    owned.remove(0).into_pyobject(py)
                } else {
                    owned.into_pyobject(py)
                };
                drop(rows);
                output
            } else {
                if single {
                    rows.into_iter()
                        .next()
                        .expect("one single-text row")
                        .into_pyobject(py)
                        .map(Bound::into_any)
                } else {
                    rows.into_pyobject(py)
                }
            }
        })?;
    let output = output?;
    let (_, release_ns, release_counts) = stage(allocations, || drop(input))?;
    // Root-list INCREF/DECREF control; element destruction is not timed here.
    let (_, refs_ns, refs_counts) = stage(allocations, || drop(output.clone()))?;
    Ok((
        output.unbind(),
        [input_ns, compute_ns, output_ns, release_ns, refs_ns],
        [
            input_counts,
            compute_counts,
            output_counts,
            release_counts,
            refs_counts,
        ],
        input_bytes,
        cloned_bytes,
        rust_input_copied_bytes,
    ))
}
