#![cfg(feature = "c_abi")]
//! Integration tests for UniqToken C-ABI GGUF vocabulary export.
use std::ffi::CString;
use std::ptr;
use uniqtoken_core::c_abi::*;
/// RAII guard ensuring allocated FFI buffers are always released even on assertion failure.
struct BufferGuard {
    buf: *mut std::os::raw::c_void,
    size: usize,
}
impl Drop for BufferGuard {
    fn drop(&mut self) {
        if !self.buf.is_null() && self.size > 0 {
            unsafe {
                uniqtoken_free_buffer(self.buf, self.size);
            }
        }
    }
}
/// Tests that null pointers return defensive error code UNIQTOKEN_ERR_NULL_PTR.
#[test]
fn test_c_abi_null_ptrs() {
    unsafe {
        let res = uniqtoken_export_gguf_vocab(ptr::null(), ptr::null_mut(), ptr::null_mut());
        assert_eq!(res, UNIQTOKEN_ERR_NULL_PTR);
    }
}
/// Tests that nonexistent paths return defensive error code UNIQTOKEN_ERR_IO.
#[test]
fn test_c_abi_nonexistent_file() {
    unsafe {
        let path = CString::new("nonexistent_vocab_path_123.json").unwrap();
        let mut buf: *mut std::os::raw::c_void = ptr::null_mut();
        let mut size: usize = 0;
        let res = uniqtoken_export_gguf_vocab(path.as_ptr(), &mut buf, &mut size);
        assert_eq!(res, UNIQTOKEN_ERR_IO);
    }
}
/// Tests that demo_vocab.json exports successfully to valid GGUF v3 binary.
#[test]
fn test_c_abi_demo_vocab_export() {
    unsafe {
        let path_str = concat!(env!("CARGO_MANIFEST_DIR"), "/demo_vocab.json");
        let path = CString::new(path_str).unwrap();
        let mut buf: *mut std::os::raw::c_void = ptr::null_mut();
        let mut size: usize = 0;
        let res = uniqtoken_export_gguf_vocab(path.as_ptr(), &mut buf, &mut size);
        assert_eq!(res, UNIQTOKEN_OK);
        assert!(size > 100);
        assert!(!buf.is_null());
        let _guard = BufferGuard { buf, size };
        let bytes = std::slice::from_raw_parts(buf as *const u8, size);
        assert_eq!(&bytes[0..4], b"GGUF");
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(version, 3);
    }
}

/// Minimal `[[token, logprob, id]]` vocabulary covering "hello world".
const TOKENIZER_VOCAB_JSON: &str = r#"[["hello",-1.0,0],["▁world",-1.0,1],["<|unk|>",-5.0,2]]"#;

/// RAII guard destroying a tokenizer handle even on assertion failure.
struct HandleGuard {
    handle: *mut UniqTokenHandle,
}
impl Drop for HandleGuard {
    fn drop(&mut self) {
        unsafe {
            uniqtoken_destroy(self.handle);
        }
    }
}

fn create_test_tokenizer() -> HandleGuard {
    let json = CString::new(TOKENIZER_VOCAB_JSON).unwrap();
    let handle = unsafe { uniqtoken_create(json.as_ptr()) };
    assert!(!handle.is_null());
    HandleGuard { handle }
}

/// Tests that null arguments are rejected without crashing.
#[test]
fn test_tokenizer_api_null_ptrs() {
    unsafe {
        assert!(uniqtoken_create(ptr::null()).is_null());
        let mut ids: *mut u32 = ptr::null_mut();
        let mut len: usize = 0;
        let text = c"hello".as_ptr();
        assert_eq!(
            uniqtoken_encode(ptr::null_mut(), text, 5, &mut ids, &mut len),
            UNIQTOKEN_ERR_NULL_PTR
        );
        // Freeing and destroying null are safe no-ops.
        uniqtoken_free_tokens(ptr::null_mut(), 0);
        uniqtoken_destroy(ptr::null_mut());
    }
}

/// Tests that malformed or empty vocabularies refuse handle creation.
#[test]
fn test_tokenizer_create_rejects_bad_vocab() {
    for bad_json in ["not json", "[]", r#"[["only-token"]]"#, "[[]]"] {
        let json = CString::new(bad_json).unwrap();
        assert!(unsafe { uniqtoken_create(json.as_ptr()) }.is_null());
    }
}

/// Tests that vocabularies without <|unk|> or with non-contiguous IDs are
/// refused: otherwise unknown spans would silently resolve to ID 0.
#[test]
fn test_tokenizer_create_requires_unk_and_contiguous_ids() {
    // No <|unk|> entry.
    let json = CString::new(r#"[["hello",-1.0,0]]"#).unwrap();
    assert!(unsafe { uniqtoken_create(json.as_ptr()) }.is_null());
    // Gap: ID 1 missing.
    let json = CString::new(r#"[["hello",-1.0,0],["x",-1.0,2],["<|unk|>",-5.0,3]]"#).unwrap();
    assert!(unsafe { uniqtoken_create(json.as_ptr()) }.is_null());
    // Duplicate ID 0.
    let json = CString::new(r#"[["hello",-1.0,0],["hello",-1.0,0],["<|unk|>",-5.0,2]]"#).unwrap();
    assert!(unsafe { uniqtoken_create(json.as_ptr()) }.is_null());
}

/// Tests the full create/encode/free/destroy cycle against known IDs.
#[test]
fn test_tokenizer_encode_roundtrip() {
    let guard = create_test_tokenizer();
    let text = c"hello world".to_bytes_with_nul();
    let mut ids: *mut u32 = ptr::null_mut();
    let mut len: usize = 0;
    let status = unsafe {
        uniqtoken_encode(
            guard.handle,
            text.as_ptr() as *const std::os::raw::c_char,
            text.len() - 1,
            &mut ids,
            &mut len,
        )
    };
    assert_eq!(status, UNIQTOKEN_OK);
    assert_eq!(len, 2);
    assert!(!ids.is_null());
    let decoded = unsafe { std::slice::from_raw_parts(ids, len) };
    assert_eq!(decoded, &[0, 1]);
    unsafe {
        uniqtoken_free_tokens(ids, len);
    }
}

/// Tests that invalid UTF-8 input is rejected and empty input yields NULL + 0.
#[test]
fn test_tokenizer_encode_edge_cases() {
    let guard = create_test_tokenizer();
    let mut ids: *mut u32 = ptr::null_mut();
    let mut len: usize = 0;
    let lone_byte = [0xFFu8];
    let status = unsafe {
        uniqtoken_encode(
            guard.handle,
            lone_byte.as_ptr() as *const std::os::raw::c_char,
            lone_byte.len(),
            &mut ids,
            &mut len,
        )
    };
    assert_eq!(status, UNIQTOKEN_ERR_INVALID_UTF8);
    let status = unsafe {
        uniqtoken_encode(
            guard.handle,
            c"".as_ptr(),
            0,
            &mut ids,
            &mut len,
        )
    };
    assert_eq!(status, UNIQTOKEN_OK);
    assert_eq!(len, 0);
    assert!(ids.is_null());
}
