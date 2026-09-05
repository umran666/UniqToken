"""Integration test for llama.cpp GGUF vocabulary table loader and C-ABI hook (Issue #52)."""

import unittest
from pathlib import Path


class TestLlamaCppGGUFAdapter(unittest.TestCase):
    """Verifies that the Rust C-ABI exports GGUF v3 vocabularies correctly."""

    def test_gguf_vocab_c_abi_export(self):
        import ctypes

        vocab_path = Path("crates/uniqtoken_core/demo_vocab.json")
        self.assertTrue(vocab_path.exists(), "demo_vocab.json must exist")
        candidates = (
            list(Path("target").glob("**/uniqtoken_core.dll"))
            + list(Path("target").glob("**/libuniqtoken_core.so"))
            + list(Path("target").glob("**/libuniqtoken_core.dylib"))
            + list(Path("crates/uniqtoken_core/target").glob("**/uniqtoken_core.dll"))
            + list(Path("crates/uniqtoken_core/target").glob("**/libuniqtoken_core.so"))
            + list(Path("crates/uniqtoken_core/target").glob("**/libuniqtoken_core.dylib"))
        )
        possible_libs = [p for p in candidates if "debug" in p.parts or "release" in p.parts]
        if not possible_libs:
            possible_libs = candidates
        if not possible_libs:
            self.skipTest("uniqtoken_core cdylib not compiled in target/ yet")
        lib = ctypes.CDLL(str(possible_libs[0]))
        lib.uniqtoken_export_gguf_vocab.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.uniqtoken_export_gguf_vocab.restype = ctypes.c_int32
        lib.uniqtoken_free_buffer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.uniqtoken_free_buffer.restype = None
        buf = ctypes.c_void_p()
        size = ctypes.c_size_t()
        path_bytes = str(vocab_path.resolve()).encode("utf-8")
        rc = lib.uniqtoken_export_gguf_vocab(path_bytes, ctypes.byref(buf), ctypes.byref(size))
        self.assertEqual(rc, 0, f"Expected UNIQTOKEN_OK (0), got {rc}")
        self.assertIsNotNone(buf.value)
        self.assertGreater(size.value, 100)
        try:
            data = ctypes.string_at(buf, size.value)
            self.assertEqual(data[:4], b"GGUF")
        finally:
            lib.uniqtoken_free_buffer(buf, size.value)


if __name__ == "__main__":
    unittest.main()
