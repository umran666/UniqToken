"""Tests for zero-copy memory-mapped binary model format (.uniqtok) (Issue #36)."""

import tempfile
import time
import unittest
from pathlib import Path

from uniqtoken.binary_format import export_binary, load_binary
from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
from uniqtoken.tokenizer import CustomTokenizer
from uniqtoken.unigram_trainer import UnigramModel


class TestBinaryMmapModel(unittest.TestCase):
    """Verifies binary format serialization, mmap loading, and sub-millisecond cold start."""

    def setUp(self):
        # Build test tokenizer
        vocab = {
            "<|unk|>": 0.0,
            "<|bos|>": 0.0,
            "<|eos|>": 0.0,
            "▁hello": -1.5,
            "▁world": -1.8,
            "hello": -2.0,
            "world": -2.1,
            "h": -3.0,
            "e": -3.1,
            "l": -3.2,
            "o": -3.3,
        }
        token_to_id = {k: i for i, k in enumerate(vocab.keys())}
        id_to_token = {i: k for k, i in token_to_id.items()}
        self.model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            special_tokens=["<|unk|>", "<|bos|>", "<|eos|>"],
            max_subword_len=16,
            byte_fallback=True,
        )
        self.normalizer = Normalizer()
        self.pre_tokenizer = RegexPreTokenizer()
        self.tokenizer = CustomTokenizer(
            model=self.model,
            normalizer=self.normalizer,
            pre_tokenizer=self.pre_tokenizer,
        )

    def test_binary_roundtrip_parity(self):
        """Verifies that mmap-loaded binary model produces bit-identical tokens and IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir) / "test_model.uniqtok"
            export_binary(self.tokenizer, bin_path)
            self.assertTrue(bin_path.exists())
            loaded = load_binary(bin_path, use_mmap=True)
            text = "hello world <|bos|>"
            ref_tokens = self.tokenizer.encode(text)
            ref_ids = self.tokenizer.encode_to_ids(text)
            loaded_tokens = loaded.encode(text)
            loaded_ids = loaded.encode_to_ids(text)
            self.assertEqual(ref_tokens, loaded_tokens)
            self.assertEqual(ref_ids, loaded_ids)

    def test_sub_millisecond_cold_start(self):
        """Asserts binary model loads in under 2 milliseconds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir) / "benchmark.uniqtok"
            export_binary(self.tokenizer, bin_path)
            # Measure loading latency
            times = []
            for _ in range(20):
                start = time.perf_counter()
                _ = load_binary(bin_path, use_mmap=True)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                times.append(elapsed_ms)
            median_ms = sorted(times)[len(times) // 2]
            self.assertLess(
                median_ms,
                2.0,
                f"Binary mmap load time took {median_ms:.2f}ms (expected < 2ms)",
            )

    def test_safe_fallback_to_json(self):
        """Verifies CustomTokenizer.load gracefully falls back to JSON when binary is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.tokenizer.save(tmp_path, save_binary=False)
            self.assertTrue((tmp_path / "tokenizer.json").exists())
            self.assertFalse((tmp_path / "tokenizer.uniqtok").exists())
            loaded = CustomTokenizer.load(tmp_path, prefer_binary=True)
            self.assertEqual(loaded.model.vocab_size, self.tokenizer.model.vocab_size)

    def test_corrupted_binary_fallback(self):
        """Verifies corrupted binary file falls back safely to tokenizer.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.tokenizer.save(tmp_path, save_binary=True)
            # Corrupt the binary file
            with open(tmp_path / "tokenizer.uniqtok", "wb") as f:
                f.write(b"CORRUPTED_BYTES_HERE")
            loaded = CustomTokenizer.load(tmp_path, prefer_binary=True)
            self.assertEqual(loaded.model.vocab_size, self.tokenizer.model.vocab_size)

    def test_non_mmap_mode(self):
        """Verifies binary loading works identically when use_mmap=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir) / "test.uniqtok"
            export_binary(self.tokenizer, bin_path)
            loaded = load_binary(bin_path, use_mmap=False)
            self.assertEqual(loaded.model.vocab_size, self.tokenizer.model.vocab_size)

    def test_sparse_id_save_fallback(self):
        """Verifies save() gracefully removes stale binary file when model has sparse IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.tokenizer.save(tmp_path, save_binary=True)
            self.assertTrue((tmp_path / "tokenizer.uniqtok").is_file())
            # Model with sparse IDs (0 and 5)
            sparse_model = UnigramModel(
                vocab={"a": 0.0, "b": -1.0},
                token_to_id={"a": 0, "b": 5},
                id_to_token={0: "a", 5: "b"},
                special_tokens=[],
            )
            sparse_tok = CustomTokenizer(
                model=sparse_model,
                normalizer=self.normalizer,
                pre_tokenizer=self.pre_tokenizer,
            )
            sparse_tok.save(tmp_path, save_binary=True)
            self.assertTrue((tmp_path / "tokenizer.json").is_file())
            self.assertFalse((tmp_path / "tokenizer.uniqtok").exists())
            # Verify the resulting JSON was actually overwritten with the sparse model
            loaded_sparse = CustomTokenizer.load(tmp_path)
            self.assertEqual(loaded_sparse.model.token_to_id, {"a": 0, "b": 5})

    def test_invalid_offsets_raise(self):
        """Verifies binary loader rejects invalid section offsets and out-of-bounds token offsets."""
        import struct

        with tempfile.TemporaryDirectory() as tmpdir:
            # Case 1: Corrupt scores_offset in header (u64 field at byte 32:
            # 8 magic + 4xI version/flags/vocab/maxlen + Q space_codepoint).
            bin_path = Path(tmpdir) / "corrupt_section_offset.uniqtok"
            export_binary(self.tokenizer, bin_path)
            with open(bin_path, "r+b") as f:
                f.seek(32)
                f.write(struct.pack("<Q", 999999))
            with self.assertRaises(ValueError):
                load_binary(bin_path, use_mmap=True)
            # Case 2: Corrupt token offset entry in offsets table to point outside string section
            bin_path2 = Path(tmpdir) / "corrupt_token_offset.uniqtok"
            export_binary(self.tokenizer, bin_path2)
            with open(bin_path2, "r+b") as f:
                # Read offsets_offset from header (byte 40)
                f.seek(40)
                (offsets_offset,) = struct.unpack("<Q", f.read(8))
                f.seek(offsets_offset)
                f.write(struct.pack("<II", 999999, 10))
            with self.assertRaises(ValueError):
                load_binary(bin_path2, use_mmap=True)
            with self.assertRaises(ValueError):
                load_binary(bin_path2, use_mmap=False)

    def test_inconsistent_vocab_raises(self):
        """Verifies binary export rejects models with inconsistent vocab and token_to_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir) / "inconsistent.uniqtok"
            inconsistent_model = UnigramModel(
                vocab={"a": 0.0},
                token_to_id={"a": 0, "b": 1},
                id_to_token={0: "a", 1: "b"},
                special_tokens=[],
            )
            bad_tok = CustomTokenizer(
                model=inconsistent_model,
                normalizer=self.normalizer,
                pre_tokenizer=self.pre_tokenizer,
            )
            with self.assertRaises(ValueError):
                export_binary(bad_tok, bin_path)

    def test_corrupted_space_codepoint_raises(self):
        """Verifies binary loader rejects invalid Unicode space codepoints."""
        import struct

        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir) / "bad_space.uniqtok"
            export_binary(self.tokenizer, bin_path)
            # Corrupt space_codepoint at byte 24 with 0x110000 (outside Unicode range)
            with open(bin_path, "r+b") as f:
                f.seek(24)
                f.write(struct.pack("<Q", 0x110000))
            with self.assertRaises(ValueError):
                load_binary(bin_path, use_mmap=True)

    def test_corrupted_binary_fallback_emits_warning(self):
        """Verifies falling back to JSON emits a UserWarning."""
        import warnings

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.tokenizer.save(tmp_path, save_binary=True)
            with open(tmp_path / "tokenizer.uniqtok", "wb") as f:
                f.write(b"CORRUPTED_BYTES_HERE")
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                loaded = CustomTokenizer.load(tmp_path, prefer_binary=True)
                self.assertEqual(loaded.model.vocab_size, self.tokenizer.model.vocab_size)
                self.assertTrue(any(issubclass(w.category, UserWarning) for w in recorded))

    def test_malformed_config_shape_fallback(self):
        """Verifies binary models with non-dict config or component configs fall back to JSON."""
        import struct

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.tokenizer.save(tmp_path, save_binary=True)
            bin_file = tmp_path / "tokenizer.uniqtok"
            with open(bin_file, "rb") as f:
                data = bytearray(f.read())
            cfg_off, cfg_len = struct.unpack_from("<QQ", data, 64)
            # Overwrite config JSON with a JSON array '[]' padded with spaces
            data[cfg_off : cfg_off + cfg_len] = b"[]".ljust(cfg_len, b" ")
            with open(bin_file, "wb") as f:
                f.write(data)
            with self.assertRaises(ValueError):
                load_binary(bin_file, use_mmap=True)
            # CustomTokenizer.load should safely catch ValueError and fall back to tokenizer.json
            loaded = CustomTokenizer.load(tmp_path, prefer_binary=True)
            self.assertEqual(loaded.model.vocab_size, self.tokenizer.model.vocab_size)


if __name__ == "__main__":
    unittest.main()
