from __future__ import annotations

"""Differential tests for the HF tokenizer.json importer.

Builds synthetic tokenizers with the real `tokenizers` package, serializes to
tokenizer.json, imports into UniqToken, and compares vocab/IDs/encodes.
"""

import json
import unittest
import warnings
from typing import Any

from uniqtoken.hf_importer import import_hf_tokenizer

try:
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE as HFBPE
    from tokenizers.models import Unigram as HFUnigram

    HAS_TOKENIZERS = True
except ImportError:  # pragma: no cover
    HAS_TOKENIZERS = False


@unittest.skipUnless(HAS_TOKENIZERS, "tokenizers package not installed")
class HFUnigramImportTests(unittest.TestCase):
    hf_json: Any
    hf: Any
    caught: Any
    cal: Any

    @classmethod
    def setUpClass(cls):
        # vocab: unk + single chars + metaspace-prefixed words; ids == list index
        chars = list("helowrd")
        entries = (
            [("<|unk|>", -10.0)]
            + [(c, -4.0) for c in chars]
            + [
                ("\u2581hello", -1.0),
                ("\u2581world", -1.0),
                ("lo", -2.0),
            ]
        )
        hf = Tokenizer(HFUnigram(entries, unk_id=0, byte_fallback=False))
        try:
            hf.pre_tokenizer = pre_tokenizers.Metaspace(replacement="\u2581", prepend_scheme="never")
        except TypeError:  # older signature
            hf.pre_tokenizer = pre_tokenizers.Metaspace("\u2581", add_prefix_space=False)
        cls.hf_json = json.loads(hf.to_str())
        cls.hf = hf
        with warnings.catch_warnings(record=True) as cls.caught:
            warnings.simplefilter("always")
            cls.cal = import_hf_tokenizer(cls.hf_json)

    def test_vocab_and_ids_preserved(self):
        model = self.cal.model
        hf_vocab = {t: s for t, s in self.hf_json["model"]["vocab"]}
        hf_ids = {t: i for i, (t, _) in enumerate(self.hf_json["model"]["vocab"])}
        self.assertEqual(dict(model.vocab), hf_vocab)
        self.assertEqual(model.token_to_id, hf_ids)
        self.assertEqual(model.id_to_token[0], "<|unk|>")

    def test_exact_encode_parity(self):
        texts = ["hello world", "hello", "world hello", "hellohello"]
        for text in texts:
            ref = self.hf.encode(text).ids
            ours = self.cal.encode_to_ids(text)
            self.assertEqual(ours, ref, f"ID mismatch for {text!r}")

    def test_decode_roundtrip(self):
        ids = self.cal.encode_to_ids("hello world")
        # UniqToken's decoder converts the metaspace char back to a real space
        self.assertEqual(self.cal.decode(ids), "hello world")


@unittest.skipUnless(HAS_TOKENIZERS, "tokenizers package not installed")
class HFByteLevelBPEImportTests(unittest.TestCase):
    hf_json: Any
    hf: Any
    cal: Any

    @classmethod
    def setUpClass(cls):
        # GPT-2-style byte-level vocab: fixed merges over b"hello world" PLUS
        # full 256-byte coverage (real vocabs always have it).
        from uniqtoken.hf_importer import _bytes_to_unicode

        byte_map = _bytes_to_unicode()
        vocab = {
            "h": 0,
            "e": 1,
            "l": 2,
            "o": 3,
            "w": 4,
            "r": 5,
            "d": 6,
            "\u0120": 7,
            "ll": 8,
            "he": 9,
            "llo": 10,
            "or": 11,
            "ld": 12,
            "orld": 13,
            "hello": 14,
            "\u0120w": 15,
            "\u0120world": 16,
        }
        next_id = max(vocab.values()) + 1
        for b in range(256):
            ch = byte_map[b]
            if ch not in vocab:
                vocab[ch] = next_id
                next_id += 1
        merges = [
            ("h", "e"),
            ("l", "l"),
            ("ll", "o"),
            ("o", "r"),
            ("l", "d"),
            ("or", "ld"),
            ("he", "llo"),
            ("\u0120", "w"),
            ("\u0120w", "orld"),
        ]
        hf = Tokenizer(HFBPE(vocab=vocab, merges=merges))
        hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        hf.decoder = decoders.ByteLevel()
        cls.hf_json = json.loads(hf.to_str())
        cls.hf = hf
        cls.cal = import_hf_tokenizer(cls.hf_json)

    def test_returns_byte_level_encoder(self):
        from uniqtoken.hf_importer import HFByteLevelBPE

        self.assertIsInstance(self.cal, HFByteLevelBPE)

    def test_exact_encode_parity(self):
        texts = ["hello world", "hello", "hello world hello", "hello, world!", "helloworld"]
        for text in texts:
            ref = self.hf.encode(text).ids
            ours = self.cal.encode(text)
            self.assertEqual(ours, ref, f"ID mismatch for {text!r}")

    def test_decode_roundtrip(self):
        ids = self.cal.encode("hello world")
        self.assertEqual(self.cal.decode(ids), "hello world")
        self.assertEqual(self.hf.decode(ids), "hello world")

    def test_special_tokens(self):
        self.cal.special_tokens["<|endoftext|>"] = 100
        self.cal._id_to_special[100] = "<|endoftext|>"
        self.assertIn(100, self.cal.encode("x <|endoftext|> y", allowed_special="all"))


class HFImporterEdgeCasesTests(unittest.TestCase):
    def test_wordpiece_rejected_with_clear_error(self):
        data = {"model": {"type": "WordPiece", "vocab": {"a": 0}}}
        with self.assertRaises(NotImplementedError):
            import_hf_tokenizer(data)

    def test_wrong_model_type_rejected(self):
        with self.assertRaises(ValueError):
            import_hf_tokenizer({"model": {"type": "Unigram", "vocab": []}})

    def test_unsupported_normalizer_warns(self):
        if not HAS_TOKENIZERS:
            self.skipTest("tokenizers package not installed")
        chars = list("ab")
        entries = [("<|unk|>", -10.0), ("a", -1.0), ("b", -1.0)]
        hf = Tokenizer(HFUnigram(entries, unk_id=0, byte_fallback=False))
        hf.normalizer = None  # direct dict injection below instead
        data = json.loads(hf.to_str())
        data["normalizer"] = {"type": "Sequence", "normalizers": [{"type": "NFKC"}, {"type": "StripAccents"}]}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import_hf_tokenizer(data)
        self.assertTrue(any("StripAccents" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
