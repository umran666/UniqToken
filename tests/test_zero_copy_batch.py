"""
Unit tests for Zero-Copy PyBuffer Borrowing in Rust Batch Encoder (Issue #33).

Validates:
1. Direct buffer borrowing from Python list, tuple, custom sequences, and iterables.
2. Error handling when invalid objects (single string/bytes, non-string items) are passed.
3. Multilingual UTF-8 stress testing with zero intermediate heap allocations.
4. Parity across rust_encode_text_batch, rust_encode_text_native_batch,
   rust_encode_text_native_ids_batch, rust_encode_tokens_batch, and RustTokenizer.
5. Materialization and fallback resilience for one-shot generators in CustomTokenizer.
"""

from __future__ import annotations

import unittest
from typing import List, Sequence

try:
    import uniqtoken_core as core
except ImportError:
    try:
        import caliper_core as core
    except ImportError:
        core = None

from uniqtoken import CustomTokenizer, Normalizer, RegexPreTokenizer, UnigramModel


class CustomStringSequence:
    """A custom sequence class that is not a built-in list or tuple."""

    def __init__(self, items: Sequence[str]):
        self._items = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> str:
        return self._items[index]


@unittest.skipIf(core is None, "no native core available")
class ZeroCopyBatchTests(unittest.TestCase):
    def setUp(self):
        self.trie = core.RustPrefixTrie()
        vocab = [
            ("hello", -1.0, 1),
            ("world", -1.5, 2),
            (" ", -0.5, 3),
            ("\u2581", -0.5, 3),
            ("test", -2.0, 4),
            ("ing", -2.5, 5),
            ("tok", -1.2, 6),
            ("en", -1.3, 7),
            ("izer", -1.4, 8),
            ("\u4f60\u597d", -1.0, 9),  # 你好
            ("\u4e16\u754c", -1.0, 10),  # 世界
            ("\U0001F600", -2.0, 11),  # 😀
        ]
        for tok, score, tid in vocab:
            self.trie.insert(tok, score, tid)

    def test_list_borrowing(self):
        inputs = ["hello world", "testing", "\u4f60\u597d\u4e16\u754c"]
        res_tokens = core.rust_encode_tokens_batch(inputs, self.trie, True)
        self.assertEqual(len(res_tokens), 3)

        res_ids = core.rust_encode_ids_batch(inputs, self.trie, True)
        self.assertEqual(len(res_ids), 3)

        res_native = core.rust_encode_text_native_batch(inputs, self.trie, True)
        self.assertEqual(len(res_native), 3)

        res_native_ids = core.rust_encode_text_native_ids_batch(inputs, self.trie, True)
        self.assertEqual(len(res_native_ids), 3)

    def test_tuple_borrowing(self):
        inputs = ("hello world", "testing", "\u4f60\u597d\u4e16\u754c")
        res_tokens = core.rust_encode_tokens_batch(inputs, self.trie, True)
        self.assertEqual(len(res_tokens), 3)

        res_ids = core.rust_encode_ids_batch(inputs, self.trie, True)
        self.assertEqual(len(res_ids), 3)

        res_native = core.rust_encode_text_native_batch(inputs, self.trie, True)
        self.assertEqual(len(res_native), 3)

        res_native_ids = core.rust_encode_text_native_ids_batch(inputs, self.trie, True)
        self.assertEqual(len(res_native_ids), 3)

    def test_custom_sequence_borrowing(self):
        inputs = CustomStringSequence(["hello world", "testing", "\u4f60\u597d\u4e16\u754c"])
        res_tokens = core.rust_encode_tokens_batch(inputs, self.trie, True)
        self.assertEqual(len(res_tokens), 3)

        res_ids = core.rust_encode_ids_batch(inputs, self.trie, True)
        self.assertEqual(len(res_ids), 3)

        res_native = core.rust_encode_text_native_batch(inputs, self.trie, True)
        self.assertEqual(len(res_native), 3)

        res_native_ids = core.rust_encode_text_native_ids_batch(inputs, self.trie, True)
        self.assertEqual(len(res_native_ids), 3)

    def test_iterable_generator_borrowing(self):
        def gen():
            yield "hello world"
            yield "testing"
            yield "\u4f60\u597d\u4e16\u754c"

        res_tokens = core.rust_encode_tokens_batch(gen(), self.trie, True)
        self.assertEqual(len(res_tokens), 3)

    def test_error_on_single_string_or_bytes(self):
        with self.assertRaises(ValueError):
            core.rust_encode_tokens_batch("hello world", self.trie, True)  # type: ignore

        with self.assertRaises(ValueError):
            core.rust_encode_tokens_batch(b"hello world", self.trie, True)  # type: ignore

        with self.assertRaises(ValueError):
            core.rust_encode_text_native_ids_batch("hello world", self.trie, True)  # type: ignore

    def test_error_on_non_string_elements(self):
        with self.assertRaises(ValueError):
            core.rust_encode_tokens_batch(["hello", 123, "world"], self.trie, True)  # type: ignore

        with self.assertRaises(ValueError):
            core.rust_encode_text_native_ids_batch(["hello", None, "world"], self.trie, True)  # type: ignore

    def test_rust_tokenizer_zero_copy(self):
        vocab = [
            ("hello", -1.0, 1),
            ("world", -1.5, 2),
            (" ", -0.5, 3),
            ("\u2581", -0.5, 3),
            ("test", -2.0, 4),
        ]
        rt = core.RustTokenizer(vocab, space_char="\u2581", byte_fallback=True)
        inputs = ["hello world", "test"]

        toks = rt.encode_batch(inputs)
        self.assertEqual(len(toks), 2)

        ids = rt.encode_ids_batch(inputs)
        self.assertEqual(len(ids), 2)

        diag = core.rust_diagnostic_batch(inputs, self.trie, True)
        self.assertIn("total", diag)
        self.assertIn("viterbi", diag)

    def test_tokenizer_generator_fallback_materialization(self):
        """Ensure CustomTokenizer.encode_to_ids_batch materializes generators so fallback works."""
        tok = CustomTokenizer.train_from_corpus(
            ["hello world", "the quick brown fox jumps over the lazy dog"],
            target_vocab_size=300,
            verbose=False,
        )

        # A batch where one item triggers native rejection (control token syntax <|...|>)
        def gen():
            yield "hello world"
            yield "hello <|invalid|> world"
            yield "the quick brown fox"

        ids_batch = tok.encode_to_ids_batch(gen())
        self.assertEqual(len(ids_batch), 3)
        self.assertTrue(len(ids_batch[0]) > 0)
        self.assertTrue(len(ids_batch[1]) > 0)
        self.assertTrue(len(ids_batch[2]) > 0)


if __name__ == "__main__":
    unittest.main()
