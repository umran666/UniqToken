"""
Unit and Parity Tests comparing uniqtoken_core Rust engine vs Pure-Python Lattice.
"""

from __future__ import annotations

import unittest
from math import log

from uniqtoken.tokenizer import CustomTokenizer
from uniqtoken.unigram_trainer import UnigramModel, UnigramTrainer

try:
    import uniqtoken_core as uniqtoken_core

    HAS_RUST = True
except ImportError:
    try:
        import uniqtoken_core  # type: ignore[no-redef]

        HAS_RUST = True
    except ImportError:
        uniqtoken_core = None  # type: ignore[assignment]
        HAS_RUST = False


class RustPythonParityTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "the quick brown fox jumps over the lazy dog",
            "machine learning and natural language processing in rust and python",
            "प्राकृतिक भाषा प्रसंस्करण नमस्ते दुनिया",
            "自然言語処理におけるトークナイザーの最適化",
            "تعتبر معالجة اللغات الطبيعية الحديثة",
            "def test_parity(a: int, b: float) -> str: return f'{a}_{b}'",
        ]
        self.tok = CustomTokenizer.train_from_corpus(
            self.corpus,
            target_vocab_size=380,
            ranking_strategy="char_savings",
            verbose=False,
        )

    def test_rust_core_is_detected_and_imported(self):
        self.assertTrue(HAS_RUST, "uniqtoken_core must be compiled and importable")
        self.assertTrue(hasattr(uniqtoken_core, "RustPrefixTrie"))
        self.assertTrue(hasattr(uniqtoken_core, "rust_viterbi_decode"))

    def test_rust_vs_python_exact_token_and_offset_parity(self):
        if not HAS_RUST:
            self.skipTest("uniqtoken_core not compiled")

        test_sentences = [
            "the quick brown fox",
            "machine learning in rust",
            "प्राकृतिक भाषा",
            "自然言語処理",
            "def step(self): pass",
            "rare_unicode_glyph_🚀_and_fallback_©",
            "1234567890 arithmetic and variables",
            "nested <0x41> literal byte sequences",
        ]

        model: UnigramModel = self.tok.model

        for sent in test_sentences:
            # 1. Native Rust decode
            rust_trie = model._get_rust_trie()
            self.assertIsNotNone(rust_trie)
            assert rust_trie is not None
            rust_spans = uniqtoken_core.rust_viterbi_decode(sent, rust_trie, model.byte_fallback)
            rust_tuples = [(s.token, s.start, s.end) for s in rust_spans]

            # 2. Pure Python lattice decode (bypassing Rust dispatch)
            from uniqtoken.unigram_lattice import UnigramLattice

            py_lattice = UnigramLattice(
                sent,
                model.vocab,
                max_subword_len=model.max_subword_len,
                byte_fallback=model.byte_fallback,
                trie=model._get_trie(),
            )
            py_edges, _ = py_lattice.viterbi_edges()
            py_tuples = [(t, e.start, e.end) for e in py_edges for t in e.tokens]

            self.assertEqual(
                rust_tuples,
                py_tuples,
                f"Rust/Python parity mismatch on sentence: {sent!r}\nRust: {rust_tuples}\nPy: {py_tuples}",
            )


if __name__ == "__main__":
    unittest.main()
