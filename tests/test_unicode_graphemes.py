"""Issue #41: UAX #29 grapheme cluster boundaries for Indic/Thai scripts."""

from __future__ import annotations

import unicodedata
import unittest
from collections import Counter

from uniqtoken.pre_tokenizer import RegexPreTokenizer
from uniqtoken.seed_builder import SeedVocabularyBuilder

try:
    import uniqtoken_core as _core

    HAS_RUST = hasattr(_core, "rust_pre_tokenize")
except ImportError:
    try:
        import caliper_core as _core  # type: ignore[no-redef]

        HAS_RUST = hasattr(_core, "rust_pre_tokenize")
    except ImportError:
        _core = None  # type: ignore[assignment]
        HAS_RUST = False

INDIC_WORDS = ["नमस्ते", "శ్రీశైలం", "தமிழ்", "বাংলা"]
THAI_WORDS = ["การศึกษา", "เชียงใหม่"]
SENTENCES = [
    "नमस्ते दुनिया",
    "శ్రీశైలం దేవస్థానం",
    "தமிழ் மொழி",
    "বাংলা ভাষা",
    "การศึกษาคืออนาคต",
    "เชียงใหม่สวยงาม",
    "नमस्ते การศึกษา hello 123",
]


def _starts_with_combining(token: str) -> bool:
    return bool(token) and unicodedata.category(token[0]) in ("Mn", "Mc", "Me")


class GraphemeBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pre = RegexPreTokenizer()

    def test_indic_no_orphan_combining_marks(self) -> None:
        for word in INDIC_WORDS:
            for tok in self.pre.pre_tokenize_with_offsets(word):
                self.assertFalse(
                    _starts_with_combining(tok.text),
                    msg=f"orphan combining mark in {word!r}: {tok.text!r}",
                )

    def test_thai_no_orphan_combining_marks(self) -> None:
        for word in THAI_WORDS:
            for tok in self.pre.pre_tokenize_with_offsets(word):
                self.assertFalse(
                    _starts_with_combining(tok.text),
                    msg=f"orphan combining mark in {word!r}: {tok.text!r}",
                )

    def test_span_tiling_exact(self) -> None:
        for text in INDIC_WORDS + THAI_WORDS + SENTENCES:
            toks = self.pre.pre_tokenize_with_offsets(text)
            self.assertEqual("".join(text[t.start : t.end] for t in toks), text, msg=f"tiling failed for {text!r}")
            for t in toks:
                self.assertEqual(text[t.start : t.end], t.text)

    def test_hex_binary_numbers_words_intact(self) -> None:
        self.assertIn("0x1A2B", [t.text for t in self.pre.pre_tokenize_with_offsets("val = 0x1A2B")])
        self.assertIn("0b10110", [t.text for t in self.pre.pre_tokenize_with_offsets("0b10110")])
        self.assertEqual(
            [t.text for t in self.pre.pre_tokenize_with_offsets("1234567")],
            ["123", "456", "7"],
        )
        self.assertIn("hello", [t.text for t in self.pre.pre_tokenize_with_offsets("hello")])

    def test_seed_builder_never_emits_standalone_combining_mark(self) -> None:
        builder = SeedVocabularyBuilder(target_vocab_size=500, min_frequency=1)
        chunks = Counter(INDIC_WORDS + THAI_WORDS + ["hello", "123"])
        for entry in builder.collect_base_alphabet(chunks):
            self.assertFalse(
                _starts_with_combining(entry.token),
                msg=f"alphabet orphan: {entry.token!r}",
            )
        ngrams = builder.mine_ngrams(chunks)
        for token in ngrams:
            self.assertFalse(
                _starts_with_combining(token),
                msg=f"ngram orphan: {token!r}",
            )


@unittest.skipUnless(HAS_RUST, "uniqtoken_core native extension not available")
class RustGraphemeParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pre = RegexPreTokenizer()

    def test_rust_no_orphan_combining_marks(self) -> None:
        assert _core is not None
        for word in INDIC_WORDS + THAI_WORDS:
            for chunk in _core.rust_pre_tokenize(word):  # type: ignore[union-attr]
                self.assertFalse(_starts_with_combining(chunk), msg=f"rust orphan in {word!r}: {chunk!r}")

    def test_rust_span_tiling_exact(self) -> None:
        assert _core is not None
        for text in INDIC_WORDS + THAI_WORDS + SENTENCES:
            chunks = _core.rust_pre_tokenize(text)  # type: ignore[union-attr]
            self.assertEqual("".join(chunks), text, msg=f"rust tiling failed for {text!r}")

    def test_python_rust_parity(self) -> None:
        assert _core is not None
        for text in INDIC_WORDS + THAI_WORDS + SENTENCES:
            py_chunks = [t.text for t in self.pre.pre_tokenize_with_offsets(text)]
            rust_chunks = _core.rust_pre_tokenize(text)  # type: ignore[union-attr]
            self.assertEqual(py_chunks, rust_chunks, msg=f"Python/Rust divergence on {text!r}")


if __name__ == "__main__":
    unittest.main()
