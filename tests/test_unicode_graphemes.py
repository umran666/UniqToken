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

# Review follow-ups (PR #60): Python/Rust parity on grapheme edge cases.
_ZWJ = "\u200d"
_KA, _VIRAMA, _SSA = "\u0915", "\u094d", "\u0937"  # क ् ष
_RI_IN, _RI_N = "\U0001f1ee", "\U0001f1f3"  # regional indicators I, N
_MAN = "\U0001f468"  # man emoji
_KHMER_COENG_TEXT = "\u1797\u17b6\u17d2\u179a\u17c1\u17a2\u1784"  # ភាស្រៀង
PARITY_CASES = [
    ("conjunct intact", _KA + _VIRAMA + _SSA, [_KA + _VIRAMA + _SSA]),
    ("virama + ZWJ + consonant", _KA + _VIRAMA + _ZWJ + _SSA, [_KA + _VIRAMA + _ZWJ + _SSA]),
    # GB11: ZWJ only joins Extended_Pictographic on both sides, not letters.
    ("ZWJ between letters", "a" + _ZWJ + "b", ["a" + _ZWJ, "b"]),
    ("ZWJ before picto at start", _ZWJ + _MAN, [_ZWJ, _MAN]),
    # GB12/13: regional indicators pair into flags; odd tail stays separate.
    ("flag pair", _RI_IN + _RI_N, [_RI_IN + _RI_N]),
    ("two flags", _RI_IN + _RI_N + _RI_IN + _RI_N, [_RI_IN + _RI_N, _RI_IN + _RI_N]),
    ("odd flag tail", _RI_IN + _RI_N + _RI_IN, [_RI_IN + _RI_N, _RI_IN]),
    # GB9c needs InCB=Consonant on both sides; Khmer coeng does not qualify.
    ("khmer coeng splits", _KHMER_COENG_TEXT, ["\u1797\u17b6\u17d2", "\u179a\u17c1", "\u17a2\u1784"]),
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

    def test_grapheme_edge_cases(self) -> None:
        """Fixed Python/Rust divergences (PR #60 review follow-ups)."""
        for name, text, expected in PARITY_CASES:
            chunks = [t.text for t in self.pre.pre_tokenize_with_offsets(text)]
            self.assertEqual(chunks, expected, msg=f"{name}: got {chunks!r}")
            self.assertEqual("".join(chunks), text, msg=f"{name}: tiling broken")

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

    def test_seed_builder_no_partial_cluster_suffix(self) -> None:
        """Issue #41: mine_ngrams never produces partial grapheme clusters.

        Also covers the forced-Python fallback path (no native extension)
        by patching caliper_core to None so the pure-Python cluster-
        windowing code is exercised.
        """
        import uniqtoken.seed_builder as _sb

        # Force the pure-Python path by hiding the native extension.
        _original_core = _sb.caliper_core
        try:
            _sb.caliper_core = None
            builder = SeedVocabularyBuilder(target_vocab_size=500, min_frequency=1)
            # 'स्क' is SA (\u0938) + VIRAMA (\u094d) + KA (\u0915).
            # It forms an indivisible conjunct cluster. Neither the base consonant
            # nor the consonant+virama prefix should be emitted as separate tokens.
            conjunct_chunks = Counter({"\u0938\u094d\u0915": 1})
            ngrams_conjunct = builder.mine_ngrams(conjunct_chunks)
            self.assertIn("\u0938\u094d\u0915", ngrams_conjunct)
            self.assertNotIn("\u0938", ngrams_conjunct)
            self.assertNotIn("\u0938\u094d", ngrams_conjunct)

            chunks = Counter({"स्क": 1, "नमस्ते": 1, "การศึกษา": 1, "hello": 1, "123": 1})
            ngrams = builder.mine_ngrams(chunks)
            for token in ngrams:
                self.assertFalse(
                    _starts_with_combining(token),
                    msg=f"partial cluster ngram: {token!r}",
                )
        finally:
            _sb.caliper_core = _original_core

    def test_unicode_version_gap_mark(self) -> None:
        """Ensure Unicode 14+ combining marks (e.g. U+0897) are recognized via regex \\p{M}."""
        from uniqtoken.pre_tokenizer import _is_mark

        # U+0897 is ARABIC PEPET (Mn), added in Unicode 14.0
        self.assertTrue(_is_mark("\u0897"), "U+0897 must be recognized as combining mark")


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
        for text in INDIC_WORDS + THAI_WORDS + SENTENCES + [c[1] for c in PARITY_CASES]:
            py_chunks = [t.text for t in self.pre.pre_tokenize_with_offsets(text)]
            rust_chunks = _core.rust_pre_tokenize(text)  # type: ignore[union-attr]
            self.assertEqual(py_chunks, rust_chunks, msg=f"Python/Rust divergence on {text!r}")


if __name__ == "__main__":
    unittest.main()
