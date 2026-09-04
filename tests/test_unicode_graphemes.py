"""Issue #41: UAX #29 grapheme cluster boundaries for Indic/Thai scripts."""

from __future__ import annotations

import unicodedata
import unittest
from collections import Counter

from uniqtoken.pre_tokenizer import RegexPreTokenizer, _is_mark
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
        """mine_ngrams must emit complete grapheme clusters only — no suffix
        ending inside a virama-linked cluster like 'स्' (SA+VIRAMA without KA)."""
        # स्क = SA + VIRAMA + KA is a single extended grapheme cluster (GB9c).
        # Codepoint slicing could emit 'स' or 'स्' as partial clusters.
        builder = SeedVocabularyBuilder(target_vocab_size=500, min_frequency=1)
        chunks = Counter(["\u0938\u094d\u0915"])  # स्क
        ngrams = builder.mine_ngrams(chunks)
        self.assertIn("\u0938\u094d\u0915", ngrams, msg="full cluster स्क must be emitted")
        self.assertNotIn("\u0938", ngrams, msg="partial prefix 'स' must not be emitted")
        self.assertNotIn("\u0938\u094d", ngrams, msg="partial suffix 'स्' must not be emitted")

    def test_unicode_version_gap_parity(self) -> None:
        """U+0897 (NKO LETTER JA) is Mn in Unicode 16.0 but unassigned in 13.0
        (Python 3.9). With regex as a required dependency, both Python and Rust
        classify it as a combining mark, so n-grams starting with it are skipped."""
        self.assertTrue(_is_mark("\u0897"), msg="U+0897 must be classified as Mn")
        builder = SeedVocabularyBuilder(target_vocab_size=500, min_frequency=1)
        chunks = Counter(["\u0897a"])  # U+0897 followed by a base
        ngrams = builder.mine_ngrams(chunks)
        for token in ngrams:
            self.assertFalse(token.startswith("\u0897"), msg=f"n-gram starts with U+0897: {token!r}")


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

    def test_rust_mine_ngrams_parity(self) -> None:
        """Rust rust_mine_ngrams must skip n-grams starting with U+0897 just like
        the Python fallback — both use the same current-Unicode data."""
        assert _core is not None
        self.assertTrue(_is_mark("\u0897"))
        chunks = {"\u0897a": 1}
        builder = SeedVocabularyBuilder(target_vocab_size=500, min_frequency=1)
        py_ngrams = builder.mine_ngrams(Counter(chunks))
        rust_ngrams = _core.rust_mine_ngrams(chunks, 16, set())  # type: ignore[union-attr]
        for token in py_ngrams:
            self.assertFalse(token.startswith("\u0897"), msg=f"python ngram starts with U+0897: {token!r}")
        for token in rust_ngrams:
            self.assertFalse(token.startswith("\u0897"), msg=f"rust ngram starts with U+0897: {token!r}")


if __name__ == "__main__":
    unittest.main()
