"""Issue #43: restrict number clumping to 1-3 digit chunks.

Default ``digit_chunking="block3"`` splits decimal numbers into 1-3 digit
blocks (LLaMA-3 / GPT-4 style) so positional place-value arithmetic works
cleanly, while hex/binary literals remain single chunks and ``encode_with_offsets``
spans still tile the raw text exactly.
"""

from __future__ import annotations

import unittest

from uniqtoken.pre_tokenizer import RegexPreTokenizer
from uniqtoken.tokenizer import CustomTokenizer

try:
    import uniqtoken_core as _core

    HAS_RUST = hasattr(_core, "rust_pre_tokenize")
except ImportError:
    try:
        import uniqtoken_core as _core  # type: ignore[no-redef]

        HAS_RUST = hasattr(_core, "rust_pre_tokenize")
    except ImportError:
        _core = None  # type: ignore[assignment]
        HAS_RUST = False


def chunks(pre_tokenizer: RegexPreTokenizer, text: str) -> list[str]:
    return [t.text for t in pre_tokenizer.pre_tokenize_with_offsets(text)]


class Block3DigitChunkingTests(unittest.TestCase):
    def test_default_is_block3(self) -> None:
        self.assertEqual(RegexPreTokenizer().digit_chunking, "block3")

    def test_long_integer_splits_into_1_to_3_digit_chunks(self) -> None:
        self.assertEqual(chunks(RegexPreTokenizer(), "1234567"), ["123", "456", "7"])
        # 2**64 - 1, the motivating example from the issue
        self.assertEqual(
            chunks(RegexPreTokenizer(), "18446744073709551615"),
            ["184", "467", "440", "737", "095", "516", "15"],
        )

    def test_decimal_number_chunks(self) -> None:
        self.assertEqual(chunks(RegexPreTokenizer(), "3.1415926"), ["3", ".", "141", "592", "6"])

    def test_hex_and_binary_literals_stay_intact(self) -> None:
        got = chunks(RegexPreTokenizer(), "val = 0x1A2B + 99999")
        self.assertIn("0x1A2B", got)
        self.assertNotIn("0x1A", got)
        self.assertEqual([c for c in got if c.isascii() and c.isdigit()], ["999", "99"])
        self.assertEqual(chunks(RegexPreTokenizer(), "0b10110"), ["0b10110"])

    def test_numbers_inside_words_split_too(self) -> None:
        self.assertEqual(
            chunks(RegexPreTokenizer(), "a1b22c333d4444"),
            ["a", "1", "b", "22", "c", "333", "d", "444", "4"],
        )

    def test_single_mode_matches_single_digits(self) -> None:
        pre_tokenizer = RegexPreTokenizer(digit_chunking="single")
        self.assertEqual(chunks(pre_tokenizer, "1234567"), ["1", "2", "3", "4", "5", "6", "7"])
        self.assertEqual(chunks(pre_tokenizer, "3.1415926"), ["3", ".", "1", "4", "1", "5", "9", "2", "6"])

    def test_greedy_mode_preserves_legacy_behavior(self) -> None:
        pre_tokenizer = RegexPreTokenizer(digit_chunking="greedy")
        self.assertEqual(chunks(pre_tokenizer, "1234567"), ["1234567"])
        self.assertEqual(chunks(pre_tokenizer, "99999"), ["99999"])
        self.assertEqual(chunks(pre_tokenizer, "3.1415926"), ["3", ".", "1415926"])

    def test_invalid_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RegexPreTokenizer(digit_chunking="bogus")  # type: ignore[arg-type]


class OffsetAlignmentTests(unittest.TestCase):
    TEXTS = [
        "1234567",
        "3.1415926",
        "val = 0x1A2B + 99999",
        "a1b22c333d4444e55555",
        "18446744073709551615 == 2**64 - 1",
    ]

    def test_pretokenizer_spans_tile_the_text_exactly(self) -> None:
        pre_tokenizer = RegexPreTokenizer()
        for text in self.TEXTS:
            tokens = pre_tokenizer.pre_tokenize_with_offsets(text)
            self.assertEqual(
                "".join(text[start:end] for start, end in (t.span for t in tokens)),
                text,
                msg=f"pre-tokenizer spans do not tile {text!r}",
            )
            for t in tokens:
                self.assertEqual(text[t.start : t.end], t.text)


@unittest.skipUnless(HAS_RUST, "uniqtoken_core native extension not available")
class RustParityTests(unittest.TestCase):
    TEXTS = [
        "1234567",
        "3.1415926",
        "val = 0x1A2B + 99999 and 0b10110",
        "a1b22c333d4444e55555",
        "18446744073709551615 == 2**64 - 1",
        "<|user|> compute 18446744073709551615 % 1000 <|end|>",
        "٠١٢٣٤٥٦٧٨٩",  # Unicode (Arabic-Indic) digits: \d must stay Unicode-aware
        " Kosten 1.234.567,89 €",
    ]

    def test_python_and_rust_chunks_are_identical(self) -> None:
        pre_tokenizer = RegexPreTokenizer()
        for text in self.TEXTS:
            self.assertEqual(
                pre_tokenizer.pre_tokenize(text),
                _core.rust_pre_tokenize(text),  # type: ignore[union-attr]
                msg=f"Python/Rust pre-tokenizer divergence on {text!r}",
            )

    def test_rust_spans_tile_the_text_exactly(self) -> None:
        pre_tokenizer = RegexPreTokenizer()
        for text in self.TEXTS:
            py_tokens = pre_tokenizer.pre_tokenize_with_offsets(text)
            py_chunks = [t.text for t in py_tokens]
            self.assertEqual(py_chunks, _core.rust_pre_tokenize(text))  # type: ignore[union-attr]
            self.assertEqual("".join(py_chunks), text)


class TokenizerOffsetTests(unittest.TestCase):
    tokenizer: CustomTokenizer

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = CustomTokenizer.train_from_corpus(
            corpus=[
                "count 1234567 items here",
                "value 3.1415926 is exact",
                "hex 0x1A2B plus 99999 bits",
                "big number 18446744073709551615 appears",
            ],
            target_vocab_size=300,
            min_frequency=1,
            verbose=False,
        )

    def test_encode_with_offsets_tiles_raw_text_exactly(self) -> None:
        for text in ["run 1234567 = 99999", "x 3.1415926 0x1A2B", "18446744073709551615"]:
            tokens = self.tokenizer.encode_with_offsets(text)
            self.assertGreater(len(tokens), 0)
            self.assertEqual(
                "".join(text[start:end] for start, end in (t.raw_span for t in tokens)),
                text,
                msg=f"encode_with_offsets spans do not tile {text!r}",
            )

    def test_trained_tokenizer_uses_block3_chunks(self) -> None:
        pre_tokens = self.tokenizer.pre_tokenizer.pre_tokenize(self.tokenizer.normalizer.normalize("value 1234567"))
        self.assertEqual(
            [chunk.lstrip("\u2581") for chunk in pre_tokens if chunk.lstrip("\u2581").isdigit()],
            ["123", "456", "7"],
        )


if __name__ == "__main__":
    unittest.main()
