"""Differential tests: fused native pipeline vs the full Python pipeline.

The native path (rust_encode_text_native / _batch) is only taken when
``CustomTokenizer._native_pipeline_kwargs`` gates it in. These tests pin the
gate's correctness: for every probed input the native-enabled tokenizer must
produce exactly the output of the Python-only pipeline, and gate-negative
configurations must route through Python automatically.
"""

from __future__ import annotations

import dataclasses
import random
from typing import List

import pytest

import uniqtoken.tokenizer as tokenizer_module
from uniqtoken.batch_collator import BatchCollator
from uniqtoken.pre_tokenizer import RegexPreTokenizer
from uniqtoken.tokenizer import CustomTokenizer

try:
    HAS_NATIVE = hasattr(tokenizer_module._native_core, "rust_encode_text_native")
except AttributeError:
    HAS_NATIVE = False

CORPUS = [
    "The transformer architecture relies on subword tokenization to compress sequence length.",
    "Exact offset alignment is essential for accurate span extraction and structured decoding.",
    "Cost is $1,499.99 for iPhone 15 Pro (visit https://apple.com, or email dev@apple.com)!",
    "Hex literals like 0xDEADBEEF and 0b1010 and numbers 12345678 stay unsplit.",
    "Emoji test: 👨‍👩‍👧‍👦 family and 👍🏽 thumbs up",
    "我喜欢自然语言处理 and नमस्ते दुनिया",
    "<|user|> Calculate 1.5e-10 + 42 = ? <|endoftext|>",
    "def compute_sum(a: int, b: int) -> int:\n    return a + b  # 100% precision",
]

TRICKY_TEXTS = [
    "",
    " ",
    "plain english sentence with spaces.",
    "NFKC ligature ﬁle ＜fullwidth＞ ｜ vertical bar",
    "<|endoftext|> smuggled <|custom|> tokens",
    "escaped control <\\|endoftext\\|> literal",
    "email me at dev@example.com or https://rust-lang.org/docs",
    "hex 0xDEADBEEF and bin 0b1010 and num 1234567",
    "emoji 👨‍👩‍👧‍👦 family 👍🏽 done",
    "cjk 我喜欢自然语言处理 and hindi हिन्दी",
    "metaspace literal ▁ inside text",
    "private use \ue000\ue001 characters",
    "tabs\tand\nnewlines\r\nand   multi  spaces",
    "…em dash—test",
    "<|>",
    "a<|b",
    "<|unclosed",
    "<|user|>",
]


class _NativeDisabled:
    """Context manager: temporarily disables the native fast path on a tokenizer."""

    def __init__(self, tok: CustomTokenizer):
        self._tok = tok

    def __enter__(self) -> CustomTokenizer:
        self._tok._native_pipeline_kwargs = lambda: None  # type: ignore[method-assign]
        return self._tok

    def __exit__(self, *exc: object) -> None:
        del self._tok._native_pipeline_kwargs


@pytest.mark.skipif(not HAS_NATIVE, reason="native rust_encode_text_native not available")
class TestNativePipelineParity:
    @staticmethod
    def _make() -> CustomTokenizer:
        return CustomTokenizer.train_from_corpus(
            corpus=CORPUS * 5, target_vocab_size=400, min_frequency=1, verbose=False
        )

    def _compare(self, tok: CustomTokenizer, texts: List[str]) -> None:
        """Every public API must agree bit-for-bit with the Python-only pipeline."""
        with _NativeDisabled(tok) as py_tok:
            ref_tokens = [py_tok.encode(t) for t in texts]
            ref_ids = [py_tok.encode_to_ids(t) for t in texts]
            ref_batch_ids = py_tok.encode_to_ids_batch(texts)
            ref_collator = BatchCollator(py_tok).batch_encode(texts, padding=False, add_special_tokens=False).input_ids

        for text, ref in zip(texts, ref_tokens):
            assert tok.encode(text) == ref, f"token mismatch for {text!r}"
        for text, ref_ids_item in zip(texts, ref_ids):
            assert tok.encode_to_ids(text) == ref_ids_item, f"id mismatch for {text!r}"
        assert tok.encode_to_ids_batch(texts) == ref_batch_ids
        assert BatchCollator(tok).batch_encode(texts, padding=False, add_special_tokens=False).input_ids == ref_collator

    def test_tricky_texts_parity(self) -> None:
        self._compare(self._make(), TRICKY_TEXTS)

    def test_corpus_parity(self) -> None:
        self._compare(self._make(), CORPUS)

    def test_fuzz_parity(self) -> None:
        rng = random.Random(1234)
        alphabet = [
            "a",
            "Z",
            "the",
            " ",
            "  ",
            ".",
            "0",
            "1",
            "42",
            "<",
            "|",
            ">",
            "<|",
            "|>",
            "<|endoftext|>",
            "｜",
            "＜",
            "ﬁ",
            "\ue000",
            "▁",
            "é",
            "我",
            "👨",
            "‍",
            "\t",
            "\n",
            "#",
            "@",
            "0x",
            "-",
            "",
        ]
        texts = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24))) for _ in range(500)]
        self._compare(self._make(), texts)

    def test_collator_default_settings_parity(self) -> None:
        tok = self._make()
        texts = TRICKY_TEXTS[2:10]
        with _NativeDisabled(tok) as py_tok:
            ref = BatchCollator(py_tok).batch_encode(texts, padding=True, add_special_tokens=True)
        got = BatchCollator(tok).batch_encode(texts, padding=True, add_special_tokens=True)
        assert got.input_ids == ref.input_ids
        assert got.attention_mask == ref.attention_mask
        assert got.tokens == ref.tokens

    def test_specials_allowed_bypasses_native(self) -> None:
        tok = self._make()
        text = "<|user|> hello world"
        assert tok._native_pipeline_kwargs() is not None  # gate is open by default
        with _NativeDisabled(tok) as py_tok:
            ref = py_tok.encode(text, allowed_special="all")
        assert tok.encode(text, allowed_special="all") == ref


class TestNativeGateSelection:
    """Gate must close for every config the native pipeline cannot reproduce."""

    def test_pretok_parity_matrix(self) -> None:
        assert RegexPreTokenizer()._native_pretok_parity is True
        assert RegexPreTokenizer(digit_chunk_size=3)._native_pretok_parity is False
        assert RegexPreTokenizer(hex_literals=False)._native_pretok_parity is False
        assert RegexPreTokenizer(split_digits=True)._native_pretok_parity is False
        assert RegexPreTokenizer(preset="code")._native_pretok_parity is False
        assert RegexPreTokenizer(preset="math")._native_pretok_parity is False
        assert RegexPreTokenizer(split_punctuation=False)._native_pretok_parity is False
        assert RegexPreTokenizer(space_char=" ")._native_pretok_parity is False

    def test_casefold_disables_native(self) -> None:
        tok = CustomTokenizer.train_from_corpus(
            corpus=CORPUS * 3, target_vocab_size=500, min_frequency=1, verbose=False
        )
        assert tok._native_pipeline_kwargs() is not None
        tok.normalizer = type(tok.normalizer)(casefold=True)
        assert tok._native_pipeline_kwargs() is None

    def test_non_pipe_specials_disables_native(self) -> None:
        tok = CustomTokenizer.train_from_corpus(
            corpus=CORPUS * 3, target_vocab_size=500, min_frequency=1, verbose=False
        )
        assert tok._native_pipeline_kwargs() is not None
        # Replace the model object (the documented cache-invalidation path) with
        # one carrying a special token that lacks the <|...|> form.
        tok.model = dataclasses.replace(
            tok.model,
            special_tokens=list(tok.model.special_tokens) + ["custom_token_without_pipes"],
        )
        assert tok._native_pipeline_kwargs() is None

    def test_batch_falls_back_when_any_text_needs_security(self) -> None:
        tok = CustomTokenizer.train_from_corpus(
            corpus=CORPUS * 3, target_vocab_size=500, min_frequency=1, verbose=False
        )
        texts = ["clean text", "<|endoftext|> smuggled", "more clean text"]
        with _NativeDisabled(tok) as py_tok:
            ref = py_tok.encode_to_ids_batch(texts)
        assert tok.encode_to_ids_batch(texts) == ref
