"""Integration tests for the native HuggingFace ``PreTrainedTokenizerFast`` adapter.

Covers issue #45 acceptance criteria:

* ``uniqtoken.hf_adapter.UniqTokenizerFast`` subclasses ``PreTrainedTokenizerFast``.
* ``save_pretrained`` / ``from_pretrained`` round-trips exactly.
* Saves are re-discoverable through ``AutoTokenizer.from_pretrained``.
* Padding, truncation and ``return_tensors`` (``np``, and ``pt``/``tf`` when
  the framework is installed) work through the standard HF API surface.
* The exported ``tokenizer.json`` encodes identically to the source
  ``CustomTokenizer``.
* ``Trainer(tokenizer=tokenizer, ...)`` accepts the adapter without warning.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from uniqtoken import CustomTokenizer
from uniqtoken.hf_adapter import HAS_TRANSFORMERS, UniqTokenizerFast, register_tokenizer

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False

try:
    import torch  # type: ignore[import-untyped]

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

try:
    import tensorflow as tf  # type: ignore[import-untyped]

    HAS_TF = True
except ImportError:  # pragma: no cover
    HAS_TF = False

CORPUS = [
    "the quick brown fox jumps over the lazy dog and runs fast",
    "unigram language model training pipeline",
    "byte fallback handles emoji and rare scripts",
    "наука и техника для всех",
]

#: Probe texts covering ASCII, byte fallback (rare scripts/digits) and default
#: no-special-token behavior that CustomTokenizer.encode uses.
TEXTS = [
    "hello world",
    "the quick brown fox",
    "byte fallback handles æøå and emoji 🎉",
    "наука и техника",
    "digits 12345 and words",
]


def _train_tiny_tokenizer(chat_template=None) -> CustomTokenizer:
    tok = CustomTokenizer.train_from_corpus(
        CORPUS, target_vocab_size=320, hex_literals=False, digit_chunking="greedy", verbose=False
    )
    if chat_template is not None:
        tok.chat_template = chat_template
    return tok


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers package not installed")
class UniqTokenizerFastIntegrationTests(unittest.TestCase):
    """Core fast-tokenizer surface (mirrors TokenizerTesterMixin coverage)."""

    def setUp(self) -> None:
        self.ct = _train_tiny_tokenizer()
        self.tokenizer = UniqTokenizerFast.from_custom_tokenizer(self.ct)
        self.tmp_dir = tempfile.mkdtemp(prefix="uniqtoken_hf_")
        self.tokenizer.save_pretrained(self.tmp_dir)
        self.reloaded = UniqTokenizerFast.from_pretrained(self.tmp_dir)

    # -- adapter construction --------------------------------------------

    def test_subclasses_pre_trained_tokenizer_fast(self) -> None:
        from transformers import PreTrainedTokenizerFast

        self.assertTrue(issubclass(UniqTokenizerFast, PreTrainedTokenizerFast))
        self.assertTrue(self.tokenizer.is_fast)
        self.assertEqual(self.tokenizer.backend_tokenizer.__class__.__name__, "Tokenizer")

    def test_from_custom_tokenizer_rejects_non_tokenizer(self) -> None:
        with self.assertRaises(TypeError):
            UniqTokenizerFast.from_custom_tokenizer("not a tokenizer")  # type: ignore[arg-type]

    # -- special tokens --------------------------------------------------

    def test_special_tokens_wired(self) -> None:
        model = self.ct.model
        self.assertEqual(self.tokenizer.pad_token, "<|pad|>")
        self.assertEqual(self.tokenizer.bos_token, "<|bos|>")
        self.assertEqual(self.tokenizer.eos_token, "<|eos|>")
        self.assertEqual(self.tokenizer.unk_token, "<|unk|>")
        for token in ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"):
            self.assertEqual(self.tokenizer.convert_tokens_to_ids(token), model.token_to_id[token])
        # Extra special tokens (e.g. <|user|>) are carried over and convert to
        # their original IDs. v5 exposes these via ``extra_special_tokens``;
        # v4 uses ``additional_special_tokens`` and ``all_special_tokens``.
        extra_tokens = (
            set(getattr(self.tokenizer, "extra_special_tokens", []) or [])
            | set(getattr(self.tokenizer, "additional_special_tokens", []) or [])
            | set(getattr(self.tokenizer, "all_special_tokens", []) or [])
        )
        self.assertIn("<|user|>", extra_tokens)
        self.assertEqual(self.tokenizer.convert_tokens_to_ids("<|user|>"), model.token_to_id["<|user|>"])

    def test_means_id_consistency(self) -> None:
        for token in ("<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"):
            assert self.tokenizer._tokenizer is not None
            self.assertEqual(self.tokenizer.convert_tokens_to_ids(token), self.tokenizer._tokenizer.token_to_id(token))

    # -- encoding parity with the native tokenizer -----------------------

    def test_encode_matches_native_tokenizer(self) -> None:
        for tokenizer in (self.tokenizer, self.reloaded):
            for text in TEXTS:
                self.assertEqual(
                    tokenizer(text, add_special_tokens=False)["input_ids"],
                    self.ct.encode_to_ids(text),
                    f"hf adapter diverged from CustomTokenizer for {text!r}",
                )

    def test_encode_batch_matches_native_tokenizer(self) -> None:
        batch = self.tokenizer(TEXTS, add_special_tokens=False, padding=False)
        self.assertEqual(batch["input_ids"], self.ct.encode_to_ids_batch(TEXTS))

    def test_add_special_tokens_behavior(self) -> None:
        ids = self.tokenizer("hello world", add_special_tokens=True)["input_ids"]
        self.assertEqual(ids[0], self.tokenizer.bos_token_id)
        self.assertEqual(ids[-1], self.tokenizer.eos_token_id)
        no_special = self.tokenizer("hello world", add_special_tokens=False)["input_ids"]
        self.assertNotIn(self.tokenizer.bos_token_id, no_special)

    def test_decode_roundtrip(self) -> None:
        ids = self.tokenizer("hello world", add_special_tokens=False)["input_ids"]
        self.assertEqual(self.tokenizer.decode(ids, skip_special_tokens=True), "hello world")

    def test_vocab_consistency(self) -> None:
        self.assertEqual(self.tokenizer.vocab_size, self.ct.vocab_size)
        vocab = self.tokenizer.get_vocab()
        self.assertEqual(len(vocab), self.ct.vocab_size)
        token = "▁hello" if "▁hello" in vocab else list(vocab)[0]
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(vocab[token]), token)

    def test_offset_mapping(self) -> None:
        for text in TEXTS:
            offsets = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            for start, end in offsets:
                self.assertTrue(0 <= start <= end <= len(text), (start, end, text))

    def test_model_input_names(self) -> None:
        self.assertIn("input_ids", self.tokenizer.model_input_names)
        self.assertIn("attention_mask", self.tokenizer.model_input_names)
        # Fast causal-LM tokenizers don't require token_type_ids in v5;
        # v4 includes them by default in PreTrainedTokenizerFast.
        self.assertIn(
            list(self.tokenizer.model_input_names),
            [
                ["input_ids", "attention_mask"],
                ["input_ids", "token_type_ids", "attention_mask"],
            ],
        )

    # -- padding & truncation --------------------------------------------

    def test_padding_longest(self) -> None:
        encoded = self.tokenizer(TEXTS, padding="longest", add_special_tokens=False)
        longest = max(len(self.ct.encode_to_ids(t)) for t in TEXTS)
        for row in encoded["input_ids"]:
            self.assertEqual(len(row), longest)
        for row, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
            self.assertEqual(len(row), len(mask))

    def test_padding_max_length(self) -> None:
        encoded = self.tokenizer("hi", padding="max_length", max_length=16, add_special_tokens=False)
        self.assertEqual(len(encoded["input_ids"]), 16)

    def test_padding_bool_flag(self) -> None:
        encoded = self.tokenizer(["a", "bb"], padding=True, add_special_tokens=False)
        self.assertEqual(len(encoded["input_ids"][0]), len(encoded["input_ids"][1]))

    def test_padding_side(self) -> None:
        self.assertEqual(self.tokenizer.padding_side, "right")
        encoded = self.tokenizer("hi", padding="max_length", max_length=10, add_special_tokens=False)
        self.assertEqual(encoded["input_ids"][-1], self.tokenizer.pad_token_id)
        self.assertEqual(encoded["attention_mask"][0], 1)

    def test_truncation(self) -> None:
        encoded = self.tokenizer(TEXTS, truncation=True, max_length=5, add_special_tokens=False)
        for row in encoded["input_ids"]:
            self.assertLessEqual(len(row), 5)

    def test_truncation_side(self) -> None:
        self.tokenizer.truncation_side = "left"
        text = TEXTS[0]
        full = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        left = self.tokenizer(text, truncation=True, max_length=6, add_special_tokens=False)["input_ids"]
        self.assertEqual(left, full[-6:])
        self.tokenizer.truncation_side = "right"

    def test_padding_and_truncation_do_not_raise(self) -> None:
        encoded = self.tokenizer(TEXTS, padding=True, truncation=True, max_length=10, add_special_tokens=False)
        for row, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
            self.assertLessEqual(len(row), 10)
            self.assertEqual(len(row), len(mask))

    def test_padding_strategy_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.tokenizer("hi", padding="bogus")

    # -- return_tensors --------------------------------------------------

    @unittest.skipUnless(HAS_NUMPY, "numpy not installed")
    def test_return_tensors_np(self) -> None:
        encoded = self.tokenizer(TEXTS, padding=True, return_tensors="np", add_special_tokens=False)
        self.assertIsInstance(encoded["input_ids"], np.ndarray)
        self.assertIsInstance(encoded["attention_mask"], np.ndarray)
        self.assertEqual(encoded["input_ids"].shape, (len(TEXTS), encoded["input_ids"].shape[1]))
        self.assertEqual(encoded["attention_mask"].dtype, np.int64)

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_return_tensors_pt(self) -> None:
        encoded = self.tokenizer(TEXTS, padding=True, return_tensors="pt", add_special_tokens=False)
        self.assertTrue(hasattr(encoded["input_ids"], "shape"))
        self.assertEqual(encoded["input_ids"].dtype, torch.long)

    @unittest.skipUnless(HAS_TF, "tensorflow not installed")
    def test_return_tensors_tf(self) -> None:
        encoded = self.tokenizer(TEXTS, padding=True, return_tensors="tf", add_special_tokens=False)
        self.assertTrue(hasattr(encoded["input_ids"], "shape"))

    # -- save / load / hub wiring ----------------------------------------

    # -- Trainer compatibility -------------------------------------------

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_trainer_accepts_tokenizer(self) -> None:
        # Acceptance criterion: Trainer(tokenizer=tokenizer, ...) runs without
        # warning or error. Training is not executed; construction + tokenize
        # path is enough to prove the tokenizer plugs in.
        import torch
        import torch.nn as nn
        from transformers import Trainer, TrainingArguments

        class TinyModel(nn.Module):
            def __init__(self, vocab_size: int) -> None:
                super().__init__()
                self.embed = nn.Embedding(vocab_size, 16)
                self.head = nn.Linear(16, vocab_size)

            def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:  # type: ignore[override]
                return {"loss": self.head(self.embed(input_ids)).mean()}

        args = TrainingArguments(output_dir=tempfile.mkdtemp(), report_to=[], num_train_epochs=0)
        trainer = Trainer(model=TinyModel(self.tokenizer.vocab_size), args=args, tokenizer=self.tokenizer)  # type: ignore[call-arg]
        train_ids = self.tokenizer(TEXTS, padding="max_length", max_length=16, truncation=True, return_tensors="pt")
        self.assertEqual(trainer.tokenizer, self.tokenizer)  # type: ignore[attr-defined]
        self.assertEqual(train_ids["input_ids"].shape[1], 16)

    # -- chat template ---------------------------------------------------

    def test_chat_template_roundtrip(self) -> None:
        ct = _train_tiny_tokenizer(chat_template="chatml")
        tok = UniqTokenizerFast.from_custom_tokenizer(ct)
        self.assertEqual(
            tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False),
            "<|im_start|>user\nhi<|im_end|>\n",
        )
        with tempfile.TemporaryDirectory() as tmp:
            tok.save_pretrained(tmp)
            reloaded = UniqTokenizerFast.from_pretrained(tmp)
            self.assertIsNotNone(reloaded.chat_template)
            self.assertEqual(
                reloaded.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False),
                "<|im_start|>user\nhi<|im_end|>\n",
            )

    # -- persistence & ecosystem wiring -----------------------------------

    def test_save_pretrained_writes_standard_files(self) -> None:
        files = {p.name for p in Path(self.tmp_dir).iterdir()}
        self.assertIn("tokenizer.json", files)
        self.assertIn("tokenizer_config.json", files)
        config = json.loads((Path(self.tmp_dir) / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["tokenizer_class"], "UniqTokenizerFast")
        self.assertEqual(config["auto_map"]["AutoTokenizer"], "uniqtoken.hf_adapter.UniqTokenizerFast")
        self.assertEqual(config["bos_token"], "<|bos|>")
        self.assertEqual(config["eos_token"], "<|eos|>")
        self.assertEqual(config["pad_token"], "<|pad|>")

    def test_from_pretrained_matches_original(self) -> None:
        self.assertIsInstance(self.reloaded, UniqTokenizerFast)
        for text in TEXTS:
            with self.subTest(text=text):
                self.assertEqual(
                    self.reloaded(text)["input_ids"],
                    self.tokenizer(text)["input_ids"],
                )

    def test_backend_tokenizer_serializes_post_processor(self) -> None:
        # bos/eos must survive a save/load round trip (they live in the
        # tokenizer.json post-processor, which from_pretrained rebuilds).
        self.assertEqual(
            self.reloaded("hello world")["input_ids"],
            [
                self.reloaded.bos_token_id,
                *self.tokenizer("hello world", add_special_tokens=False)["input_ids"],
                self.reloaded.eos_token_id,
            ],
        )

    def test_export_to_huggingface_directory_loads(self) -> None:
        # A directory written by the (older) exporter entry point must be
        # loadable by the custom class too.
        with tempfile.TemporaryDirectory() as tmp:
            self.ct.export_to_huggingface(tmp)
            loaded = UniqTokenizerFast.from_pretrained(tmp)
            self.assertEqual(
                loaded("hello world", add_special_tokens=False)["input_ids"],
                self.ct.encode_to_ids("hello world"),
            )

    def test_auto_tokenizer_resolves_custom_class(self) -> None:
        from transformers import AutoTokenizer

        if not register_tokenizer():
            self.skipTest("this transformers version has no tokenizer class registry")
        auto = AutoTokenizer.from_pretrained(self.tmp_dir)
        self.assertIsInstance(auto, UniqTokenizerFast)
        for text in TEXTS:
            self.assertEqual(
                auto(text, add_special_tokens=False)["input_ids"],
                self.ct.encode_to_ids(text),
            )


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers package not installed")
class UniqTokenizerFastExporterWiringTests(unittest.TestCase):
    """tokenizer.json <-> adapter wiring through HuggingFaceExporter."""

    def test_exporter_writes_adapter_class_metadata(self) -> None:
        ct = _train_tiny_tokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            ct.export_to_huggingface(tmp)
            config = json.loads((Path(tmp) / "tokenizer_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["tokenizer_class"], "UniqTokenizerFast")
            self.assertEqual(config["model_type"], "uniqtoken")
            self.assertIn("UniqTokenizerFast", config["auto_map"]["AutoTokenizer"])

            backend_json = json.loads((Path(tmp) / "tokenizer.json").read_text(encoding="utf-8"))
            self.assertEqual(backend_json["model"]["type"], "Unigram")
            self.assertTrue(backend_json["model"]["byte_fallback"])
            self.assertIsNone(backend_json["post_processor"])

    def test_lazy_import_from_package_namespace(self) -> None:
        # ``from uniqtoken import UniqTokenizerFast`` must work through the
        # package-level lazy attr and keep the registered class identity.
        import uniqtoken

        self.assertIs(uniqtoken.UniqTokenizerFast, UniqTokenizerFast)
        self.assertTrue(uniqtoken.HAS_TRANSFORMERS)
        self.assertIs(uniqtoken.register_tokenizer, register_tokenizer)


if __name__ == "__main__":
    unittest.main()
