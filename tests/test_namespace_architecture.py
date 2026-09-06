"""Architecture tests for the compat/train engine split (issue #49).

Verifies:
- the import structure promised by the issue acceptance criteria,
- that compat-loaded models never mutate token IDs and reject vocabulary
  mutation or re-ranking,
- that the two namespaces do not leak each other's engines.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from collections.abc import Mapping

import uniqtoken
import uniqtoken.compat
import uniqtoken.train
from uniqtoken import SuperBPE, UnigramTrainer
from uniqtoken.cem_merger import CrossEntropyMerging
from uniqtoken.compat import (
    FrozenCompatModel,
    SentencePieceCompat,
    TiktokenCompat,
    VocabularyMutationError,
    from_tiktoken,
)

try:
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram as HFUnigram

    HAS_TOKENIZERS = True
except ImportError:  # pragma: no cover
    HAS_TOKENIZERS = False


def _write_synthetic_ranks() -> str:
    """Builds a tiny but valid .tiktoken file: all 256 bytes + a few merges."""
    lines = []
    for b in range(256):
        lines.append(base64.b64encode(bytes([b])) + b" " + str(b).encode())
    for token, rank in [(b"th", 256), (b"e ", 257), (b" the", 258), (b"the", 259)]:
        lines.append(base64.b64encode(token) + b" " + str(rank).encode())
    fd, path = tempfile.mkstemp(suffix=".tiktoken")
    with os.fdopen(fd, "wb") as f:
        f.write(b"\n".join(lines) + b"\n")
    return path


class ImportStructureTests(unittest.TestCase):
    def test_compat_namespace_surface(self):
        self.assertTrue(issubclass(TiktokenCompat, uniqtoken.compat.TiktokenEncoding))
        self.assertIs(TiktokenCompat, uniqtoken.compat.TiktokenCompat)
        # Facade names are frozen loaders, not raw mutable importers.
        self.assertIsNot(uniqtoken.compat.HuggingFaceCompat, uniqtoken.compat.import_hf_tokenizer)
        self.assertIsNot(uniqtoken.compat.SentencePieceCompat, uniqtoken.compat.import_sentencepiece)
        for name in (
            "TiktokenCompat",
            "HuggingFaceCompat",
            "SentencePieceCompat",
            "from_tiktoken",
            "from_huggingface",
            "from_sentencepiece",
        ):
            self.assertTrue(callable(getattr(uniqtoken.compat, name)))

    def test_train_namespace_surface(self):
        from uniqtoken.train import (
            BPETrainer,
            CrossEntropyMerging,
            SeedVocabularyBuilder,
            UnigramLattice,
            UnigramModel,
            VocabularyAdapter,
        )

        self.assertIs(uniqtoken.train.UnigramTrainer, UnigramTrainer)

    def test_top_level_exports_superbpe_and_trainer(self):
        self.assertTrue(issubclass(SuperBPE, CrossEntropyMerging))

    def test_engines_do_not_leak_into_each_other(self):
        for research_name in ("UnigramTrainer", "SeedVocabularyBuilder", "SuperBPE", "BPETrainer"):
            self.assertFalse(hasattr(uniqtoken.compat, research_name))
        for compat_name in ("TiktokenCompat", "import_sentencepiece", "load_tiktoken_ranks"):
            self.assertFalse(hasattr(uniqtoken.train, compat_name))


class CompatFrozenModelTests(unittest.TestCase):
    def test_compat_model_never_mutates_token_ids(self):
        path = _write_synthetic_ranks()
        try:
            model = from_tiktoken(path, name="synthetic", pattern="gpt2")
            self.assertIsInstance(model, FrozenCompatModel)
            before = model.encode("the quick brown fox")
            self.assertEqual(model.encode("the quick brown fox"), before)
            self.assertEqual(model.decode(before), "the quick brown fox")
            # Zero token count delta / exact ID parity with the raw ranks:
            self.assertEqual(model.encode("the"), [259])
            self.assertEqual(model.encode(" the"), [258])
        finally:
            os.unlink(path)

    def test_compat_model_rejects_mutation_and_reranking(self):
        path = _write_synthetic_ranks()
        try:
            model = from_tiktoken(path, name="synthetic", pattern="gpt2")
            with self.assertRaises(VocabularyMutationError):
                model.ranks = {}
            with self.assertRaises(VocabularyMutationError):
                model.special_tokens = {"<|x|>": 99999}
            with self.assertRaises(VocabularyMutationError):
                model.train_from_corpus(["corpus"])
            # IDs unchanged after rejected mutation attempts.
            self.assertEqual(model.encode("the"), [259])
        finally:
            os.unlink(path)

    def test_superbpe_forces_cross_word(self):
        self.assertTrue(SuperBPE().cross_word)
        self.assertFalse(CrossEntropyMerging().cross_word)


class CompatDeepFreezeTests(unittest.TestCase):
    def test_delegated_state_is_read_only(self):
        path = _write_synthetic_ranks()
        try:
            model = from_tiktoken(path, name="synthetic", pattern="gpt2")
            # Mutable container types surface as immutable views.
            self.assertIsInstance(model.ranks, Mapping)
            self.assertIsInstance(model.special_tokens, Mapping)
            with self.assertRaises(TypeError):
                model.ranks[b"x"] = 999  # type: ignore[index]
            with self.assertRaises(TypeError):
                model.special_tokens["<|x|>"] = 99999  # type: ignore[index]
            # IDs unchanged after rejected in-place mutations.
            self.assertEqual(model.encode("the"), [259])
        finally:
            os.unlink(path)

    def test_tiktoken_compat_is_frozen_subclass(self):
        path = _write_synthetic_ranks()
        try:
            enc = TiktokenCompat.from_file(path, name="synthetic", pattern="gpt2")
            self.assertIsInstance(enc, uniqtoken.compat.TiktokenEncoding)
            with self.assertRaises(VocabularyMutationError):
                enc.name = "renamed"  # type: ignore[attr-defined]
            with self.assertRaises(TypeError):
                enc.ranks[b"x"] = 999  # type: ignore[index]
            self.assertEqual(enc.encode("the"), [259])
        finally:
            os.unlink(path)

    @unittest.skipUnless(HAS_TOKENIZERS, "tokenizers package not installed")
    def test_nested_model_state_is_frozen(self):
        """The HF/SentencePiece path returns a CustomTokenizer; its model's
        vocab mappings must be immutable through the compat surface."""
        from uniqtoken.compat import from_huggingface

        entries = (
            [("<|unk|>", -10.0)] + [(c, -4.0) for c in "helo wrd"] + [("\u2581hello", -1.0), ("\u2581world", -1.0)]
        )
        hf = Tokenizer(HFUnigram(entries, unk_id=0, byte_fallback=False))
        fm = from_huggingface(json.loads(hf.to_str()))
        self.assertIsInstance(fm, FrozenCompatModel)
        self.assertIsInstance(fm.model, FrozenCompatModel)
        before = fm.encode_to_ids("hello world")
        with self.assertRaises(TypeError):
            fm.model.vocab["injected"] = -99.0  # type: ignore[index]
        with self.assertRaises(TypeError):
            fm.model.token_to_id["injected"] = 99999  # type: ignore[index]
        self.assertEqual(fm.encode_to_ids("hello world"), before)


if __name__ == "__main__":
    unittest.main()
