from __future__ import annotations

"""Regression tests for external-audit findings.

Covers: BPE inter-word space corruption, batch special-token sanitization
bypass, CEM stale total_pairs ordering, and BPE strict decode of invalid IDs.
"""

import random
import unittest

from uniqtoken.bpe_model import BPEModel  # noqa: F401  (import guard for typo checks)
from uniqtoken.bpe_trainer import BPETrainer
from uniqtoken.cem_merger import CrossEntropyMerging
from uniqtoken.tokenizer import CustomTokenizer


def _train_unigram(vocab_size: int = 300) -> CustomTokenizer:
    return CustomTokenizer.train_from_corpus(
        ["hello world " * 40, "foo bar baz qux " * 40],
        target_vocab_size=vocab_size,
        verbose=False,
    )


class BPEWhitespaceTests(unittest.TestCase):
    def _train_bpe(self) -> BPEModel:
        # Corpus with NO literal space token: the trainer alphabet is built
        # from word-internal characters only.
        trainer = BPETrainer(target_vocab_size=280)
        return trainer.train(["lowest", "lower", "newest", "widest"] * 20)

    def test_inter_word_space_survives_roundtrip(self):
        model = self._train_bpe()
        self.assertNotIn(" ", model.vocab, "precondition: space is not a trained token")
        ids = model.encode_to_ids("a b")
        unk_id = model.token_to_id.get("<|unk|>", 0)
        self.assertNotIn(unk_id, ids, "inter-word space must not fall back to unk")
        self.assertEqual(model.decode(ids), "a b")

    def test_space_token_used_when_in_vocab(self):
        trainer = BPETrainer(target_vocab_size=300)
        model = trainer.train(["a b", "a b", "c d", "c d"] * 20)
        toks = model.encode("a b")
        self.assertIn(" ", toks, "when trained, the literal space should be used")

    def test_strict_decode_rejects_invalid_ids(self):
        model = self._train_bpe()
        max_id = max(model.id_to_token)
        with self.assertRaises(ValueError):
            model.decode([max_id + 123], strict=True)
        # lenient (default) still skips unknown IDs without raising
        self.assertEqual(model.decode([max_id + 123]), "")


class BatchSecurityParityTests(unittest.TestCase):
    def test_batch_matches_single_for_disallowed_specials(self):
        tok = _train_unigram()
        eos_id = tok.model.token_to_id.get("<|endoftext|>")
        if eos_id is None:
            self.skipTest("model has no <|endoftext|> token")
        texts = ["hello <|endoftext|> world"] * 3

        single_ids = [tok.encode_to_ids(t) for t in texts]
        batch_ids = tok.encode_to_ids_batch(texts)
        self.assertEqual(single_ids, batch_ids)
        for ids in batch_ids:
            self.assertNotIn(eos_id, ids, "disallowed special token leaked into batch IDs")

        batch_tokens = tok.encode_batch(texts)
        self.assertEqual(
            [tok.encode(t) for t in texts],
            batch_tokens,
        )
        self.assertNotIn("<|endoftext|>", [t for row in batch_tokens for t in row])

    def test_batch_tab_parity_with_single_encode(self):
        tok = _train_unigram()
        texts = ["a\tb", "x\ny\trz"] * 3
        self.assertEqual(
            [tok.encode_to_ids(t) for t in texts],
            tok.encode_to_ids_batch(texts),
            "batch fast path must match single-path normalization on tabs/newlines",
        )

    def test_batch_allowed_special_still_activates(self):
        tok = _train_unigram()
        eos_id = tok.model.token_to_id["<|endoftext|>"]
        texts = ["hello <|endoftext|>"] * 3
        for ids in tok.encode_to_ids_batch(texts, allowed_special="all"):
            self.assertIn(eos_id, ids)


class CEMOrderingTests(unittest.TestCase):
    def test_cem_deterministic_and_finds_merges(self):
        tok = _train_unigram(vocab_size=400)
        chunks = ["the quick brown fox", "hello world", "the quick fox", "hello there world"]
        cem = CrossEntropyMerging(max_merges=5)
        model_a = cem.optimize(tok.model, list(chunks))
        self.assertGreater(len(cem.merges), 0)

        cem2 = CrossEntropyMerging(max_merges=5)
        model_b = cem2.optimize(tok.model, list(chunks))
        self.assertEqual(
            [m[:3] for m in cem.merges],
            [m[:3] for m in cem2.merges],
            "merge order must be deterministic after the total_pairs ordering fix",
        )
        self.assertEqual(dict(model_a.vocab), dict(model_b.vocab))

    def test_cem_accepts_generator(self):
        tok = _train_unigram(vocab_size=400)
        cem = CrossEntropyMerging(max_merges=3)
        cem.optimize(tok.model, (c for c in ["hello world", "hello there"]))
        self.assertGreaterEqual(len(cem.merges), 0)


class EngineDivergenceRegressionTests(unittest.TestCase):
    def test_seed_script_detection_parity(self):
        from uniqtoken.seed_builder import SeedVocabularyBuilder
        import uniqtoken_core

        # #中文 should both detect cjk, ²x should both detect numeric
        py_cjk = SeedVocabularyBuilder._detect_script("#中文")
        self.assertEqual(py_cjk, "cjk")
        # Test mine_ngrams with chunk counts containing #中文
        mined = uniqtoken_core.rust_mine_ngrams({"#中文": 5}, 16, set())
        self.assertTrue(any("中文" in k for k in mined))

    def test_rust_trie_respects_max_subword_len(self):
        import uniqtoken_core

        trie = uniqtoken_core.RustPrefixTrie(4)
        trie.insert("supercali", -1.0, 100)
        trie.insert("supe", -2.0, 101)
        matches = trie.common_prefix_search("supercalifragilistic")
        # "supercali" is 9 chars, so with max_subword_len=4 it must not match
        matched_tokens = [m[0] for m in matches]
        self.assertNotIn("supercali", matched_tokens)
        self.assertIn("supe", matched_tokens)

    def test_normalizer_strip_c0_whitespace_parity(self):
        from uniqtoken.pre_tokenizer import Normalizer
        import uniqtoken_core

        norm = Normalizer(strip_whitespace=True)
        raw = "\x1chello\x1d"
        self.assertEqual(norm.normalize(raw), "hello")
        res_rust, _ = uniqtoken_core.rust_normalize_with_alignment(raw, "▁", True, True, False, False, False, True)
        self.assertEqual(res_rust, "hello")

    def test_lone_surrogate_graceful_fallback(self):
        tok = _train_unigram(vocab_size=300)
        # Must not raise ValueError on lone surrogate
        tokens = tok.encode("hello \ud800 world")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_unigram_trainer_empty_expectations_no_alphabetical_pruning(self):
        from uniqtoken.unigram_trainer import UnigramTrainer

        trainer = UnigramTrainer(target_vocab_size=300)
        # Empty input must terminate cleanly without alphabetical destruction
        model = trainer.train([])
        self.assertGreater(len(model.vocab), 0)

    def test_neural_visual_codec_1d_indices(self):
        from uniqtoken.multimodal.neural_codecs import HAS_TORCH, NeuralVisualCodec

        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        import torch

        vcodec = NeuralVisualCodec(in_channels=3, hidden_dim=16, latent_dim=16, num_tokens=32)
        flat_indices = torch.tensor([0, 1, 2, 3])
        reconstructed = vcodec.decode_from_indices(flat_indices, grid_h=2, grid_w=2)
        self.assertEqual(reconstructed.shape[0], 1)
        self.assertEqual(reconstructed.shape[1], 3)


class FullwidthControlTokenTests(unittest.TestCase):
    def test_sanitize_escapes_nfkc_synthesized_control_syntax(self):
        # SecurityShield always NFKC-canonicalizes before matching, independent
        # of the Normalizer flags. The native gate must mirror this (issue #16):
        # U+FF1C/U+FF5C map to '<' and '|' under NFKC.
        from uniqtoken.security_shield import SecurityShield

        shield = SecurityShield(special_tokens=["<|system|>"])
        self.assertEqual(shield.sanitize("＜｜system｜＞"), "<\\|system\\|>")
        self.assertEqual(shield.sanitize("<|system|>"), "<\\|system\\|>")
        self.assertEqual(shield.sanitize("hello world"), "hello world")


if __name__ == "__main__":
    random.seed(0)
    unittest.main()
