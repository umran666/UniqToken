"""
Unit tests for Downstream Model Pretraining & BPB Benchmark.
"""

from __future__ import annotations

import math
import unittest
from benchmarks.train_toy_transformer import (
    _split_documents,
    create_tokenizers,
    train_toy_transformer,
    PRETRAINING_CORPUS,
)


class DownstreamTransformerTests(unittest.TestCase):
    def test_downstream_tokenizers_creation(self):
        tokenizers = create_tokenizers(target_vocab=500)
        self.assertIn("UniqToken (Unigram)", tokenizers)
        self.assertIn("UniqToken (SuperBPE)", tokenizers)
        self.assertIn("Standard BPE", tokenizers)

    def test_downstream_pretraining_step_and_bpb(self):
        tokenizers = create_tokenizers(target_vocab=500)
        tok = tokenizers["UniqToken (SuperBPE)"]
        metrics = train_toy_transformer(
            tok,
            "UniqToken (SuperBPE)",
            PRETRAINING_CORPUS[:4],
            steps=5,
            seq_len=24,
            batch_size=2,
            dim=32,
            heads=2,
            layers=1,
        )
        self.assertGreater(metrics.total_tokens, 0)
        self.assertGreater(metrics.compression_ratio, 0.0)
        self.assertGreater(metrics.validation_evaluated_tokens, 0)
        self.assertTrue(math.isfinite(metrics.validation_loss))
        self.assertGreater(metrics.final_loss, 0.0)
        self.assertTrue(math.isfinite(metrics.final_loss))
        self.assertTrue(math.isfinite(metrics.bits_per_byte))
        # BPB must use the same held-out population as validation loss.
        expected_bpb = metrics.final_loss * metrics.evaluated_tokens / (metrics.evaluated_bytes * math.log(2.0))
        self.assertAlmostEqual(metrics.bits_per_byte, expected_bpb, places=6)

    def test_duplicate_documents_do_not_cross_validation_boundary(self):
        corpus = ["alpha", "beta", "gamma", "alpha", "beta", "gamma"]
        train_docs, validation_docs, test_docs = _split_documents(corpus)
        self.assertTrue(train_docs)
        self.assertTrue(validation_docs)
        self.assertTrue(test_docs)
        self.assertTrue(set(train_docs).isdisjoint(validation_docs))
        self.assertTrue(set(train_docs).isdisjoint(test_docs))
        self.assertTrue(set(validation_docs).isdisjoint(test_docs))

    def test_rejects_invalid_benchmark_inputs(self):
        tokenizers = create_tokenizers(target_vocab=500)
        tok = tokenizers["UniqToken (Unigram)"]
        with self.assertRaises(ValueError):
            train_toy_transformer(tok, "test", [], steps=1)
        with self.assertRaises(ValueError):
            train_toy_transformer(tok, "test", ["text"], steps=0)
        with self.assertRaises(ValueError):
            _split_documents(["same", "same"])


if __name__ == "__main__":
    unittest.main()
