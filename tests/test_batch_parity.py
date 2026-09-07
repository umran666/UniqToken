from __future__ import annotations

import unittest
from uniqtoken.pre_tokenizer import Normalizer
from uniqtoken.tokenizer import CustomTokenizer


class BatchParityTests(unittest.TestCase):
    def setUp(self):
        corpus = ["the quick brown fox " * 5 for _ in range(20)]
        self.tok = CustomTokenizer.train_from_corpus(corpus, target_vocab_size=800, verbose=False)
        self.norm = Normalizer()
        self.texts = [
            "the quick brown fox jumps",
            "hello world",
            "  a  b  ",
            "visit https://example.com",
            "fi \u00a0 test",
            "A\u030a",
            "\u0928\u092e\u0938\u094d\u0924\u0947",
        ]

    def test_model_encode_vs_batch(self):
        chunks = []
        for t in self.texts:
            n = self.norm.normalize(t)
            chunks.extend(self.tok.pre_tokenizer.pre_tokenize(n))
        batch = self.tok.model.encode_batch(chunks)
        single = [self.tok.model.encode(c) for c in chunks]
        self.assertEqual(batch, single)

    def test_tokenizer_encode_vs_batch(self):
        # encode uses encode_batch internally now; verify against manual per-chunk
        for t in self.texts:
            n = self.norm.normalize(t)
            chunks = self.tok.pre_tokenizer.pre_tokenize(n)
            manual = []
            for c in chunks:
                if c in self.tok.model.special_tokens:
                    manual.append(c)
                else:
                    manual.extend(self.tok.model.encode(c))
            # apply cross merges
            expected = self.tok._apply_cross_word_merges(manual)
            got = self.tok.encode(t)
            self.assertEqual(got, expected)

    def test_offsets_vs_batch(self):
        # encode_with_offsets per text vs per chunk spans
        for t in self.texts:
            a = self.tok.encode_with_offsets(t)
            b = self.tok.encode(t)
            self.assertEqual([x.text for x in a], b)
            if a:
                for tok in a:
                    self.assertGreaterEqual(tok.raw_span[0], 0)
                    self.assertLessEqual(tok.raw_span[1], len(t))
                for i in range(len(a) - 1):
                    self.assertLessEqual(a[i].raw_span[0], a[i + 1].raw_span[0])

    def test_rust_token_vs_span_path(self):
        try:
            import uniqtoken_core as uniqtoken_core
        except ImportError:
            try:
                import caliper_core as uniqtoken_core
            except ImportError:
                self.skipTest("no native core")

        if not hasattr(uniqtoken_core, "rust_encode_tokens_batch"):
            self.skipTest("rust_encode_tokens_batch not built")
        trie = self.tok.model._get_rust_trie()
        if trie is None:
            self.skipTest("no trie")
        chunks = ["the", " quick", " brown"]
        toks = uniqtoken_core.rust_encode_tokens_batch(chunks, trie, True)
        spans = uniqtoken_core.rust_viterbi_decode_batch(chunks, trie, True)
        self.assertEqual([s.token for s in spans[0]], toks[0])


if __name__ == "__main__":
    unittest.main()
