"""Predeclared matrices and release-only timing checks for residual native work."""

from pathlib import Path
import unittest

from benchmarks import profile_residual_native as p


class ResidualNativeProfileTests(unittest.TestCase):
    def test_predeclared_matrix_has_all_lengths_scripts_vocabularies_and_batches(self):
        rows = list(p.specifications({b: Path(f"model-{b}") for b in (8192, 16384, 32768)}))
        segmentation = [r for r in rows if r["surface"] == "segmentation"]
        batches = [r for r in rows if r["surface"] == "batch"]
        self.assertEqual(len(segmentation), 3 * 3 * 5)
        self.assertEqual(len(batches), 3 * 4 * 2 * 2)
        self.assertEqual({r["batch_size"] for r in batches}, {1, 8, 32, 128})
        self.assertEqual({r["cache"] for r in batches}, {"warm", "cold"})
        self.assertEqual({r["output"] for r in batches}, {"strings", "ids"})

    def test_fixtures_are_exact_fixed_length_and_long_runs_skip_existing_cache(self):
        rows = p.specifications({b: Path(f"model-{b}") for b in (8192, 16384, 32768)})
        for row in rows:
            texts = p.fixture(row)
            self.assertEqual(len(texts), row["batch_size"])
            self.assertTrue(all(len(text) == p.LENGTHS[row["length"]] for text in texts))
            if row["length"] == "long":
                self.assertTrue(all(len(text.encode("utf-8")) > 1024 for text in texts))
                self.assertTrue(all(not any(char.isspace() for char in text) for text in texts))

    def test_peak_rss_is_positive_process_memory(self):
        self.assertGreater(p.peak_rss(), 0)

    def test_span_decode_and_grouped_fallback_score_are_exact(self):
        tokens = ["a", "<0xE4>", "<0xB8>", "<0xAD>"]
        ids = dict(zip(tokens, range(4)))
        scores = {"a": -1.0, "<0xE4>": -2.0, "<0xB8>": -3.0, "<0xAD>": -4.0}
        stream = [(token, ids[token], 0 if token == "a" else 1, 1 if token == "a" else 2) for token in tokens]
        self.assertEqual(p.decode_pieces(tokens), "a\u4e2d")
        self.assertEqual(p.validate_spans("a\u4e2d", stream, ids, scores), (-10.0).hex())
        with self.assertRaisesRegex(ValueError, "token/ID"):
            p.validate_spans("a\u4e2d", [("a", 999, 0, 1), *stream[1:]], ids, scores)
        with self.assertRaisesRegex(ValueError, "tile"):
            p.validate_spans("a\u4e2d", [("a", 0, 1, 2)], ids, scores)


if __name__ == "__main__":
    unittest.main()
