"""Paired uncertainty and independent dense-prefix stress protocol."""

from pathlib import Path
import tempfile
import unittest

from benchmarks import compare_residual_native as c
from benchmarks import run_research_experiments as h


class CompareResidualNativeTests(unittest.TestCase):
    def test_paired_interval_preserves_constant_ratio_and_rejects_invalid_samples(self):
        self.assertEqual(c.interval([2.0] * 11), [2.0, 2.0])
        for samples in ([], [0.0], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                c.interval(samples)
        self.assertEqual(c.interval([0.5, 1.0, 2.0]), c.interval([0.5, 1.0, 2.0]))

    def test_complete_matrix_and_dense_prefix_bound(self):
        rows = list(c.specs({b: Path(str(b)) for b in (8192, 16384, 32768)}, Path("stress")))
        self.assertEqual(len(rows), 97)
        self.assertEqual(
            len(
                {
                    (
                        r["surface"],
                        r["vocab_budget"],
                        r["length"],
                        r["script"],
                        r["batch_size"],
                        r["output"],
                        r["cache"],
                        r.get("characters"),
                    )
                    for r in rows
                }
            ),
            97,
        )
        self.assertEqual([len(c.fixture(row)[0]) for row in rows[-4:]], [32, 256, 4096, 16384])

    def test_synthetic_fixture_has_all_prefixes_controls_and_fallback_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.json"
            c.synthetic_model(model)
            payload = h.read_json(model)
        self.assertEqual(len(payload["vocab"]), 276)
        self.assertEqual(set(payload["token_to_id"].values()), set(range(276)))
        self.assertEqual(payload["max_subword_len"], 16)
        self.assertTrue(all("a" * length in payload["vocab"] for length in range(1, 17)))
        self.assertTrue(all(f"<0x{byte:02X}>" in payload["vocab"] for byte in range(256)))


if __name__ == "__main__":
    unittest.main()
