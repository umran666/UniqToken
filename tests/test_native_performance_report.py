"""Repeatability gates must expose uncertain results and every batch-one regression."""

import unittest

from benchmarks import native_performance_report as r


class NativePerformanceReportTests(unittest.TestCase):
    def test_repeatability_requires_both_rounds_and_strict_intervals(self):
        def comparison(low, high):
            return {"bootstrap_95_interval": [low, high]}

        self.assertEqual(r.classify([comparison(1.1, 1.2), comparison(1.2, 1.3)]), "repeatable_gain")
        self.assertEqual(r.classify([comparison(0.8, 0.9), comparison(0.9, 0.95)]), "repeatable_degradation")
        for rows in (
            [comparison(1.1, 1.2), comparison(0.9, 1.1)],
            [comparison(1.0, 1.2), comparison(1.1, 1.2)],
            [comparison(0.8, 0.9), comparison(0.9, 1.0)],
            [comparison(0.8, 0.9), comparison(1.1, 1.2)],
        ):
            self.assertEqual(r.classify(rows), "inconclusive_or_mixed")

    def test_rejection_report_does_not_hide_a_batch_one_degradation(self):
        row = {"variant": "compact", "surface": "batch", "batch_size": 1, "classification": "repeatable_degradation"}
        payload = {"build_sources": {"baseline": "a" * 40}, "compiler": "rustc test"}
        text = r.report([row], payload)
        self.assertIn("rejection gate: FAIL", text)
        self.assertIn("repeatable degraded cells: 1", text)
        row["classification"] = "inconclusive_or_mixed"
        self.assertIn("rejection gate: PASS", r.report([row], payload))

    def test_partial_rounds_cannot_be_labeled_repeatable(self):
        for comparisons in ([], [{"bootstrap_95_interval": [1.1, 1.2]}]):
            with self.assertRaises(ValueError):
                r.classify(comparisons)


if __name__ == "__main__":
    unittest.main()
