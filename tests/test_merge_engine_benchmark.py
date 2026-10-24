"""Smoke tests for the merge-engine benchmark (Issue #111).

Keeps CI honest about the benchmark's contract without running the full
matrix: the parity gate must pass on real fixtures before any timing, the
percentile helper must be nearest-rank, and records must carry the fields
the #111 measurement contract requires.
"""

from __future__ import annotations

import unittest

from benchmarks.fast_merge_engine_benchmark import build_synthetic_benchmark_fixture
from benchmarks.merge_engine_benchmark import (
    COMPARISON_PATHS,
    ParityGateError,
    build_synthetic_leaf_tokens,
    measure_call,
    percentile,
    run_parity_gate,
)
from uniqtoken.merge_engine import cross_word_membership_table, production_constraints


class ParityGateTests(unittest.TestCase):
    def test_parity_gate_passes_on_synthetic_fixture(self):
        pieces, table, tok = build_synthetic_benchmark_fixture(256, "medium")
        tokens = build_synthetic_leaf_tokens(tok, pieces)
        report = run_parity_gate(tok, list(pieces), tokens, table, production_constraints(table))
        self.assertEqual(report["parity"], "pass")
        self.assertEqual(report["leaf_pieces"], len(pieces))
        self.assertGreater(report["applied_merges"], 0)

    def test_parity_gate_fails_loudly_on_corrupted_table(self):
        """A table that disagrees with the production path must abort with ParityGateError."""
        from uniqtoken.merge_engine import MembershipMergeTable

        pieces, table, tok = build_synthetic_benchmark_fixture(256, "medium")
        tokens = build_synthetic_leaf_tokens(tok, pieces)
        # Empty the eligible mapping: engines then merge nothing while the
        # production path still uses the tokenizer's own cross set, so the
        # 3-way outputs must diverge and the gate must abort before timing.
        broken = MembershipMergeTable(eligible={}, vocabulary_identity=table.vocabulary_identity)
        with self.assertRaises(ParityGateError):
            run_parity_gate(tok, list(pieces), tokens, broken, production_constraints(broken))


class MeasurementHelpersTests(unittest.TestCase):
    def test_percentile_is_nearest_rank(self):
        values = sorted([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(percentile(values, 0.5), 4.0)
        self.assertEqual(percentile(values, 0.95), 7.0)
        self.assertEqual(percentile(values, 1.0), 7.0)
        self.assertEqual(percentile(values, 0.01), 1.0)

    def test_measure_call_records_repetitions_and_ordering(self):
        stats = measure_call(lambda: [0] * 100, warmup=1, repetitions=3)
        self.assertEqual(stats["repetitions"], 3.0)
        self.assertLessEqual(stats["min_ms"], stats["p50_ms"])
        self.assertLessEqual(stats["p50_ms"], stats["max_ms"])
        self.assertGreaterEqual(stats["p95_ms"], stats["p50_ms"])

    def test_comparison_paths_declared(self):
        self.assertEqual(COMPARISON_PATHS, ("production", "fast", "reference"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
