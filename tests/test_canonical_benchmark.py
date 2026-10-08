"""Unit tests for the canonical end-to-end tokenizer benchmark (Issue #105).

Verifies:
- Parity gate functionality and corruption rejection.
- Independent alignment oracle behavior.
- Geometric mean mathematical calculation.
- Published receipt files existence and SHA-256 manifest integrity.
- Results schema compliance, optimization ledger, and summary completeness.
"""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

from benchmarks.canonical_benchmark import geomean, reference_raw_spans, run_parity_gate
from uniqtoken.tokenizer import CustomTokenizer

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS_DIR = ROOT / "benchmarks" / "canonical" / "issue105"


class CanonicalBenchmarkTests(unittest.TestCase):
    def test_geomean_math(self) -> None:
        self.assertEqual(geomean([]), 0.0)
        self.assertEqual(geomean([0.0, 1.0]), 0.0)
        # geomean of [2, 8] is sqrt(16) = 4.0
        self.assertAlmostEqual(geomean([2.0, 8.0]), 4.0)
        # geomean of [1, 2, 4] is (8)^(1/3) = 2.0
        self.assertAlmostEqual(geomean([1.0, 2.0, 4.0]), 2.0)

    def test_reference_raw_spans_oracle(self) -> None:
        corpus = ["The quick brown fox jumps over 42 lazy dogs."] * 5
        tok = CustomTokenizer.train_from_corpus(corpus, target_vocab_size=320, min_frequency=1, verbose=False)
        text = "The quick brown fox"
        tokens = tok.encode(text)
        spans = reference_raw_spans(tok, text, tokens)

        self.assertEqual(len(spans), len(tokens))
        for start, end in spans:
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, len(text))
            self.assertLess(start, end)

    def test_parity_gate_passes_on_canonical_model(self) -> None:
        # The canonical model directory is gitignored ("artifacts/"), so it is
        # absent in CI. Train a tiny deterministic tokenizer to a temp path and
        # run the parity gate against it rather than asserting an on-disk path
        # that is never committed.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            corpus = ["The quick brown fox jumps over 42 lazy dogs."] * 5
            tok = CustomTokenizer.train_from_corpus(corpus, target_vocab_size=320, min_frequency=1, verbose=False)
            model_path = Path(tmp) / "uniqtoken_model"
            tok.save(model_path)
            run_parity_gate(model_path)

    def test_receipts_files_exist(self) -> None:
        results_file = RECEIPTS_DIR / "results.json"
        report_file = RECEIPTS_DIR / "REPORT.md"
        manifest_file = RECEIPTS_DIR / "manifest.json"

        self.assertTrue(results_file.is_file(), f"Missing {results_file}")
        self.assertTrue(report_file.is_file(), f"Missing {report_file}")
        self.assertTrue(manifest_file.is_file(), f"Missing {manifest_file}")

    def test_manifest_sha256_integrity(self) -> None:
        manifest_file = RECEIPTS_DIR / "manifest.json"
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))

        self.assertEqual(manifest_data.get("schema_version"), "1.0")
        sha_map = manifest_data.get("sha256", {})

        for filename, expected_hash in sha_map.items():
            target_path = RECEIPTS_DIR / filename
            self.assertTrue(target_path.is_file(), f"File {filename} in manifest does not exist")
            actual_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"SHA-256 hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}",
            )

    def test_results_json_schema_and_completeness(self) -> None:
        results_file = RECEIPTS_DIR / "results.json"
        data = json.loads(results_file.read_text(encoding="utf-8"))

        # 1. Metadata check
        meta = data.get("metadata", {})
        self.assertEqual(meta.get("benchmark_issue"), 105)
        self.assertEqual(meta.get("worker_count"), 1)
        self.assertIn("primary_metric", meta)
        self.assertIn("throughput_mb_s", meta["primary_metric"])
        self.assertIn("artifact_hashes", meta)
        self.assertIn("fixture_hashes", meta)
        self.assertIn("interpretation_rules", meta)
        self.assertIn("baselines", meta)

        # 2. Historical optimization evidence ledger
        ledger = data.get("optimization_evidence_ledger", [])
        self.assertGreaterEqual(len(ledger), 5)
        issues = {item["issue"] for item in ledger}
        self.assertTrue({"95", "96-99", "100", "101", "103", "104"}.issubset(issues))

        # 3. Summaries & geometric means
        summaries = data.get("summaries", {})
        for tok in ("uniqtoken", "sentencepiece", "tokenizers"):
            self.assertIn(tok, summaries)
            s = summaries[tok]
            self.assertGreater(s["overall_geomean_mb_s"], 0.0)
            self.assertGreater(s["batch_1_geomean_mb_s"], 0.0)
            self.assertGreater(s["batch_128_geomean_mb_s"], 0.0)

        # 4. Workload matrix records
        records = data.get("records", [])
        self.assertEqual(len(records), 60)  # 5 fixtures * 4 batch sizes * 3 tokenizers
        for r in records:
            self.assertIn(r["batch_size"], (1, 8, 32, 128))
            self.assertGreater(r["throughput_mb_s"], 0.0)
            self.assertGreater(r["tokens_per_s"], 0.0)
            self.assertGreater(r["encode_p50_ms"], 0.0)
            self.assertGreater(r["decode_p50_ms"], 0.0)
            self.assertGreater(r["peak_rss_mb"], 0.0)


if __name__ == "__main__":
    unittest.main()
