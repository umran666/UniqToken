"""Exact matrix, failure receipts, count invariants and native memory measurement."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import vocabulary_scaling as s
from benchmarks import run_research_experiments as h
from tests.test_token_density import fixture


IDENTITY = {"commit_hash": "a" * 40, "working_tree_dirty": False}


def matrix():
    records = fixture()["records"]
    conditions = []
    for budget in s.BUDGETS:
        for name in s.COHORT:
            rows = [{**row, "tokenizer": name, "vocab_budget": budget, "actual_vocab_size": budget} for row in records]
            conditions.append(
                {
                    "tokenizer": name,
                    "vocab_budget": budget,
                    "actual_vocab_size": budget,
                    "status": "complete",
                    "identity": IDENTITY,
                    "model_hashes": {"model": "b" * 64},
                    "training_seconds": 1.0,
                    "training_cpu_seconds": 0.8,
                    "training_process_peak_rss_bytes": 1000,
                    "records": rows,
                }
            )
    return {
        "schema_version": 1,
        "identity": IDENTITY,
        "configuration": {"cohort": list(s.COHORT), "budgets": list(s.BUDGETS)},
        "conditions": conditions,
    }


class VocabularyScalingTests(unittest.TestCase):
    def test_full_exact_matrix_and_explicit_failures(self):
        payload = matrix()
        self.assertEqual(len(s.validate_results(payload)["conditions"]), 15)
        for status in ("budget_not_reached", "resource_limit", "failed_training", "failed_validation", "worker_failed"):
            changed = copy.deepcopy(payload)
            changed["conditions"][0] = {**changed["conditions"][0], "status": status, "records": []}
            s.validate_results(changed)

    def test_missing_duplicate_relabeling_and_padding_rejected(self):
        baseline = matrix()
        absent = copy.deepcopy(baseline)
        absent["conditions"].pop()
        duplicate = copy.deepcopy(baseline)
        duplicate["conditions"].append(duplicate["conditions"][0])
        mismatch = copy.deepcopy(baseline)
        mismatch["conditions"][0]["actual_vocab_size"] -= 1
        renamed = copy.deepcopy(baseline)
        renamed["configuration"]["cohort"][0] = "substitute"
        for payload in (absent, duplicate, mismatch, renamed):
            with self.assertRaises(ValueError):
                s.validate_results(payload)

    def test_failed_conditions_cannot_keep_favorable_metrics(self):
        payload = matrix()
        payload["conditions"][0]["status"] = "resource_limit"
        with self.assertRaisesRegex(ValueError, "favorable"):
            s.validate_results(payload)

    def test_source_model_cost_and_record_identity_gates(self):
        for field, value in (
            ("identity", {}),
            ("model_hashes", {}),
            ("training_seconds", float("nan")),
            ("training_process_peak_rss_bytes", 0),
        ):
            payload = matrix()
            payload["conditions"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                s.validate_results(payload)
        payload = matrix()
        payload["conditions"][0]["records"][0]["tokenizer"] = "another"
        with self.assertRaisesRegex(ValueError, "identity"):
            s.validate_results(payload)

    def test_corrupt_histogram_or_test_split_rejected(self):
        for key, value in (("token_length_bytes_histogram", {"1": 999}), ("split", "test")):
            payload = matrix()
            payload["conditions"][0]["records"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                s.validate_results(payload)

    def test_process_high_water_rss_is_available(self):
        self.assertIsInstance(s.peak_rss_bytes(), int)
        self.assertGreater(s.peak_rss_bytes(), 0)

    def test_worker_quarantines_training_failure_with_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "request.json"
            h.write_new_json(request, {"identity": IDENTITY, "documents": {"train": [], "validation": []}})
            output = Path(directory) / "failed"
            with (
                patch.object(h, "runtime_identity", return_value=IDENTITY),
                patch.object(
                    s, "train_condition", side_effect=ValueError("exact vocabulary budget/ID invariant failed")
                ),
            ):
                result = s.worker(request, "uniq_superbpe_r64", 8192, output)
            self.assertEqual(result["status"], "budget_not_reached")
            self.assertEqual(result["records"], [])
            self.assertEqual(h.read_json(output / "response.json"), result)
            self.assertEqual(result["request_sha256"], h.file_hash(request))


if __name__ == "__main__":
    unittest.main()
