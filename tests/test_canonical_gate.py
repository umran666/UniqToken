"""Unit tests for the performance regression gate (Issue #106).

Verifies:
- Baseline construction with provenance, expiry, and fingerprint metadata.
- Pass / advisory / fail decision logic for throughput and memory metrics.
- Statistical-significance requirement before any merge-blocking failure.
- Soft-skip behavior for missing, expired, provenance-broken, or
  hardware-mismatched baselines and drifted fixtures.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone

from benchmarks.canonical_gate import build_baseline, evaluate, main

FINGERPRINT_A = {
    "platform": "Windows-11-10.0.26200-SP0",
    "machine": "AMD64",
    "processor": "Intel64 Family 6 Model 154",
    "physical_cpus": 8,
    "logical_cpus": 12,
    "total_ram_gb": 7.65,
}
FINGERPRINT_A_STORED = dict(FINGERPRINT_A, total_ram_gb=8)  # RAM compared at GB granularity
FINGERPRINT_B = dict(FINGERPRINT_A, platform="Linux-6.8.0-x86_64", physical_cpus=4, logical_cpus=4)


def make_results(
    fingerprint: dict,
    throughput: float = 100.0,
    ci: tuple = (90.0, 110.0),
    rss: float = 50.0,
    fixtures: dict | None = None,
    timestamp: str = "2026-10-08T00:00:00Z",
) -> dict:
    return {
        "metadata": {
            "schema_version": "1.0",
            "timestamp_utc": timestamp,
            "git_commit": "a" * 40,
            **fingerprint,
            "fixture_hashes": fixtures if fixtures is not None else {"short": "h1", "medium": "h2"},
        },
        "records": [
            {
                "tokenizer": "uniqtoken",
                "workload": "short_b1",
                "batch_size": 1,
                "fixture": "short",
                "throughput_mb_s": throughput,
                "throughput_ci_95_mb_s": list(ci),
                "peak_rss_mb": rss,
            }
        ],
    }


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


class BaselineConstructionTests(unittest.TestCase):
    def test_build_baseline_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_json(root / "results.json", make_results(FINGERPRINT_A))
            baseline_path = root / "gate-baseline.json"

            exit_code = main(["--write-baseline", "--results", str(source), "--baseline", str(baseline_path)])
            self.assertEqual(exit_code, 0)

            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            self.assertEqual(baseline["schema_version"], "1.0")
            self.assertEqual(baseline["thresholds"]["throughput_drop_pct"], 5.0)
            self.assertEqual(baseline["thresholds"]["memory_rise_pct"], 10.0)
            self.assertEqual(baseline["fingerprint"], FINGERPRINT_A_STORED)
            self.assertEqual(len(baseline["records"]), 1)
            self.assertIn("uniqtoken|short_b1", baseline["records"])
            self.assertEqual(baseline["expiry"]["max_age_days"], 30)

            # Identical measurements must pass.
            artifact, code = evaluate(baseline_path, source)
            self.assertEqual(code, 0)
            self.assertEqual(artifact["status"], "pass")

    def test_build_baseline_rejects_non_canonical_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_json(root / "results.json", {"metadata": {}, "records": []})
            with self.assertRaises(ValueError):
                build_baseline(json.loads(source.read_text(encoding="utf-8")), source)


class DecisionLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.source = write_json(self.root / "results.json", make_results(FINGERPRINT_A))
        self.baseline_path = self.root / "gate-baseline.json"
        main(["--write-baseline", "--results", str(self.source), "--baseline", str(self.baseline_path)])

    def evaluate_against(self, results: dict, enforce: bool = False):
        current = write_json(self.root / "current.json", results)
        return evaluate(self.baseline_path, current, enforce=enforce)

    def test_nominal_drop_within_noise_is_advisory(self) -> None:
        # 8% nominal drop, but uncertainty intervals overlap the baseline.
        artifact, code = self.evaluate_against(make_results(FINGERPRINT_A, throughput=92.0, ci=(88.0, 96.0)))
        self.assertEqual(code, 0)
        self.assertEqual(artifact["status"], "advisory")
        self.assertEqual(artifact["summary"]["throughput_regressions_nominal"], ["uniqtoken|short_b1"])
        self.assertFalse(artifact["merge_blocking"])

    def test_significant_drop_fails_only_under_enforcement(self) -> None:
        current = make_results(FINGERPRINT_A, throughput=80.0, ci=(70.0, 85.0))
        artifact, code = self.evaluate_against(current, enforce=False)
        self.assertEqual((artifact["status"], code), ("advisory", 0))

        artifact, code = self.evaluate_against(current, enforce=True)
        self.assertEqual((artifact["status"], code), ("fail", 1))
        self.assertTrue(artifact["merge_blocking"])
        self.assertEqual(artifact["summary"]["throughput_regressions_significant"], ["uniqtoken|short_b1"])

    def test_memory_regression_follows_controlled_environment_rule(self) -> None:
        current = make_results(FINGERPRINT_A, rss=56.0)  # +12% peak RSS
        artifact, code = self.evaluate_against(current, enforce=False)
        self.assertEqual((artifact["status"], code), ("advisory", 0))

        artifact, code = self.evaluate_against(current, enforce=True)
        self.assertEqual((artifact["status"], code), ("fail", 1))
        self.assertEqual(artifact["summary"]["memory_regressions"], ["uniqtoken|short_b1"])

    def test_drop_exactly_at_threshold_is_not_a_breach(self) -> None:
        artifact, code = self.evaluate_against(make_results(FINGERPRINT_A, throughput=95.0))
        self.assertEqual((artifact["status"], code), ("pass", 0))
        self.assertEqual(artifact["summary"]["throughput_regressions_nominal"], [])

    def test_hardware_mismatch_is_advisory(self) -> None:
        artifact, code = self.evaluate_against(make_results(FINGERPRINT_B, throughput=50.0), enforce=True)
        self.assertEqual(code, 0)
        self.assertEqual(artifact["status"], "hardware-mismatch")
        self.assertTrue(artifact["fingerprint"]["differences"])

    def test_ram_reporting_noise_keeps_fingerprint_match(self) -> None:
        # Same host with slightly different total RAM reporting must stay controlled.
        rescaled = make_results(dict(FINGERPRINT_A, total_ram_gb=7.71), throughput=92.0)
        artifact, code = self.evaluate_against(rescaled)
        self.assertTrue(artifact["fingerprint"]["controlled_match"])
        self.assertEqual(artifact["status"], "advisory")

    def test_fixture_drift_blocks_comparison(self) -> None:
        current = make_results(FINGERPRINT_A, fixtures={"short": "changed", "medium": "h2"})
        artifact, code = self.evaluate_against(current, enforce=True)
        self.assertEqual((artifact["status"], code), ("fixture-drift", 0))

    def test_expired_baseline_is_advisory(self) -> None:
        baseline = json.loads(self.baseline_path.read_text(encoding="utf-8"))
        past = datetime.now(timezone.utc) - timedelta(days=1)
        baseline["expiry"]["expires_at"] = past.strftime("%Y-%m-%dT%H:%M:%SZ")
        write_json(self.baseline_path, baseline)

        artifact, code = self.evaluate_against(make_results(FINGERPRINT_A, throughput=50.0), enforce=True)
        self.assertEqual((artifact["status"], code), ("baseline-expired", 0))

    def test_provenance_mismatch_is_advisory(self) -> None:
        # Mutate the baseline's source receipt after recording.
        source = json.loads(self.source.read_text(encoding="utf-8"))
        source["records"][0]["throughput_mb_s"] = 999.0
        write_json(self.source, source)

        artifact, code = self.evaluate_against(make_results(FINGERPRINT_A, throughput=50.0), enforce=True)
        self.assertEqual((artifact["status"], code), ("baseline-provenance-mismatch", 0))
        self.assertFalse(artifact["baseline"]["provenance_verified"])

    def test_missing_baseline_is_advisory(self) -> None:
        current = write_json(self.root / "current.json", make_results(FINGERPRINT_A))
        artifact, code = evaluate(self.root / "does-not-exist.json", current)
        self.assertEqual((artifact["status"], code), ("baseline-missing", 0))

    def test_unreadable_results_is_hard_error(self) -> None:
        current = self.root / "current.json"
        current.write_text("not json", encoding="utf-8")
        artifact, code = evaluate(self.baseline_path, current)
        self.assertEqual((artifact["status"], code), ("results-unavailable", 2))


if __name__ == "__main__":
    unittest.main()
