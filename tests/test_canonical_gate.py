"""Unit tests for the performance regression gate (Issue #106).

Verifies:
- Baseline construction with provenance, expiry, and fingerprint metadata.
- Pass / advisory decision logic and hard malformed-input failures.
- Observed ranges never authorize statistical performance enforcement.
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
    "build_mode": "release",
    "python_version": "test-python",
    "worker_count": 1,
    "warmup": 3,
    "repetitions": 7,
    "iterations_per_repetition": 5,
    "baselines": {"uniqtoken": {"available": True, "version": "1.0.0", "engine": "test-native"}},
}
FINGERPRINT_A_STORED = dict(FINGERPRINT_A, total_ram_gb=8)  # RAM compared at GB granularity
FINGERPRINT_B = dict(FINGERPRINT_A, platform="Linux-6.8.0-x86_64", physical_cpus=4, logical_cpus=4)


def make_results(
    fingerprint: dict,
    throughput: float = 100.0,
    ci: tuple | None = None,
    rss: float = 50.0,
    fixtures: dict | None = None,
    timestamp: str = "2026-10-08T00:00:00Z",
) -> dict:
    """Build a canonical results fixture with one workload and configurable metrics."""
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
                "throughput_ci_95_mb_s": list(ci) if ci is not None else [throughput * 0.9, throughput * 1.1],
                "peak_rss_mb": rss,
                "repetitions": 7,
                "iterations_per_rep": 5,
            }
        ],
    }


def write_json(path: Path, payload: dict) -> Path:
    """Write a UTF-8 JSON fixture and return its path."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


class BaselineConstructionTests(unittest.TestCase):
    def test_build_baseline_round_trip(self) -> None:
        """Verify CLI baseline recording preserves metadata and identical results pass."""
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
        """Verify baseline construction rejects results without a schema or records."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_json(root / "results.json", {"metadata": {}, "records": []})
            with self.assertRaises(ValueError):
                build_baseline(json.loads(source.read_text(encoding="utf-8")), source)


class DecisionLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        """Record a temporary baseline and register cleanup for each decision test."""
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.source = write_json(self.root / "results.json", make_results(FINGERPRINT_A))
        self.baseline_path = self.root / "gate-baseline.json"
        main(["--write-baseline", "--results", str(self.source), "--baseline", str(self.baseline_path)])

    def evaluate_against(self, results: dict, enforce: bool = False):
        """Write current results and evaluate them against the test's baseline."""
        current = write_json(self.root / "current.json", results)
        return evaluate(self.baseline_path, current, enforce=enforce)

    def test_nominal_drop_within_noise_is_advisory(self) -> None:
        """Verify a threshold breach with overlapping intervals remains advisory."""
        # 8% nominal drop, but uncertainty intervals overlap the baseline.
        artifact, code = self.evaluate_against(make_results(FINGERPRINT_A, throughput=92.0, ci=(88.0, 96.0)))
        self.assertEqual(code, 0)
        self.assertEqual(artifact["status"], "advisory")
        self.assertEqual(artifact["summary"]["throughput_regressions_nominal"], ["uniqtoken|short_b1"])
        self.assertFalse(artifact["merge_blocking"])

    def test_observed_range_separation_never_proves_significance(self) -> None:
        """Canonical extrema do not provide calibrated confidence, even with --enforce."""
        current = make_results(FINGERPRINT_A, throughput=80.0, ci=(70.0, 85.0))
        artifact, code = self.evaluate_against(current, enforce=False)
        self.assertEqual((artifact["status"], code), ("advisory", 0))

        artifact, code = self.evaluate_against(current, enforce=True)
        self.assertEqual((artifact["status"], code), ("advisory", 0))
        self.assertFalse(artifact["merge_blocking"])
        self.assertFalse(artifact["enforcement_eligible"])
        self.assertEqual(artifact["summary"]["throughput_regressions_outside_range"], ["uniqtoken|short_b1"])

    def test_enforcement_request_is_not_reported_as_active(self) -> None:
        """Requesting enforcement must not claim an advisory gate is enforcing."""
        artifact, code = self.evaluate_against(make_results(FINGERPRINT_A), enforce=True)
        self.assertEqual(code, 0)
        self.assertTrue(artifact["enforcement_requested"])
        self.assertFalse(artifact["enforced"])

    def test_memory_regression_follows_controlled_environment_rule(self) -> None:
        """A single peak RSS value has no calibrated memory uncertainty."""
        current = make_results(FINGERPRINT_A, rss=56.0)  # +12% peak RSS
        artifact, code = self.evaluate_against(current, enforce=False)
        self.assertEqual((artifact["status"], code), ("advisory", 0))

        artifact, code = self.evaluate_against(current, enforce=True)
        self.assertEqual((artifact["status"], code), ("advisory", 0))
        self.assertEqual(artifact["summary"]["memory_regressions"], ["uniqtoken|short_b1"])

    def test_drop_exactly_at_threshold_is_not_a_breach(self) -> None:
        """Verify a throughput drop equal to the threshold still passes."""
        artifact, code = self.evaluate_against(make_results(FINGERPRINT_A, throughput=95.0))
        self.assertEqual((artifact["status"], code), ("pass", 0))
        self.assertEqual(artifact["summary"]["throughput_regressions_nominal"], [])

    def test_hardware_mismatch_is_advisory(self) -> None:
        """Verify different hardware prevents enforcement of a throughput regression."""
        artifact, code = self.evaluate_against(make_results(FINGERPRINT_B, throughput=50.0), enforce=True)
        self.assertEqual(code, 0)
        self.assertEqual(artifact["status"], "hardware-mismatch")
        self.assertTrue(artifact["fingerprint"]["differences"])

    def test_ram_reporting_noise_keeps_fingerprint_match(self) -> None:
        """Verify small RAM reporting differences preserve a hardware match."""
        # Same host with slightly different total RAM reporting must stay controlled.
        rescaled = make_results(dict(FINGERPRINT_A, total_ram_gb=7.71), throughput=92.0)
        artifact, code = self.evaluate_against(rescaled)
        self.assertTrue(artifact["fingerprint"]["controlled_match"])
        self.assertEqual(artifact["status"], "advisory")

    def test_nondeterministic_artifact_hashes_do_not_break_comparability(self) -> None:
        """SentencePiece training shuffles its corpus, so artifact hashes vary per run."""
        current = make_results(FINGERPRINT_A)
        current["metadata"]["artifact_hashes"] = {"sentencepiece_model": "different-run-digest"}
        artifact, code = self.evaluate_against(current)
        self.assertTrue(artifact["fingerprint"]["controlled_match"])
        self.assertEqual((artifact["status"], code), ("pass", 0))

    def test_fixture_drift_blocks_comparison(self) -> None:
        """Verify changed fixture hashes soft-skip comparison even under enforcement."""
        current = make_results(FINGERPRINT_A, fixtures={"short": "changed", "medium": "h2"})
        artifact, code = self.evaluate_against(current, enforce=True)
        self.assertEqual((artifact["status"], code), ("fixture-drift", 0))

    def test_expired_baseline_is_advisory(self) -> None:
        """Verify an expired baseline soft-skips comparison under enforcement."""
        baseline = json.loads(self.baseline_path.read_text(encoding="utf-8"))
        past = datetime.now(timezone.utc) - timedelta(days=1)
        baseline["expiry"]["expires_at"] = past.strftime("%Y-%m-%dT%H:%M:%SZ")
        write_json(self.baseline_path, baseline)

        artifact, code = self.evaluate_against(make_results(FINGERPRINT_A, throughput=50.0), enforce=True)
        self.assertEqual((artifact["status"], code), ("baseline-expired", 0))

    def test_provenance_mismatch_is_advisory(self) -> None:
        """Verify a changed source receipt soft-skips comparison under enforcement."""
        # Mutate the baseline's source receipt after recording.
        source = json.loads(self.source.read_text(encoding="utf-8"))
        source["records"][0]["throughput_mb_s"] = 999.0
        write_json(self.source, source)

        artifact, code = self.evaluate_against(make_results(FINGERPRINT_A, throughput=50.0), enforce=True)
        self.assertEqual((artifact["status"], code), ("baseline-provenance-mismatch", 0))
        self.assertFalse(artifact["baseline"]["provenance_verified"])

    def test_missing_baseline_is_advisory(self) -> None:
        """Verify a missing baseline produces an advisory outcome and zero exit code."""
        current = write_json(self.root / "current.json", make_results(FINGERPRINT_A))
        artifact, code = evaluate(self.root / "does-not-exist.json", current)
        self.assertEqual((artifact["status"], code), ("baseline-missing", 0))

    def test_unreadable_results_is_hard_error(self) -> None:
        """Verify invalid current JSON produces a hard input error with exit code two."""
        current = self.root / "current.json"
        current.write_text("not json", encoding="utf-8")
        artifact, code = evaluate(self.baseline_path, current)
        self.assertEqual((artifact["status"], code), ("results-unavailable", 2))

    def test_malformed_or_nonfinite_records_produce_input_artifact(self) -> None:
        """Bad measurements must not crash or silently count as passing cells."""
        for record in ({}, None, {"tokenizer": "uniqtoken"}):
            with self.subTest(record=record):
                current = make_results(FINGERPRINT_A)
                current["records"] = [record]
                artifact, code = self.evaluate_against(current, enforce=True)
                self.assertEqual((artifact["status"], code), ("results-unavailable", 2))
        for field in ("throughput_mb_s", "peak_rss_mb"):
            for value in (float("nan"), float("inf"), 10**400, -1, 0, True, "100"):
                with self.subTest(field=field, value=value):
                    current = make_results(FINGERPRINT_A)
                    current["records"][0][field] = value
                    artifact, code = self.evaluate_against(current, enforce=True)
                    self.assertEqual((artifact["status"], code), ("results-unavailable", 2))

    def test_duplicate_or_missing_workloads_cannot_pass(self) -> None:
        """Repeated cells and partial matrices must remain visible failures."""
        current = make_results(FINGERPRINT_A)
        current["records"].append(dict(current["records"][0]))
        artifact, code = self.evaluate_against(current)
        self.assertEqual((artifact["status"], code), ("results-unavailable", 2))
        current["records"] = [dict(current["records"][0], workload="medium_b1", fixture="medium")]
        artifact, code = self.evaluate_against(current)
        self.assertEqual((artifact["status"], code), ("workload-mismatch", 0))

    def test_missing_provenance_cannot_enforce(self) -> None:
        """A missing baseline source must not authorize a blocking comparison."""
        self.source.unlink()
        artifact, code = self.evaluate_against(make_results(FINGERPRINT_A, throughput=1, ci=(1, 1)), enforce=True)
        self.assertEqual((artifact["status"], code), ("baseline-provenance-unavailable", 0))

    def test_software_and_measurement_mismatch_cannot_be_controlled(self) -> None:
        """Same hardware does not make a different runtime or protocol comparable."""
        for key, value in (("build_mode", "debug"), ("python_version", "0.0"), ("worker_count", 32)):
            with self.subTest(key=key):
                current = make_results(FINGERPRINT_A)
                current["metadata"][key] = value
                artifact, code = self.evaluate_against(current, enforce=True)
                self.assertEqual((artifact["status"], code), ("hardware-mismatch", 0))
                self.assertFalse(artifact["fingerprint"]["controlled_match"])

    def test_missing_repetition_evidence_cannot_pass(self) -> None:
        """One observation must not qualify as a repeated performance check."""
        current = make_results(FINGERPRINT_A)
        current["metadata"]["repetitions"] = 1
        current["records"][0]["repetitions"] = 1
        artifact, code = self.evaluate_against(current, enforce=True)
        self.assertEqual(code, 0)
        self.assertNotEqual(artifact["status"], "pass")

    def test_cli_writes_comparison_for_invalid_records(self) -> None:
        """CI must retain a structured diagnostic even when a current cell is invalid."""
        current = make_results(FINGERPRINT_A)
        current["records"] = [{}]
        path = write_json(self.root / "invalid.json", current)
        output = self.root / "comparison.json"
        code = main(["--results", str(path), "--baseline", str(self.baseline_path), "--out", str(output)])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.read_text())["status"], "results-unavailable")

    def test_invalid_ranges_and_schemas_cannot_pass(self) -> None:
        """Malformed ranges and unsupported result schemas must produce diagnostics."""
        for bounds in (None, [], [110, 90], [0, 110], [float("nan"), 110], [90, "110"], [90, 95]):
            with self.subTest(bounds=bounds):
                current = make_results(FINGERPRINT_A)
                current["records"][0]["throughput_ci_95_mb_s"] = bounds
                artifact, code = self.evaluate_against(current)
                self.assertEqual((artifact["status"], code), ("results-unavailable", 2))
        current = make_results(FINGERPRINT_A)
        current["metadata"]["schema_version"] = "unknown"
        artifact, code = self.evaluate_against(current)
        self.assertEqual((artifact["status"], code), ("results-unavailable", 2))

    def test_malformed_provenance_is_advisory(self) -> None:
        """Bad source paths cannot bypass verification or crash the CLI."""
        baseline = json.loads(self.baseline_path.read_text())
        for path in (None, [], {}):
            with self.subTest(path=path):
                baseline["source"]["path"] = path
                write_json(self.baseline_path, baseline)
                artifact, code = self.evaluate_against(make_results(FINGERPRINT_A), enforce=True)
                self.assertEqual((artifact["status"], code), ("baseline-provenance-unavailable", 0))

    def test_altered_baseline_cells_do_not_match_source_provenance(self) -> None:
        """Hashing the source does not authenticate hand-edited derivative records."""
        baseline = json.loads(self.baseline_path.read_text())
        baseline["records"]["uniqtoken|short_b1"]["throughput_mb_s"] = 101.0
        write_json(self.baseline_path, baseline)
        artifact, code = self.evaluate_against(make_results(FINGERPRINT_A), enforce=True)
        self.assertEqual((artifact["status"], code), ("baseline-provenance-mismatch", 0))

    def test_source_metadata_cannot_be_rewritten_in_baseline(self) -> None:
        """Changing copied metadata cannot authenticate a different environment or date."""
        baseline = json.loads(self.baseline_path.read_text())
        for section, key, value in (
            ("fingerprint", "build_mode", "debug"),
            ("source", "measured_at", "2026-10-20T00:00:00Z"),
            ("source", "git_commit", "b" * 40),
            ("fixture_hashes", "short", "changed"),
        ):
            with self.subTest(section=section, key=key):
                changed = json.loads(json.dumps(baseline))
                changed[section][key] = value
                write_json(self.baseline_path, changed)
                current = make_results(FINGERPRINT_A)
                if section == "fingerprint":
                    current["metadata"][key] = value
                artifact, code = self.evaluate_against(current, enforce=True)
                self.assertEqual((artifact["status"], code), ("baseline-provenance-mismatch", 0))

    def test_per_cell_repetitions_must_match_metadata(self) -> None:
        """Metadata must not conceal a workload measured only once."""
        current = make_results(FINGERPRINT_A)
        current["records"][0]["repetitions"] = 1
        artifact, code = self.evaluate_against(current, enforce=True)
        self.assertEqual((artifact["status"], code), ("insufficient-evidence", 0))


if __name__ == "__main__":
    unittest.main()
