"""Explicit experimental benchmark selection and pre-timing native parity."""

import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


class NativeMergeBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("uniqtoken.native_merge_engine")

    def test_native_cli_requires_explicit_output_destination(self):
        from benchmarks import fast_merge_engine_benchmark

        # Run a real copy so the pre-fix default cannot overwrite tracked evidence.
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "benchmark.py"
            script.write_bytes(Path(fast_merge_engine_benchmark.__file__).read_bytes())
            result = subprocess.run(
                [sys.executable, str(script), "--engine", "native"],
                env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])),
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("--out", result.stderr)
            self.assertFalse((Path(tmp) / "fast_merge_engine_results.json").exists())

    def test_benchmark_selects_actual_native_engine(self):
        from benchmarks.fast_merge_engine_benchmark import run_benchmark

        records = run_benchmark(engine="native")
        self.assertEqual(len(records), 11)
        self.assertTrue(all(r["candidate_engine"] == "native-prototype" and r["parity_verified"] for r in records))

    def test_benchmark_rejects_wrong_native_ids_before_timing(self):
        from benchmarks.fast_merge_engine_benchmark import run_benchmark

        original = self.module.NativeFastMergeEngine.apply

        def wrong_id(engine, *args, **kwargs):
            plan = original(engine, *args, **kwargs)
            groups = tuple(replace(g, merged_id=999999) if g.merged_id is not None else g for g in plan.groups)
            return replace(plan, groups=groups)

        with patch.object(self.module.NativeFastMergeEngine, "apply", wrong_id):
            with patch(
                "benchmarks.fast_merge_engine_benchmark.time.perf_counter",
                side_effect=AssertionError("timed before parity"),
            ):
                with self.assertRaisesRegex(RuntimeError, "Parity mismatch"):
                    run_benchmark(engine="native")
