"""Public container ownership, exact outputs, and direct diagnostic contracts."""

import json
import copy
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import unittest

from benchmarks import profile_native_boundary as profiler
from benchmarks.profile_boundary_overhead import make_benchmark_tokenizer
import uniqtoken.tokenizer as tokenizer_module


class NativeBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = make_benchmark_tokenizer()
        cls.native = tokenizer_module._native_core
        if not hasattr(cls.native, "rust_encode_text_native"):
            raise unittest.SkipTest("native extension required")

    def test_mutating_output_and_clearing_cache_preserves_retained_strings(self):
        texts = ["hello", "Cafe\u0301 \u4e2d\u6587 \U0001f469\u200d\U0001f4bb", "", "hello"] * 8
        expected = [self.tok.encode(t) for t in texts]
        batch = self.tok.encode_batch(texts)
        retained = list(batch[1])
        self.assertIs(type(batch), list)
        self.assertTrue(all(type(row) is list and all(type(t) is str for t in row) for row in batch))
        self.assertEqual(batch, expected)
        batch[0].append("mutation")
        batch[1].clear()
        self.tok.model.clear_cache()
        self.assertEqual(retained, expected[1])
        self.assertEqual(self.tok.encode_batch(texts), expected)

    def test_native_batch_sequence_types_and_errors_match(self):
        trie, kwargs = self.tok.model._get_rust_trie(), self.tok._native_pipeline_kwargs()
        texts = ["hello", "\u4e2d\u6587", ""]
        expected = [self.native.rust_encode_text_native(t, trie, **kwargs) for t in texts]
        for source in (list(texts), tuple(texts), iter(texts)):
            self.assertEqual(self.native.rust_encode_text_native_batch(source, trie, **kwargs), expected)
        for source in ("hello", b"hello", ["hello", 123], ["hello", "<|unk|>"]):
            with self.assertRaises(ValueError):
                self.native.rust_encode_text_native_batch(source, trie, **kwargs)

    def test_crossing_counts_pin_changed_paths_and_decode_control(self):
        texts = ["hello"] * 32
        calls = profiler.api_calls(self.tok, texts, single=False)
        counts, batch = profiler.observed_calls(self.native, calls["tokens"])
        self.assertEqual(counts, {"rust_encode_text_native_batch": 1})
        counts, iterative = profiler.observed_calls(self.native, calls["iterative_tokens"])
        self.assertEqual(counts, {"rust_encode_text_native": 32})
        self.assertEqual(batch, iterative)
        self.assertEqual(profiler.observed_calls(self.native, calls["decode"])[0], {})

    def test_opt_in_stage_output_and_allocations(self):
        if not hasattr(self.native, "rust_profile_boundary"):
            self.skipTest("diagnostic build only")
        texts = ["hello", "\u4e2d\u6587", ""] * 11
        trie, kwargs = self.tok.model._get_rust_trie(), self.tok._native_pipeline_kwargs()
        records = []
        for reference in (True, False):
            result, ns, counts, size, copies, input_copies = self.native.rust_profile_boundary(
                texts, trie, reference=reference, allocations=True, **kwargs
            )
            self.assertEqual(result, self.tok.encode_batch(texts))
            self.assertEqual(len(ns), len(profiler.STAGES))
            self.assertEqual(size, sum(len(t.encode()) for t in texts))
            self.assertEqual(input_copies, 0)
            self.assertTrue(all(len(c) == 5 for c in counts))
            self.assertEqual(copies, sum(len(t.encode()) for row in result for t in row) if reference else 0)
            records.append(counts[2])
        self.assertGreater(records[0][0], records[1][0])
        self.assertGreater(records[0][1], records[1][1])

    def test_published_native_evidence_hashes_and_parity(self):
        directory = profiler.ROOT / "benchmarks/boundary_overhead/issue101/native"
        if not directory.exists():
            self.skipTest("evidence not yet published")
        manifest = json.loads((directory / "manifest.json").read_bytes())
        for filename, expected in manifest.items():
            self.assertEqual(profiler.sha256(directory / filename), expected)
        records = [json.loads(line) for line in (directory / "public_api.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 50)
        for row in records:
            before, after = (row["variants"][v] for v in ("before", "after"))
            self.assertEqual(before["output_sha256"], after["output_sha256"])
            self.assertEqual(before["native_calls"], after["native_calls"])
            self.assertEqual(len(before["latency"]["samples_ms"]), len(after["latency"]["samples_ms"]))

    def test_before_after_comparison_rejects_output_drift(self):
        run = {
            "model_sha256": "model",
            "parity": [],
            "public_native_signatures": {},
            "rows": [
                {
                    "workload": "short_single",
                    "api": "tokens",
                    "samples_ms": [1.0],
                    "native_calls": {"rust_encode_text_native": 1},
                    "output_sha256": "same",
                    "python_heap": {},
                    "process_peak_rss_bytes": 1,
                    "normalized_utf8_bytes": 1,
                }
            ],
        }
        changed = copy.deepcopy(run)
        changed["rows"][0]["output_sha256"] = "different"
        with self.assertRaisesRegex(AssertionError, "output or crossing"):
            profiler.compare_runs([("before", run), ("after", changed)])

    def test_evidence_writer_completes_and_refuses_overwrite(self):
        payload = {
            "native_sha256": "a" * 64,
            "model_sha256": "b" * 64,
            "parity": [],
            "public_native_signatures": {},
            "rows": [
                {
                    "workload": "short_single",
                    "api": "tokens",
                    "samples_ms": [1.0],
                    "native_calls": {},
                    "output_sha256": "same",
                    "python_heap": {},
                    "process_peak_rss_bytes": 1,
                    "normalized_utf8_bytes": 1,
                }
            ],
        }

        def version(*command):
            self.assertEqual(command, ("rustc", "--version"))
            return "rustc fixture"

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-evidence"
            argv = [
                "profiler",
                "--before-native",
                "before.so",
                "--after-native",
                "after.so",
                "--profile-native",
                "profile.so",
                "--baseline-commit",
                "baseline",
                "--output",
                str(output),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(profiler, "tool_version", side_effect=version),
                patch.object(profiler, "git_value", side_effect=lambda *args: "" if args[0] == "status" else "c" * 40),
                patch.object(profiler, "source_sha256", return_value="d" * 64),
                patch.object(
                    profiler, "run_worker", side_effect=lambda *args, **kwargs: copy.deepcopy(payload)
                ) as worker,
            ):
                profiler.main()
                self.assertEqual(worker.call_count, 5)
                metadata = json.loads((output / "metadata.json").read_bytes())
                self.assertEqual(metadata["rustc"], "rustc fixture")
                manifest = json.loads((output / "manifest.json").read_bytes())
                for name, expected in manifest.items():
                    self.assertEqual(profiler.sha256(output / name), expected)
                with self.assertRaises(SystemExit):
                    profiler.main()
                self.assertEqual(worker.call_count, 5)


if __name__ == "__main__":
    unittest.main()
