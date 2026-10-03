"""Exact archived measurement bytes, safe extraction and retained native evidence."""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from benchmarks import native_receipt_archive as a
from benchmarks import native_performance_report as r
from benchmarks import profile_prefix_allocations as prefix
from benchmarks import profile_residual_native as baseline
from benchmarks import run_research_experiments as h


class NativeReceiptArchiveTests(unittest.TestCase):
    def test_archive_roundtrip_is_exact_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            h.write_new_json(root / "results.json", {"value": 1.25, "source": "measured"})
            h.write_new_json(root / "manifest.json", {"status": "complete", "artifacts": h.artifact_hashes(root)})
            one, two = Path(directory) / "one.zip", Path(directory) / "two.zip"
            a.pack(root, one)
            a.pack(root, two)
            self.assertEqual(h.file_hash(one), h.file_hash(two))
            extracted = a.unpack(one, Path(directory) / "extracted")
            for relative, digest in h.artifact_hashes(root).items():
                self.assertEqual(h.file_hash(extracted / relative), digest)

    def test_corrupt_receipt_and_path_escape_are_rejected_before_extraction(self):
        for name in ("results.json", "../outside.json", "/absolute.json", "C:drive-relative.json", "dir\\escape.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.zip"
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr(name, b"{}")
                    archive.writestr("manifest.json", json.dumps({"status": "complete", "artifacts": {name: "0" * 64}}))
                output = Path(directory) / "extracted"
                with self.assertRaises(ValueError):
                    a.unpack(path, output)
                self.assertFalse(output.exists())


class RetainedNativeEvidenceTests(unittest.TestCase):
    def test_full_receipts_parity_uncertainty_memory_and_rejection_gate(self):
        root = Path(a.__file__).parent / "native_performance" / "issue96-99"
        receipt = h.read_json(root / "manifest.json")
        self.assertEqual(receipt["status"], "complete")
        for relative, digest in receipt["artifacts"].items():
            self.assertEqual(h.file_hash(root / relative), digest, relative)
        with tempfile.TemporaryDirectory() as directory:
            unpacked = {}
            for name in ("baseline", "paired", "prefix"):
                unpacked[name] = a.unpack(root / f"{name}-receipts.zip", Path(directory) / name)
                self.assertEqual(
                    h.file_hash(unpacked[name] / "manifest.json"),
                    receipt["sources"][name]["measurement_manifest_sha256"],
                )
            baseline.validate_profile(h.read_json(unpacked["baseline"] / "results.json"))
            path = unpacked["paired"] / "results.json"
            payload, _ = r.verified(path)
            supplementary, _ = prefix.verified(unpacked["prefix"] / "results.json", path, payload)
            rows = r.analyze(payload, supplementary)
        summary = h.read_json(root / "summary" / "summary.json")
        self.assertEqual(summary["records"], rows)
        self.assertEqual(len(payload["records"]), 194)
        self.assertEqual(len(payload["allocations"]), 194)
        self.assertEqual(len(supplementary["records"]), 49)
        self.assertEqual(len(rows), 291)
        self.assertFalse(
            any(
                row["variant"] == "compact"
                and row["surface"] == "batch"
                and row["batch_size"] == 1
                and row["classification"] == "repeatable_degradation"
                for row in rows
            )
        )
        replay = h.read_json(root / "summary" / "batch-span-parity.json")
        self.assertEqual(replay["status"], "complete")
        self.assertEqual(len(replay["records"]), 48)


if __name__ == "__main__":
    unittest.main()
