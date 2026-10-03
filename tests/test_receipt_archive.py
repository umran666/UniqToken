"""Archive packing preserves original bytes and rejects incomplete or corrupt evidence."""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from benchmarks import receipt_archive as a
from benchmarks import run_research_experiments as h


class ReceiptArchiveTests(unittest.TestCase):
    def fixture(self, root):
        root.mkdir()
        (root / "results.json").write_bytes(b'{"raw": "\\u093e"}\r\n')
        (root / "REPORT.md").write_bytes(b"# Original report\r\n")
        h.write_new_json(root / "manifest.json", {"status": "complete", "artifacts": h.artifact_hashes(root)})

    def test_deterministic_publication_preserves_every_original_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            self.fixture(root)
            one, two = Path(directory) / "one", Path(directory) / "two"
            a.publish(root, one)
            a.publish(root, two)
            self.assertEqual(h.artifact_hashes(one), h.artifact_hashes(two))
            with a.retained_bundle(one) as extracted:
                self.assertEqual(h.artifact_hashes(extracted), h.artifact_hashes(root))
            self.assertEqual((one / "REPORT.md").read_bytes(), (root / "REPORT.md").read_bytes())

    def test_source_files_missing_from_receipt_cannot_be_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            self.fixture(root)
            (root / "unreceipted.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "coverage"):
                a.pack(root, Path(directory) / "evidence.zip")

    def test_corrupt_archive_preview_or_manifest_binding_is_rejected(self):
        for changed in ("evidence.zip", "REPORT.md", "manifest_binding"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root, output = Path(directory) / "source", Path(directory) / "published"
                self.fixture(root)
                a.publish(root, output)
                if changed == "manifest_binding":
                    index = h.read_json(output / "archive.json")
                    index["original_manifest_sha256"] = "0" * 64
                    (output / "archive.json").write_text(json.dumps(index), encoding="utf-8")
                else:
                    (output / changed).write_bytes(b"tampered")
                with self.assertRaises(ValueError), a.retained_bundle(output):
                    self.fail("corrupt evidence was exposed")

    def test_unsafe_duplicate_or_unreceipted_members_reject_before_extraction(self):
        for name in ("../outside", "/absolute", "C:drive", "dir\\escape", "dir//file", "manifest.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path, output = Path(directory) / "bad.zip", Path(directory) / "extracted"
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr(name, b"{}")
                    archive.writestr("manifest.json", json.dumps({"status": "complete", "artifacts": {name: "0" * 64}}))
                with self.assertRaises(ValueError):
                    a.unpack(path, output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
