"""NFKC fast-path contract, including transforms that must not be skipped."""

import itertools
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from benchmarks import profile_nfkc_fast_path as profiler
import uniqtoken.pre_tokenizer as pre_tokenizer


@unittest.skipUnless(pre_tokenizer._HAS_RUST_NORM, "native normalizer required")
class NfkcFastPathTests(unittest.TestCase):
    def test_all_flags_preserve_text_and_alignment(self):
        # Python is an independent oracle on these Unicode-13-compatible
        # fixtures. The full before/after gate also covers contextual cases
        # where existing token-only and aligned lowercasing can differ.
        texts = profiler.TEXTS[:14]
        for values, text in itertools.product(itertools.product((False, True), repeat=6), texts):
            normalizer = pre_tokenizer.Normalizer(**dict(zip(profiler.FLAGS, values)))
            actual = normalizer.normalize_with_alignment(text)
            with (
                patch.object(pre_tokenizer, "_HAS_RUST_NORM", False),
                patch.object(pre_tokenizer, "native_function", return_value=None),
            ):
                expected = normalizer.normalize_with_alignment(text)
            self.assertEqual(actual, expected, (values, repr(text)))

    def test_token_only_ascii_still_applies_all_transforms(self):
        for values in itertools.product((False, True), repeat=6):
            normalizer = pre_tokenizer.Normalizer(space_char="@", **dict(zip(profiler.FLAGS, values)))
            text = "  A@ B\t\tC\r\nD \x1c"
            self.assertEqual(normalizer.normalize(text), normalizer.normalize_with_alignment(text)[0])

    def test_native_security_is_independent_of_normalizer_flag(self):
        from benchmarks.profile_hot_paths import make_tokenizer

        tok = make_tokenizer()
        native = pre_tokenizer._uniqtoken_core
        trie = tok.model._get_rust_trie()
        for flag, text in itertools.product(
            (False, True), ("<|unk|>", "\uff1c\uff5cunk\uff5c\uff1e", "\ufe64\uff5csystem\uff5c\uff1e")
        ):
            with self.assertRaisesRegex(ValueError, "control-token syntax after NFKC"):
                native.rust_encode_text_native(text, trie, normalize_unicode=flag)
        for text in ("\ue000", "\ue001"):
            with self.assertRaisesRegex(ValueError, "private-use metaspace"):
                native.rust_encode_text_native(text, trie)

    def test_unsupported_native_inputs_are_rejected(self):
        native = pre_tokenizer._uniqtoken_core
        for value in (None, 1, b"ASCII", b"\xff", []):
            with self.assertRaises(TypeError):
                native.rust_normalize(value)
        with self.assertRaises(UnicodeEncodeError):
            native.rust_normalize("\ud800")
        for space in ("\ue000", "\ue001"):
            with self.assertRaisesRegex(ValueError, "reserved metaspace"):
                native.rust_normalize("ASCII", space)


class NfkcEvidenceTests(unittest.TestCase):
    def test_receipt_format_preserves_nested_values(self):
        value = {"samples": [0.001, 1e-20, -1.5, 42], "rows": [{"name": "[1, 2]", "interval": [0.9, 1.1]}]}
        self.assertEqual(json.loads(profiler.json_text(value)), value)
        self.assertIn('"interval": [0.9,1.1]', profiler.json_text(value))

    def test_published_fixture_and_parity_receipt(self):
        path = Path(__file__).resolve().parents[1] / "benchmarks/profiles/issue102/results.json"
        if not path.exists():
            self.skipTest("evidence has not yet been recorded")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["fixtures_sha256"], profiler.digest(profiler.fixtures()))
        self.assertEqual(receipt["parity"]["misses"], 0)
        self.assertEqual(receipt["parity"]["configurations"], 64)
        self.assertEqual(receipt["parity"]["texts"], len(profiler.TEXTS))
        manifest = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8"))
        self.assertEqual({item["path"] for item in manifest["artifacts"]}, {"REPORT.md", "SAFETY.md", "results.json"})
        for item in manifest["artifacts"]:
            contents = path.with_name(item["path"]).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(contents).hexdigest(), item["sha256"])
        for variant in ("baseline", "optimized"):
            self.assertEqual(manifest["native_execution"][variant], {"single_cases": 10, "batch_cases": 10})
        expected = {f"{name}_{batch}" for name in profiler.fixtures() for batch in (1, 32)}
        self.assertEqual({row["name"] for row in receipt["workloads"]}, expected)
        for row in receipt["workloads"]:
            for variant in ("baseline", "optimized"):
                samples = row[variant]["samples_ms"]
                self.assertEqual(len(samples), receipt["repetitions"])
                self.assertEqual(row[variant], profiler.timing_summary(samples))
            ratios = [a / b for b, a in zip(row["baseline"]["samples_ms"], row["optimized"]["samples_ms"])]
            self.assertEqual(row["paired_ratio_interval_95"], profiler.paired_interval(ratios))


if __name__ == "__main__":
    unittest.main()
