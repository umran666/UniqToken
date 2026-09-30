"""The diagnostic replay must fail closed before reporting measurements."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from benchmarks import profile_hot_paths as profiler


class HotPathProfilerTests(unittest.TestCase):
    def test_observed_range_includes_all_samples(self):
        self.assertEqual(profiler.observed_range([1.0, 3.0, 2.0]), [1.0, 3.0])

    def test_reference_spans_match_offsets_for_escaped_special_token(self):
        tok = profiler.make_tokenizer()
        text = "  hello <|endoftext|> world  "
        expected = [token.raw_span for token in tok.encode_with_offsets(text)]
        self.assertEqual(profiler.reference_raw_spans(tok, text, tok.encode(text)), expected)

    def test_profile_parity_gate_rejects_corrupt_offsets(self):
        import uniqtoken_core

        tok = profiler.make_tokenizer()
        text = profiler.FIXTURES["short"]
        original = tok.encode_with_offsets

        def corrupt_offsets(value):
            rows = original(value)
            return [replace(rows[0], raw_span=(0, 0)), *rows[1:]]

        with patch.object(tok, "encode_with_offsets", side_effect=corrupt_offsets):
            with self.assertRaisesRegex(AssertionError, "exact offset parity"):
                profiler.validate(tok, uniqtoken_core, [("short_single", [text])])

    def test_profile_parity_gate_rejects_corrupt_ids(self):
        import uniqtoken_core

        tok = profiler.make_tokenizer()
        text = profiler.FIXTURES["short"]
        original = uniqtoken_core.rust_profile_native_batch

        def corrupt_ids(*args, **kwargs):
            rows = original(*args, **kwargs)
            tokens, ids, stages = rows[0]
            return [(tokens, [ids[0] + 1, *ids[1:]], stages)]

        with patch.object(uniqtoken_core, "rust_profile_native_batch", side_effect=corrupt_ids):
            with self.assertRaisesRegex(AssertionError, "diagnostic native replay parity"):
                profiler.validate(tok, uniqtoken_core, [("short_single", [text])])
