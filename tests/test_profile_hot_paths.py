"""The diagnostic replay must fail closed before reporting measurements."""

import unittest
from unittest.mock import patch

from benchmarks import profile_hot_paths as profiler


class HotPathProfilerTests(unittest.TestCase):
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
