"""Unit tests for the matched-budget vocab quality race harness.

The harness itself is exercised end-to-end by
``benchmarks/vocab_quality_race.py``. These tests focus on the report
*shape* and the BPB *invariant*, not on running the full race, because
a full race trains three tokenizers from scratch and is too slow for a
unit test.

The integration tests are skipped if the heavy import chain
``transformers + sklearn + pandas + pyarrow`` is not importable in
the current environment (some Windows pyarrow builds crash on import,
see https://github.com/apache/arrow/issues).
"""

from __future__ import annotations

import json
import math
import unittest
import warnings
from typing import Any

from benchmarks.vocab_quality_race import (
    RaceEntry,
    RaceReport,
    run_vocab_quality_race,
)


def _env_can_run_race() -> bool:
    """Return True if the heavy import chain is importable.

    The chain is the same one pytest's collection may pull in on some
    platforms. If any link raises (e.g. a Windows pyarrow native
    crash), we skip the integration tests cleanly rather than fail.

    The probe runs in a short-lived subprocess so that a *native*
    crash (e.g. Windows access violation in pyarrow) does not take
    down the pytest process. The subprocess has a 10-second timeout.

    Result is memoized; the probe is only run once per process.
    """
    cached = getattr(_env_can_run_race, "_result", None)
    if cached is not None:
        return bool(cached)

    import subprocess
    import sys

    probe = (
        "import sys\n"
        "try:\n"
        "    import transformers\n"
        "    import sklearn\n"
        "    import pandas\n"
        "    import pyarrow\n"
        "    print('PROBE_OK')\n"
        "except Exception as e:\n"
        "    sys.exit(1)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=10,
        )
        ok = result.returncode == 0 and "PROBE_OK" in result.stdout
    except (subprocess.TimeoutExpired, OSError):
        ok = False

    setattr(_env_can_run_race, "_result", ok)
    return ok


@unittest.skipUnless(
    True,
    "transformers/sklearn/pandas/pyarrow import chain is not healthy in this environment",
)
class VocabQualityRaceHarnessTests(unittest.TestCase):
    """Run a single small race per class and assert structural invariants."""

    _TEST_BUDGET = 500
    report: Any

    @classmethod
    def setUpClass(cls):
        if not _env_can_run_race():
            raise unittest.SkipTest(
                "transformers/sklearn/pandas/pyarrow import chain is not healthy in this environment"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.report = run_vocab_quality_race(
                budget=cls._TEST_BUDGET,
                steps=2,
                include_pretrained=False,
                include_sentencepiece=False,
                train_kwargs={
                    "seq_len": 16,
                    "batch_size": 2,
                    "dim": 32,
                    "heads": 2,
                    "layers": 1,
                },
            )

    def test_report_shape(self):
        names = [e.tokenizer for e in self.report.entries]
        self.assertIn("UniqToken (Unigram)", names)
        self.assertIn("UniqToken (BPE)", names)
        self.assertIn("UniqToken (SuperBPE)", names)
        for e in self.report.entries:
            self.assertGreater(e.actual_vocab, 0)
            self.assertGreater(e.evaluated_bytes, 0)
            self.assertGreater(e.evaluated_tokens, 0)
            self.assertTrue(math.isfinite(e.final_loss))
            self.assertTrue(math.isfinite(e.bits_per_byte))

    def test_bpb_invariant(self):
        for e in self.report.entries:
            if e.evaluated_bytes == 0:
                continue
            expected = e.final_loss * e.evaluated_tokens / (e.evaluated_bytes * math.log(2.0))
            self.assertAlmostEqual(e.bits_per_byte, expected, places=4)

    def test_categories_set_correctly(self):
        cats = {e.category for e in self.report.entries}
        self.assertIn("uniqtoken", cats)
        for e in self.report.entries:
            self.assertTrue(e.trained_fresh)
            self.assertEqual(e.category, "uniqtoken")

    def test_to_dict_is_json_serializable(self):
        json.dumps(self.report.to_dict())


class RaceEntryConstructionTests(unittest.TestCase):
    """Pure dataclass tests that do not run any race."""

    def test_race_entry_required_fields(self):
        e = RaceEntry(
            tokenizer="test",
            category="uniqtoken",
            target_vocab=500,
            actual_vocab=500,
            trained_fresh=True,
            bytes_per_token=3.0,
            evaluated_tokens=100,
            evaluated_bytes=300,
            final_loss=1.0,
            bits_per_byte=0.5,
            tokens_per_sec=1000.0,
            bytes_per_sec=3000.0,
            wallclock_sec=1.0,
        )
        for k in (
            "tokenizer",
            "category",
            "target_vocab",
            "actual_vocab",
            "trained_fresh",
            "bytes_per_token",
            "evaluated_tokens",
            "evaluated_bytes",
            "final_loss",
            "bits_per_byte",
            "tokens_per_sec",
            "bytes_per_sec",
            "wallclock_sec",
        ):
            self.assertIn(k, e.__dict__)

    def test_race_report_to_dict_round_trip(self):
        report = RaceReport(
            budget=500,
            corpus_size_documents=10,
            corpus_size_bytes=1000,
            steps=4,
            seed=42,
        )
        d = report.to_dict()
        self.assertEqual(d["budget"], 500)
        self.assertEqual(d["steps"], 4)
        self.assertEqual(d["entries"], [])
        json.dumps(d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
