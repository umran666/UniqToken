"""Script-safe denominators and schema accounting for issue #90."""

import copy
from pathlib import Path
import unittest

from benchmarks import token_density as d
from benchmarks import run_research_experiments as h
from benchmarks.ledger import SCHEMA_VERSION, validate_ledger
from benchmarks.tokenizer_failure_metrics import analyze_condition
from tests.test_tokenizer_failure_analysis import FixtureTokenizer, row


def fixture():
    cases = {
        "zh": ("\u4e2d\u6587", ["\u4e2d", "\u6587"]),
        "hi": ("\u0915\u093f", ["<0xE0>", "<0xA4>", "<0x95>", "<0xE0>", "<0xA4>", "<0xBF>"]),
        "ar": ("\u0639\u0631\u0628\u064a", ["\u0639\u0631", "\u0628\u064a"]),
        "ru": ("\u0442\u0435\u0441\u0442", ["\u0442\u0435\u0441\u0442"]),
        "code": ("x += 1;", ["x", "\u2581", "+=", "\u2581", "1;"]),
        "punct": ("!?!...", ["!?!", "..."]),
    }
    tok = FixtureTokenizer({text: pieces for text, pieces in cases.values()})
    assignments = {
        split: [(row(text, lang, "code" if lang == "code" else "web"), text) for lang, (text, _) in cases.items()]
        for split in ("train", "validation")
    }
    records = [d.project_record(r) for r in analyze_condition(tok, assignments)]
    return {
        "density_schema_version": d.SCHEMA,
        "metadata": {
            "ledger_schema_version": SCHEMA_VERSION,
            "commit_hash": "a" * 40,
            "working_tree_dirty": False,
            "data_split": "document_disjoint_fixture",
        },
        "records": records,
    }


class TokenDensityTests(unittest.TestCase):
    def test_multiscript_density_and_byte_lengths(self):
        payload = d.validate_density_ledger(fixture())
        langs = {r["language"]: r for r in payload["records"] if r["scope"] == "language" and r["split"] == "train"}
        self.assertEqual(set(langs), {"zh", "hi", "ar", "ru", "code", "punct"})
        self.assertEqual(langs["zh"]["tokens_per_unicode_character"], 1)
        self.assertEqual(langs["zh"]["tokens_per_normalized_utf8_byte"], 1 / 3)
        self.assertEqual(langs["hi"]["tokens_per_unicode_character"], 3)
        self.assertEqual(langs["hi"]["token_length_bytes_histogram"], {"1": 6})
        self.assertEqual(langs["ar"]["bytes_per_token"], 4)
        self.assertEqual(langs["ru"]["token_length_bytes_p95"], 8)
        self.assertEqual(langs["code"]["unicode_characters"], 7)
        self.assertEqual(langs["punct"]["token_length_bytes_histogram"], {"3": 2})

    def test_empty_ratios_and_percentiles_are_null(self):
        result = d.density_metrics(0, 0, 0, {})
        self.assertIsNone(result["tokens_per_unicode_character"])
        self.assertIsNone(result["tokens_per_normalized_utf8_byte"])
        self.assertIsNone(result["token_length_bytes_p99"])

    def test_percentiles_are_nearest_rank_and_order_independent(self):
        a = d.density_metrics(100, 199, 199, {"100": 1, "1": 99})
        b = d.density_metrics(100, 199, 199, {"1": 99, "100": 1})
        self.assertEqual(a, b)
        self.assertEqual(a["token_length_bytes_p99"], 1)
        self.assertEqual(a["token_length_bytes_max"], 100)

    def test_invalid_histograms_rejected(self):
        for args in [
            (2, 3, 3, {"1": 2}),
            (3, 2, 2, {"1": 2}),
            (2, 2, 2, {"01": 2}),
            (2, 2, 2, {"1": True}),
            (2, 2, 2, {"0": 2}),
            (1, 8, 1, {"8": 1}),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                d.density_metrics(*args)

    def test_nested_universal_fertility_rejected_by_both_schemas(self):
        for key in ("fertility", "tokens_per_word", "WORD_FERTILITY"):
            payload = fixture()
            payload["records"][0]["metrics"] = [{"nested": {key: 2.0}}]
            for validator in (validate_ledger, d.validate_density_ledger):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "ambiguous"):
                    validator(payload)

    def test_aggregate_is_pooled_not_mean_of_stratum_ratios(self):
        payload = fixture()
        total = next(r for r in payload["records"] if r["scope"] == "aggregate" and r["split"] == "train")
        strata = [r for r in payload["records"] if r["scope"] == "stratum" and r["split"] == "train"]
        self.assertEqual(
            total["tokens_per_unicode_character"],
            sum(r["tokens"] for r in strata) / sum(r["unicode_characters"] for r in strata),
        )
        self.assertNotEqual(
            total["tokens_per_unicode_character"], sum(r["tokens_per_unicode_character"] for r in strata) / len(strata)
        )
        total["tokens_per_unicode_character"] = 999
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            d.validate_density_ledger(payload)

    def test_missing_duplicate_or_test_rows_rejected(self):
        baseline = fixture()
        missing = copy.deepcopy(baseline)
        missing["records"] = [r for r in missing["records"] if r["scope"] != "language"]
        duplicate = copy.deepcopy(baseline)
        duplicate["records"].append(duplicate["records"][0])
        heldout = copy.deepcopy(baseline)
        heldout["records"][0]["split"] = "test"
        metric = copy.deepcopy(baseline)
        del metric["records"][0]["bytes_per_token"]
        for payload in (missing, duplicate, heldout, metric):
            with self.assertRaises(ValueError):
                d.validate_density_ledger(payload)

    def test_missing_canonical_fields_fail_with_schema_error(self):
        record = fixture()["records"][0]
        for name in ("vocab_budget", "tokens", "token_length_bytes_histogram"):
            invalid = {k: v for k, v in record.items() if k != name}
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "canonical"):
                d.project_record(invalid)
        with self.assertRaisesRegex(ValueError, "object"):
            d.project_record(None)

    def test_retained_failure_diagnostics_project_without_retokenization(self):
        path = Path(d.__file__).parent / "failure_analysis" / "issue85" / "results.json"
        source = h.read_json(path)
        payload = fixture()
        payload["records"] = [d.project_record(r) for r in source["records"]]
        d.validate_density_ledger(payload)
        self.assertGreater(len(payload["records"]), 100)


if __name__ == "__main__":
    unittest.main()
