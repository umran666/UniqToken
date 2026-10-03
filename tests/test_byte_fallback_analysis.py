"""Regression tests for reviewed PR #124; no external dataset required."""

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import byte_fallback_analysis as b
from benchmarks import run_research_experiments as h
from uniqtoken.byte_codec import ByteFallbackEngine as Bytes
from uniqtoken.tokenizer import CustomTokenizer
from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
from uniqtoken.unigram_trainer import UnigramModel


def byte_model():
    pieces = [*h.SPECIALS, *(Bytes.byte_to_token(i) for i in range(256)), "a", "b", "\u2581"]
    ids = {t: i for i, t in enumerate(pieces)}
    return UnigramModel({t: -math.log(len(ids)) for t in pieces}, ids, {i: t for t, i in ids.items()}, list(h.SPECIALS))


class ByteFallbackReviewTests(unittest.TestCase):
    def test_retained_exact_budget_receipts_and_span_accounting(self):
        root = Path(b.__file__).parent / "byte_fallback" / "issue86"
        receipt = h.read_json(root / "manifest.json")
        self.assertEqual(receipt["status"], "complete")
        for name, digest in receipt["artifacts"].items():
            self.assertEqual(h.file_hash(root / name), digest, name)
        result = h.read_json(root / "results.json")
        self.assertEqual(set(result["runs"]), {"8192", "16384", "32768"})
        self.assertFalse(result["identity"]["working_tree_dirty"])
        self.assertEqual(result["assignments"]["source"]["test_access"], "forbidden_not_opened")
        for budget, conditions in result["runs"].items():
            self.assertEqual(set(conditions), set(b.CONDITIONS))
            baseline = conditions["baseline"]["strata"]
            for condition, measured in conditions.items():
                self.assertEqual(measured["actual_vocab_size"], int(budget))
                self.assertEqual(measured["incomplete_prefix_additions"], 0)
                self.assertEqual(len(measured["merges"]) + len(measured["recovered"]), 64)
                passed, regressions = b.evaluate_regressions(baseline, measured["strata"], set(baseline))
                self.assertTrue(passed)
                self.assertEqual(measured["regression_gate_passed"], passed)
                self.assertEqual(measured["regressions_pct"], regressions)
                for stratum in measured["strata"].values():
                    histogram = stratum["span_stats"]["histogram_bytes"]
                    self.assertEqual(sum(int(k) * v for k, v in histogram.items()), stratum["fallback_tokens"])
                directory = root / f"{budget}-{condition}"
                self.assertEqual(h.artifact_hashes(directory), measured["artifact_hashes"])
                tok = CustomTokenizer.load(directory, prefer_binary=False)
                h.validate_tokenizer(
                    h.ResearchTokenizer("uniq_superbpe", tok, tok.model.token_to_id, len(measured["merges"])),
                    int(budget),
                )
                for text in ("literal <0xE0><0xA4>", "a\u093e\U0001f600 b", "\u4e2d\u6587", "\u0639\u0631\u0628\u064a"):
                    self.assertEqual(tok.decode(tok.encode_to_ids(text)), h.normalize(text))

    def test_span_units_are_bytes(self):
        self.assertEqual(b.extract_fallback_spans(["a", "<0xE0>", "<0xA4>", "<0xBE>", "b", "<0xC3>", "<0xA9>"]), [3, 2])
        self.assertEqual(b.extract_fallback_spans([]), [])
        self.assertEqual(b.extract_fallback_spans(["a"]), [])

    def test_exact_histogram_and_nearest_rank_percentiles(self):
        result = b.compute_span_metrics([1, 2, 3, 3, 4, 5, 6, 8, 12])
        self.assertEqual(result["histogram_bytes"], {"1": 1, "2": 1, "3": 2, "4": 1, "5": 1, "6": 1, "8": 1, "12": 1})
        self.assertEqual(result["p50"], 4)
        self.assertEqual(result["p95"], 12)
        self.assertIsNone(b.compute_span_metrics([])["p95"])

    def test_no_multibyte_notation_extension(self):
        self.assertFalse(Bytes.is_byte_token("<0xE0><0xA4>"))
        self.assertEqual(Bytes.decode_tokens(["<0xE0><0xA4>"]), "<0xE0><0xA4>")

    def test_atomic_recovery_all_utf8_widths(self):
        model = byte_model()
        chars = "\u00e9\u093e\U0001f600"
        updated, records = b.recover_characters(model, [chars] * 4, 3)
        self.assertEqual({row["token"] for row in records}, set(chars))
        self.assertEqual({row["utf8_bytes"] for row in records}, {2, 3, 4})
        self.assertEqual(len(updated.vocab), len(model.vocab) + 3)
        self.assertEqual(updated.encode(chars), list(chars))
        self.assertEqual(Bytes.decode_tokens(updated.encode(chars)), chars)
        self.assertTrue(all(updated.token_to_id[t] == i for t, i in model.token_to_id.items()))
        self.assertFalse(any(t.startswith("<0x") for t in set(updated.vocab) - set(model.vocab)))
        self.assertNotIn(chars[0], model.vocab)

    def test_singletons_and_zero_limit_do_not_add_tokens(self):
        model = byte_model()
        self.assertEqual(b.recover_characters(model, ["\u00e9"], 10), (model, []))
        self.assertEqual(b.recover_characters(model, ["\u00e9"] * 3, 0), (model, []))

    def test_weight_and_limit_validation(self):
        for weight in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                b.recover_characters(byte_model(), [], 1, weight)
        for limit in (-1, 1.5):
            with self.assertRaises(ValueError):
                b.recover_characters(byte_model(), [], limit)

    def test_weight_formula_and_candidate_set(self):
        model = byte_model()
        chunks = ["\u00e9\u093e\U0001f600"] * 4
        plain = {r["token"]: r for r in b.recovery_candidates(model, chunks)}
        weighted = {r["token"]: r for r in b.recovery_candidates(model, chunks, 5)}
        self.assertEqual(set(plain), set(weighted))
        for token, row in plain.items():
            self.assertAlmostEqual(weighted[token]["score"], row["score"] - 5 * row["frequency"] * row["utf8_bytes"])

    def test_deterministic_under_training_order(self):
        chunks = ["\u00e9", "\u093e", "\U0001f600"] * 4
        self.assertEqual(
            b.recovery_candidates(byte_model(), chunks), b.recovery_candidates(byte_model(), reversed(chunks))
        )

    def test_roundtrip_ids_offsets_and_reload(self):
        model, _ = b.recover_characters(byte_model(), ["\u00e9\u093e\U0001f600"] * 4, 3)
        tok = CustomTokenizer(Normalizer(), RegexPreTokenizer(), model)
        text = "a\u00e9\u093e\U0001f600 b"
        ids = tok.encode_to_ids(text)
        self.assertEqual(tok.decode(ids), text)
        self.assertEqual([item.text for item in tok.encode_with_offsets(text)], tok.encode(text))
        with tempfile.TemporaryDirectory() as directory:
            tok.save(directory, save_binary=False)
            loaded = CustomTokenizer.load(directory, prefer_binary=False)
            self.assertEqual(loaded.encode_to_ids(text), ids)
            self.assertEqual(loaded.decode(ids), text)

    def test_aggregate_bytes_are_normalized_source(self):
        tok = CustomTokenizer(Normalizer(), RegexPreTokenizer(), byte_model())
        metrics, emitted = b.evaluate_stratum(tok, ["a b", "\u093e"])
        self.assertEqual(metrics["normalized_utf8_bytes"], 6)
        self.assertEqual(metrics["fallback_tokens"], 3)
        self.assertEqual(sum(emitted.values()), metrics["total_tokens"])

    def test_regression_gate_rejects_missing_or_undefined_strata(self):
        baseline = {"hi": {"bytes_per_token": 2.0}, "code": {"bytes_per_token": 2.0}}
        for candidate in ({}, {"hi": {"bytes_per_token": None}}, {"hi": {"bytes_per_token": float("nan")}}):
            with self.assertRaises(ValueError):
                b.evaluate_regressions(baseline, candidate, {"hi", "code"})
        with self.assertRaises(ValueError):
            b.evaluate_regressions(baseline, baseline, set())

    def test_regression_gate_includes_tail_languages(self):
        baseline = {"hi": {"bytes_per_token": 2.0}, "code": {"bytes_per_token": 2.0}}
        candidate = {"hi": {"bytes_per_token": 1.96}, "code": {"bytes_per_token": 2.1}}
        passed, values = b.evaluate_regressions(baseline, candidate, set(baseline))
        self.assertFalse(passed)
        self.assertAlmostEqual(values["hi"], 2.0)
        self.assertEqual(values["code"], 0.0)

    def test_exact_budget_rejected_for_undersized_seed(self):
        with self.assertRaisesRegex(ValueError, "exact vocabulary"):
            b.train_base(["a b"] * 4, 8192)

    def test_final_budget_rejected_when_merges_exhausted(self):
        base = CustomTokenizer(Normalizer(), RegexPreTokenizer(), byte_model())
        with self.assertRaisesRegex(ValueError, "exact vocabulary"):
            b.make_condition(base, ["a"], 300, 37, "baseline", 1, 5)

    def test_stratum_sample_order_and_excerpt_are_fixed(self):
        rows = [{"id": str(i), "domain": "web", "language": lang} for i, lang in enumerate(["hi", "en", "hi", "en"])]
        picked = b.select_records(rows, ["abcd", "efgh", "ijkl", "mnop"], 1, 2)
        self.assertEqual([(r["id"], t) for r, t in picked], [("0", "ab"), ("1", "ef")])

    def test_split_alias_guard_runs_before_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = {
                "splits": {
                    "train": {"path": "test.jsonl"},
                    "validation": {"path": "v.jsonl"},
                    "test": {"path": "test.jsonl"},
                }
            }
            with (
                patch.object(h, "read_json", return_value=manifest),
                patch.object(b.stages, "load_stage_source") as loader,
            ):
                with self.assertRaisesRegex(ValueError, "alias"):
                    b.assignments(path)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
