"""Independent metric accounting and deterministic diagnostic artifact checks."""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import analyze_tokenizer_failures as runner
from benchmarks import run_phase_a as stages
from benchmarks import run_research_experiments as h
from benchmarks.tokenizer_failure_metrics import Counts, analyze_condition, encode_document, readonly_tokenizer


class FixtureTokenizer:
    name = "sp_unigram"

    def __init__(self, encodings):
        pieces = [*h.SPECIALS, *(f"<0x{i:02X}>" for i in range(256))]
        for sequence in encodings.values():
            for piece in sequence:
                if piece not in pieces:
                    pieces.append(piece)
        self.vocab = {piece: i for i, piece in enumerate(pieces)}
        self.inverse = {i: piece for piece, i in self.vocab.items()}
        self.encodings = encodings
        self.model = self

    def encode(self, text):
        return [self.vocab[piece] for piece in self.encodings[text]]

    def piece_for_id(self, token_id):
        return self.inverse[token_id]


def row(text, language="en", domain="web", identity="row"):
    return {"id": identity, "language": language, "domain": domain, "raw_utf8_bytes": len(text.encode("utf-8"))}


def aggregate(records, split):
    return next(record for record in records if record["scope"] == "aggregate" and record["split"] == split)


def fixed_analysis():
    tokenizer = FixtureTokenizer(
        {"a a": ["a", "\u2581", "a"], "\u00e9!": ["<0xC3>", "<0xA9>", "!"], "a b": ["a", "\u2581", "b"]}
    )
    assignments = {
        "train": [(row("a a", identity="t1"), "a a"), (row("\u00e9!", "fr", identity="t2"), "\u00e9!")],
        "validation": [(row("a b", identity="v1"), "a b")],
    }
    return tokenizer, assignments, analyze_condition(tokenizer, assignments, rare_threshold=1)


class FailureMetricTests(unittest.TestCase):
    def _boundary_tokenizer(self):
        from uniqtoken.bpe_model import BPEModel

        pieces = [*h.SPECIALS, *(f"<0x{i:02X}>" for i in range(256)), "a", "b", "ab", " "]
        vocab = {piece: i for i, piece in enumerate(pieces)}
        model = BPEModel(
            set(vocab), vocab, {i: piece for piece, i in vocab.items()}, {("a", "b"): 0}, list(h.SPECIALS), True
        )
        return h.ResearchTokenizer("boundary_bpe", model, vocab)

    def test_immutable_boundary_reader_matches_production_ids_and_restores_state(self):
        import random

        tokenizer = self._boundary_tokenizer()
        rng = random.Random(85)
        texts = ["ab ab", "ab\tab\n", "\u00e9\u2014!ab"]
        texts.extend("".join(rng.choice("ab \t\n\u00e9\u2014!") for _ in range(rng.randint(1, 50))) for _ in range(100))
        expected = [tokenizer.encode(text) for text in texts]
        model = tokenizer.model
        originals = {name: getattr(model, name) for name in ("vocab", "token_to_id", "id_to_token", "merges")}
        with patch.object(model, "_validate_byte_fallback", wraps=model._validate_byte_fallback) as validator:
            with readonly_tokenizer(tokenizer):
                self.assertEqual([tokenizer.encode(text) for text in texts], expected)
                with self.assertRaises(TypeError):
                    model.token_to_id["new"] = 999
                with self.assertRaises(TypeError):
                    model.merges[("b", "a")] = 1
                self.assertEqual(validator.call_count, 1)
        for name, value in originals.items():
            self.assertIs(getattr(model, name), value)
        self.assertIs(tokenizer.vocab, originals["token_to_id"])
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with readonly_tokenizer(tokenizer):
                raise RuntimeError("fixture failure")
        for name, value in originals.items():
            self.assertIs(getattr(model, name), value)

    def test_immutable_boundary_reader_rejects_invalid_maps_before_encoding(self):
        tokenizer = self._boundary_tokenizer()
        tokenizer.model.id_to_token[tokenizer.vocab["<0x00>"]] = "broken"
        with self.assertRaisesRegex(ValueError, "byte fallback IDs"):
            with readonly_tokenizer(tokenizer):
                self.fail("invalid frozen snapshot was accepted")

    def test_byte_fallback_length_and_unicode_punctuation_fragmentation(self):
        text = "\u00e9\u2014!"
        tokenizer = FixtureTokenizer({text: ["<0xC3>", "<0xA9>", "<0xE2>", "<0x80>", "<0x94>", "!"]})
        counts = Counts()
        counts.observe(tokenizer, row(text), text)
        metric = counts.metrics(tokenizer.vocab, counts.frequencies, 1)
        self.assertEqual((metric["normalized_utf8_bytes"], metric["unicode_characters"], metric["tokens"]), (6, 3, 6))
        self.assertEqual(metric["byte_fallback_tokens"], 5)
        self.assertEqual(metric["token_length_bytes_histogram"], {"1": 6})
        self.assertEqual(metric["punctuation_runs"], 1)
        self.assertEqual(metric["punctuation_unicode_characters"], 2)
        self.assertEqual(metric["punctuation_token_intersections"], 4)
        self.assertEqual(metric["punctuation_excess_fragments"], 3)
        self.assertEqual(metric["punctuation_split_run_percent"], 100.0)
        self.assertIsNone(metric["whitespace_tokens_per_run"])

    def test_rarity_uses_training_only_and_includes_unseen_tokens(self):
        tokenizer, _, records = fixed_analysis()
        training, validation = aggregate(records, "train"), aggregate(records, "validation")
        self.assertEqual(training["tokens"], 6)
        self.assertEqual(training["rare_tokens"], 4)
        self.assertEqual(validation["rare_tokens"], 2)
        self.assertEqual(validation["unseen_in_diagnostic_training_tokens"], 1)
        self.assertEqual(validation["vocabulary_utilization_percent"], 100 * 3 / (len(tokenizer.vocab) - 4))

    def test_aggregate_pools_counts_and_unions_vocabulary(self):
        long = "a" * 100
        tokenizer = FixtureTokenizer({long: [long], "b": ["b"]})
        assignments = {split: [(row(long, "en"), long), (row("b", "fr"), "b")] for split in ("train", "validation")}
        records = analyze_condition(tokenizer, assignments)
        metric = aggregate(records, "validation")
        self.assertEqual(metric["tokens_per_unicode_character"], 2 / 101)
        self.assertNotEqual(metric["tokens_per_unicode_character"], (1 / 100 + 1) / 2)
        self.assertEqual(metric["observed_token_types"], 2)
        self.assertEqual(metric["token_length_bytes_p50"], 1)
        self.assertEqual(metric["token_length_bytes_p95"], 100)
        self.assertEqual(metric["token_length_bytes_histogram"], {"1": 1, "100": 1})

    def test_cross_field_emission_is_distinct_from_merge_applications(self):
        text = "a b!"
        tokenizer = FixtureTokenizer({text: ["a\u2581b", "!"]})
        counts = Counts()
        counts.observe(tokenizer, row(text), text)
        metric = counts.metrics(tokenizer.vocab, counts.frequencies, 5)
        self.assertEqual(metric["cross_word_token_percent"], 50.0)
        self.assertEqual(metric["applied_cross_word_merges"], 0)
        self.assertEqual(metric["whitespace_mixed_tokens"], 1)
        self.assertEqual(metric["punctuation_isolated_tokens"], 1)

    def test_observer_preserves_production_superbpe_output_and_method(self):
        from tests.test_subword_regularization import _make_two_pair_tokenizer, TEXT

        model = _make_two_pair_tokenizer()
        pieces = [*h.SPECIALS, *model.model.vocab]
        model.model.vocab = {piece: model.model.vocab.get(piece, -100.0) for piece in pieces}
        model.model.token_to_id = {piece: i for i, piece in enumerate(pieces)}
        model.model.id_to_token = {i: piece for piece, i in model.model.token_to_id.items()}
        model.model.special_tokens = list(h.SPECIALS)
        tokenizer = h.ResearchTokenizer("uniq_superbpe", model, model.model.token_to_id, 1)
        original = model._apply_cross_word_merges
        baseline = tokenizer.encode(TEXT)
        ids, parts, _, initial = encode_document(tokenizer, TEXT)
        self.assertEqual(ids, baseline)
        self.assertEqual(b"".join(parts), TEXT.encode())
        self.assertEqual(initial - len(ids), 2)
        self.assertEqual(model._apply_cross_word_merges, original)

    def test_reconstruction_failure_is_not_silently_scored(self):
        tokenizer = FixtureTokenizer({"ab": ["a"]})
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            encode_document(tokenizer, "ab")

    def test_source_case_and_character_counts_precede_metaspace_conversion(self):
        text = "A\tB\n"
        tokenizer = FixtureTokenizer({text: ["A", "<0x09>", "B", "<0x0A>"]})
        counts = Counts()
        counts.observe(tokenizer, row(text), text)
        metric = counts.metrics(tokenizer.vocab, counts.frequencies, 5)
        self.assertEqual(metric["whitespace_runs"], 2)
        self.assertEqual(metric["whitespace_unicode_characters"], 2)
        self.assertEqual(metric["whitespace_tokens_per_run"], 1.0)
        self.assertEqual(metric["byte_fallback_percent"], 50.0)


class FailureIntegrityTests(unittest.TestCase):
    def test_retained_frozen_run_receipt_and_accounting(self):
        evidence = Path(runner.__file__).parent / "failure_analysis" / "issue85"
        receipt = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "complete")
        for name, expected in receipt["artifacts"].items():
            self.assertEqual(hashlib.sha256((evidence / name).read_bytes()).hexdigest(), expected, name)
        result = json.loads((evidence / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(h.digest(result), receipt["results_content_sha256"])
        self.assertEqual(result["test_access"], "forbidden_not_opened_or_hashed")
        self.assertFalse(result["identity"]["working_tree_dirty"])
        self.assertEqual(len(result["models"]), 9)
        self.assertEqual(len(result["inputs"]["coverage"]), 30)
        for record in result["records"]:
            histogram = record["token_length_bytes_histogram"]
            self.assertEqual(sum(histogram.values()), record["tokens"])
            self.assertEqual(
                sum(int(length) * count for length, count in histogram.items()), record["normalized_utf8_bytes"]
            )

    def test_invalid_configuration_rejected_before_data_access(self):
        import argparse

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            args = argparse.Namespace(
                dataset=base / "inputs" / "manifest.json",
                phase_a=base / "models" / "ledger.json",
                output=base / "output",
                rare_threshold=5,
                max_training_documents_per_stratum=-1,
            )
            with patch.object(h, "read_json", side_effect=AssertionError("must not read inputs")):
                with self.assertRaisesRegex(ValueError, "training document cap"):
                    runner.run(args)

    def test_deterministic_json_csv_report_and_plots_on_fixed_fixture(self):
        _, assignments, records = fixed_analysis()
        all_records = []
        for name in runner.COHORT:
            all_records.extend({**record, "tokenizer": name} for record in records)
        payload = {
            "configuration": {"max_training_documents_per_stratum": 32, "rare_threshold": 1, "plots": True},
            "identity": {"commit_hash": "a" * 40, "source_hash": "b" * 64},
            "inputs": {
                "dataset": {"manifest_sha256": "c" * 64},
                "phase_a_ledger_sha256": "d" * 64,
                "coverage": [
                    {
                        "training_domain": "web",
                        "language": "en",
                        "training_diagnostic_available": True,
                        "screening_validation_available": True,
                        "validation_source_domains": ["flores200"],
                    }
                ],
                "diagnostic_assignments": {
                    split: runner.assignment_info(pairs) for split, pairs in assignments.items()
                },
            },
            "records": all_records,
        }
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "first", Path(temporary) / "second"
            runner.write_outputs(payload, first)
            runner.write_outputs(copy.deepcopy(payload), second)
            expected = sorted(path.name for path in first.iterdir())
            self.assertEqual(expected, sorted(path.name for path in second.iterdir()))
            for name in expected:
                self.assertEqual(first.joinpath(name).read_bytes(), second.joinpath(name).read_bytes(), name)
            manifest = json.loads(first.joinpath("manifest.json").read_text())
            for name, identity in manifest["artifacts"].items():
                self.assertEqual(hashlib.sha256(first.joinpath(name).read_bytes()).hexdigest(), identity)
            with self.assertRaises(FileExistsError):
                runner.write_outputs(payload, first)

    def test_training_cap_is_ordered_stratified_and_explicit(self):
        pairs = [
            (row("a", "en", identity="t1"), "a"),
            (row("b", "en", identity="t2"), "b"),
            (row("c", "fr", identity="t3"), "c"),
        ]
        self.assertEqual(runner.select_training(pairs, 1), [pairs[0], pairs[2]])
        self.assertEqual(runner.select_training(pairs, 0), pairs)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            runner.select_training(pairs, -1)

    def test_split_aliases_and_path_escape_rejected_without_opening_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            manifest = {"splits": {split: {"path": split + ".jsonl"} for split in ("train", "validation", "test")}}
            with patch.object(Path, "open", side_effect=AssertionError("test must not be opened")):
                paths = runner.guard_split_paths(base / "manifest.json", manifest)
                self.assertEqual(paths["test"], (base / "test.jsonl").resolve())
                alias = copy.deepcopy(manifest)
                alias["splits"]["train"]["path"] = "test.jsonl"
                with self.assertRaisesRegex(ValueError, "alias"):
                    runner.guard_split_paths(base / "manifest.json", alias)
                escape = copy.deepcopy(manifest)
                escape["splits"]["validation"]["path"] = "../test.jsonl"
                with self.assertRaisesRegex(ValueError, "escapes"):
                    runner.guard_split_paths(base / "manifest.json", escape)

    def test_shared_frozen_loader_never_opens_or_hashes_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            splits = {}
            for split, text in (("train", "training only"), ("validation", "validation only")):
                path = base / (split + ".jsonl")
                record = {**row(text, identity=split), "text": text, "normalized_utf8_bytes": len(text.encode())}
                path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                splits[split] = {"path": path.name, "sha256": h.file_hash(path)}
            splits["test"] = {"path": "DO_NOT_OPEN.jsonl", "sha256": "f" * 64}
            manifest = {
                "schema_version": h.DATASET_MANIFEST_SCHEMA,
                "dataset_id": "fixed-fixture",
                "normalization": h.NORMALIZATION,
                "freeze": {"immutable": True, "source_revisions": {}},
                "splits": splits,
            }
            manifest_path = base / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            original_open = Path.open

            def guarded_open(path, *args, **kwargs):
                self.assertNotEqual(path.name, "DO_NOT_OPEN.jsonl")
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", guarded_open):
                _, train, _, validation, source = stages.load_stage_source(manifest_path)
            self.assertEqual(train, ["training only"])
            self.assertEqual(validation, ["validation only"])
            self.assertEqual(source["untouched_test_file_sha256"], "f" * 64)

    def test_legacy_validation_gate_detects_drift(self):
        _, _, records = fixed_analysis()
        metric = aggregate(records, "validation")
        original = {"validation": dict(metric)}
        runner.check_legacy_metrics(records, original)
        original["validation"]["tokens"] += 1
        with self.assertRaisesRegex(ValueError, "metric drift"):
            runner.check_legacy_metrics(records, original)


if __name__ == "__main__":
    unittest.main()
