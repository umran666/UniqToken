"""Regression tests for benchmark research-integrity contracts."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.benchmark_suite import TokenizerBenchmarkSuite
from benchmarks.benchmark_throughput import _train_sentencepiece
from benchmarks.downstream_eval import DownstreamEvaluator
from benchmarks.run_matched_budget_eval import (
    BenchmarkRecord,
    EXPERIMENT_VERSION,
    LEDGER_SCHEMA_VERSION,
    LM_CONFIGS,
    generate_balanced_multilingual_corpus,
    run_benchmark,
)
from benchmarks.train_toy_transformer import (
    PRETRAINING_CORPUS,
    PretrainingMetrics,
    _require_vocab_budget,
    _split_documents,
    run_pretraining_benchmark,
    train_superbpe_tokenizer,
    train_toy_transformer,
)
from benchmarks.vocab_quality_race import RaceEntry, RaceReport


class DataSeparationTests(unittest.TestCase):
    def test_downstream_evaluator_rejects_overlapping_documents(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            DownstreamEvaluator(
                vocab_size=500,
                training_corpus=["same document"],
                evaluation_corpus=["same document"],
            )

    def test_benchmark_suite_rejects_overlapping_documents(self):
        fake_tokenizer = SimpleNamespace(vocab_size=500)
        with self.assertRaisesRegex(ValueError, "overlap"):
            TokenizerBenchmarkSuite(
                tokenizer=fake_tokenizer,
                training_corpus=["same document"],
                evaluation_corpora={"test": "same document"},
            )

    def test_generated_matched_budget_documents_are_disjoint(self):
        train_docs, val_by_domain = generate_balanced_multilingual_corpus(num_docs_per_lang=10, seed=91)
        validation_docs = [doc for text in val_by_domain.values() for doc in text.splitlines()]
        self.assertTrue(set(train_docs).isdisjoint(validation_docs))

    def test_downstream_metrics_use_only_explicit_evaluation_documents(self):
        evaluator = DownstreamEvaluator(
            vocab_size=500,
            training_corpus=["training material only"],
            evaluation_corpus=["held out material only"],
        )
        seen = []

        def encode(text: str):
            seen.append(text)
            return [1, 2, 3]

        metrics = evaluator.evaluate_tokenizer("fake", encode, vocab_size=3)
        self.assertEqual(seen, ["held out material only"])
        self.assertEqual(metrics.total_bytes, len("held out material only".encode("utf-8")))
        self.assertEqual(metrics.model_kind, "tokenizer_only")


class BudgetAndConfigurationTests(unittest.TestCase):
    def test_exact_vocab_budget_is_required(self):
        with self.assertRaisesRegex(RuntimeError, "499 pieces"):
            _require_vocab_budget(SimpleNamespace(vocab_size=499), 500, "candidate")

    def test_throughput_sentencepiece_budget_mismatch_fails(self):
        fake_spm = SimpleNamespace(
            SentencePieceTrainer=SimpleNamespace(train=lambda **_kwargs: None),
            SentencePieceProcessor=lambda **_kwargs: SimpleNamespace(get_piece_size=lambda: 499),
        )
        with self.assertRaisesRegex(RuntimeError, "499 pieces"):
            _train_sentencepiece(fake_spm, ["training only"], target_vocab=500)

    def test_superbpe_rejects_zero_merge_result(self):
        import benchmarks.train_toy_transformer as module

        class FakeTokenizer:
            def __init__(self, normalizer=None, pre_tokenizer=None, model=None):
                self.normalizer = normalizer
                self.pre_tokenizer = pre_tokenizer
                self.model = model

            @staticmethod
            def train_from_corpus(**kwargs):
                return SimpleNamespace(
                    normalizer=SimpleNamespace(normalize=lambda text: text),
                    pre_tokenizer=SimpleNamespace(pre_tokenize=lambda text: [text]),
                    model=SimpleNamespace(vocab={str(i): 0.0 for i in range(kwargs["target_vocab_size"])}),
                )

        class NoOpCEM:
            def __init__(self, **_kwargs):
                self.merges = []

            def optimize(self, model, chunks):
                return model

        with (
            patch.object(module, "CustomTokenizer", FakeTokenizer),
            patch.object(module, "CrossEntropyMerging", NoOpCEM),
        ):
            with self.assertRaisesRegex(RuntimeError, "no-op"):
                train_superbpe_tokenizer(["alpha beta"], target_vocab=500)

    def test_superbpe_rejects_zero_requested_merges(self):
        with self.assertRaisesRegex(ValueError, "max_merges"):
            train_superbpe_tokenizer(["alpha beta"], target_vocab=500, max_merges=0)

    def test_matched_harness_rejects_invalid_configuration_before_running(self):
        with self.assertRaisesRegex(ValueError, "vocab_budgets"):
            run_benchmark([], [next(iter(LM_CONFIGS))], verbose=False)
        with self.assertRaisesRegex(ValueError, "invalid tiers"):
            run_benchmark([1024], ["not-a-tier"], verbose=False)

    def test_transformer_harness_does_not_substitute_short_sequence_model(self):
        fake_tokenizer = SimpleNamespace(
            vocab_size=8,
            model=SimpleNamespace(token_to_id={str(i): i for i in range(8)}),
            encode_to_ids=lambda text: [ord(char) % 8 for char in text],
        )
        with self.assertRaisesRegex(ValueError, "requires at least"):
            train_toy_transformer(
                fake_tokenizer,
                "fake",
                ["train", "validation", "test"],
                steps=1,
                seq_len=100,
            )


class LedgerContractTests(unittest.TestCase):
    @staticmethod
    def _metrics() -> PretrainingMetrics:
        return PretrainingMetrics(
            model_name="fake",
            model_kind="causal_transformer",
            vocab_size=500,
            total_tokens=30,
            total_bytes=60,
            evaluated_tokens=5,
            evaluated_bytes=10,
            compression_ratio=2.0,
            validation_loss=1.2,
            validation_evaluated_tokens=5,
            final_loss=1.1,
            bits_per_byte=0.8,
            tokens_per_sec=100.0,
            bytes_per_sec=200.0,
        )

    def test_pretraining_json_records_model_kind(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ledger.json"
            with (
                patch("benchmarks.train_toy_transformer.create_tokenizers", return_value={"fake": object()}),
                patch("benchmarks.train_toy_transformer.train_toy_transformer", return_value=self._metrics()),
                redirect_stdout(io.StringIO()),
            ):
                run_pretraining_benchmark(steps=1, export_json=str(output))
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["records"][0]["model_kind"], "causal_transformer")

    def test_all_active_lm_ledger_rows_declare_model_kind(self):
        self.assertIn("model_kind", BenchmarkRecord.__annotations__)
        self.assertIn("lm_bits_per_byte", BenchmarkRecord.__annotations__)
        self.assertNotIn("true_lm_bpb", BenchmarkRecord.__annotations__)
        race_entry = RaceEntry(
            tokenizer="fake",
            model_kind="causal_transformer",
            category="uniqtoken",
            target_vocab=500,
            actual_vocab=500,
            trained_fresh=True,
            bytes_per_token=2.0,
            evaluated_tokens=5,
            evaluated_bytes=10,
            final_loss=1.0,
            bits_per_byte=0.7,
            tokens_per_sec=10.0,
            bytes_per_sec=20.0,
            wallclock_sec=0.1,
        )
        report = RaceReport(500, 3, 100, 1, 42, entries=[race_entry])
        self.assertEqual(report.to_dict()["entries"][0]["model_kind"], "causal_transformer")
        self.assertEqual(report.to_dict()["ledger_schema_version"], 3)

    def test_current_schema_cannot_be_confused_with_archived_ledger(self):
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / "benchmarks" / "matched_budget_eval_records.json").exists())
        archived = root / "benchmarks" / "legacy" / "matched_budget_eval_records_pre_integrity.json"
        self.assertTrue(archived.exists())
        archived_payload = json.loads(archived.read_text(encoding="utf-8"))
        self.assertNotEqual(archived_payload.get("metadata", {}).get("ledger_schema_version"), LEDGER_SCHEMA_VERSION)
        self.assertEqual(EXPERIMENT_VERSION, "research-integrity-heldout-v3")

    def test_three_way_split_keeps_test_bytes_held_out(self):
        train, validation, test = _split_documents(PRETRAINING_CORPUS)
        self.assertTrue(set(train).isdisjoint(validation))
        self.assertTrue(set(train).isdisjoint(test))
        self.assertTrue(set(validation).isdisjoint(test))


class DocumentationClaimTests(unittest.TestCase):
    def test_retired_claims_do_not_reappear_in_active_documents(self):
        root = Path(__file__).resolve().parents[1]
        active_documents = "\n".join(
            (root / filename).read_text(encoding="utf-8") for filename in ("README.md", "PAPER_DRAFT.md")
        ).casefold()
        retired_claim_fragments = (
            "40% lower llm api inference cost",
            "f(2, 8) = 425.71",
            "higher byte efficiency than",
            "lower token-level cross-entropy than",
            "universal pareto frontier",
            "root-boundary preservation",
        )
        for fragment in retired_claim_fragments:
            self.assertNotIn(fragment, active_documents)


if __name__ == "__main__":
    unittest.main()
