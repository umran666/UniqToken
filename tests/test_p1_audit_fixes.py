"""Regression coverage for confirmed P1 audit fixes."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.downstream_eval import DownstreamMetrics
from benchmarks.train_toy_transformer import _split_documents, train_superbpe_tokenizer
from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
from uniqtoken.tokenizer import CustomTokenizer
from uniqtoken.unigram_trainer import UnigramModel


class ModelInvariantTests(unittest.TestCase):
    def _model(self) -> UnigramModel:
        return UnigramModel(
            vocab={"a": -1.0, "b": -1.0, "ab": -10.0, "<|unk|>": -20.0},
            token_to_id={"a": 0, "b": 1, "ab": 2, "<|unk|>": 3},
            id_to_token={0: "a", 1: "b", 2: "ab", 3: "<|unk|>"},
            special_tokens=["<|unk|>"],
            byte_fallback=False,
        )

    def test_in_place_score_mutation_invalidates_segmentation_cache(self):
        model = self._model()
        self.assertEqual(model.encode("ab"), ["a", "b"])
        model.vocab["ab"] = 0.0
        self.assertEqual(model.encode("ab"), ["ab"])

    def test_rejects_non_bijective_ids_at_construction(self):
        with self.assertRaisesRegex(ValueError, "unique ID"):
            UnigramModel(
                vocab={"a": 0.0, "b": 0.0, "<|unk|>": -1.0},
                token_to_id={"a": 0, "b": 0, "<|unk|>": 1},
                id_to_token={0: "b", 1: "<|unk|>"},
                special_tokens=["<|unk|>"],
                byte_fallback=False,
            )

    def test_load_rejects_conflicting_json_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "tokenizer.json").write_text(
                json.dumps(
                    {
                        "vocab": {"a": 0.0, "b": 0.0, "<|unk|>": -1.0},
                        "token_to_id": {"a": 0, "b": 0, "<|unk|>": 1},
                        "special_tokens": ["<|unk|>"],
                        "byte_fallback": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unique ID"):
                CustomTokenizer.load(directory, prefer_binary=False)


class NativeEMFailureTests(unittest.TestCase):
    def test_native_em_runtime_failure_is_not_silently_retrained_in_python(self):
        import uniqtoken.unigram_trainer as module

        class FakeTrie:
            def __init__(self, _max_len):
                pass

            def insert(self, _token, _score, _token_id):
                pass

        fake_core = SimpleNamespace(
            RustPrefixTrie=FakeTrie,
            rust_forward_backward_expectations=lambda *_args: (_ for _ in ()).throw(RuntimeError("native EM failed")),
        )
        trainer = module.UnigramTrainer(target_vocab_size=300, min_frequency=1, show_progress=False)
        with patch.object(module, "_HAS_UNIQTOKEN_CORE", True), patch.object(module, "uniqtoken_core", fake_core):
            with self.assertRaisesRegex(RuntimeError, "native EM failed"):
                trainer.train(["alpha beta gamma delta"] * 20, verbose=False)


class BenchmarkContractTests(unittest.TestCase):
    def test_document_splits_are_three_way_and_duplicate_safe(self):
        train, validation, test = _split_documents(["a", "b", "c", "d", "a", "b"])
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertTrue(test)
        self.assertTrue(set(train).isdisjoint(validation))
        self.assertTrue(set(train).isdisjoint(test))
        self.assertTrue(set(validation).isdisjoint(test))

    def test_superbpe_reserves_nonzero_merge_capacity(self):
        import benchmarks.train_toy_transformer as module

        calls = []

        class FakeTokenizer:
            def __init__(self, normalizer=None, pre_tokenizer=None, model=None):
                self.normalizer = normalizer
                self.pre_tokenizer = pre_tokenizer
                self.model = model

            @staticmethod
            def train_from_corpus(**kwargs):
                calls.append(kwargs["target_vocab_size"])
                return SimpleNamespace(
                    normalizer=SimpleNamespace(normalize=lambda text: text),
                    pre_tokenizer=SimpleNamespace(pre_tokenize=lambda text: [text]),
                    model=SimpleNamespace(vocab={str(i): 0.0 for i in range(kwargs["target_vocab_size"])}),
                )

        class FakeCEM:
            def __init__(self, max_merges, **_kwargs):
                self.max_merges = max_merges
                self.merges = [("a", "b", "ab")]
                calls.append(max_merges)

            def optimize(self, model, chunks):
                self.chunks = chunks
                return model

        with patch.object(module, "CustomTokenizer", FakeTokenizer), patch.object(module, "CrossEntropyMerging", FakeCEM):
            train_superbpe_tokenizer(["alpha beta"], target_vocab=500)

        self.assertEqual(calls, [470, 30])

    def test_uniform_code_length_metric_is_not_reported_as_bpb(self):
        self.assertNotIn("estimated_bits_per_byte", DownstreamMetrics.__annotations__)

    def test_sentencepiece_budget_is_not_silently_reduced(self):
        import benchmarks.vocab_quality_race as module

        calls = []

        class FakeSentencePieceTrainer:
            @staticmethod
            def Train(**kwargs):
                calls.append(kwargs["vocab_size"])
                raise RuntimeError("Vocabulary size too high")

        fake_sentencepiece = SimpleNamespace(SentencePieceTrainer=FakeSentencePieceTrainer)
        with patch.dict(sys.modules, {"sentencepiece": fake_sentencepiece}):
            with self.assertRaisesRegex(RuntimeError, "Vocabulary size too high"):
                module._train_sentencepiece(500, ["repeated training input"])
        self.assertEqual(calls, [500])


class WheelDiscoveryTests(unittest.TestCase):
    def test_wheel_contains_compat_and_train_namespaces(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--wheel-dir", directory],
                cwd=root,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            wheel = next(Path(directory).glob("*.whl"))
            with zipfile.ZipFile(wheel) as archive:
                members = set(archive.namelist())
            self.assertIn("uniqtoken/compat/__init__.py", members)
            self.assertIn("uniqtoken/train/__init__.py", members)


class UnicodeFixtureTests(unittest.TestCase):
    def test_native_parity_fixtures_are_real_unicode(self):
        from tests.test_native_pipeline import CORPUS, TRICKY_TEXTS

        fixtures = CORPUS + TRICKY_TEXTS
        self.assertTrue(any("👨‍👩‍👧‍👦" in text for text in fixtures))
        self.assertTrue(any("我喜欢自然语言处理" in text for text in fixtures))
        self.assertTrue(any("ﬁle" in text for text in fixtures))
        self.assertFalse(any("�" in text for text in fixtures))


class MultimodalSupportTests(unittest.TestCase):
    def test_untrained_audio_codecs_are_not_public_api(self):
        import uniqtoken
        import uniqtoken.multimodal as multimodal

        for namespace in (uniqtoken, multimodal):
            self.assertFalse(hasattr(namespace, "AudioSegment"))
            self.assertFalse(hasattr(namespace, "ResidualVectorQuantizer"))
            self.assertFalse(hasattr(namespace, "NeuralAudioCodec"))
        self.assertNotIn("NeuralCodecFacade", uniqtoken.__all__)


if __name__ == "__main__":
    unittest.main()
