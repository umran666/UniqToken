from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmarks.downstream_eval import DownstreamEvaluator
from benchmarks.benchmark_suite import TokenizerBenchmarkSuite
from benchmarks.ledger import SCHEMA_VERSION, load_ledger, validate_ledger, write_ledger
from uniqtoken._native import native_function
from uniqtoken.bpe_model import BPEModel
from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
from uniqtoken.seed_builder import SeedVocabularyBuilder
from uniqtoken.tokenizer import CustomTokenizer
from uniqtoken.unigram_lattice import UnigramLattice
from uniqtoken.unigram_trainer import UnigramModel


def model_with_bytes(extra=None):
    vocab = {f"<0x{i:02X}>": -10.0 for i in range(256)}
    vocab.update(extra or {})
    ids = {token: i for i, token in enumerate(vocab)}
    return UnigramModel(vocab, ids, {i: t for t, i in ids.items()}, [])


def add_token(model, token):
    model.vocab[token] = -1.0
    model.token_to_id[token] = len(model.token_to_id)
    model.id_to_token[model.token_to_id[token]] = token


def test_tokenizer_cross_word_cache_tracks_content_and_space_marker():
    model = model_with_bytes({"a": -1.0, "b": -1.0, "\u2581": -1.0})
    tok = CustomTokenizer(Normalizer(), RegexPreTokenizer(), model)
    assert not tok._cross_word_tokens()
    add_token(model, "a\u2581")
    assert tok._apply_cross_word_merges(["a", "\u2581", "b"]) == ["a\u2581", "b"]
    tok.normalizer.space_char = " "
    assert not tok._cross_word_tokens()


def test_security_cache_tracks_in_place_special_tokens_and_model_replacement():
    model = model_with_bytes({"[MASK]": -1.0})
    tok = CustomTokenizer(Normalizer(), RegexPreTokenizer(), model)
    assert tok._special_tokens_contain_pipe_marker()
    model.special_tokens.append("[MASK]")
    assert not tok._special_tokens_contain_pipe_marker()
    assert "[MASK]" in tok.security.special_tokens
    add_token(model, "<|NEW|>")
    model.special_tokens.append("<|NEW|>")
    assert tok._prepare_text("<|NEW|>", "all", "raise") == "<|NEW|>"
    tok.model = model_with_bytes({"<|REPLACED|>": -1.0})
    tok.model.special_tokens = ["<|REPLACED|>"]
    assert tok._prepare_text_with_alignment("<|REPLACED|>", "all", "raise")[0] == "<|REPLACED|>"
    assert tok.security.special_tokens == {"<|REPLACED|>"}


@pytest.mark.parametrize("kind", ["unigram", "bpe"])
def test_incomplete_byte_vocabulary_rejected(kind):
    with pytest.raises(ValueError, match="all 256"):
        if kind == "unigram":
            UnigramModel({"a": 0.0}, {"a": 0}, {0: "a"}, [])
        else:
            BPEModel({"a"}, {"a": 0}, {0: "a"}, {})


def test_byte_token_removed_after_construction_is_rejected_before_encoding():
    model = model_with_bytes({"a": -1.0})
    token = "<0xFF>"
    del model.vocab[token]
    del model.id_to_token[model.token_to_id.pop(token)]
    with pytest.raises(ValueError, match="all 256"):
        model.encode_to_ids("a")


@pytest.mark.parametrize("error", [ValueError, TypeError, AttributeError, RuntimeError])
def test_native_viterbi_errors_propagate(error):
    model = model_with_bytes()
    with (
        patch.object(model, "_get_rust_trie", return_value=object()),
        patch(
            "uniqtoken.unigram_trainer.uniqtoken_core",
            SimpleNamespace(rust_viterbi_decode=lambda *args: (_ for _ in ()).throw(error("native failed"))),
        ),
    ):
        with pytest.raises(error, match="native failed"):
            model.encode_with_spans("unknown")


def test_missing_native_function_logs_and_uses_python(caplog):
    assert native_function(SimpleNamespace(), "missing") is None
    assert "unavailable; using Python" in caplog.text


def test_native_normalization_error_is_not_hidden():
    with (
        patch("uniqtoken.pre_tokenizer._HAS_RUST_NORM", True),
        patch(
            "uniqtoken.pre_tokenizer._uniqtoken_core",
            SimpleNamespace(rust_normalize=lambda *args: (_ for _ in ()).throw(ValueError("native failed"))),
        ),
    ):
        with pytest.raises(ValueError, match="native failed"):
            Normalizer().normalize("hello")


@pytest.mark.parametrize(
    "method,function",
    [
        ("encode", "rust_encode_text_native"),
        ("encode_to_ids", "rust_encode_text_native_ids"),
        ("_encode_tokens_native_batch", "rust_encode_text_native_batch"),
        ("_encode_ids_native_batch", "rust_encode_text_native_ids_batch"),
    ],
)
def test_fused_native_computation_error_propagates(method, function):
    model = model_with_bytes()
    tok = CustomTokenizer(Normalizer(), RegexPreTokenizer(), model)
    core = SimpleNamespace(**{function: lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("native failed"))})
    with (
        patch.object(tok, "_native_pipeline_kwargs", return_value={}),
        patch.object(model, "_get_rust_trie", return_value=object()),
        patch("uniqtoken.tokenizer._native_core", core),
    ):
        value = ["hello"] if "batch" in method else "hello"
        with pytest.raises(ValueError, match="native failed"):
            getattr(tok, method)(value)


def test_native_batch_error_propagates():
    model = model_with_bytes()
    core = SimpleNamespace(rust_encode_tokens_batch=lambda *args: (_ for _ in ()).throw(ValueError("native failed")))
    with (
        patch.object(model, "_get_rust_trie", return_value=object()),
        patch("uniqtoken.unigram_trainer.uniqtoken_core", core),
    ):
        with pytest.raises(ValueError, match="native failed"):
            model.encode_batch(["hello"])


@pytest.mark.parametrize("error", [ValueError, TypeError, AttributeError, RuntimeError])
def test_native_mining_error_propagates(error):
    core = SimpleNamespace(rust_mine_ngrams=lambda *args: (_ for _ in ()).throw(error("native failed")))
    with patch("uniqtoken.seed_builder.uniqtoken_core", core):
        with pytest.raises(error, match="native failed"):
            SeedVocabularyBuilder(target_vocab_size=500).mine_ngrams({"hello": 1})


def test_missing_native_viterbi_uses_python_with_warning(caplog):
    model = model_with_bytes()
    with (
        patch.object(model, "_get_rust_trie", return_value=object()),
        patch("uniqtoken.unigram_trainer.uniqtoken_core", SimpleNamespace()),
    ):
        assert model.encode("z") == ["<0x7A>"]
    assert "rust_viterbi_decode unavailable" in caplog.text


def test_incomplete_byte_model_rejected_on_load(tmp_path):
    payload = {"vocab": {"a": 0.0}, "token_to_id": {"a": 0}, "special_tokens": [], "byte_fallback": True}
    (tmp_path / "tokenizer.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="all 256"):
        CustomTokenizer.load(str(tmp_path))


@pytest.mark.parametrize("text", ["abc", "\u4e2d\u6587", "\u00e9a"])
def test_rust_python_fallback_parity_with_excluded_long_prefix(text):
    core = pytest.importorskip("uniqtoken_core")
    model = model_with_bytes({text: -0.1})
    model.max_subword_len = 1
    trie = core.RustPrefixTrie(1)
    for token, score in model.vocab.items():
        trie.insert(token, score, model.token_to_id[token])
    rust = core.rust_viterbi_decode(text, trie, True)
    edges, _ = UnigramLattice(text, model.vocab, max_subword_len=1, byte_fallback=True).viterbi_edges()
    python_spans = [(token, edge.start, edge.end) for edge in edges for token in edge.tokens]
    assert [(s.token, s.start, s.end) for s in rust] == python_spans


def test_cjk_uses_explicit_unicode_character_denominator():
    evaluator = DownstreamEvaluator(
        training_corpus=["train"], evaluation_corpus=["\u4e2d\u6587\u6d4b\u8bd5"], vocab_size=500
    )
    result = evaluator.evaluate_tokenizer("fake", lambda _: [1, 2], 500)
    assert result.tokens_per_unicode_character == 0.5
    assert not hasattr(result, "tokens_per_word")


def test_multilingual_suite_uses_same_character_metric_for_cjk_and_latin():
    tok = SimpleNamespace(
        encode_to_ids=lambda text: [1, 2],
        decode=lambda ids: "",
        encode_with_offsets=lambda text: [SimpleNamespace(text="piece"), SimpleNamespace(text="piece")],
    )
    suite = TokenizerBenchmarkSuite(
        tokenizer=tok, training_corpus=["training only"], evaluation_corpora={"CJK": "中文测试"}
    )
    for name, text in [("CJK", "中文测试"), ("Latin", "abcd")]:
        result = suite.evaluate_dataset(name, text, warmup=0, iterations=1)
        assert result.tokens_per_unicode_character == 0.5
        assert not hasattr(result, "tokens_per_word")


def ledger():
    return {
        "metadata": {
            "ledger_schema_version": SCHEMA_VERSION,
            "commit_hash": "a" * 40,
            "working_tree_dirty": False,
            "data_split": "document_disjoint_train_validation",
        },
        "records": [
            {
                "model_kind": "causal_transformer",
                "vocab_budget": 500,
                "actual_vocab_size": 500,
                "tokens_per_unicode_character": 0.5,
            }
        ],
    }


def test_ledger_round_trip_validates_provenance(tmp_path):
    path = tmp_path / "results.json"
    write_ledger(path, ledger())
    assert load_ledger(path, expected_commit="a" * 40) == ledger()
    with pytest.raises(ValueError, match="expected commit"):
        load_ledger(path, expected_commit="b" * 40)


@pytest.mark.parametrize(
    "field,value", [("ledger_schema_version", 2), ("commit_hash", None), ("working_tree_dirty", None)]
)
def test_ledger_rejects_missing_or_old_provenance_on_load(tmp_path, field, value):
    payload = ledger()
    payload["metadata"][field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_ledger(path)


@pytest.mark.parametrize(
    "patch_row", [{"fertility": 400}, {"tokens_per_word": 400}, {"actual_vocab_size": 499}, {"model_kind": "laplace"}]
)
def test_ledger_rejects_misleading_or_invalid_rows(patch_row):
    payload = copy.deepcopy(ledger())
    payload["records"][0].update(patch_row)
    with pytest.raises(ValueError):
        validate_ledger(payload)
