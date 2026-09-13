"""Synthetic harness regressions only. These fixtures are not research results."""

import copy
from dataclasses import asdict
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks import run_research_experiments as h


def document_bytes(docs):
    # Most fixtures are identity-normalized; normalization-specific tests supply originals separately.
    return {
        s: [
            {"source_utf8_bytes": len(t.encode("utf-8")), "normalized_utf8_bytes": len(h.normalize(t).encode("utf-8"))}
            for t in texts
        ]
        for s, texts in docs.items()
    }


@pytest.fixture(autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def manifest(tmp_path):
    source_path = tmp_path / "source.txt"
    source_path.write_text("unit-test pinned source", encoding="utf-8")
    source = {
        "dataset": "unit-test/source",
        "revision": "a" * 40,
        "url": "https://example.invalid/unit-test/source/a",
        "local_path": source_path.name,
        "source_file_sha256": h.file_hash(source_path),
        "license": "test",
        "release_variant": "unit-test-v1",
    }

    def row(identifier, text):
        return {
            "id": identifier,
            "text": text,
            "language": "test",
            "domain": "test",
            "raw_utf8_bytes": len(text.encode("utf-8")),
            "normalized_utf8_bytes": len(h.normalize(text).encode("utf-8")),
            "source": source,
            "dedup": {"status": "accepted_after_exact_and_near_eval_check"},
        }

    data = {
        "schema_version": h.DATASET_MANIFEST_SCHEMA,
        "dataset_id": "synthetic-test-only",
        "source": "unit test",
        "license": "test fixture",
        "deduplication": "exact normalized documents checked",
        "normalization": h.NORMALIZATION,
        "freeze": {
            "immutable": True,
            "source_files": [
                {
                    "local_path": source_path.name,
                    "sha256": h.file_hash(source_path),
                    "dataset": source["dataset"],
                    "revision": source["revision"],
                    "url": source["url"],
                    "license": source["license"],
                    "release_variant": source["release_variant"],
                    "file_bytes": source_path.stat().st_size,
                }
            ],
            "source_revisions": {"unit_test": source["revision"]},
        },
        "splits": {},
    }
    groups = []
    for split, text in (("train", "training text"), ("validation", "validation text"), ("test", "\u4e2d\u6587")):
        path = tmp_path / f"{split}.jsonl"
        record = row(split, text)
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        data["splits"][split] = {"path": path.name, "sha256": h.file_hash(path)}
        groups.append(
            {
                "split": split,
                "dataset": source["dataset"],
                "release_variant": source["release_variant"],
                "language": record["language"],
                "domain": record["domain"],
                "documents": 1,
                "raw_utf8_bytes": record["raw_utf8_bytes"],
                "normalized_utf8_bytes": record["normalized_utf8_bytes"],
            }
        )
    groups.sort(
        key=lambda group: (
            group["split"],
            group["dataset"],
            group["release_variant"],
            group["language"],
            group["domain"],
        )
    )
    data["freeze"]["selection"] = {"byte_unit": "MB_decimal", "groups": groups}
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def replace_split(manifest, split, row):
    data = h.read_json(manifest)
    source = data["freeze"]["source_files"][0]
    row = {
        "language": "test",
        "domain": "test",
        "raw_utf8_bytes": len(row["text"].encode("utf-8")),
        "normalized_utf8_bytes": len(h.normalize(row["text"]).encode("utf-8")),
        "source": {**source, "source_file_sha256": source["sha256"]},
        "dedup": {"status": "accepted_after_exact_and_near_eval_check"},
        **row,
    }
    target = manifest.parent / data["splits"][split]["path"]
    target.write_text(json.dumps(row), encoding="utf-8")
    data["splits"][split]["sha256"] = h.file_hash(target)
    group = next(group for group in data["freeze"]["selection"]["groups"] if group["split"] == split)
    group["raw_utf8_bytes"] = row["raw_utf8_bytes"]
    group["normalized_utf8_bytes"] = row["normalized_utf8_bytes"]
    manifest.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    "mutation",
    ["raw_duplicate", "normalized_duplicate", "duplicate_id", "hash", "missing_split", "empty", "reserved", "metadata"],
)
def test_dataset_failures(manifest, mutation):
    if mutation == "raw_duplicate":
        replace_split(manifest, "test", {"id": "test", "text": "training text"})
    elif mutation == "normalized_duplicate":
        replace_split(manifest, "train", {"id": "train", "text": "\ufb01"})
        replace_split(manifest, "test", {"id": "test", "text": "fi"})
    elif mutation == "duplicate_id":
        replace_split(manifest, "test", {"id": "train", "text": "other"})
    elif mutation == "empty":
        replace_split(manifest, "test", {"id": "test", "text": " "})
    elif mutation == "reserved":
        replace_split(manifest, "test", {"id": "test", "text": "<|bos|>"})
    else:
        data = h.read_json(manifest)
        if mutation == "hash":
            data["splits"]["test"]["sha256"] = "bad"
        elif mutation == "missing_split":
            del data["splits"]["test"]
        else:
            del data["license"]
        manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        h.load_dataset(manifest)


def test_normalization_and_cjk_denominator(manifest):
    replace_split(manifest, "train", {"id": "train", "text": "\ufb01\u00a0x"})
    docs, data = h.load_dataset(manifest)
    assert docs["train"] == ["fi x"]
    assert data["splits"]["test"]["utf8_bytes"] == 6
    assert h.token_metrics(byte_tokenizer(), docs["test"], source_utf8_bytes=6)["tokens_per_unicode_character"] == 3


@pytest.mark.parametrize("mutation", ["byte_counts", "untracked_source", "selection_accounting"])
def test_dataset_requires_verified_document_accounting(manifest, mutation):
    data = h.read_json(manifest)
    if mutation == "selection_accounting":
        data["freeze"]["selection"]["groups"][0]["normalized_utf8_bytes"] += 1
        manifest.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError):
            h.load_dataset(manifest)
        return
    target = manifest.parent / data["splits"]["train"]["path"]
    row = json.loads(target.read_text(encoding="utf-8"))
    if mutation == "byte_counts":
        row["normalized_utf8_bytes"] += 1
    else:
        row["source"]["local_path"] = "not-in-inventory.txt"
    target.write_text(json.dumps(row), encoding="utf-8")
    data["splits"]["train"]["sha256"] = h.file_hash(target)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        h.load_dataset(manifest)


def byte_tokenizer(name="boundary_bpe"):
    vocab = {**h.SPECIAL_IDS, **{f"<0x{i:02X}>": i + 4 for i in range(256)}}
    return h.ResearchTokenizer(
        name,
        SimpleNamespace(
            encode_to_ids=lambda t: [b + 4 for b in t.encode("utf-8")],
            decode=lambda ids: bytes(i - 4 for i in ids).decode("utf-8"),
            id_to_token={i + 4: f"<0x{i:02X}>" for i in range(256)},
        ),
        vocab,
    )


@pytest.mark.parametrize("mutation", ["short", "sparse", "special", "bytes", "no_merges"])
def test_tokenizer_invariants(mutation):
    tok = byte_tokenizer()
    if mutation == "short":
        del tok.vocab["<0x00>"]
    elif mutation == "sparse":
        tok.vocab["<0x00>"] = 999
    elif mutation == "special":
        tok.vocab[h.SPECIALS[0]], tok.vocab[h.SPECIALS[1]] = 1, 0
    elif mutation == "bytes":
        tok.vocab["not-a-byte"] = tok.vocab.pop("<0x00>")
    else:
        tok.name = "uniq_superbpe"
    with pytest.raises(ValueError):
        h.validate_tokenizer(tok, 260)


def test_encoding_rejects_lossy_or_control_output():
    tok = byte_tokenizer()
    tok.model.encode_to_ids = lambda t: [0]
    with pytest.raises(ValueError, match="control"):
        tok.encode("x")
    tok.model.encode_to_ids = lambda t: [5]
    with pytest.raises(ValueError, match="roundtrip"):
        tok.encode("x")


@pytest.mark.parametrize("name", h.COHORT)
def test_real_small_tokenizer_artifact_roundtrip(tmp_path, name):
    rng = random.Random(123)
    words = ["".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=8)) for _ in range(100)]
    texts = [f"the {word} alpha beta gamma delta" for word in words] * 3
    tok, _ = h.train_tokenizer(name, texts, 384, tmp_path / name)
    h.validate_tokenizer(tok, 384)
    row = {
        "artifact": name,
        "artifact_hashes": h.artifact_hashes(tmp_path / name),
        "tokenizer": name,
        "learned_merges": tok.merges,
        "vocab_budget": 384,
    }
    restored = h.load_tokenizer(row, tmp_path)
    for text in ("held out text", "\u4e2d\u6587\t\n  x", "fi  x"):
        assert restored.encode(text) == tok.encode(text)
    with (tmp_path / name / "unexpected.txt").open("w") as stream:
        stream.write("stale")
    with pytest.raises(ValueError, match="stale"):
        h.load_tokenizer(row, tmp_path)


@pytest.mark.parametrize("vocab", h.VOCABS)
@pytest.mark.parametrize("cfg,context", [(h.SCREEN, 128), (h.CONFIRM, 1024)])
def test_parameter_spec_matches_actual_architecture(vocab, cfg, context):
    with torch.device("meta"):
        model = h.CausalMiniTransformer(vocab, cfg, context)
    counts = h.parameter_accounting(vocab, cfg, context)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == counts["total_params"]
    assert model.embed.weight.numel() == counts["input_embedding_params"] == vocab * cfg.d_model
    assert model.head.weight.numel() == counts["output_head_params"] == vocab * cfg.d_model
    assert model.embed.weight is not model.head.weight
    assert actual - model.embed.weight.numel() - model.head.weight.numel() == counts["core_params"]
    assert (
        counts["core_params"]
        == counts["non_embedding_params"]
        == h.parameter_accounting(16384, cfg, context)["core_params"]
    )


def test_causal_mask_prevents_future_information():
    cfg = h.LMArchConfig("test", 1, 8, 2, 16, 1, 0.001)
    model = h.CausalMiniTransformer(16, cfg, 4).eval()
    with torch.no_grad():
        a = model(torch.tensor([[2, 4, 5, 6]]))
        b = model(torch.tensor([[2, 4, 8, 9]]))
    torch.testing.assert_close(a[:, :2], b[:, :2])


def test_evaluation_scores_first_token_tail_eos_and_utf8_bytes():
    class Uniform(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((*x.shape, 260))

    docs = ["a", "\u4e2d"]
    encoded = [byte_tokenizer().encode(t) for t in docs]
    metric = h.evaluate(Uniform(), encoded, docs, 2, "cpu", source_utf8_bytes=4)
    assert metric["target_tokens_including_eos"] == 6
    assert metric["utf8_bytes"] == 4
    assert metric["total_nll_nats"] == pytest.approx(6 * math.log(260))
    assert metric["bits_per_byte"] == pytest.approx(6 * math.log2(260) / 4)
    assert metric["token_perplexity"] == pytest.approx(260)
    assert list(h.windows([4, 5, 6], 2)) == [([2, 4], [4, 5]), ([5, 6], [6, 3])]


@pytest.mark.parametrize("regime", h.REGIMES)
def test_tiny_causal_training_budget_and_heldout(regime):
    cfg = h.LMArchConfig("test", 1, 8, 2, 16, 1, 0.001)
    docs = {"train": ["ab", "cd"], "validation": ["ef"], "test": ["\u4e2d"]}
    budget = 4 if regime == "bytes" else 20 * h.training_flops(260, cfg, 3)
    result = h.train_lm(
        byte_tokenizer(), docs, cfg, 4, regime, budget, 0, "cpu", True, document_bytes=document_bytes(docs)
    )
    assert result["validation"]["utf8_bytes"] == 2
    assert result["test"]["utf8_bytes"] == 3
    if regime == "bytes":
        assert result["completed_document_bytes"] == budget
    else:
        assert 0.99 * budget <= result["actual_analytical_flops"] <= budget


def test_screening_does_not_encode_test_or_swallow_computation(monkeypatch):
    cfg = h.LMArchConfig("test", 1, 8, 2, 16, 1, 0.001)
    docs = {"train": ["ab"], "validation": ["cd"], "test": ["DO NOT READ"]}
    tok = byte_tokenizer()
    seen = []
    original = tok.encode
    tok.encode = lambda t: (seen.append(t), original(t))[1]
    result = h.train_lm(tok, docs, cfg, 4, "bytes", 2, 0, "cpu", False, document_bytes=document_bytes(docs))
    assert "test" not in result and "DO NOT READ" not in seen
    monkeypatch.setattr(
        h.CausalMiniTransformer, "forward", lambda *a: (_ for _ in ()).throw(RuntimeError("native failure"))
    )
    with pytest.raises(RuntimeError, match="native failure"):
        h.train_lm(tok, docs, cfg, 4, "bytes", 2, 0, "cpu", False, document_bytes=document_bytes(docs))


@pytest.mark.parametrize("budget", [0, -1, 1, 3, 1.5])
def test_byte_budget_cannot_round_or_split_documents(budget):
    with pytest.raises(ValueError):
        h.byte_schedule(["ab", "cd"], budget)


@pytest.fixture
def staged(tmp_path, manifest, monkeypatch):
    identity = {
        "ledger_schema_version": 3,
        "commit_hash": "a" * 40,
        "working_tree_dirty": False,
        "source_hash": "b" * 64,
        "extension_hash": "c" * 64,
        "versions": {},
        "extension_status": "installed",
    }
    monkeypatch.setattr(h, "runtime_identity", lambda: identity)
    trained = []

    def trainer(name, texts, vocab, directory):
        trained.append(list(texts))
        directory.mkdir()
        h.write_new_json(directory / "fixture.json", {"synthetic_unit_test": True})
        tok = byte_tokenizer(name)
        tok.vocab.update({f"synthetic{i}": i for i in range(260, vocab)})
        tok.encode = byte_tokenizer().encode
        tok.merges = 1
        return tok, 0.1

    monkeypatch.setattr(h, "train_tokenizer", trainer)
    monkeypatch.setattr(h, "load_tokenizer", lambda *a: byte_tokenizer())

    def lm(tok, docs, cfg, context, regime, budget, seed, device, evaluate_test, *, document_bytes):
        targets = int(budget // h.training_flops(16384, cfg, 1)) if regime == "flops" else len(docs["train"][0]) + 1
        completed = 1 if regime == "bytes" else 0
        result = {
            **h.parameter_accounting(16384, cfg, context),
            "training_steps": targets,
            "training_target_tokens": targets,
            "training_sequence_length_squared_sum": targets,
            "completed_document_bytes": budget if regime == "bytes" else 0,
            "completed_training_documents": completed,
            "training_byte_scope": "complete_document_prefix",
            "training_bytes": h.byte_totals(document_bytes["train"], completed),
            **h.flop_accounting(16384, cfg, targets, targets),
        }
        for split in ("validation", "test") if evaluate_test else ("validation",):
            result[split] = h.nll_metrics(
                3.0,
                sum(len(t.encode()) + 1 for t in docs[split]),
                sum(len(t.encode()) for t in docs[split]),
                source_utf8_bytes=h.byte_totals(document_bytes[split])["source_utf8_bytes"],
            )
        return result

    monkeypatch.setattr(h, "train_lm", lm)
    monkeypatch.setattr(h, "VOCABS", (16384,))
    args = SimpleNamespace(
        phase="A",
        dataset=manifest,
        output=tmp_path / "a",
        phase_a=None,
        screening=None,
        selection=None,
        flops=None,
        bytes=None,
        seeds=None,
        device="cpu",
    )
    a = h.run(args)
    assert len(trained) == 5 and all(d == ["training text"] for d in trained)
    return args, identity, a


def test_complete_staged_run_and_selection(staged):
    args, identity, a = staged
    args.phase, args.phase_a = "B", args.output / "ledger.json"
    args.output = args.output.parent / "b"
    args.flops, args.bytes = 1e10, len("training text")
    b = h.run(args)
    assert len(b["records"]) == 10
    assert {r["budget_regime"] for r in b["records"]} == set(h.REGIMES)
    args.screening = args.output / "ledger.json"
    args.selection = args.output.parent / "selection.json"
    h.write_new_json(
        args.selection,
        {
            "schema_version": 1,
            "screening_sha256": h.file_hash(args.screening),
            "rationale": "synthetic test selection",
            "conditions": [["boundary_bpe", 16384, "bytes"]],
        },
    )
    args.phase, args.output = "C", args.output.parent / "c"
    c = h.run(args)
    assert len(c["records"]) == 3
    assert {r["seed"] for r in c["records"]} == {1, 2, 3}
    assert all("test" in r for r in c["records"])
    assert all(r["model_kind"] == "causal_transformer" for r in c["records"])


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "missing",
        "duplicate",
        "cohort",
        "model",
        "specials",
        "dataset",
        "extension",
        "source",
        "vocab",
        "nan",
        "no_merges",
    ],
)
def test_stale_or_incomplete_ledgers_rejected(staged, mutation):
    args, identity, a = staged
    data = copy.deepcopy(a)
    row, meta = data["records"][0], data["metadata"]
    if mutation == "schema":
        meta["research_schema_version"] = 0
    elif mutation == "missing":
        data["records"].pop()
    elif mutation == "duplicate":
        data["records"].append(copy.deepcopy(row))
    elif mutation == "cohort":
        row["tokenizer"] = "other"
    elif mutation == "model":
        row["model_kind"] = "laplace"
    elif mutation == "specials":
        row["special_tokens"] = {}
    elif mutation == "dataset":
        row["dataset_hash"] = "bad"
    elif mutation in ("extension", "source"):
        meta["identity"][mutation + "_hash"] = "bad"
    elif mutation == "vocab":
        row["actual_vocab_size"] -= 1
    elif mutation == "nan":
        row["test"]["tokens"] = float("nan")
    else:
        data["records"][-1]["learned_merges"] = 0
    with pytest.raises(ValueError):
        h.validate_research_ledger(data, identity, a["metadata"]["dataset"])


@pytest.mark.parametrize("mutation", ["wrong_seeds", "no_phase_a", "no_budgets", "screen_too_large", "overwrite"])
def test_invalid_stage_configuration(staged, mutation):
    args, _, _ = staged
    original = args.output
    args.phase, args.phase_a = "B", original / "ledger.json"
    args.output, args.flops, args.bytes = original.parent / "b", 1e10, len("training text")
    if mutation == "wrong_seeds":
        args.seeds = [1, 2]
    elif mutation == "no_phase_a":
        args.phase_a = None
    elif mutation == "no_budgets":
        args.flops = None
    elif mutation == "screen_too_large":
        args.flops = 1e12
    else:
        args.output = original
    with pytest.raises(ValueError):
        h.run(args)


def test_selection_bound_to_screening_and_known_conditions(tmp_path):
    ledger = tmp_path / "b.json"
    h.write_new_json(ledger, {"fixture": True})
    selection = tmp_path / "selection.json"
    h.write_new_json(
        selection,
        {
            "schema_version": 1,
            "screening_sha256": "stale",
            "rationale": "test",
            "conditions": [["other", 16384, "bytes"]],
        },
    )
    with pytest.raises(ValueError, match="stale"):
        h.select_conditions(selection, {"records": []}, ledger)
    data = h.read_json(selection)
    data["screening_sha256"] = h.file_hash(ledger)
    selection.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="un-screened"):
        h.select_conditions(selection, {"records": []}, ledger)


def test_no_complete_ledger_on_failed_condition(tmp_path, manifest, monkeypatch):
    identity = {"working_tree_dirty": False}
    monkeypatch.setattr(h, "runtime_identity", lambda: identity)
    monkeypatch.setattr(h, "train_tokenizer", lambda *a: (_ for _ in ()).throw(RuntimeError("training failed")))
    args = SimpleNamespace(
        phase="A",
        dataset=manifest,
        output=tmp_path / "out",
        phase_a=None,
        screening=None,
        selection=None,
        flops=None,
        bytes=None,
        seeds=None,
        device="cpu",
    )
    identity.update(commit_hash="a" * 40, extension_hash=None)
    with pytest.raises(RuntimeError, match="training failed"):
        h.run(args)
    assert (args.output / "plan.json").exists()
    assert not (args.output / "ledger.json").exists()


@pytest.mark.parametrize(
    "mutation",
    ["bpb", "targets", "bytes", "architecture", "params", "regime", "budget", "artifact", "test_access", "zero_steps"],
)
def test_invalid_lm_ledgers(staged, mutation):
    args, identity, a = staged
    args.phase, args.phase_a = "B", args.output / "ledger.json"
    args.output, args.flops, args.bytes = args.output.parent / "b", 1e10, len("training text")
    data = h.run(args)
    row = data["records"][0]
    if mutation == "bpb":
        row["validation"]["bits_per_byte"] /= 2
    elif mutation == "targets":
        row["validation"]["target_tokens_including_eos"] -= 1
    elif mutation == "bytes":
        row["validation"]["utf8_bytes"] += 1
    elif mutation == "architecture":
        row["model_config"]["head"] = "tied"
    elif mutation == "params":
        row["total_params"] = 125_000_000
    elif mutation == "regime":
        row["budget_regime"] = "unspecified"
    elif mutation == "budget":
        row["actual_analytical_flops"] *= 2
    elif mutation == "artifact":
        row["tokenizer_artifact_hash"] = "old"
    elif mutation == "test_access":
        row["test"] = row["validation"]
    else:
        row["training_steps"] = 0
    with pytest.raises(ValueError):
        h.validate_research_ledger(data, identity, a["metadata"]["dataset"])


def test_runtime_identity_hashes_installed_extension_and_current_source():
    identity = h.runtime_identity()
    assert len(identity["commit_hash"]) == 40
    assert len(identity["source_hash"]) == 64
    assert identity["extension_status"] in ("installed", "unavailable")
    assert (identity["extension_hash"] is None) == (identity["extension_status"] == "unavailable")
    assert set(identity["versions"]) == {"python", "torch", "sentencepiece", "numpy", "regex"}


def test_dirty_tree_does_not_start_experiments(tmp_path, manifest, monkeypatch):
    monkeypatch.setattr(h, "runtime_identity", lambda: {"working_tree_dirty": True})
    args = SimpleNamespace(dataset=manifest, output=tmp_path / "out")
    with pytest.raises(ValueError, match="commit"):
        h.run(args)
    assert not args.output.exists()


def test_too_small_flop_budget_fails():
    cfg = h.LMArchConfig("test", 1, 8, 2, 16, 1, 0.001)
    docs = {"train": ["ab"], "validation": ["cd"], "test": ["ef"]}
    with pytest.raises(ValueError, match="FLOP budget"):
        h.train_lm(byte_tokenizer(), docs, cfg, 4, "flops", 1, 0, "cpu", False, document_bytes=document_bytes(docs))


def test_fractional_flop_budget_never_overshoots():
    cfg = h.LMArchConfig("test", 1, 8, 2, 16, 1, 0.001)
    docs = {"train": ["abcdef"], "validation": ["gh"], "test": ["ij"]}
    budget = 100 * h.training_flops(260, cfg, 4) + h.training_flops(260, cfg, 2)
    result = h.train_lm(
        byte_tokenizer(), docs, cfg, 4, "flops", budget, 0, "cpu", False, document_bytes=document_bytes(docs)
    )
    assert 0.99 * budget <= result["actual_analytical_flops"] <= budget


def test_exclusive_result_write_keeps_original(tmp_path):
    path = tmp_path / "result.json"
    h.write_new_json(path, {"original": True})
    with pytest.raises(FileExistsError):
        h.write_new_json(path, {"original": False})
    assert h.read_json(path) == {"original": True}


def test_frozen_doc_order_is_identical_for_byte_matching():
    texts = ["ab", "\u4e2d", "cdef"]
    assert h.byte_schedule(texts, 5) == 2
    assert h.byte_schedule(texts, 18) == 6


@pytest.mark.parametrize("field", ["flops", "bytes"])
def test_phase_a_rejects_even_zero_lm_options(staged, field):
    args, _, _ = staged
    args.output = args.output.parent / "invalid-a"
    setattr(args, field, 0)
    with pytest.raises(ValueError, match="LM inputs"):
        h.run(args)


def test_seed_type_is_not_silently_coerced(staged):
    _, identity, a = staged
    a["records"][0]["seed"] = False
    with pytest.raises(ValueError, match="seed"):
        h.validate_research_ledger(a, identity, a["metadata"]["dataset"])


def test_flop_components_preserve_estimator_and_vocab_dependence():
    cfg = h.CONFIRM
    lengths = [1024, 17, 1]
    tokens, squares = sum(lengths), sum(s * s for s in lengths)
    rows = [h.flop_accounting(v, cfg, tokens, squares) for v in h.VOCABS]
    assert len({r["core_analytical_flops"] for r in rows}) == 1
    for vocab, row in zip(h.VOCABS, rows):
        assert row["output_projection_flops"] == 6 * tokens * cfg.d_model * vocab
        assert row["actual_analytical_flops"] == row["core_analytical_flops"] + row["output_projection_flops"]
        assert row["actual_analytical_flops"] == sum(
            6 * cfg.num_layers * s * (4 * cfg.d_model**2 + 2 * cfg.d_model * cfg.d_ff)
            + 12 * cfg.num_layers * s * s * cfg.d_model
            + 6 * s * cfg.d_model * vocab
            for s in lengths
        )
        assert row["flop_estimator_version"] == h.FLOP_ESTIMATOR_VERSION


def test_source_and_normalized_bytes_are_retained(manifest):
    replace_split(manifest, "train", {"id": "train", "text": "\ufb01"})
    docs, data = h.load_dataset(manifest)
    assert docs["train"] == ["fi"]
    assert data["splits"]["train"]["source_utf8_bytes"] == 3
    assert data["splits"]["train"]["normalized_utf8_bytes"] == data["splits"]["train"]["utf8_bytes"] == 2
    assert h.byte_totals(data["document_bytes"]["train"], 3) == {"source_utf8_bytes": 9, "normalized_utf8_bytes": 6}
    metric = h.token_metrics(byte_tokenizer(), docs["train"], source_utf8_bytes=3)
    assert metric["source_utf8_bytes"] == 3 and metric["normalized_utf8_bytes"] == 2
    nll = h.nll_metrics(6.0, 3, 2, source_utf8_bytes=3)
    assert nll["bits_per_byte"] == 6.0 / (2 * math.log(2))
    assert nll["source_utf8_bytes"] == 3


def test_byte_matching_keeps_source_exposure_identical_across_segmentations():
    source = {"train": ["\ufb01", "xy"], "validation": ["ab"], "test": ["cd"]}
    docs = {s: [h.normalize(t) for t in texts] for s, texts in source.items()}
    cfg = h.LMArchConfig("test", 1, 8, 2, 16, 1, 0.001)
    tokenizers = [byte_tokenizer(), byte_tokenizer()]
    # A synthetic alternative segmentation; no tokenizer training or research condition.
    tokenizers[1].encode = lambda text: [4]
    results = [
        h.train_lm(tok, docs, cfg, 4, "bytes", 2, 0, "cpu", False, document_bytes=document_bytes(source))
        for tok in tokenizers
    ]
    assert results[0]["training_target_tokens"] != results[1]["training_target_tokens"]
    for row in results:
        assert row["completed_training_documents"] == 1
        assert row["training_bytes"] == {"source_utf8_bytes": 3, "normalized_utf8_bytes": 2}
        assert row["completed_document_bytes"] == 2
        assert row["actual_analytical_flops"] == row["core_analytical_flops"] + row["output_projection_flops"]


@pytest.mark.parametrize(
    "field",
    [
        "core_params",
        "non_embedding_params",
        "input_embedding_params",
        "output_head_params",
        "total_params",
        "core_analytical_flops",
        "output_projection_flops",
        "actual_analytical_flops",
        "flop_estimator_version",
        "training_sequence_length_squared_sum",
        "completed_training_documents",
        "training_byte_scope",
        "training_bytes",
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "incorrect"])
def test_accounting_fields_are_required_and_validated(staged, field, mutation):
    args, identity, a = staged
    args.phase, args.phase_a = "B", args.output / "ledger.json"
    args.output, args.flops, args.bytes = args.output.parent / "b", 1e10, len("training text")
    data = h.run(args)
    row = data["records"][0]
    if mutation == "missing":
        row.pop(field)
    elif field == "training_bytes":
        row[field]["source_utf8_bytes"] += 1
    elif isinstance(row[field], str):
        row[field] = "incorrect"
    else:
        row[field] += 1
    with pytest.raises((ValueError, KeyError)):
        h.validate_research_ledger(data, identity, a["metadata"]["dataset"])


@pytest.mark.parametrize("field", ["source_utf8_bytes", "normalized_utf8_bytes"])
@pytest.mark.parametrize("phase", ["A", "B"])
def test_heldout_byte_fields_validated(staged, field, phase):
    args, identity, a = staged
    data = a
    if phase == "B":
        args.phase, args.phase_a = "B", args.output / "ledger.json"
        args.output, args.flops, args.bytes = args.output.parent / "b", 1e10, len("training text")
        data = h.run(args)
    data["records"][0]["validation"][field] += 1
    with pytest.raises(ValueError):
        h.validate_research_ledger(data, identity, a["metadata"]["dataset"])


@pytest.mark.parametrize("field", ["byte_budget_field", "byte_audit_field", "research_schema_version"])
def test_old_or_ambiguous_accounting_schema_rejected(staged, field):
    _, identity, a = staged
    a["metadata"][field] = 1 if field == "research_schema_version" else "utf8_bytes"
    with pytest.raises(ValueError):
        h.validate_research_ledger(a, identity, a["metadata"]["dataset"])
