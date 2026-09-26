"""Regression tests for the tokenizer-only objective ablation harness.

These fixtures are synthetic; they are not research results.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import ledger as ledger_module
from benchmarks import run_ablation_harness as ablation
from benchmarks import run_phase_a as stages
from benchmarks import run_research_experiments as research


def frozen_manifest(tmp_path):
    def row(identifier, text, language, domain):
        return {
            "id": identifier,
            "text": text,
            "language": language,
            "domain": domain,
            "raw_utf8_bytes": len(text.encode("utf-8")),
            "normalized_utf8_bytes": len(research.normalize(text).encode("utf-8")),
        }

    split_rows = {
        "train": [row("t1", "aa", "en", "latin"), row("t2", "bbb", "hi", "indic")],
        "validation": [row("v1", "screen candidate", "en", "flores"), row("v2", "confirm candidate", "hi", "flores")],
        "test": [row("never", "DO NOT OPEN", "test", "flores")],
    }
    splits = {}
    for split, rows in split_rows.items():
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
        splits[split] = {"path": path.name, "sha256": research.file_hash(path)}
    manifest = {
        "schema_version": research.DATASET_MANIFEST_SCHEMA,
        "dataset_id": "frozen-test",
        "normalization": research.NORMALIZATION,
        "freeze": {"immutable": True, "source_revisions": {"fixture": "a" * 40}},
        "splits": splits,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _fake_rows(languages, domains, texts):
    return [
        {"domain": domain, "language": language, "raw_utf8_bytes": len(text.encode("utf-8"))}
        for language, domain, text in zip(languages, domains, texts)
    ]


class Tokenizer:
    """A fake that satisfies the exact-budget and special-token invariants of
    research.validate_tokenizer without touching the Rust core."""

    def __init__(self, name, budget):
        self.name = name
        self.merges = 1  # uniq_superbpe requires at least one learned merge
        self.vocab = {}
        for special, token_id in research.SPECIAL_IDS.items():
            self.vocab[special] = token_id
        for byte in range(256):
            self.vocab[f"<0x{byte:02X}>"] = 4 + byte
        for index in range(len(self.vocab), budget):
            self.vocab[str(index)] = index

    def piece_for_id(self, token_id):
        return f"<0x{token_id:02X}>"


def _fake_metrics(tokenizer, texts, source_utf8_bytes):
    """Model validation metrics after the shape of research.token_metrics."""
    normalized = sum(len(text.encode("utf-8")) for text in texts)
    characters = sum(map(len, texts))
    tokens = characters  # one token per character, stable and deterministic
    return {
        "tokens": tokens,
        "utf8_bytes": normalized,
        "normalized_utf8_bytes": normalized,
        "source_utf8_bytes": source_utf8_bytes,
        "unicode_characters": characters,
        "tokens_per_unicode_character": tokens / characters,
        "bytes_per_token": normalized / tokens,
        "byte_fallback_tokens": 0,
        "byte_fallback_percent": 0.0,
    }


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    manifest = frozen_manifest(tmp_path)
    train_rows = _fake_rows(("en", "hi", "en", "hi"), ("latin", "indic", "latin", "indic"), ("aa", "bb", "cc", "dd"))
    train_texts = ["aa", "bb", "cc", "dd"]
    validation_texts = [f"heldout-{index}" for index in range(4)]
    validation_rows = _fake_rows(("en", "hi", "en", "hi"), ("flores", "flores", "flores", "flores"), validation_texts)
    source = {
        "manifest_sha256": research.file_hash(manifest),
        "dataset_id": "fixture",
        "normalization": research.NORMALIZATION,
        "source_revisions": {"fixture": "r" * 40},
        "train_file_sha256": "a" * 64,
        "validation_file_sha256": "b" * 64,
        "untouched_test_file_sha256": "c" * 64,
        "test_access": "forbidden_not_opened",
    }

    def fake_load_stage_source(_path):
        return (train_rows, train_texts, validation_rows, validation_texts, source)

    monkeypatch.setattr(stages, "load_stage_source", fake_load_stage_source)
    monkeypatch.setattr(stages, "VALIDATION_PARTITION_VERSION", "test_partition")
    identity = {
        "working_tree_dirty": False,
        "commit_hash": "d" * 40,
        "ledger_schema_version": ledger_module.SCHEMA_VERSION,
        "extension_hash": "e" * 64,
    }
    monkeypatch.setattr(research, "runtime_identity", lambda: identity)

    def trainer(name, texts, budget, directory):
        directory.mkdir()
        research.write_new_json(directory / "fixture.json", {"name": name, "budget": budget})
        return Tokenizer(name, budget), 0.5

    monkeypatch.setattr(research, "train_tokenizer", trainer)
    monkeypatch.setattr(research, "load_tokenizer", lambda row, root: Tokenizer(row["tokenizer"], row["vocab_budget"]))
    monkeypatch.setattr(research, "token_metrics", _fake_metrics)
    return manifest


def test_module_never_opens_or_imports_test_modeling():
    source = Path(ablation.__file__).read_text(encoding="utf-8")
    for forbidden in ("load_dataset", "train_lm", "evaluate", "CausalMiniTransformer", "lm_"):
        assert forbidden not in source
    assert "from benchmarks import run_phase_a as stages" in source
    assert "from benchmarks import run_research_experiments as h" in source


def test_preflight_reports_configuration_without_experiments(synthetic):
    conditions = ablation._build_plan(
        synthetic, [(name, budget) for name in research.COHORT for budget in research.VOCABS]
    )
    assert conditions["status"] == "planned"
    assert conditions["test_access"] == "forbidden_not_opened"
    assert conditions["data_split"] == "document_disjoint_train_validation_test_not_opened"
    assert conditions["language_model"] == {"initialized": False, "reachable": False}
    assert len(conditions["conditions"]) == len(research.COHORT) * len(research.VOCABS)
    assert conditions["objectives"]["metrics"] == list(ablation.METRICS)


def test_plan_records_frozen_split_hashes_and_assignments(synthetic):
    plan = ablation._build_plan(synthetic, [(name, budget) for name in research.COHORT for budget in research.VOCABS])
    assert plan["dataset"]["test_access"] == "forbidden_not_opened"
    assert plan["dataset"]["untouched_test_file_sha256"] == "c" * 64
    assert plan["training"]["assignment_hash"] == research.digest(
        [research.digest(t) for t in ("aa", "bb", "cc", "dd")]
    )
    assert plan["validation"]["assignment_hash"] == research.digest(
        [research.digest(t) for t in (f"heldout-{i}" for i in range(4))]
    )
    assert set(plan["validation"]["strata"]["domain/language"]) == {"flores::en", "flores::hi"}


def test_run_produces_deterministic_atomic_ledger(synthetic, tmp_path):
    first = ablation.run(SimpleNamespace(dataset=synthetic, output=tmp_path / "out1", resume=False))
    second = ablation.run(SimpleNamespace(dataset=synthetic, output=tmp_path / "out2", resume=False))
    assert first == second
    assert (tmp_path / "out1" / "ledger.json").exists()
    assert first["metadata"]["status"] == "complete"
    assert len(first["records"]) == len(research.COHORT) * len(research.VOCABS)
    assert all(record["model_kind"] == "tokenizer_only" for record in first["records"])
    assert all("test" not in record for record in first["records"])
    assert first["comparisons"]["metrics"] == list(ablation.METRICS)


def test_provenance_mismatch_is_rejected_on_resume(synthetic, tmp_path, monkeypatch):
    output = tmp_path / "out"
    ablation.run(SimpleNamespace(dataset=synthetic, output=output, resume=False))
    (output / "ledger.json").unlink()
    condition = research.read_json(output / "condition-000.json")
    condition["record"]["git_commit"] = "f" * 40
    (output / "condition-000.json").write_text(json.dumps(condition), encoding="utf-8")
    monkeypatch.setattr(
        research,
        "train_tokenizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("validated conditions must be skipped")),
    )
    with pytest.raises(ValueError, match="commit"):
        ablation.run(SimpleNamespace(dataset=synthetic, output=output, resume=True))


def test_resume_discards_artifact_lacking_a_checkpoint(synthetic, tmp_path):
    output = tmp_path / "out"
    ablation.run(SimpleNamespace(dataset=synthetic, output=output, resume=False))
    (output / "ledger.json").unlink()
    envelope = research.read_json(output / "condition-007.json")
    orphan = output / envelope["record"]["artifact"]
    assert orphan.is_dir()
    (output / "condition-007.json").unlink()
    resumed = ablation.run(SimpleNamespace(dataset=synthetic, output=output, resume=True))
    assert len(resumed["records"]) == len(research.COHORT) * len(research.VOCABS)
    assert (output / "ledger.json").exists()


def test_incomplete_run_fails_loud(synthetic, tmp_path):
    output = tmp_path / "out"
    ablation.run(SimpleNamespace(dataset=synthetic, output=output, resume=False))
    (output / "ledger.json").unlink()
    # A checkpoint present but never completed (e.g. a crashed writer) must abort
    # loudly instead of being silently rebuilt and republished as evidence.
    status = "condition_started: interrupted"
    envelope = research.read_json(output / "condition-002.json")
    envelope["status"] = status
    (output / "condition-002.json").write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        ablation.run(SimpleNamespace(dataset=synthetic, output=output, resume=True))


def test_comparison_reports_all_predeclared_metrics_and_regressions(synthetic, tmp_path):
    ledger = ablation.run(SimpleNamespace(dataset=synthetic, output=tmp_path / "out", resume=False))
    comparisons = ledger["comparisons"]
    assert set(comparisons["metrics"]) == set(ablation.METRICS)
    assert all(
        {f"{metric}_baseline", f"{metric}_ratio", f"{metric}_regression"} <= set(row)
        for row in comparisons["rows"]
        for metric in ablation.METRICS
    )
    assert all(row["is_baseline"] or "regressions" in row for row in comparisons["rows"])
