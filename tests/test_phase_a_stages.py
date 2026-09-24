"""Regression tests for the two-stage tokenizer experiment gate."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_stage_loader_never_opens_test_split(tmp_path, monkeypatch):
    manifest = frozen_manifest(tmp_path)
    original = research.jsonl_rows

    def guarded(path):
        if Path(path).name == "test.jsonl":
            raise AssertionError("test split was opened")
        return original(path)

    monkeypatch.setattr(research, "jsonl_rows", guarded)
    train_rows, train, validation_rows, validation, source = stages.load_stage_source(manifest)
    assert len(train_rows) == len(train) == 2
    assert len(validation_rows) == len(validation) == 2
    assert source["test_access"] == "forbidden_not_opened"
    assert source["untouched_test_file_sha256"] == research.read_json(manifest)["splits"]["test"]["sha256"]


def test_screen_training_is_stratified_deterministic_and_shared():
    rows = [
        {"language": "en", "domain": "latin", "raw_utf8_bytes": 2},
        {"language": "en", "domain": "latin", "raw_utf8_bytes": 2},
        {"language": "hi", "domain": "indic", "raw_utf8_bytes": 2},
        {"language": "hi", "domain": "indic", "raw_utf8_bytes": 2},
    ]
    texts = ["aa", "bb", "cc", "dd"]
    first = stages.stratified_screen_training(rows, texts, 4)
    second = stages.stratified_screen_training(rows, texts, 4)
    assert first == second
    assert first[0] == ["aa", "cc"]
    assert {group["language"] for group in first[1]["groups"]} == {"en", "hi"}


def test_validation_partitions_are_disjoint_and_deterministic():
    texts = [f"validation-{index}" for index in range(20)]
    rows = [{"raw_utf8_bytes": len(text)} for text in texts]
    screen, confirm = stages.partition_validation(rows, texts)
    assert (screen, confirm) == stages.partition_validation(rows, texts)
    assert {text for _, text in screen}.isdisjoint(text for _, text in confirm)
    assert {text for _, text in screen} | {text for _, text in confirm} == set(texts)


def screening_ledger():
    records = []
    for vocab in research.VOCABS:
        for index, name in enumerate(research.COHORT):
            records.append(
                {
                    "tokenizer": name,
                    "vocab_budget": vocab,
                    "training_wall_clock_seconds": 1000 - index,
                    "validation": {
                        "tokens_per_unicode_character": 1.0 + index,
                        "byte_fallback_percent": float(index),
                    },
                }
            )
    return {"metadata": {"stage": "A-SCREEN", "result_label": "SCREENING", "status": "complete"}, "records": records}


def test_selection_is_deterministic_validation_only_and_has_no_test_input():
    ledger = screening_ledger()
    selected = stages.selection_from_screening(ledger)
    assert selected["conditions"] == [[research.COHORT[0], vocab] for vocab in research.VOCABS]
    assert selected["test_metrics_used"] is False
    assert "training_wall_clock_seconds" not in selected["validation_fields"]
    for row in ledger["records"]:
        row["training_wall_clock_seconds"] *= 100
        row["test"] = {"tokens_per_unicode_character": 0.0}
    assert stages.selection_from_screening(ledger)["conditions"] == selected["conditions"]


@pytest.fixture
def synthetic_stage(monkeypatch, tmp_path):
    train_texts = ["aa", "bb", "cc", "dd"]
    train_rows = [
        {"language": "en", "domain": "latin", "raw_utf8_bytes": 2},
        {"language": "en", "domain": "latin", "raw_utf8_bytes": 2},
        {"language": "hi", "domain": "indic", "raw_utf8_bytes": 2},
        {"language": "hi", "domain": "indic", "raw_utf8_bytes": 2},
    ]
    validation_texts = [f"heldout-{index}" for index in range(12)]
    validation_rows = [{"raw_utf8_bytes": len(text.encode("utf-8"))} for text in validation_texts]
    source = {
        "manifest_sha256": "m" * 64,
        "dataset_id": "fixture",
        "normalization": research.NORMALIZATION,
        "source_revisions": {"fixture": "r" * 40},
        "train_file_sha256": "a" * 64,
        "validation_file_sha256": "b" * 64,
        "untouched_test_file_sha256": "c" * 64,
        "test_access": "forbidden_not_opened",
    }
    monkeypatch.setattr(
        stages,
        "load_stage_source",
        lambda path: (train_rows, train_texts, validation_rows, validation_texts, source),
    )
    monkeypatch.setattr(stages, "SCREEN_TRAINING_TARGET_BYTES", 4)
    monkeypatch.setattr(stages, "SCREEN_TRAINING_MIN_BYTES", 1)
    monkeypatch.setattr(stages, "SCREEN_TRAINING_MAX_BYTES", 10)
    monkeypatch.setattr(research, "VOCABS", (260,))
    identity = {"working_tree_dirty": False, "commit_hash": "d" * 40, "extension_hash": "e" * 64}
    monkeypatch.setattr(research, "runtime_identity", lambda: identity)

    class Tokenizer:
        def __init__(self, name):
            self.name = name
            self.vocab = {str(index): index for index in range(260)}
            self.merges = 1

    def trainer(name, texts, budget, directory):
        directory.mkdir()
        research.write_new_json(directory / "fixture.json", {"texts": texts})
        return Tokenizer(name), 0.5

    monkeypatch.setattr(research, "train_tokenizer", trainer)
    monkeypatch.setattr(research, "load_tokenizer", lambda *args: None)

    def metric(tokenizer, texts, source_bytes):
        index = research.COHORT.index(tokenizer.name)
        tokens = len("".join(texts)) + index
        normalized = sum(len(text.encode("utf-8")) for text in texts)
        characters = sum(map(len, texts))
        return {
            "tokens": tokens,
            "utf8_bytes": normalized,
            "normalized_utf8_bytes": normalized,
            "source_utf8_bytes": source_bytes,
            "unicode_characters": characters,
            "tokens_per_unicode_character": tokens / characters,
            "bytes_per_token": normalized / tokens,
            "byte_fallback_tokens": 0,
            "byte_fallback_percent": 0.0,
        }

    monkeypatch.setattr(stages, "validation_metric", metric)
    return tmp_path, source


def test_screen_command_labels_results_and_confirmation_uses_independent_validation(synthetic_stage):
    tmp_path, _ = synthetic_stage
    screen_output = tmp_path / "screen"
    screen_args = SimpleNamespace(
        stage="A-SCREEN",
        dataset=tmp_path / "manifest.json",
        output=screen_output,
        resume=False,
        screening=None,
        selection=None,
    )
    screen = stages.run_stage(screen_args)
    assert len(screen["records"]) == 5
    assert all(row["result_label"] == "SCREENING" and "test" not in row for row in screen["records"])
    assert screen["metadata"]["test_access"] == "forbidden_not_opened"

    selection_path = tmp_path / "selection.json"
    stages.write_selection(screen_output / "ledger.json", selection_path, tmp_path / "manifest.json")
    confirm_output = tmp_path / "confirm"
    confirm = stages.run_stage(
        SimpleNamespace(
            stage="A-CONFIRM",
            dataset=tmp_path / "manifest.json",
            output=confirm_output,
            resume=False,
            screening=screen_output / "ledger.json",
            selection=selection_path,
        )
    )
    assert len(confirm["records"]) == 1
    assert confirm["metadata"]["training"]["scope"] == "full_frozen_training_corpus"
    assert confirm["metadata"]["validation"]["assignment_hash"] != screen["metadata"]["validation"]["assignment_hash"]
    assert all("test" not in row for row in confirm["records"])


def test_confirmation_rejects_modified_selection(synthetic_stage):
    tmp_path, _ = synthetic_stage
    screen_output = tmp_path / "screen"
    stages.run_stage(
        SimpleNamespace(
            stage="A-SCREEN",
            dataset=tmp_path / "manifest.json",
            output=screen_output,
            resume=False,
            screening=None,
            selection=None,
        )
    )
    selection_path = tmp_path / "selection.json"
    stages.write_selection(screen_output / "ledger.json", selection_path, tmp_path / "manifest.json")
    selection = research.read_json(selection_path)
    selection["conditions"][0][0] = research.COHORT[-1]
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(ValueError, match="deterministically"):
        stages.run_stage(
            SimpleNamespace(
                stage="A-CONFIRM",
                dataset=tmp_path / "manifest.json",
                output=tmp_path / "confirm",
                resume=False,
                screening=screen_output / "ledger.json",
                selection=selection_path,
            )
        )


def test_screen_stage_resumes_only_validated_complete_conditions(synthetic_stage, monkeypatch):
    tmp_path, _ = synthetic_stage
    output = tmp_path / "screen"
    args = SimpleNamespace(
        stage="A-SCREEN",
        dataset=tmp_path / "manifest.json",
        output=output,
        resume=False,
        screening=None,
        selection=None,
    )
    original = stages.run_stage(args)
    (output / "ledger.json").unlink()
    args.resume = True
    monkeypatch.setattr(
        research,
        "train_tokenizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("validated conditions must be skipped")),
    )
    resumed = stages.run_stage(args)
    assert resumed == original


def test_screen_resume_rejects_tokenizer_configuration_change(synthetic_stage):
    tmp_path, _ = synthetic_stage
    output = tmp_path / "screen"
    args = SimpleNamespace(
        stage="A-SCREEN",
        dataset=tmp_path / "manifest.json",
        output=output,
        resume=False,
        screening=None,
        selection=None,
    )
    stages.run_stage(args)
    (output / "ledger.json").unlink()
    condition = research.read_json(output / "condition-000.json")
    condition["record"]["tokenizer_config"]["byte_fallback"] = False
    (output / "condition-000.json").write_text(json.dumps(condition), encoding="utf-8")
    args.resume = True
    with pytest.raises(ValueError, match="configuration"):
        stages.run_stage(args)
