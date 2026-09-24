from __future__ import annotations

import copy

import pytest

from benchmarks import run_phase_c_confirm as phase_c
from benchmarks import run_research_experiments as h


def fixture():
    cfg = h.model_config("B", "cpu")
    identity = {"commit_hash": "a" * 40, "extension_hash": "ext"}
    plan = {
        "identity": identity,
        "model_config": cfg,
        "selection_sha256": "selection",
        "training": {"documents": 1, "assignment_hash": "train"},
        "validation": {"assignment_hash": "valid", "normalized_utf8_bytes": 10, "source_utf8_bytes": 11},
        "provenance": {"source_manifest_sha256": "source"},
    }
    source = {"artifact_hashes": {"model": "hash"}}
    byte_rows = {
        "train": [{h.BYTE_BUDGET_FIELD: phase_c.EXPOSURE_BYTES, h.BYTE_AUDIT_FIELD: phase_c.EXPOSURE_BYTES + 1}]
    }
    targets, squares = 2, 4
    row = {
        "tokenizer": "sp_unigram",
        "vocab_budget": phase_c.VOCAB,
        "budget_regime": "bytes",
        "seed": 1,
        "model_kind": "causal_transformer",
        "result_label": phase_c.LABEL,
        "actual_vocab_size": phase_c.VOCAB,
        "git_commit": identity["commit_hash"],
        "extension_hash": identity["extension_hash"],
        "selection_sha256": "selection",
        "model_config": cfg,
        "special_tokens": h.SPECIAL_IDS,
        "requested_budget": phase_c.EXPOSURE_BYTES,
        "tokenizer_artifact_hash": h.digest(source["artifact_hashes"]),
        "dataset_manifest_hash": "source",
        "training_assignment_hash": "train",
        "validation_assignment_hash": "valid",
        "completed_training_documents": 1,
        "completed_document_bytes": phase_c.EXPOSURE_BYTES,
        "training_bytes": h.byte_totals(byte_rows["train"]),
        "training_target_tokens": targets,
        "training_sequence_length_squared_sum": squares,
        "training_steps": 1,
        "training_byte_scope": "complete_document_prefix",
        **h.parameter_accounting(phase_c.VOCAB, h.SCREEN, 128),
        **h.flop_accounting(phase_c.VOCAB, h.SCREEN, targets, squares),
        "validation": h.nll_metrics(3.0, 2, 10, source_utf8_bytes=11),
    }
    return row, plan, source, byte_rows


def test_grid_is_exact_three_tokenizers_by_three_paired_seeds():
    assert len(phase_c.CONDITIONS) == 9
    assert {condition[0] for condition in phase_c.CONDITIONS} == set(phase_c.NAMES)
    assert {condition[3] for condition in phase_c.CONDITIONS} == {1, 2, 3}
    assert all(condition[1:3] == (16_384, "bytes") for condition in phase_c.CONDITIONS)


def test_complete_result_validates_without_test_metrics():
    row, plan, source, byte_rows = fixture()
    phase_c.validate_result(row, phase_c.CONDITIONS[0], plan, source, byte_rows)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("completed_training_documents", 0, "incomplete"),
        ("dataset_manifest_hash", "other", "provenance"),
        ("validation_assignment_hash", "other", "provenance"),
    ],
)
def test_result_rejects_shortened_or_mismatched_conditions(field, value, match):
    row, plan, source, byte_rows = fixture()
    row[field] = value
    with pytest.raises(ValueError, match=match):
        phase_c.validate_result(row, phase_c.CONDITIONS[0], plan, source, byte_rows)


def test_result_rejects_test_access():
    row, plan, source, byte_rows = fixture()
    row["test"] = {"bits_per_byte": 1.0}
    with pytest.raises(ValueError, match="cannot score test"):
        phase_c.validate_result(row, phase_c.CONDITIONS[0], plan, source, byte_rows)


def test_resume_rejects_completed_ledger(tmp_path):
    (tmp_path / "ledger.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot resume"):
        phase_c.resume_records(None, tmp_path, {}, None, None)
