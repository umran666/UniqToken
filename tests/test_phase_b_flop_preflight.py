"""Synthetic checks for the read-only Phase B tokenizer/FLOP preflight."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import run_research_experiments as h
from tools import phase_b_flop_preflight as p
from tools import phase_b_exposure as exposure_policy
from tools import verify_phase_b_flop_preflight as independent


def test_flop_schedule_matches_budget_without_constructing_an_lm():
    encoded = [[4] * 50 for _ in range(40)]
    budget = 1e11
    result = p.schedule(encoded, 16384, "flops", budget, 128)
    assert budget * 0.99 <= result["actual_analytical_flops"] <= budget
    assert result["training_steps"] > 0
    assert result["training_target_tokens"] <= result["training_sequence_length_squared_sum"]


def test_byte_schedule_consumes_all_frozen_documents_once():
    encoded = [[4], [5, 6], [7] * 3]
    result = p.schedule(encoded, 16384, "bytes", 6, 128)
    assert result["completed_training_documents"] == 3


def test_training_reloader_opens_only_the_train_artifact(tmp_path):
    train = tmp_path / "train.jsonl"
    rows = []
    for index, text in enumerate(("first", "second")):
        rows.append({"id": f"train-{index}", "text": text, "domain": "fixture", "language": "fixture",
                     "normalized_utf8_bytes": len(text.encode()), "raw_utf8_bytes": len(text.encode()),
                     "source": {}, "dedup": {"status": "accepted_after_exact_and_near_eval_check", "truncated_to_quota": False}})
    train.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    ordered = [{"position": index, "train_row_index": index, "id": row["id"], "domain": row["domain"],
                "language": row["language"], "normalized_text_hash": h.digest(row["text"]),
                "normalized_utf8_bytes": row["normalized_utf8_bytes"], "source_utf8_bytes": row["raw_utf8_bytes"],
                "source": row["source"], "dedup": row["dedup"]}
               for index, row in enumerate(rows)]
    sample = {"ordered_documents": ordered, "normalized_utf8_bytes": 11}
    assert p.load_frozen_documents(sample, train) == ["first", "second"]


def test_source_check_never_resolves_or_hashes_validation_or_test(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    source = {"schema_version": h.DATASET_MANIFEST_SCHEMA, "dataset_id": "fixture", "normalization": h.NORMALIZATION,
              "freeze": {"immutable": True, "source_revisions": {"r": "1"}},
              "splits": {"train": {"path": train.name, "sha256": h.file_hash(train)},
                         "validation": {"path": "must-not-open.jsonl", "sha256": "v"},
                         "test": {"path": "must-not-open-test.jsonl", "sha256": "t"}}}
    selection = {"dataset": {"dataset_id": "fixture", "normalization": h.NORMALIZATION,
                               "source_revisions": {"r": "1"}, "train_file_sha256": h.file_hash(train),
                               "validation_file_sha256": "v", "untouched_test_file_sha256": "t"}}
    assert p.check_source(selection, source, tmp_path / "manifest.json") == train.resolve()


def test_rejected_or_interrupted_preflight_never_exposes_a_success_bundle(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path / "attempt")
    monkeypatch.setattr(p, "preflight", lambda _: (_ for _ in ()).throw(ValueError("blocked")))
    with pytest.raises(ValueError, match="blocked"):
        p.run(args)
    rejection = h.read_json(args.output / "rejection-receipt.json")
    assert rejection["status"] == p.REJECTED
    assert rejection["final_preflight_sha256"] is None
    assert not (args.output / "frozen").exists()


def test_complete_preflight_is_atomic_and_not_lm_authorization(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path / "attempt")
    result = {"conditions": [{"tokenizer": "fixture"}], "all_conditions_feasible": True,
              "gate_passed": True, "content_sha256": "unused"}
    monkeypatch.setattr(p, "preflight", lambda _: result)
    receipt = p.run(args)
    assert receipt["status"] == p.STATUS
    assert receipt["experiments_started"] is False
    assert receipt["authorized_for_lm_execution"] is False
    assert (args.output / "frozen" / "preflight.json").exists()
    assert not (args.output / ".publication-incomplete").exists()


def test_blocked_condition_publishes_rejection_not_a_success_bundle(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path / "attempt")
    result = {"conditions": [{"tokenizer": "fixture", "vocab_budget": 16,
                              "status": "blocked", "regimes": {"flops": {"coverage_documents_completed": 2}}}],
              "all_conditions_feasible": False, "gate_passed": False, "runtime_blockers": [],
              "upstream_source_truncations_for_review": [], "content_sha256": "unused"}
    monkeypatch.setattr(p, "preflight", lambda _: result)
    receipt = p.run(args)
    assert receipt["reason"] == "FLOP_COVERAGE_BLOCKED"
    assert receipt["blocked_conditions"][0]["completed_coverage_documents"] == 2
    assert (args.output / "preflight-rejection.json").exists()
    assert not (args.output / "frozen").exists()


@pytest.fixture
def condition(tmp_path, monkeypatch):
    texts = [f"document-{i}" for i in range(32)]
    ids = [[4] for _ in range(30)] + [[5] * 200, [6] * 400]
    entries = [{"position": i, "id": f"id-{i}", "normalized_text_hash": h.digest(text),
                "normalized_utf8_bytes": len(text), "source_utf8_bytes": len(text) + 1,
                "domain": "fixture", "language": str(i % 30)} for i, text in enumerate(texts)]
    calls = []
    def encode(text):
        calls.append(text)
        return ids[texts.index(text)]
    tok = SimpleNamespace(vocab=range(16384), encode=encode)
    monkeypatch.setattr(h, "load_tokenizer", lambda *args: tok)
    for name in ("train_lm", "evaluate", "train_tokenizer", "CausalMiniTransformer"):
        monkeypatch.setattr(h, name, lambda *a, **kw: pytest.fail("training/evaluation invoked"))
    directory = tmp_path / "artifact"
    directory.mkdir()
    (directory / "fixture.json").write_text("{}")
    source = {"tokenizer": "fixture", "vocab_budget": 16384, "artifact": "artifact",
              "artifact_hashes": h.artifact_hashes(directory), "tokenizer_config": {},
              "git_commit": "a" * 40, "extension_hash": "e" * 64}
    sample = {"ordered_documents": entries, "strata": [{"domain": "fixture", "language": str(i)} for i in range(30)],
              "normalized_utf8_bytes": sum(map(len, texts)), "source_utf8_bytes": sum(map(len, texts)) + len(texts)}
    context = {"artifact_root": tmp_path, "context": 128, "model_config": h.model_config("B", "cpu")}
    return source, sample, texts, ids, calls, context


def test_full_schedule_advances_after_coverage_prefix(condition):
    source, sample, texts, ids, calls, context = condition
    # First 30 short documents, then 100 targets in the long 31st document.
    budget = 30 * h.training_flops(16384, h.SCREEN, 2) + h.training_flops(16384, h.SCREEN, 100)
    row = p.condition_report(source, sample, texts, flops_budget=budget,
                             byte_budget=sample["normalized_utf8_bytes"], context=context)
    assert calls == texts
    flop = row["regimes"]["flops"]
    assert flop["completed_training_documents"] == 30
    assert flop["terminal_document"]["id"] == "id-30"
    assert flop["terminal_document"]["target_tokens_consumed_including_eos"] == 100
    assert flop["training_target_tokens"] == 160
    assert flop["actual_analytical_flops"] == budget
    assert row["regimes"]["bytes"]["training_target_tokens"] == sum(map(len, ids)) + len(ids)


def test_blocked_flops_still_get_complete_byte_accounting(condition):
    source, sample, texts, ids, calls, context = condition
    row = p.condition_report(source, sample, texts, flops_budget=h.training_flops(16384, h.SCREEN, 1),
                             byte_budget=sample["normalized_utf8_bytes"], context=context)
    assert calls == texts and row["status"] == "blocked"
    byte = row["regimes"]["bytes"]
    assert byte["status"] == "feasible" and byte["completed_training_documents"] == len(texts)
    assert byte["complete_document_bytes"][h.BYTE_BUDGET_FIELD] == sample["normalized_utf8_bytes"]
    assert sum(s[h.BYTE_BUDGET_FIELD] for s in byte["strata"]) == sample["normalized_utf8_bytes"]
    assert len(row["encoded_document_accounting"]) == len(texts)


def test_one_complete_terminal_document_passes_approved_policy(condition):
    source, sample, texts, ids, calls, context = condition
    budget = h.training_flops(16384, h.SCREEN, 2)
    row = p.condition_report(source, sample, texts, flops_budget=budget,
                             byte_budget=sample["normalized_utf8_bytes"], context=context)
    run = row["regimes"]["flops"]
    assert row["status"] == "feasible"
    assert run["completed_training_documents"] == 0
    assert run["fully_predicted_documents"] == 1
    assert run["terminal_document"]["fully_predicted"] is True
    assert run["coverage_gate"] == "PASS"
    assert run["flop_interval"] == [0.99 * budget, budget]


def test_changed_order_and_byte_budget_rejected(condition):
    source, sample, texts, _, _, context = condition
    with pytest.raises(ValueError, match="order"):
        p.condition_report(source, sample, list(reversed(texts)), flops_budget=1e11,
                           byte_budget=sample["normalized_utf8_bytes"], context=context)
    with pytest.raises(ValueError, match="byte budget"):
        p.condition_report(source, sample, texts, flops_budget=1e11,
                           byte_budget=sample["normalized_utf8_bytes"] - 1, context=context)


def test_unreachable_flop_budget_rejects_noop():
    with pytest.raises(ValueError, match="FLOP budget"):
        p.schedule([[4]], 16384, "flops", 1, 128)


def test_input_mutation_is_rejected_at_final_pin_check(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{}")
    pins = {str(path): h.file_hash(path)}
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed or stale"):
        p.check_pins(pins)


@pytest.mark.parametrize("error", [RuntimeError("native computation failed"), KeyboardInterrupt()])
def test_encoder_failure_propagates_without_substitution(condition, monkeypatch, error):
    source, sample, texts, _, _, context = condition
    def encode(text):
        raise error
    monkeypatch.setattr(h, "load_tokenizer", lambda *args: SimpleNamespace(vocab=range(16384), encode=encode))
    with pytest.raises(type(error)):
        p.condition_report(source, sample, texts, flops_budget=1e11,
                           byte_budget=sample["normalized_utf8_bytes"], context=context)


def test_altered_artifact_rejected_before_encode(condition):
    source, sample, texts, _, calls, context = condition
    (context["artifact_root"] / source["artifact"] / "fixture.json").write_text("modified")
    with pytest.raises(ValueError, match="artifact changed"):
        p.condition_report(source, sample, texts, flops_budget=1e11,
                           byte_budget=sample["normalized_utf8_bytes"], context=context)
    assert calls == []


@pytest.mark.parametrize("field", ["actual_vocab_size", "artifact_hashes", "tokenizer_config", "extension_hash",
                                   "training_assignment_hash", "validation_assignment_hash"])
def test_selected_condition_changes_rejected(tmp_path, field):
    records = [{"tokenizer": name, "vocab_budget": vocab, "actual_vocab_size": vocab,
                "artifact": f"{name}-{vocab}", "artifact_hashes": {"model": "f" * 64},
                "tokenizer_config": h.tokenizer_configuration(name, vocab), "extension_hash": "e" * 64,
                "training_assignment_hash": "train", "validation_assignment_hash": "validation"}
               for name, vocab in p.screen.CONDITIONS]
    selection = {"conditions": json.loads(json.dumps(records)), "phase_a_identity": {}, "dataset": {},
                 "training": {}, "validation": {}, "special_tokens": h.SPECIAL_IDS}
    records[0][field] = "changed"
    ledger = {"metadata": {"identity": {}, "dataset": {}, "training": {}, "validation": {},
                           "stage": "A-SCREEN", "status": "complete"}, "records": records}
    path = tmp_path / "ledger.json"
    h.write_new_json(path, ledger)
    selection["phase_a_ledger_sha256"] = h.file_hash(path)
    with pytest.raises(ValueError):
        p.selected_phase_a_records(selection, ledger, path)


@pytest.mark.parametrize("blocker", ["runtime", "source_truncation"])
def test_budget_feasibility_does_not_override_provenance_blocker(tmp_path, monkeypatch, blocker):
    result = {"conditions": [], "all_conditions_feasible": True, "gate_passed": False,
              "runtime_blockers": ["pinned extension mismatch"] if blocker == "runtime" else [],
              "upstream_source_truncations_for_review": [{"id": "truncated"}] if blocker != "runtime" else []}
    monkeypatch.setattr(p, "preflight", lambda _: result)
    output = tmp_path / "attempt"
    receipt = p.run(SimpleNamespace(output=output))
    assert receipt["reason"] == "PROVENANCE_REVIEW_REQUIRED"
    assert not (output / "frozen").exists()


def test_success_publication_failure_retains_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "preflight", lambda _: {"conditions": [], "all_conditions_feasible": True, "gate_passed": True})
    def fail(*args):
        raise OSError("disk failure")
    monkeypatch.setattr(p.os, "rename", fail)
    args = SimpleNamespace(output=tmp_path / "attempt")
    with pytest.raises(OSError):
        p.run(args)
    assert not (args.output / "frozen").exists()
    assert h.read_json(args.output / "rejection-receipt.json")["status"] == p.REJECTED


@pytest.fixture
def frozen_exposure():
    documents = []
    for (domain, language), quota in p.quotas().items():
        for suffix, size in (("short", 2), ("rest", quota - 2)):
            identifier = f"{domain}-{language}-{suffix}"
            documents.append({"id": identifier, "domain": domain, "language": language,
                              "normalized_text_hash": h.digest(identifier), "normalized_utf8_bytes": size,
                              "source_utf8_bytes": size + 1, "train_row_index": len(documents)})
    sample = exposure_policy.construct(documents, p.quotas())
    body = {"exposure_schema_version": 2, "status": "exposure_frozen", "gate": "exact_packing_only",
            "provenance": {"selection_sha256": "s", "source_manifest_sha256": "m"},
            "sample": sample, "verification": {"status": "PASS"}}
    for field in ("tokenizer_outputs_used_for_sampling", "token_counts_used_for_sampling", "flops_used_for_sampling",
                  "validation_or_test_text_used_for_sampling", "selection_metrics_used", "lm_results_used",
                  "tokenizer_flop_preflight_run", "lm_training_run", "authorized_for_lm_execution"):
        body[field] = False
    return body, {"status": "exact_packing_feasible", "independent_verification": "PASS"}


def test_all_strata_recomputed_without_resampling(frozen_exposure, monkeypatch):
    body, receipt = frozen_exposure
    monkeypatch.setattr(exposure_policy, "construct", lambda *args: pytest.fail("preflight resampled"))
    body["content_sha256"] = h.digest(body)
    assert p.check_exposure(body, receipt, "s", "m")["normalized_utf8_bytes"] == 1_000_000


@pytest.mark.parametrize("mutation", ["quota", "total", "order", "coverage", "duplicate", "source", "verification", "test_used"])
def test_exposure_invariants_rejected_even_with_refreshed_content_hash(frozen_exposure, mutation):
    body, receipt = frozen_exposure
    if mutation == "quota":
        body["sample"]["strata"][0]["allocated_normalized_bytes"] += 1
    elif mutation == "total":
        body["sample"]["normalized_utf8_bytes"] -= 1
    elif mutation == "order":
        body["sample"]["ordered_documents"].reverse()
    elif mutation == "coverage":
        body["sample"]["ordered_documents"][0]["coverage_document"] = False
    elif mutation == "duplicate":
        body["sample"]["ordered_documents"][1]["id"] = body["sample"]["ordered_documents"][0]["id"]
    elif mutation == "source":
        body["provenance"]["source_manifest_sha256"] = "wrong"
    elif mutation == "verification":
        body["verification"]["status"] = "FAIL"
    else:
        body["validation_or_test_text_used_for_sampling"] = True
    body["content_sha256"] = h.digest(body)
    with pytest.raises(ValueError):
        p.check_exposure(body, receipt, "s", "m")


@pytest.mark.parametrize("vocab", [16384, 32768, 65536])
@pytest.mark.parametrize("counts", [[1, 300, 19], [127, 128, 129], [10] * 30 + [400]])
def test_independent_scalar_schedule_agrees_on_nonuniform_full_stream(vocab, counts):
    for regime in ("flops", "bytes"):
        result = independent.reference_schedule(counts, vocab, regime, 1e11)
        expected = p.schedule([[4] * n for n in counts], vocab, regime, 1e11, 128)
        for key, value in result.items():
            if key != "fully_predicted_documents":
                assert expected[key] == value, (vocab, counts, regime, key)


def test_independent_verification_distinguishes_eos_completion_from_byte_counter():
    cost = h.training_flops(16384, h.SCREEN, 2)
    result = independent.reference_schedule([1] * 30 + [300], 16384, "flops", 30 * cost)
    assert result["completed_training_documents"] == 29
    assert result["fully_predicted_documents"] == 30
    assert result["terminal_document_target_tokens"] == 2


def test_explicit_hash_pin_rejects_changed_evidence(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed or stale"):
        p.require_sha(path, "0" * 64, "fixture")


def test_rejected_historical_exposure_cannot_be_reused_under_new_policy():
    args = SimpleNamespace(exposure_sha256="8569e8045997b20d7cafa64ea4f02394bcc8e240d7586a9365c35ea38fdc6893")
    with pytest.raises(ValueError, match="historical exposure rejected"):
        p.preflight(args)
