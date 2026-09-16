"""Synthetic gate regressions; no research training or benchmark results."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import run_phase_b_screen as b
from benchmarks import run_research_experiments as h
from benchmarks import run_phase_a as a


@pytest.fixture
def gate(tmp_path, monkeypatch):
    identity = {
        "ledger_schema_version": 3, "commit_hash": "c" * 40, "source_hash": "s" * 64,
        "working_tree_dirty": False, "extension_status": "installed", "extension_hash": "e" * 64,
        "versions": {k: "fixture" for k in ("python", "torch", "numpy", "regex", "sentencepiece")},
    }
    monkeypatch.setattr(h, "runtime_identity", lambda: copy.deepcopy(identity))
    monkeypatch.setattr(b, "implementation_hash", lambda: "f" * 64)
    monkeypatch.setattr(b, "verify_historical_code", lambda identity: None)
    original_stratify = a.stratified_screen_training
    monkeypatch.setattr(a, "stratified_screen_training", lambda rows, texts: original_stratify(rows, texts, 5))
    splits = {}
    for split, texts in {"train": ["aa", "bbb"], "validation": [f"heldout-{i}" for i in range(20)]}.items():
        rows = [{"id": f"{split}-{i}", "text": t, "raw_utf8_bytes": len(t), "normalized_utf8_bytes": len(t),
                 "language": "fixture", "domain": "fixture"} for i, t in enumerate(texts)]
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        splits[split] = {"path": path.name, "sha256": h.file_hash(path)}
    # Deliberately absent. Any attempt to read test fails, including a hash read.
    splits["test"] = {"path": "missing-test.jsonl", "sha256": "t" * 64}
    manifest = tmp_path / "manifest.json"
    h.write_new_json(manifest, {"schema_version": h.DATASET_MANIFEST_SCHEMA, "dataset_id": "fixture",
                              "normalization": h.NORMALIZATION, "freeze": {"immutable": True, "source_revisions": {"test": "a" * 40}},
                              "splits": splits})
    docs, byte_rows, source, training, validation = b.dataset_context(manifest)
    exposure_training = {
        "scope": "frozen_exact_phase_b_exposure", "normalized_utf8_bytes": 5,
        "source_utf8_bytes": 5, "documents": 2, "assignment_hash": h.digest(["fixture-exposure"]),
    }
    source_manifest = tmp_path / "source.json"
    exposure_manifest = tmp_path / "exposure.json"
    source_manifest.write_text("{}", encoding="utf-8")
    exposure_manifest.write_text("{}", encoding="utf-8")
    source_sha = h.file_hash(source_manifest)
    exposure_sha = h.file_hash(exposure_manifest)
    def exposure_context(args, selection):
        h.require(args.source_manifest_sha256 == source_sha, "source manifest changed or stale")
        h.require(args.exposure_manifest_sha256 == exposure_sha, "exposure manifest changed or stale")
        provenance = {
            "source_manifest_sha256": source_sha, "exposure_manifest_sha256": exposure_sha,
            "exposure_content_sha256": "3" * 64, "exposure_order_sha256": exposure_training["assignment_hash"],
            "exposure_receipt_sha256": "4" * 64,
        }
        return (copy.deepcopy(docs), copy.deepcopy(byte_rows), copy.deepcopy(exposure_training),
                copy.deepcopy(validation), provenance)
    monkeypatch.setattr(b, "frozen_exposure_context", exposure_context)
    phase_dir = tmp_path / "phase-a"
    phase_dir.mkdir()
    records = []
    for name in h.COHORT:
        for vocab in h.VOCABS:
            artifact = f"{name}-{vocab}"
            directory = phase_dir / artifact
            directory.mkdir()
            h.write_new_json(directory / "fixture.json", {"vocab_size": vocab})
            count = validation["unicode_characters"]
            records.append({
                "tokenizer": name, "vocab_budget": vocab, "actual_vocab_size": vocab,
                "git_commit": identity["commit_hash"], "extension_hash": identity["extension_hash"],
                "dataset_manifest_hash": source["manifest_sha256"], "tokenizer_config": h.tokenizer_configuration(name, vocab),
                "training_assignment_hash": training["assignment_hash"], "validation_assignment_hash": validation["assignment_hash"],
                "training_wall_clock_seconds": 1.0, "learned_merges": 1, "artifact": artifact,
                "artifact_hashes": h.artifact_hashes(directory), "result_label": "SCREENING",
                "validation": {"tokens": count, "unicode_characters": count, "utf8_bytes": validation["normalized_utf8_bytes"],
                               **h.byte_totals(byte_rows["validation"]), "tokens_per_unicode_character": 1.0,
                               "bytes_per_token": validation["normalized_utf8_bytes"] / count,
                               "byte_fallback_tokens": 0, "byte_fallback_percent": 0.0},
            })
    meta = a._stage_plan("A-SCREEN", identity, source, training, validation,
                         [(n, v) for n in h.COHORT for v in h.VOCABS])
    ledger = {"metadata": {**meta, "status": "complete"}, "records": records}
    path = phase_dir / "ledger.json"
    h.write_new_json(path, ledger)
    monkeypatch.setattr(h, "load_tokenizer", lambda row, root: SimpleNamespace(vocab=range(row["vocab_budget"])))
    selection = tmp_path / "selection.json"
    args = SimpleNamespace(phase_a=path, dataset=manifest, output=selection)
    report = b.freeze(args)
    args = SimpleNamespace(**vars(args))
    args.output = tmp_path / "phase-b"
    args.selection, args.selection_sha256 = selection, report["sha256"]
    args.flops, args.bytes, args.device = 1e11, 5, "cpu"
    args.source_manifest, args.source_manifest_sha256 = source_manifest, source_sha
    args.exposure_manifest, args.exposure_manifest_sha256 = exposure_manifest, exposure_sha
    args.resume = False
    return args, ledger, identity


def test_freeze_is_deterministic_nine_conditions_both_regimes(gate):
    args, ledger, _ = gate
    selection = h.read_json(args.selection)
    assert selection == b.selection_payload(ledger, h.file_hash(args.phase_a), b.implementation_hash())
    assert [(r["tokenizer"], r["vocab_budget"]) for r in selection["conditions"]] == list(b.CONDITIONS)
    assert selection["seeds"] == [0]
    assert selection["regimes"] == ["flops", "bytes"]
    assert selection["lm_run_count"] == 18
    assert "NOT the current" in selection["rationale"]["boundary_bpe"]
    before = args.selection.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        b.freeze(SimpleNamespace(phase_a=args.phase_a, dataset=args.dataset, output=args.selection))
    assert args.selection.read_bytes() == before


@pytest.mark.parametrize("mutation", ["cohort", "seed", "regimes", "model", "test", "implementation", "metrics"])
def test_modified_selection_rejected_even_with_new_external_hash(gate, mutation):
    args, _, _ = gate
    data = h.read_json(args.selection)
    if mutation == "cohort":
        data["conditions"][0]["tokenizer"] = "sp_bpe"
    elif mutation == "seed":
        data["seeds"] = [0, 1, 2]
    elif mutation == "regimes":
        data["regimes"] = ["bytes"]
    elif mutation == "model":
        data["model_config_cpu_template"]["head"] = "tied"
    elif mutation == "test":
        data["test_metrics_used"] = True
    elif mutation == "implementation":
        data["implementation_git_blob_hash"] = "a" * 64
    else:
        data["conditions"][0]["phase_a_validation_metrics"]["bytes_per_token"] = 100
    args.selection.write_text(json.dumps(data), encoding="utf-8")
    args.selection_sha256 = h.file_hash(args.selection)
    with pytest.raises(ValueError, match="selection/configuration"):
        b.prepare(args)


@pytest.mark.parametrize("mutation", ["artifact", "dataset", "config", "vocab", "specials", "assignment", "test", "incomplete"])
def test_stale_phase_a_inputs_rejected_before_execution(gate, mutation):
    args, ledger, _ = gate
    row = ledger["records"][0]
    if mutation == "artifact":
        (args.phase_a.parent / row["artifact"] / "fixture.json").write_text("corrupt")
    elif mutation == "dataset":
        ledger["metadata"]["dataset"]["manifest_sha256"] = "old"
    elif mutation == "config":
        row["tokenizer_config"]["normalization"] = "none"
    elif mutation == "vocab":
        row["actual_vocab_size"] -= 1
    elif mutation == "specials":
        row["tokenizer_config"]["special_tokens"] = {}
    elif mutation == "assignment":
        row["training_assignment_hash"] = "other"
    elif mutation == "test":
        row["test"] = {"bits_per_byte": 1.0}
    else:
        ledger["records"].pop()
    args.phase_a.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(ValueError):
        b.prepare(args, execution=True)
    assert not args.output.exists()


@pytest.mark.parametrize("flops,byte_budget,device", [(0, 5, "cpu"), (1, 5, "cpu"), (math.nan, 5, "cpu"),
                                                    (1e12, 5, "cpu"), (1e11, 4, "cpu"), (1e11, 0, "cpu"),
                                                    (1e11, 1_000_001, "cpu"), (1e11, 5, "unknown")])
def test_invalid_screening_configuration_fails(gate, flops, byte_budget, device):
    args, _, _ = gate
    args.flops, args.bytes, args.device = flops, byte_budget, device
    with pytest.raises(ValueError):
        b.prepare(args)


@pytest.mark.parametrize("mutation", ["dirty", "extension", "versions"])
def test_incompatible_execution_runtime_rejected(gate, mutation):
    args, _, identity = gate
    if mutation == "dirty":
        identity["working_tree_dirty"] = True
    elif mutation == "extension":
        identity["extension_hash"] = "wrong"
    else:
        identity["versions"]["sentencepiece"] = "wrong"
    with pytest.raises(ValueError):
        b.prepare(args, execution=True)


def test_gpu_build_is_recorded_but_tokenizer_runtime_must_match(gate):
    _, _, identity = gate
    current = copy.deepcopy(identity)
    current["versions"]["torch"] = "fixture+cuda"
    b.check_runtime(current, identity)


@pytest.mark.parametrize("available,device,workspace", [(False, "cuda", ":4096:8"),
                                                       (True, "cuda:5", ":4096:8"), (True, "cuda", "")])
def test_cuda_preflight_fails_loudly(gate, monkeypatch, available, device, workspace):
    import torch
    _, _, identity = gate
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    with pytest.raises(ValueError):
        b.check_runtime(identity, identity, device)


def fake_lm(gate, calls):
    _, ledger, _ = gate
    def train(tok, docs, regime, budget, device, document_bytes):
        cfg, context, seed = h.SCREEN, 128, 0
        assert set(docs) == set(document_bytes) == {"train", "validation"}
        assert docs["train"] == ["aa", "bbb"]
        assert seed == 0 and cfg == h.SCREEN
        calls.append((len(tok.vocab), regime, seed))
        targets = int(budget // h.training_flops(len(tok.vocab), cfg, 1)) if regime == "flops" else 3
        completed = 0 if regime == "flops" else 2
        heldout = ledger["metadata"]["validation"]
        evaluation_targets = sum(map(len, docs["validation"])) + len(docs["validation"])
        return {**h.parameter_accounting(len(tok.vocab), cfg, context),
                **h.flop_accounting(len(tok.vocab), cfg, targets, targets),
                "training_steps": targets, "training_target_tokens": targets, "training_sequence_length_squared_sum": targets,
                "training_byte_scope": "complete_document_prefix", "completed_training_documents": completed,
                "fully_predicted_documents": max(1, completed), "partial_final_window": False,
                "terminal_document_target_tokens": 1 if regime == "flops" else 0,
                "coverage_policy_version": "flop_fixed_budget_one_complete_document_v1",
                "minimum_fully_predicted_documents": 1, "coverage_gate": "PASS",
                "completed_document_bytes": 0 if completed == 0 else 5,
                "training_bytes": h.byte_totals(document_bytes["train"], completed),
                "validation": h.nll_metrics(15.0, evaluation_targets, heldout["normalized_utf8_bytes"],
                                            source_utf8_bytes=heldout["source_utf8_bytes"])}
    return train


def test_only_eighteen_frozen_runs_and_complete_ledger(gate, monkeypatch):
    args, _, _ = gate
    calls = []
    monkeypatch.setattr(b, "train_phase_b_condition", fake_lm(gate, calls))
    report = b.run(args)
    result = h.read_json(report["ledger"])
    assert len(calls) == len(result["records"]) == 18
    assert {r["tokenizer"] for r in result["records"]} == set(b.NAMES)
    assert all("test" not in r for r in result["records"])
    assert result["metadata"]["test_access"] == "forbidden_not_opened"
    assert {r["training_bytes"]["normalized_utf8_bytes"] for r in result["records"] if r["budget_regime"] == "bytes"} == {5}
    with pytest.raises(ValueError, match="output exists"):
        b.run(args)


def test_phase_b_resume_validates_and_skips_complete_conditions(gate, monkeypatch):
    args, _, _ = gate
    calls = []
    monkeypatch.setattr(b, "train_phase_b_condition", fake_lm(gate, calls))
    ledger_path = Path(b.run(args)["ledger"])
    assert len(calls) == 18
    ledger_path.unlink()
    args.resume = True
    calls.clear()
    result = b.run(args)
    assert calls == []
    assert len(h.read_json(result["ledger"])["records"]) == 18


@pytest.mark.parametrize("mutation", ["plan", "selection", "condition", "artifact", "source", "exposure", "runtime"])
def test_phase_b_resume_refuses_changed_provenance(gate, monkeypatch, mutation):
    args, ledger, identity = gate
    monkeypatch.setattr(b, "train_phase_b_condition", fake_lm(gate, []))
    ledger_path = Path(b.run(args)["ledger"])
    ledger_path.unlink()
    args.resume = True
    if mutation == "plan":
        (args.output / "plan.json").write_text("{}", encoding="utf-8")
    elif mutation == "selection":
        (args.output / "selection.json").write_text("{}", encoding="utf-8")
    elif mutation == "condition":
        condition = h.read_json(args.output / "condition-000.json")
        condition["record"]["requested_budget"] = -1
        (args.output / "condition-000.json").write_text(json.dumps(condition), encoding="utf-8")
    elif mutation == "artifact":
        source = ledger["records"][0]
        (args.phase_a.parent / source["artifact"] / "fixture.json").write_text("changed", encoding="utf-8")
    elif mutation == "source":
        args.source_manifest_sha256 = "9" * 64
    elif mutation == "exposure":
        args.exposure_manifest_sha256 = "8" * 64
    else:
        identity["versions"]["torch"] = "changed"
    with pytest.raises((ValueError, TypeError)):
        b.run(args)
    assert not ledger_path.exists()


@pytest.mark.parametrize("target", ["selection", "plan", "snapshot", "artifact", "no_op", "runtime", "error"])
def test_mutation_or_failure_stops_run_without_final_ledger(gate, monkeypatch, target):
    args, _, identity = gate
    wrapped = fake_lm(gate, [])
    calls = []
    def fail(*pos, **kw):
        measured = wrapped(*pos, **kw)
        calls.append(1)
        if target == "selection":
            args.selection.write_text("{}")
        elif target == "plan":
            (args.output / "plan.json").write_text("{}")
        elif target == "snapshot":
            (args.output / "selection.json").write_text("{}")
        elif target == "runtime":
            identity["versions"]["torch"] = "changed"
        elif target == "artifact":
            (args.phase_a.parent / "sp_unigram-16384" / "fixture.json").write_text("changed")
        elif target == "no_op":
            measured["training_target_tokens"] = 0
        else:
            raise RuntimeError("native failure")
        return measured
    monkeypatch.setattr(b, "train_phase_b_condition", fail)
    with pytest.raises((ValueError, RuntimeError)):
        b.run(args)
    assert len(calls) == 1
    assert not (args.output / "ledger.json").exists()


@pytest.mark.parametrize("field", ["core_params", "input_embedding_params", "output_head_params", "total_params",
                                   "core_analytical_flops", "output_projection_flops", "actual_analytical_flops",
                                   "flop_estimator_version", "training_bytes", "validation", "seed", "test"])
def test_lm_ledger_rejects_bad_accounting_or_selection(gate, monkeypatch, field):
    args, _, _ = gate
    monkeypatch.setattr(b, "train_phase_b_condition", fake_lm(gate, []))
    plan, ledger, _, byte_rows = b.prepare(args)
    result = h.read_json(b.run(args)["ledger"])
    result["records"][0][field] = "invalid"
    with pytest.raises((ValueError, TypeError)):
        b.validate_ledger(result, plan, ledger, byte_rows)
    result["records"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        b.validate_ledger(result, plan, ledger, byte_rows)


def test_no_test_metrics_accepted_by_selection_builder(gate):
    _, ledger, _ = gate
    ledger["records"][0]["test"] = {"loss": 0}
    with pytest.raises(ValueError, match="test metrics"):
        b.selection_payload(ledger, "a" * 64, "b" * 64)


def test_historical_code_behavior_change_is_rejected(tmp_path, monkeypatch):
    identity = {"commit_hash": "a" * 40, "working_tree_dirty": False,
                "extension_status": "installed", "extension_hash": "e" * 64, "source_hash": "s"}
    monkeypatch.setattr(b.lineage, "source_hash", lambda commit: "s")
    monkeypatch.setattr(b, "executable_paths", lambda commit: ["uniqtoken/tokenizer.py"])
    monkeypatch.setattr(b.lineage, "tree", lambda commit: {"uniqtoken/tokenizer.py": ("100644", "blob", "original")})
    monkeypatch.setattr(b.lineage, "git", lambda *args: b"changed")
    with pytest.raises(ValueError, match="changed historical executable"):
        b.verify_historical_code(identity)
