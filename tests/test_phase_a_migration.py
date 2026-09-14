"""Migration tests use tiny synthetic conditions; never run the research corpus."""
import copy
import json
from types import SimpleNamespace

import pytest

from benchmarks import phase_a_migrate as migration
from benchmarks import run_phase_a as stages
from benchmarks import run_research_experiments as research
from tests.test_phase_a_stages import synthetic_stage  # noqa: F401

REAL_LOAD_TOKENIZER = research.load_tokenizer


@pytest.fixture
def saved(synthetic_stage, monkeypatch):
    root, dataset = synthetic_stage
    identity = research.runtime_identity()
    identity.update(source_hash="old-source", versions={"python": "test"})
    args = SimpleNamespace(stage="A-SCREEN", dataset=root / "manifest.json", output=root / "source", resume=False)
    stages.run_stage(args)
    # Keep only the three approved baseline families in this tiny grid.
    for index in (3, 4):
        (args.output / f"condition-{index:03d}.json").unlink()
    (args.output / "ledger.json").unlink()
    active = {**identity, "commit_hash": "f" * 40, "source_hash": "new-source"}
    monkeypatch.setattr(research, "runtime_identity", lambda: active)
    monkeypatch.setattr(migration, "approve_code_change", lambda a, b: {"policy": migration.POLICY, "trained_commit": a, "revalidated_commit": b})
    monkeypatch.setattr(migration, "source_hash", lambda c: "new-source" if c == active["commit_hash"] else "old-source")
    monkeypatch.setattr(research, "load_tokenizer", lambda row, root: SimpleNamespace(name=row["tokenizer"]))
    monkeypatch.setattr(research, "train_tokenizer", lambda *a, **kw: pytest.fail("migration must not train"))
    return SimpleNamespace(source=args.output, output=root / "migrated", dataset=args.dataset, dry_run=False), active


def mutate_record(args, fn):
    path = args.source / "condition-000.json"
    envelope = research.read_json(path)
    fn(envelope["record"])
    path.write_text(json.dumps(envelope))


def test_valid_migration_preserves_provenance_and_recomputes_metrics(saved, monkeypatch):
    args, identity = saved
    original_files = {p.name: p.read_bytes() for p in args.source.glob("*.json")}
    calls = []
    metric = stages.validation_metric

    def evaluate(tok, texts, source_bytes):
        calls.append(tok.name)
        result = metric(tok, texts, source_bytes)
        result["byte_fallback_tokens"] = 1
        result["byte_fallback_percent"] = 100 / result["tokens"]
        return result

    monkeypatch.setattr(stages, "validation_metric", evaluate)
    assert migration.migrate(args)["revalidated_conditions"] == 3
    assert calls == ["sp_unigram", "sp_bpe", "boundary_bpe"]
    assert not (args.output / "ledger.json").exists()
    for name, contents in original_files.items():
        assert (args.source / name).read_bytes() == contents
        assert (args.output / "originals" / name).read_bytes() == contents
    row = research.read_json(args.output / "condition-000.json")["record"]
    original = research.read_json(args.source / "condition-000.json")["record"]
    assert row["migration"]["trained_commit"] == original["git_commit"]
    assert row["migration"]["revalidated_commit"] == identity["commit_hash"]
    assert row["training_wall_clock_seconds"] == original["training_wall_clock_seconds"]
    assert row["validation"] != original["validation"]


@pytest.mark.parametrize("change", [
    lambda r: r.update(artifact_hashes={"fixture.json": "bad"}),
    lambda r: r.update(dataset_manifest_hash="wrong"),
    lambda r: r["tokenizer_config"].update(byte_fallback=False),
    lambda r: r.update(vocab_budget=261),
    lambda r: r["tokenizer_config"]["special_tokens"].update({"<|unk|>": 9}),
    lambda r: r.update(training_assignment_hash="wrong"),
    lambda r: r.update(validation_assignment_hash="wrong"),
    lambda r: r.update(artifact="../outside"),
])
def test_rejects_invalid_original_record(saved, change):
    args, _ = saved
    mutate_record(args, change)
    with pytest.raises(ValueError):
        migration.migrate(args)
    assert not args.output.exists()


def test_missing_artifact_rejected(saved):
    args, _ = saved
    (args.source / "sp_unigram-260" / "fixture.json").unlink()
    with pytest.raises(ValueError, match="artifact"):
        migration.migrate(args)


@pytest.mark.parametrize("stage", ["B", "C", "A-CONFIRM"])
def test_other_phases_rejected_before_loading_dataset(saved, monkeypatch, stage):
    args, _ = saved
    path = args.source / "plan.json"
    plan = research.read_json(path)
    plan["stage"] = stage
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(stages, "prepare_stage", lambda a: pytest.fail("must reject phase first"))
    with pytest.raises(ValueError, match="A-SCREEN"):
        migration.migrate(args)


def test_duplicate_migration_rejected(saved):
    args, _ = saved
    migration.migrate(args)
    with pytest.raises(ValueError, match="duplicate"):
        migration.migrate(args)


def test_dry_run_checks_without_evaluation_or_writes(saved, monkeypatch):
    args, _ = saved
    args.dry_run = True
    monkeypatch.setattr(stages, "validation_metric", lambda *a: pytest.fail("dry run must not evaluate"))
    result = migration.migrate(args)
    assert result["eligible_conditions"] == 3
    assert not args.output.exists()


@pytest.mark.parametrize("field,value", [("extension_hash", "other"), ("versions", {}), ("working_tree_dirty", True)])
def test_runtime_mismatch_rejected(saved, field, value):
    args, active = saved
    active[field] = value
    with pytest.raises(ValueError):
        migration.migrate(args)


def test_changed_current_dataset_rejected(saved, monkeypatch):
    args, _ = saved
    load = stages.load_stage_source

    def changed(path):
        *rows, dataset = load(path)
        return (*rows, {**dataset, "manifest_sha256": "different"})

    monkeypatch.setattr(stages, "load_stage_source", changed)
    with pytest.raises(ValueError, match="plan mismatch"):
        migration.migrate(args)


def test_resume_skips_migrated_conditions_and_final_ledger_validates(saved, monkeypatch):
    args, _ = saved
    migration.migrate(args)
    trained = []

    def train(name, texts, budget, directory):
        assert name in ("uniq_unigram", "uniq_superbpe")
        trained.append(name)
        directory.mkdir()
        research.write_new_json(directory / "fixture.json", {})
        return SimpleNamespace(name=name, vocab=dict.fromkeys(range(budget)), merges=1), 1.0

    monkeypatch.setattr(research, "train_tokenizer", train)
    ledger = stages.run_stage(SimpleNamespace(stage="A-SCREEN", dataset=args.dataset, output=args.output, resume=True))
    assert trained == ["uniq_unigram", "uniq_superbpe"]
    assert len(ledger["records"]) == 5
    assert sum("migration" in r for r in ledger["records"]) == 3


def test_tampered_migrated_training_metrics_rejected(saved):
    args, identity = saved
    migration.migrate(args)
    path = args.output / "condition-000.json"
    envelope = research.read_json(path)
    envelope["record"]["training_wall_clock_seconds"] = 999
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="training provenance"):
        stages.run_stage(SimpleNamespace(stage="A-SCREEN", dataset=args.dataset, output=args.output, resume=True))


def test_code_policy_accepts_exact_reviewed_blobs(monkeypatch):
    before = {"uniqtoken/tokenizer.py": ("100644", "blob", "unchanged")}
    after = {**before, **{p: ("100644", "blob", h) for p, h in migration.APPROVED_BLOBS.items()}}
    monkeypatch.setattr(migration, "tree", lambda c: before if c == migration.TRAINED_COMMIT else after)
    assert migration.approve_code_change(migration.TRAINED_COMMIT, "f" * 40)["policy"] == migration.POLICY


@pytest.mark.parametrize("path", ["uniqtoken/tokenizer.py", "crates/uniqtoken_core/Cargo.lock", "pyproject.toml",
                                   "benchmarks/run_research_experiments.py", "benchmarks/run_phase_a.py"])
def test_behavior_changes_rejected_even_in_approved_file(monkeypatch, path):
    before = {"uniqtoken/tokenizer.py": ("100644", "blob", "unchanged")}
    after = {**before, **{p: ("100644", "blob", h) for p, h in migration.APPROVED_BLOBS.items()}}
    after[path] = ("100644", "blob", "unreviewed")
    monkeypatch.setattr(migration, "tree", lambda c: before if c == migration.TRAINED_COMMIT else after)
    with pytest.raises(ValueError, match="unapproved"):
        migration.approve_code_change(migration.TRAINED_COMMIT, "f" * 40)


@pytest.fixture
def real_saved(saved, monkeypatch):
    args, active = saved
    for index in (0, 1):
        (args.source / f"condition-{index:03d}.json").unlink()
    path = args.source / "condition-002.json"
    envelope = research.read_json(path)
    artifact = args.source / envelope["record"]["artifact"]
    (artifact / "fixture.json").unlink()
    vocab = {**research.SPECIAL_IDS, **{f"<0x{i:02X}>": i + 4 for i in range(256)}}
    research.write_new_json(artifact / "bpe.json", {"vocab": vocab, "merges": []})
    envelope["record"]["artifact_hashes"] = research.artifact_hashes(artifact)
    path.write_text(json.dumps(envelope))
    monkeypatch.setattr(research, "load_tokenizer", REAL_LOAD_TOKENIZER)
    monkeypatch.setattr(stages, "validation_metric", lambda tok, texts, source_bytes:
                        research.token_metrics(tok, texts, source_utf8_bytes=source_bytes))
    return args


def test_actual_saved_artifact_is_loaded_and_revalidated_without_training(real_saved):
    args = real_saved
    result = migration.migrate(args)
    assert result["revalidated_conditions"] == 1
    row = research.read_json(args.output / "condition-002.json")["record"]
    assert row["validation"]["byte_fallback_percent"] == 100.0
    assert row["actual_vocab_size"] == 260


@pytest.mark.parametrize("mutation", ["special", "vocab"])
def test_actual_model_accounting_rejected_even_with_matching_artifact_hash(real_saved, mutation):
    args = real_saved
    record_path = args.source / "condition-002.json"
    envelope = research.read_json(record_path)
    artifact = args.source / envelope["record"]["artifact"]
    path = artifact / "bpe.json"
    model = research.read_json(path)
    if mutation == "special":
        model["vocab"]["<|unk|>"], model["vocab"]["<|pad|>"] = 1, 0
    else:
        model["vocab"]["extra"] = 260
    path.write_text(json.dumps(model))
    envelope["record"]["artifact_hashes"] = research.artifact_hashes(artifact)
    record_path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="special-token|vocabulary"):
        migration.migrate(args)


def test_failed_revalidation_never_publishes_resume_plan(saved, monkeypatch):
    args, _ = saved
    monkeypatch.setattr(stages, "validation_metric", lambda *a: (_ for _ in ()).throw(ValueError("evaluation failed")))
    with pytest.raises(ValueError, match="evaluation failed"):
        migration.migrate(args)
    assert not (args.output / "plan.json").exists()
    assert not (args.output / "ledger.json").exists()


def test_source_evidence_tampering_rejected_at_resume(saved):
    args, _ = saved
    migration.migrate(args)
    (args.output / "originals" / "condition-000.json").write_text("{}")
    with pytest.raises(ValueError, match="original condition hash"):
        stages.run_stage(SimpleNamespace(stage="A-SCREEN", dataset=args.dataset, output=args.output, resume=True))


def test_reviewed_blob_pins_match_working_files():
    for path, blob in migration.APPROVED_BLOBS.items():
        assert migration.git("hash-object", path).decode().strip() == blob


def test_original_linux_source_hash_is_reproducible():
    assert migration.source_hash(migration.TRAINED_COMMIT) == "a589dadf64355e6b359034b8a182a6a4fc64b11edbf9745da70760f87d357b11"
