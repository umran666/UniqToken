"""Two-stage tokenizer experiment gate: screening, deterministic selection, confirmation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from benchmarks import run_research_experiments as research

STAGE_SCHEMA_VERSION = 1
SCREEN_TRAINING_TARGET_BYTES = 75_000_000
SCREEN_TRAINING_MIN_BYTES = 50_000_000
SCREEN_TRAINING_MAX_BYTES = 100_000_000
VALIDATION_PARTITION_VERSION = "normalized_sha256_parity_v1"
SELECTION_POLICY_VERSION = "lowest_validation_tokens_per_unicode_character_v1"


def _load_rows(path, manifest, split):
    entry = manifest["splits"][split]
    data_path = path.parent / entry["path"]
    research.require(research.file_hash(data_path) == entry["sha256"], f"{split} file hash mismatch")
    rows, texts, seen = [], [], set()
    for row in research.jsonl_rows(data_path):
        text = research.normalize(row["text"])
        research.require(text and not research.RESERVED_CORPUS_TEXT.search(text), "invalid frozen text")
        research.require(
            row["normalized_utf8_bytes"] == len(text.encode("utf-8"))
            and row["raw_utf8_bytes"] == len(row["text"].encode("utf-8")),
            "frozen byte accounting mismatch",
        )
        identity = research.digest(text)
        research.require(row["id"] not in seen and identity not in seen, "duplicate frozen document")
        seen.update((row["id"], identity))
        rows.append(row)
        texts.append(text)
    research.require(texts, f"empty {split} split")
    return rows, texts


def load_stage_source(manifest_path):
    """Load train/validation only; the declared test artifact is never opened."""
    path = Path(manifest_path)
    manifest = research.read_json(path)
    research.require(manifest.get("schema_version") == research.DATASET_MANIFEST_SCHEMA, "unsupported manifest")
    research.require(manifest.get("normalization") == research.NORMALIZATION, "normalization mismatch")
    research.require(manifest.get("freeze", {}).get("immutable") is True, "dataset is not frozen")
    research.require(set(manifest.get("splits", {})) == {"train", "validation", "test"}, "three frozen splits required")
    train_rows, train = _load_rows(path, manifest, "train")
    validation_rows, validation = _load_rows(path, manifest, "validation")
    train_hashes = {research.digest(text) for text in train}
    research.require(not train_hashes.intersection(map(research.digest, validation)), "train/validation leakage")
    source = {
        "manifest_sha256": research.file_hash(path),
        "dataset_id": manifest["dataset_id"],
        "normalization": manifest["normalization"],
        "source_revisions": manifest["freeze"]["source_revisions"],
        "train_file_sha256": manifest["splits"]["train"]["sha256"],
        "validation_file_sha256": manifest["splits"]["validation"]["sha256"],
        "untouched_test_file_sha256": manifest["splits"]["test"]["sha256"],
        "test_access": "forbidden_not_opened",
    }
    return train_rows, train, validation_rows, validation, source


def stratified_screen_training(rows, texts, target_bytes=SCREEN_TRAINING_TARGET_BYTES):
    research.require(len(rows) == len(texts), "training row/text mismatch")
    groups = defaultdict(list)
    totals = defaultdict(int)
    for index, (row, text) in enumerate(zip(rows, texts)):
        key = (row["domain"], row["language"])
        size = len(text.encode("utf-8"))
        groups[key].append((index, text, size))
        totals[key] += size
    research.require(groups and target_bytes <= sum(totals.values()), "invalid screening corpus target")
    raw_quotas = {key: target_bytes * total / sum(totals.values()) for key, total in totals.items()}
    quotas = {key: int(value) for key, value in raw_quotas.items()}
    remainder = target_bytes - sum(quotas.values())
    for key in sorted(groups, key=lambda item: (-(raw_quotas[item] - quotas[item]), item))[:remainder]:
        quotas[key] += 1
    selected, accounting = [], []
    for key in sorted(groups):
        chosen, used = [], 0
        for index, text, size in groups[key]:
            if used + size > quotas[key]:
                continue
            chosen.append((index, text))
            used += size
        research.require(chosen, f"screening quota selected no documents for {key}")
        selected.extend(chosen)
        accounting.append(
            {"domain": key[0], "language": key[1], "target_bytes": quotas[key], "actual_bytes": used, "documents": len(chosen)}
        )
    selected.sort()
    selected_texts = [text for _, text in selected]
    actual = sum(len(text.encode("utf-8")) for text in selected_texts)
    source_bytes = sum(rows[index]["raw_utf8_bytes"] for index, _ in selected)
    return selected_texts, {
        "selection_version": "proportional_language_domain_whole_document_v1",
        "target_normalized_utf8_bytes": target_bytes,
        "actual_normalized_utf8_bytes": actual,
        "source_utf8_bytes": source_bytes,
        "documents": len(selected_texts),
        "assignment_hash": research.digest([research.digest(text) for text in selected_texts]),
        "groups": accounting,
    }


def partition_validation(rows, texts):
    screen, confirm = [], []
    for row, text in zip(rows, texts):
        (screen if int(research.digest(text), 16) % 2 == 0 else confirm).append((row, text))
    research.require(screen and confirm, "validation partition is empty")
    research.require(
        not {research.digest(text) for _, text in screen}.intersection(research.digest(text) for _, text in confirm),
        "validation overlap",
    )
    return screen, confirm


def validation_metric(tokenizer, texts, source_bytes):
    return research.token_metrics(tokenizer, texts, source_utf8_bytes=source_bytes)


def selection_from_screening(ledger):
    metadata = ledger.get("metadata", {})
    research.require(
        metadata.get("stage") == "A-SCREEN"
        and metadata.get("result_label") == "SCREENING"
        and metadata.get("status") == "complete",
        "selection requires a complete screening ledger",
    )
    rows = ledger.get("records", [])
    selected = []
    for vocab in research.VOCABS:
        candidates = [row for row in rows if row["vocab_budget"] == vocab]
        research.require({row["tokenizer"] for row in candidates} == set(research.COHORT), "incomplete screening cohort")
        winner = min(
            candidates,
            key=lambda row: (
                row["validation"]["tokens_per_unicode_character"],
                row["validation"]["byte_fallback_percent"],
                research.COHORT.index(row["tokenizer"]),
            ),
        )
        selected.append([winner["tokenizer"], vocab])
    return {
        "selection_schema_version": 1,
        "policy": SELECTION_POLICY_VERSION,
        "screening_ledger_sha256": None,
        "conditions": selected,
        "validation_fields": [
            "tokens_per_unicode_character",
            "byte_fallback_percent",
        ],
        "test_metrics_used": False,
    }


def validate_stage_ledger(ledger, identity, source, output, stage, conditions):
    metadata = ledger.get("metadata", {})
    research.require(
        metadata.get("stage_schema_version") == STAGE_SCHEMA_VERSION
        and metadata.get("stage") == stage
        and metadata.get("status") == "complete"
        and metadata.get("identity") == identity
        and metadata.get("dataset") == source
        and metadata.get("test_access") == "forbidden_not_opened",
        "stale or invalid Phase A stage ledger",
    )
    research.require(metadata.get("conditions") == [list(condition) for condition in conditions], "stage condition grid mismatch")
    rows = ledger.get("records", [])
    research.require(len(rows) == len(conditions), "incomplete Phase A stage ledger")
    for row, expected in zip(rows, conditions):
        research.require(row.get("result_label") == metadata["result_label"], "stage result label mismatch")
        _validate_condition(row, expected, identity, source, metadata["training"], metadata["validation"], output)
    return ledger


def write_selection(screening_path, output, dataset_path):
    ledger = research.read_json(screening_path)
    _, _, _, _, source = load_stage_source(dataset_path)
    identity = research.runtime_identity()
    research.require(not identity["working_tree_dirty"], "commit the stage harness before selection")
    research.require(not identity["working_tree_dirty"], "commit the stage harness before selecting conditions")
    conditions = [(name, vocab) for name in research.COHORT for vocab in research.VOCABS]
    validate_stage_ledger(ledger, identity, source, Path(screening_path).parent, "A-SCREEN", conditions)
    selection = selection_from_screening(ledger)
    selection["screening_ledger_sha256"] = research.file_hash(screening_path)
    research.write_new_json_atomic(output, selection)
    return selection


def _validate_condition(row, expected, identity, source, training, validation, output):
    name, vocab = expected
    research.require((row.get("tokenizer"), row.get("vocab_budget")) == expected, "condition mismatch")
    research.require(row.get("actual_vocab_size") == vocab, "vocabulary target mismatch")
    research.require(row.get("git_commit") == identity["commit_hash"], "condition commit mismatch")
    research.require(row.get("extension_hash") == identity["extension_hash"], "condition extension mismatch")
    research.require(row.get("dataset_manifest_hash") == source["manifest_sha256"], "condition dataset mismatch")
    research.require(row.get("tokenizer_config") == research.tokenizer_configuration(name, vocab), "tokenizer configuration mismatch")
    research.require(row.get("training_assignment_hash") == training["assignment_hash"], "training assignment mismatch")
    research.require(row.get("validation_assignment_hash") == validation["assignment_hash"], "validation assignment mismatch")
    expected_metric = row["validation"]
    research.require(
        expected_metric.get("tokens", 0) > 0
        and expected_metric.get("unicode_characters") == validation["unicode_characters"]
        and expected_metric.get("normalized_utf8_bytes") == validation["normalized_utf8_bytes"]
        and expected_metric.get("source_utf8_bytes") == validation["source_utf8_bytes"]
        and expected_metric.get("utf8_bytes") == validation["normalized_utf8_bytes"]
        and expected_metric.get("tokens_per_unicode_character")
        == expected_metric["tokens"] / validation["unicode_characters"]
        and expected_metric.get("bytes_per_token")
        == validation["normalized_utf8_bytes"] / expected_metric["tokens"],
        "invalid validation metrics",
    )
    research.require("test" not in row, "test metrics are forbidden in Phase A stages")
    artifact = output / row["artifact"]
    research.require(research.artifact_hashes(artifact) == row["artifact_hashes"], "tokenizer artifact mismatch")
    research.load_tokenizer(row, output)
    json.dumps(row, allow_nan=False)
    return row


def _stage_plan(stage, identity, source, training, validation, conditions, selection=None):
    return {
        "stage_schema_version": STAGE_SCHEMA_VERSION,
        "stage": stage,
        "result_label": "SCREENING" if stage == "A-SCREEN" else "CONFIRMATION",
        "status": "planned",
        "identity": identity,
        "dataset": source,
        "training": training,
        "validation": validation,
        "test_access": "forbidden_not_opened",
        "conditions": [list(condition) for condition in conditions],
        "selection": selection,
    }


def run_stage(args):
    stage = args.stage
    research.require(stage in ("A-SCREEN", "A-CONFIRM"), "invalid Phase A stage")
    train_rows, full_train, validation_rows, full_validation, source = load_stage_source(args.dataset)
    screen_train, screen_info = stratified_screen_training(train_rows, full_train, SCREEN_TRAINING_TARGET_BYTES)
    screen_pairs, confirm_pairs = partition_validation(validation_rows, full_validation)
    screen_validation = [text for _, text in screen_pairs]
    confirm_validation = [text for _, text in confirm_pairs]
    identity = research.runtime_identity()
    research.require(not identity["working_tree_dirty"], "commit the stage harness before experiments")
    if stage == "A-SCREEN":
        research.require(
            SCREEN_TRAINING_MIN_BYTES <= screen_info["actual_normalized_utf8_bytes"] <= SCREEN_TRAINING_MAX_BYTES,
            "screening corpus is outside the predeclared 50-100 MB range",
        )
        training_texts, validation_texts = screen_train, screen_validation
        conditions = [(name, vocab) for name in research.COHORT for vocab in research.VOCABS]
        selection = None
    else:
        research.require(args.screening and args.selection, "confirmation requires screening ledger and selection")
        screening = research.read_json(args.screening)
        screening_conditions = [(name, vocab) for name in research.COHORT for vocab in research.VOCABS]
        validate_stage_ledger(
            screening, identity, source, Path(args.screening).parent, "A-SCREEN", screening_conditions
        )
        expected_selection = selection_from_screening(screening)
        expected_selection["screening_ledger_sha256"] = research.file_hash(args.screening)
        selection = research.read_json(args.selection)
        research.require(selection == expected_selection, "selection is stale or not deterministically derived")
        research.require(screening["metadata"]["dataset"] == source, "screening used a different dataset")
        training_texts, validation_texts = full_train, confirm_validation
        conditions = [tuple(condition) for condition in selection["conditions"]]
    training = {
        "scope": "representative_stratified_screen" if stage == "A-SCREEN" else "full_frozen_training_corpus",
        "normalized_utf8_bytes": sum(len(text.encode("utf-8")) for text in training_texts),
        "source_utf8_bytes": (
            screen_info["source_utf8_bytes"]
            if stage == "A-SCREEN"
            else sum(row["raw_utf8_bytes"] for row in train_rows)
        ),
        "documents": len(training_texts),
        "assignment_hash": research.digest([research.digest(text) for text in training_texts]),
        "screen_selection": screen_info if stage == "A-SCREEN" else None,
    }
    validation = {
        "partition_version": VALIDATION_PARTITION_VERSION,
        "partition": "screening" if stage == "A-SCREEN" else "confirmation",
        "normalized_utf8_bytes": sum(len(text.encode("utf-8")) for text in validation_texts),
        "source_utf8_bytes": sum(
            row["raw_utf8_bytes"] for row, _ in (screen_pairs if stage == "A-SCREEN" else confirm_pairs)
        ),
        "unicode_characters": sum(map(len, validation_texts)),
        "documents": len(validation_texts),
        "assignment_hash": research.digest([research.digest(text) for text in validation_texts]),
    }
    plan = _stage_plan(stage, identity, source, training, validation, conditions, selection)
    output = Path(args.output)
    if output.exists():
        research.require(args.resume and output.is_dir(), "output exists; use --resume for an incomplete stage")
        research.require(not (output / "ledger.json").exists(), "stage already has a complete ledger")
        research.require(research.digest(research.read_json(output / "plan.json")) == research.digest(plan), "resume plan mismatch")
    else:
        research.require(not args.resume, "resume output does not exist")
        output.mkdir(parents=True)
        research.write_new_json_atomic(output / "plan.json", plan)
    existing = {}
    for path in sorted(output.glob("condition-*.json")):
        research.require(path.name == f"condition-{int(path.stem.split('-')[1]):03d}.json", "invalid condition filename")
        index = int(path.stem.split("-")[1])
        research.require(index < len(conditions) and index not in existing, "unexpected condition index")
        envelope = research.read_json(path)
        research.require(envelope.get("status") == "condition_complete", "incomplete condition")
        existing[index] = _validate_condition(envelope["record"], conditions[index], identity, source, training, validation, output)
    records = []
    for index, (name, vocab) in enumerate(conditions):
        if index in existing:
            records.append(existing[index])
            continue
        artifact_name = f"{name}-{vocab}"
        tokenizer, elapsed = research.train_tokenizer(name, training_texts, vocab, output / artifact_name)
        row = {
            "tokenizer": name,
            "vocab_budget": vocab,
            "actual_vocab_size": len(tokenizer.vocab),
            "git_commit": identity["commit_hash"],
            "extension_hash": identity["extension_hash"],
            "dataset_manifest_hash": source["manifest_sha256"],
            "tokenizer_config": research.tokenizer_configuration(name, vocab),
            "training_assignment_hash": training["assignment_hash"],
            "validation_assignment_hash": validation["assignment_hash"],
            "training_wall_clock_seconds": elapsed,
            "training_normalized_mb_per_sec": training["normalized_utf8_bytes"] / (elapsed * 1_000_000),
            "learned_merges": tokenizer.merges,
            "artifact": artifact_name,
            "artifact_hashes": research.artifact_hashes(output / artifact_name),
            "result_label": plan["result_label"],
            "validation": validation_metric(tokenizer, validation_texts, validation["source_utf8_bytes"]),
        }
        _validate_condition(row, conditions[index], identity, source, training, validation, output)
        research.write_new_json_atomic(output / f"condition-{index:03d}.json", {"status": "condition_complete", "record": row})
        records.append(row)
    research.require(len(records) == len(conditions), "incomplete Phase A stage")
    ledger = {"metadata": {**plan, "status": "complete"}, "records": records}
    validate_stage_ledger(ledger, identity, source, output, stage, conditions)
    research.write_new_json_atomic(output / "ledger.json", ledger)
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, stage in (("screen", "A-SCREEN"), ("confirm", "A-CONFIRM")):
        child = subparsers.add_parser(command)
        child.set_defaults(stage=stage)
        child.add_argument("--dataset", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--resume", action="store_true")
        if command == "confirm":
            child.add_argument("--screening", type=Path, required=True)
            child.add_argument("--selection", type=Path, required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--dataset", type=Path, required=True)
    select.add_argument("--screening", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        write_selection(args.screening, args.output, args.dataset)
    else:
        run_stage(args)


if __name__ == "__main__":
    main()
