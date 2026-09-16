"""Strict nine-condition Phase C confirmation runner.

The runner consumes only the committed 300 MB exposure and confirmation half of
FLORES dev.  It never resolves or opens the test split and publishes a final
ledger only after all nine provenance-validated checkpoints exist.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import time

from benchmarks import run_phase_a as stages
from benchmarks import run_phase_b_screen as phase_b
from benchmarks import run_research_experiments as h
from tools import phase_c_exposure as exposure_gate


VERSION = 1
STAGE = "C-CONFIRM"
LABEL = "CONFIRMATORY"
VOCAB = 16_384
NAMES = ("sp_unigram", "boundary_bpe", "uniq_superbpe")
SEEDS = (1, 2, 3)
CONDITIONS = tuple((name, VOCAB, "bytes", seed) for seed in SEEDS for name in NAMES)
EXPOSURE_BYTES = 300_000_000


def require_hash(path: Path, expected: str, label: str) -> None:
    h.require(re.fullmatch(r"[0-9a-f]{64}", expected or "") is not None, f"explicit {label} SHA-256 required")
    h.require(h.file_hash(path) == expected, f"{label} changed or stale")


def load_exposure(source_path: Path, exposure_path: Path, validation_path: Path):
    require_hash(source_path, exposure_gate.SOURCE_SHA256, "source manifest")
    source = h.read_json(source_path)
    exposure = h.read_json(exposure_path)
    validation_receipt = h.read_json(validation_path)
    receipt_path = exposure_path.parent.parent / "receipt.json"
    receipt = h.read_json(receipt_path)
    h.require(receipt["status"] == "phase_c_exposure_frozen" and receipt["test_split_opened"] is False,
              "Phase C exposure receipt is not usable")
    h.require(receipt["exposure_sha256"] == h.file_hash(exposure_path), "exposure/receipt mismatch")
    h.require(receipt["confirmation_validation_sha256"] == h.file_hash(validation_path),
              "confirmation-validation/receipt mismatch")
    h.require(exposure["status"] == "phase_c_exposure_frozen" and exposure["test_split_opened"] is False,
              "invalid Phase C exposure")
    h.require(exposure["normalized_utf8_bytes"] == EXPOSURE_BYTES and len(exposure["strata"]) == 30,
              "Phase C exposure accounting mismatch")
    body = dict(exposure)
    content = body.pop("content_sha256")
    h.require(h.digest(body) == content, "Phase C exposure content hash mismatch")
    verified = h.read_json(exposure_path.with_name("verification.json"))
    h.require(verified["status"] == "PASS" and verified["test_split_opened"] is False,
              "independent exposure verification missing")
    root = source_path.resolve().parent
    train_path = (root / source["splits"]["train"]["path"]).resolve()
    h.require(train_path.is_relative_to(root), "training path escapes frozen source")
    h.require(h.file_hash(train_path) == source["splits"]["train"]["sha256"], "source train hash mismatch")
    texts, byte_rows = [], []
    selected_hashes = set()
    with train_path.open(encoding="utf-8") as stream:
        wanted = {row["train_row_index"]: row for row in exposure["ordered_documents"]}
        found = {}
        for index, line in enumerate(stream):
            expected = wanted.get(index)
            if expected is None:
                continue
            row = json.loads(line)
            text = h.normalize(row["text"])
            actual = exposure_gate.document(row, index)
            for key, value in expected.items():
                if key != "position":
                    h.require(actual[key] == value, "frozen exposure record changed")
            found[index] = text
        h.require(set(found) == set(wanted), "frozen exposure record absent")
        for expected in exposure["ordered_documents"]:
            text = found[expected["train_row_index"]]
            texts.append(text)
            selected_hashes.add(h.digest(text))
            byte_rows.append({h.BYTE_BUDGET_FIELD: expected["normalized_utf8_bytes"],
                              h.BYTE_AUDIT_FIELD: expected["source_utf8_bytes"]})
    h.require(sum(row[h.BYTE_BUDGET_FIELD] for row in byte_rows) == EXPOSURE_BYTES,
              "reloaded exposure byte mismatch")
    validation_rows, validation_texts = stages._load_rows(source_path, source, "validation")
    _, confirmation = stages.partition_validation(validation_rows, validation_texts)
    heldout = [text for _, text in confirmation]
    hashes = [h.digest(text) for text in heldout]
    h.require(not selected_hashes.intersection(hashes), "training/validation overlap")
    expected_validation = {
        "partition_version": stages.VALIDATION_PARTITION_VERSION,
        "partition": "confirmation", "documents": len(heldout),
        "assignment_hash": h.digest(hashes), "ordered_normalized_text_hashes": hashes,
        "normalized_utf8_bytes": sum(len(text.encode("utf-8")) for text in heldout),
        "source_utf8_bytes": sum(row["raw_utf8_bytes"] for row, _ in confirmation),
        "screening_assignment_hash": h.digest([h.digest(text) for _, text in stages.partition_validation(validation_rows, validation_texts)[0]]),
        "test_split_opened": False,
    }
    expected_validation["content_sha256"] = h.digest(expected_validation)
    h.require(validation_receipt == expected_validation, "confirmation validation assignment changed")
    validation_bytes = [{h.BYTE_BUDGET_FIELD: len(text.encode("utf-8")),
                         h.BYTE_AUDIT_FIELD: row["raw_utf8_bytes"]} for row, text in confirmation]
    training = {
        "scope": "frozen_exact_phase_c_exposure", "documents": len(texts),
        "normalized_utf8_bytes": EXPOSURE_BYTES,
        "source_utf8_bytes": sum(row[h.BYTE_AUDIT_FIELD] for row in byte_rows),
        "assignment_hash": exposure["ordered_exposure_hash"],
    }
    return {"train": texts, "validation": heldout}, {"train": byte_rows, "validation": validation_bytes}, training, validation_receipt, receipt_path


def prepare(args, *, execution=False):
    phase_a_ledger, _, _ = phase_b.verify_phase_a(args.phase_a, args.dataset)
    identity = h.runtime_identity()
    selection = phase_b.check_selection(args.selection, args.selection_sha256,
                                        phase_a_ledger, args.phase_a, identity)
    require_hash(args.protocol, exposure_gate.PHASE_C_PROTOCOL_SHA256, "Phase C protocol")
    require_hash(args.phase_b_report, exposure_gate.PHASE_B_REPORT_SHA256, "Phase B report")
    require_hash(args.phase_b_ledger, exposure_gate.PHASE_B_LEDGER_SHA256, "Phase B ledger")
    require_hash(args.exposure_manifest, args.exposure_manifest_sha256, "Phase C exposure")
    require_hash(args.validation_receipt, args.validation_receipt_sha256, "confirmation validation")
    docs, byte_rows, training, validation, exposure_receipt = load_exposure(
        args.source_manifest, args.exposure_manifest, args.validation_receipt)
    h.require(args.exposure_manifest_sha256 == h.file_hash(args.exposure_manifest), "exposure hash mismatch")
    h.require(args.device == "cpu" or re.fullmatch(r"cuda(?::\d+)?", args.device), "invalid device")
    selected = [row for row in selection["conditions"] if row["vocab_budget"] == VOCAB and row["tokenizer"] in NAMES]
    h.require([(row["tokenizer"], row["vocab_budget"]) for row in selected] == [(name, VOCAB) for name in NAMES],
              "frozen Phase C tokenizer cohort mismatch")
    if execution:
        phase_b.check_runtime(identity, selection["phase_a_identity"], args.device)
    pins = {
        "phase_c_protocol_sha256": exposure_gate.PHASE_C_PROTOCOL_SHA256,
        "phase_b_report_sha256": exposure_gate.PHASE_B_REPORT_SHA256,
        "phase_b_ledger_sha256": exposure_gate.PHASE_B_LEDGER_SHA256,
        "source_manifest_sha256": exposure_gate.SOURCE_SHA256,
        "exposure_manifest_sha256": args.exposure_manifest_sha256,
        "exposure_receipt_sha256": h.file_hash(exposure_receipt),
        "validation_receipt_sha256": args.validation_receipt_sha256,
    }
    plan = {
        **identity, "identity": identity, "phase_c_schema_version": VERSION,
        "stage": STAGE, "result_label": LABEL, "status": "planned",
        "selection_sha256": args.selection_sha256, "selection": selection,
        "conditions": [list(item) for item in CONDITIONS], "seeds": list(SEEDS),
        "data_split": "document_disjoint_train_confirmation_validation_test_not_opened",
        "test_access": "forbidden_not_opened", "training": training, "validation": validation,
        "model_config": h.model_config("B", args.device), "budget_regime": "bytes",
        "byte_budget": EXPOSURE_BYTES, "provenance": pins,
        "cuda_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if args.device.startswith("cuda") else None,
    }
    return plan, phase_a_ledger, docs, byte_rows


def source_record(ledger, condition):
    return next(row for row in ledger["records"]
                if (row["tokenizer"], row["vocab_budget"]) == tuple(condition[:2]))


def validate_result(row, condition, plan, source, byte_rows):
    h.require(h.condition_key(row) == tuple(condition), "condition mismatch")
    cfg = h.LMArchConfig(**plan["model_config"]["architecture"])
    required = {
        "model_kind": "causal_transformer", "result_label": LABEL, "actual_vocab_size": VOCAB,
        "git_commit": plan["identity"]["commit_hash"], "extension_hash": plan["identity"]["extension_hash"],
        "selection_sha256": plan["selection_sha256"], "model_config": plan["model_config"],
        "special_tokens": h.SPECIAL_IDS, "requested_budget": EXPOSURE_BYTES,
        "tokenizer_artifact_hash": h.digest(source["artifact_hashes"]),
        "dataset_manifest_hash": plan["provenance"]["source_manifest_sha256"],
        "training_assignment_hash": plan["training"]["assignment_hash"],
        "validation_assignment_hash": plan["validation"]["assignment_hash"],
        **h.parameter_accounting(VOCAB, cfg, plan["model_config"]["context"]),
    }
    h.require(all(row.get(key) == value for key, value in required.items()), "result provenance/configuration mismatch")
    h.require("test" not in row and "test_metrics" not in row, "Phase C cannot score test")
    h.require(row["completed_training_documents"] == plan["training"]["documents"], "incomplete one-pass exposure")
    h.require(row["training_bytes"] == h.byte_totals(byte_rows["train"]), "training byte accounting mismatch")
    h.require(row["completed_document_bytes"] == EXPOSURE_BYTES, "training exposure shortened")
    h.require(all(row.get(key) == value for key, value in h.flop_accounting(
        VOCAB, cfg, row["training_target_tokens"], row["training_sequence_length_squared_sum"]).items()),
        "analytical FLOP accounting mismatch")
    metric = row["validation"]
    h.require(metric == h.nll_metrics(metric["total_nll_nats"], metric["target_tokens_including_eos"],
                                      plan["validation"]["normalized_utf8_bytes"],
                                      source_utf8_bytes=plan["validation"]["source_utf8_bytes"]),
              "validation NLL/BPB mismatch")
    h.digest(row)


def check_lock(args, output, plan):
    h.require(h.read_json(output / "plan.json") == plan, "execution plan changed")
    for path, expected in (
        (args.selection, plan["selection_sha256"]), (args.protocol, plan["provenance"]["phase_c_protocol_sha256"]),
        (args.phase_b_report, plan["provenance"]["phase_b_report_sha256"]),
        (args.phase_b_ledger, plan["provenance"]["phase_b_ledger_sha256"]),
        (args.source_manifest, plan["provenance"]["source_manifest_sha256"]),
        (args.exposure_manifest, plan["provenance"]["exposure_manifest_sha256"]),
        (args.validation_receipt, plan["provenance"]["validation_receipt_sha256"]),
    ):
        h.require(h.file_hash(path) == expected, f"pinned input changed: {path}")
    h.require(h.runtime_identity() == plan["identity"], "runtime changed")


def resume_records(args, output, plan, ledger, byte_rows):
    h.require(not (output / "ledger.json").exists(), "complete Phase C ledger cannot resume")
    records = {}
    for path in sorted(output.glob("condition-*.json")):
        h.require(re.fullmatch(r"condition-\d{3}\.json", path.name), "invalid checkpoint filename")
        index = int(path.stem.removeprefix("condition-"))
        h.require(index < len(CONDITIONS) and index not in records, "unexpected/duplicate checkpoint")
        envelope = h.read_json(path)
        h.require(envelope.get("status") == "condition_complete", "incomplete checkpoint")
        source = source_record(ledger, CONDITIONS[index])
        h.require(h.artifact_hashes(Path(args.phase_a).parent / source["artifact"]) == source["artifact_hashes"],
                  "tokenizer artifact changed")
        validate_result(envelope["record"], CONDITIONS[index], plan, source, byte_rows)
        records[index] = envelope["record"]
    return records


def run(args):
    plan, ledger, docs, byte_rows = prepare(args, execution=True)
    output = args.output
    if output.exists():
        h.require(args.resume and output.is_dir(), "output exists; use --resume only for exact provenance")
        records = resume_records(args, output, plan, ledger, byte_rows)
    else:
        h.require(not args.resume, "resume requires existing output")
        output.mkdir(parents=True)
        h.write_new_json_atomic(output / "plan.json", plan)
        records = {}
    for index, condition in enumerate(CONDITIONS):
        check_lock(args, output, plan)
        if index in records:
            continue
        name, vocab, regime, seed = condition
        source = source_record(ledger, condition)
        tok = h.load_tokenizer(source, Path(args.phase_a).parent)
        started = time.perf_counter()
        measured = h.train_lm(tok, docs, h.SCREEN, 128, regime, EXPOSURE_BYTES, seed,
                              args.device, False, document_bytes=byte_rows)
        row = {
            **measured, "tokenizer": name, "vocab_budget": vocab, "actual_vocab_size": len(tok.vocab),
            "budget_regime": regime, "seed": seed, "model_kind": "causal_transformer",
            "result_label": LABEL, "model_config": plan["model_config"], "requested_budget": EXPOSURE_BYTES,
            "git_commit": plan["identity"]["commit_hash"], "extension_hash": plan["identity"]["extension_hash"],
            "selection_sha256": plan["selection_sha256"], "special_tokens": h.SPECIAL_IDS,
            "tokenizer_artifact_hash": h.digest(source["artifact_hashes"]),
            "dataset_manifest_hash": plan["provenance"]["source_manifest_sha256"],
            "training_assignment_hash": plan["training"]["assignment_hash"],
            "validation_assignment_hash": plan["validation"]["assignment_hash"],
            "wall_clock_seconds": time.perf_counter() - started,
        }
        validate_result(row, condition, plan, source, byte_rows)
        check_lock(args, output, plan)
        h.write_new_json_atomic(output / f"condition-{index:03d}.json",
                                {"status": "condition_complete", "record": row})
        records[index] = row
    h.require(set(records) == set(range(9)), "incomplete Phase C checkpoints")
    result = {"metadata": {**plan, "status": "complete"},
              "records": [records[index] for index in range(9)]}
    h.validate_ledger(result, expected_commit=plan["identity"]["commit_hash"])
    for row, condition in zip(result["records"], CONDITIONS):
        validate_result(row, condition, plan, source_record(ledger, condition), byte_rows)
    check_lock(args, output, plan)
    h.write_new_json_atomic(output / "ledger.json", result)
    return {"ledger": str(output / "ledger.json"), "conditions": 9, "test_opened": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--phase-a", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--exposure-manifest", type=Path, required=True)
    parser.add_argument("--exposure-manifest-sha256", required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--validation-receipt-sha256", required=True)
    parser.add_argument("--phase-b-ledger", type=Path, required=True)
    parser.add_argument("--phase-b-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        h.require(args.output is not None, "run requires --output")
        result = run(args)
    else:
        plan, _, _, _ = prepare(args)
        result = {"configuration_valid": True, "experiments_started": False, "plan": plan}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
