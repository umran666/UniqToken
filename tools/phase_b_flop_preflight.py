"""Read-only Phase B FLOP/byte coverage gate for a frozen exact exposure.

This command never constructs an LM, optimizer, loss, validation set, or test
set.  It loads the nine frozen tokenizer artifacts solely to encode the already
selected training documents and simulates the existing deterministic scheduler.
It does not create an executable Phase B plan.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import time

from benchmarks import run_phase_b_screen as screen
from benchmarks import run_research_experiments as h
from tools.phase_b_exposure import quotas


VERSION = 2
STATUS = "tokenizer_flop_preflight_feasible"
REJECTED = "tokenizer_flop_preflight_rejected"


def publish(path, value):
    """Atomically create evidence, never replacing an earlier attempt."""
    h.write_new_json_atomic(path, value)


def require_sha(path, expected, label):
    h.require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), f"explicit {label} SHA-256 required")
    h.require(h.file_hash(path) == expected, f"{label} changed or stale")


def content_hash(payload):
    h.require("content_sha256" in payload, "missing content SHA-256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    h.require(payload["content_sha256"] == h.digest(body), "content SHA-256 mismatch")


def check_source(selection, source, source_path):
    """Check split identities without opening validation or test files."""
    h.require(source["schema_version"] == h.DATASET_MANIFEST_SCHEMA, "unsupported source manifest schema")
    h.require(source["freeze"]["immutable"] is True, "source manifest is not immutable")
    dataset = selection["dataset"]
    h.require(source["dataset_id"] == dataset["dataset_id"], "dataset identity mismatch")
    h.require(source["normalization"] == dataset["normalization"] == h.NORMALIZATION, "normalization mismatch")
    h.require(source["freeze"]["source_revisions"] == dataset["source_revisions"], "source revision mismatch")
    split_fields = {
        "train": "train_file_sha256",
        "validation": "validation_file_sha256",
        "test": "untouched_test_file_sha256",
    }
    for split, field in split_fields.items():
        h.require(source["splits"][split]["sha256"] == dataset[field], f"{split} split provenance mismatch")
    root = Path(source_path).resolve().parent
    train_path = (root / source["splits"]["train"]["path"]).resolve()
    h.require(train_path.is_relative_to(root), "training path escapes frozen source directory")
    h.require(h.file_hash(train_path) == source["splits"]["train"]["sha256"], "training file hash mismatch")
    # Deliberately do not resolve, hash, or open validation/test paths.
    return train_path


def check_exposure(exposure, receipt, selection_sha, source_sha):
    content_hash(exposure)
    h.require(exposure["exposure_schema_version"] == 2 and exposure["status"] == "exposure_frozen",
              "exposure is not frozen")
    h.require(exposure["gate"] == "exact_packing_only", "unexpected exposure gate")
    h.require(exposure["provenance"]["selection_sha256"] == selection_sha, "exposure selection mismatch")
    h.require(exposure["provenance"]["source_manifest_sha256"] == source_sha, "exposure source mismatch")
    h.require(receipt["status"] == "exact_packing_feasible", "exact-packing receipt is not feasible")
    # Receipt hashes the serialized manifest, while content_sha256 hashes its JSON body.
    for key in ("tokenizer_outputs_used_for_sampling", "token_counts_used_for_sampling", "flops_used_for_sampling",
                "validation_or_test_text_used_for_sampling", "selection_metrics_used", "lm_results_used",
                "tokenizer_flop_preflight_run", "lm_training_run", "authorized_for_lm_execution"):
        h.require(exposure[key] is False, f"exposure unexpectedly used {key}")
    sample = exposure["sample"]
    h.require(sample["normalized_utf8_bytes"] == 1_000_000, "exact normalized-byte cap missing")
    h.require(sample["selected_documents"] == len(sample["ordered_documents"]), "selected-document count mismatch")
    h.require(sample["source_utf8_bytes"] == sum(d["source_utf8_bytes"] for d in sample["ordered_documents"]),
              "source-byte accounting mismatch")
    h.require(sample["normalized_utf8_bytes"] == sum(d["normalized_utf8_bytes"] for d in sample["ordered_documents"]),
              "normalized-byte accounting mismatch")
    h.require(len(sample["strata"]) == 30 and all(s["packing_status"] == s["exact_quota_status"] == "PASS"
                                                     for s in sample["strata"]), "incomplete exact strata")
    h.require(all(d["position"] == index for index, d in enumerate(sample["ordered_documents"])),
              "frozen exposure order mismatch")
    h.require(all(d["coverage_document"] for d in sample["ordered_documents"][:30]) and
              not any(d["coverage_document"] for d in sample["ordered_documents"][30:]),
              "mandatory coverage prefix mismatch")
    h.require(len({d["id"] for d in sample["ordered_documents"]}) == sample["selected_documents"],
              "duplicate frozen document ID")
    h.require(len({d["normalized_text_hash"] for d in sample["ordered_documents"]}) == sample["selected_documents"],
              "duplicate frozen document text")
    h.require(sample["ordered_exposure_hash"] == h.digest(sample["ordered_documents"]), "ordered exposure hash mismatch")
    h.require(exposure["verification"]["status"] == receipt["independent_verification"] == "PASS",
              "independent exposure verification missing")
    allocation = quotas()
    h.require({(s["domain"], s["language"]) for s in sample["strata"]} == set(allocation), "strata mismatch")
    for stratum in sample["strata"]:
        key = stratum["domain"], stratum["language"]
        entries = [d for d in sample["ordered_documents"] if (d["domain"], d["language"]) == key]
        h.require(sum(d["normalized_utf8_bytes"] for d in entries) ==
                  stratum["actual_normalized_bytes"] == stratum["allocated_normalized_bytes"] == allocation[key],
                  "per-stratum exact quota mismatch")
        h.require(len(entries) == stratum["selected_documents"] and
                  sum(d["coverage_document"] for d in entries) == 1, "stratum coverage/count mismatch")
    return sample


def load_frozen_documents(sample, train_path):
    """Reload selected training rows only, in the manifest's immutable order."""
    wanted = {entry["train_row_index"]: entry for entry in sample["ordered_documents"]}
    h.require(len(wanted) == len(sample["ordered_documents"]), "duplicate frozen train row index")
    found = {}
    with Path(train_path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            expected = wanted.get(index)
            if expected is None:
                continue
            row = json.loads(line)
            text = h.normalize(row["text"])
            h.require(text and h.digest(text) == expected["normalized_text_hash"], "frozen text hash mismatch")
            h.require(row["id"] == expected["id"], "frozen document ID mismatch")
            h.require(row["domain"] == expected["domain"] and row["language"] == expected["language"],
                      "frozen stratum mismatch")
            h.require(row["source"] == expected["source"] and row["dedup"] == expected["dedup"] and
                      row["dedup"]["status"] == "accepted_after_exact_and_near_eval_check",
                      "frozen-record source/dedup mismatch")
            h.require(len(text.encode("utf-8")) == expected["normalized_utf8_bytes"] == row["normalized_utf8_bytes"],
                      "frozen normalized byte mismatch")
            h.require(len(row["text"].encode("utf-8")) == expected["source_utf8_bytes"] == row["raw_utf8_bytes"],
                      "frozen source byte mismatch")
            found[index] = text
    h.require(set(found) == set(wanted), "frozen document is absent from train split")
    documents = [found[entry["train_row_index"]] for entry in sample["ordered_documents"]]
    h.require(sum(len(text.encode("utf-8")) for text in documents) == sample["normalized_utf8_bytes"],
              "reloaded normalized byte total mismatch")
    return documents


def selected_phase_a_records(selection, ledger, ledger_path):
    h.require(h.file_hash(ledger_path) == selection["phase_a_ledger_sha256"], "Phase A ledger changed or stale")
    meta = ledger["metadata"]
    h.require(meta["identity"] == selection["phase_a_identity"], "Phase A identity mismatch")
    h.require(meta["dataset"] == selection["dataset"], "Phase A dataset mismatch")
    h.require(meta["training"] == selection["training"], "Phase A training assignment mismatch")
    h.require(meta["validation"] == selection["validation"], "Phase A validation assignment mismatch")
    h.require(meta["stage"] == "A-SCREEN" and meta["status"] == "complete", "incomplete Phase A ledger")
    by_key = {(row["tokenizer"], row["vocab_budget"]): row for row in ledger["records"]}
    h.require(len(by_key) == len(ledger["records"]), "duplicate Phase A condition")
    selected = []
    for expected in selection["conditions"]:
        key = expected["tokenizer"], expected["vocab_budget"]
        row = by_key.get(key)
        h.require(row is not None, "selected Phase A artifact is absent")
        h.require(row["actual_vocab_size"] == expected["actual_vocab_size"] == expected["vocab_budget"],
                  "vocabulary budget mismatch")
        for field in ("artifact", "artifact_hashes", "tokenizer_config", "extension_hash",
                      "training_assignment_hash", "validation_assignment_hash"):
            h.require(row[field] == expected[field], f"selected tokenizer {field} mismatch")
        h.require(row["tokenizer_config"] == h.tokenizer_configuration(*key), "current tokenizer configuration mismatch")
        h.require(row["tokenizer_config"]["special_tokens"] == selection["special_tokens"],
                  "special-token accounting mismatch")
        selected.append(row)
    h.require([(row["tokenizer"], row["vocab_budget"]) for row in selected] == list(screen.CONDITIONS),
              "frozen tokenizer condition assignment mismatch")
    return selected


def schedule(encoded, vocab, regime, budget, context):
    """Pure simulation of run_research_experiments.train_lm's training schedule."""
    h.require(regime in h.REGIMES and encoded, "invalid empty preflight schedule")
    h.require(math.isfinite(budget) and budget > 0, "invalid budget")
    doc_limit = len(encoded) if regime == "bytes" else None
    steps = targets = squares = completed = 0
    flops = 0
    partial_final_window = False
    terminal_targets = 0
    while doc_limit is None or completed < doc_limit:
        index = completed % len(encoded)
        terminal_targets = 0
        for x, y in h.windows(encoded[index], context):
            original_length = len(x)
            cost = h.training_flops(vocab, h.SCREEN, original_length)
            if regime == "flops" and flops + cost > budget:
                length = len(x)
                while length and flops + h.training_flops(vocab, h.SCREEN, length) > budget:
                    length -= 1
                if not length:
                    h.require(steps > 0 and flops >= budget * 0.99,
                              "FLOP budget cannot be matched within 1%; increase budget")
                    break
                x, y = x[:length], y[:length]
                cost = h.training_flops(vocab, h.SCREEN, length)
                partial_final_window = length != original_length
            steps += 1
            targets += len(y)
            terminal_targets += len(y)
            squares += len(y) ** 2
            flops += cost
            if regime == "flops" and flops >= budget * 0.99:
                break
        else:
            completed += 1
            continue
        break
    h.require(steps > 0 and targets > 0, "no-op training schedule")
    if regime == "flops":
        h.require(budget * 0.99 <= flops <= budget, "FLOP budget cannot be matched within 1%")
    else:
        h.require(completed == len(encoded), "byte budget did not consume every frozen document")
    return {
        "training_steps": steps,
        "training_target_tokens": targets,
        "training_sequence_length_squared_sum": squares,
        "completed_training_documents": completed,
        "partial_final_window": partial_final_window,
        "terminal_document_target_tokens": terminal_targets if regime == "flops" else 0,
        **h.flop_accounting(vocab, h.SCREEN, targets, squares),
    }


def condition_report(source, sample, documents, *, flops_budget, byte_budget, context):
    h.require(h.artifact_hashes(context["artifact_root"] / source["artifact"]) == source["artifact_hashes"],
              "tokenizer artifact changed before preflight")
    tok = h.load_tokenizer(source, context["artifact_root"])
    h.require(len(tok.vocab) == source["vocab_budget"], "loaded tokenizer vocabulary mismatch")
    input_order = [{key: row[key] for key in ("position", "id", "normalized_text_hash",
                                                "normalized_utf8_bytes", "source_utf8_bytes")}
                   for row in sample["ordered_documents"]]
    coverage_count = len(sample["strata"])
    document_bytes = [{h.BYTE_BUDGET_FIELD: row["normalized_utf8_bytes"],
                       h.BYTE_AUDIT_FIELD: row["source_utf8_bytes"]} for row in sample["ordered_documents"]]
    h.require(h.byte_schedule(documents, byte_budget) == len(documents), "byte budget must consume exact frozen sample once")
    encoded, token_rows = [], []
    for position, (text, entry) in enumerate(zip(documents, input_order)):
        h.require(h.digest(text) == entry["normalized_text_hash"], "tokenization input order changed")
        ids = tok.encode(text)
        h.require(ids and all(type(i) is int and 4 <= i < len(tok.vocab) for i in ids), "invalid encoding")
        encoded.append(ids)
        token_rows.append({**entry, "text_tokens": len(ids), "token_ids_sha256": h.digest(ids)})
        if (position + 1) % 25 == 0 or position + 1 == len(documents):
            print(f"{source['tokenizer']} {source['vocab_budget']}: {position + 1}/{len(documents)} documents encoded", flush=True)
    h.require(len(encoded) == len(documents) == len(input_order), "document membership changed")
    runs = {}
    for regime, budget in (("flops", flops_budget), ("bytes", byte_budget)):
        run = schedule(encoded, len(tok.vocab), regime, budget, context["context"])
        count = run["completed_training_documents"]
        run.update({"budget_regime": regime, "requested_budget": budget,
                    "complete_document_bytes": h.byte_totals(document_bytes, count),
                    "training_byte_scope": "complete_document_prefix",
                    "coverage_documents_completed": min(count, coverage_count),
                    "status": "blocked_coverage_prefix" if count < coverage_count else "feasible"})
        run["completed_document_bytes"] = run["complete_document_bytes"][h.BYTE_BUDGET_FIELD]
        run["completed_document_order_sha256"] = h.digest([input_order[i % len(documents)] for i in range(count)])
        run["strata"] = []
        for stratum in sample["strata"]:
            indices = [i for i in range(count) if
                       (sample["ordered_documents"][i % len(documents)]["domain"],
                        sample["ordered_documents"][i % len(documents)]["language"]) ==
                       (stratum["domain"], stratum["language"])]
            totals = {key: sum(document_bytes[i % len(documents)][key] for i in indices)
                      for key in (h.BYTE_BUDGET_FIELD, h.BYTE_AUDIT_FIELD)}
            run["strata"].append({"domain": stratum["domain"], "language": stratum["language"],
                                  "completed_documents": len(indices), **totals,
                                  "normalized_byte_share": totals[h.BYTE_BUDGET_FIELD] / run["completed_document_bytes"]
                                  if run["completed_document_bytes"] else 0})
        run["terminal_document"] = None
        if regime == "flops":
            index = count % len(documents)
            used = run["terminal_document_target_tokens"]
            run["terminal_document"] = {**input_order[index], "cycle": count // len(documents),
                                         "target_tokens_consumed_including_eos": used,
                                         "total_target_tokens_including_eos": len(encoded[index]) + 1,
                                         "text_tokens_consumed": min(used, len(encoded[index])),
                                         "fully_predicted": used == len(encoded[index]) + 1,
                                         "included_in_complete_document_bytes": False}
        runs[regime] = run
    h.require(runs["bytes"]["complete_document_bytes"] == {h.BYTE_BUDGET_FIELD: byte_budget,
                                                             h.BYTE_AUDIT_FIELD: sample["source_utf8_bytes"]},
              "byte-matched source/normalized exposure mismatch")
    h.require(h.artifact_hashes(context["artifact_root"] / source["artifact"]) == source["artifact_hashes"],
              "tokenizer artifact changed during preflight")
    base = {
        "tokenizer": source["tokenizer"], "vocab_budget": source["vocab_budget"],
        "actual_vocab_size": len(tok.vocab), "tokenizer_artifact_hash": h.digest(source["artifact_hashes"]),
        "tokenizer_config": source["tokenizer_config"], "special_tokens": h.SPECIAL_IDS,
        "input_exposure_order_sha256": h.digest(input_order), "input_documents": len(documents),
        "input_normalized_utf8_bytes": sample["normalized_utf8_bytes"],
        "input_source_utf8_bytes": sample["source_utf8_bytes"],
        "model_config": context["model_config"],
        "encoded_document_accounting": token_rows,
        "trained_commit": source["git_commit"], "original_extension_hash": source["extension_hash"],
        **h.parameter_accounting(len(tok.vocab), h.SCREEN, context["context"]),
    }
    return {**base, "status": "blocked" if runs["flops"]["status"] != "feasible" else "feasible",
            "encoded_documents": len(documents), "encoded_tokens": sum(map(len, encoded)), "regimes": runs}


def preflight(args):
    selection_path, source_path, exposure_path, phase_a_path = map(Path, (args.selection, args.source, args.exposure, args.phase_a))
    require_sha(selection_path, args.selection_sha256, "selection")
    require_sha(source_path, args.source_sha256, "source manifest")
    require_sha(exposure_path, args.exposure_sha256, "frozen exposure")
    require_sha(phase_a_path, args.phase_a_sha256, "Phase A ledger")
    selection = h.read_json(selection_path)
    source = h.read_json(source_path)
    exposure = h.read_json(exposure_path)
    receipt = h.read_json(exposure_path.parent / "receipt.json")
    ledger = h.read_json(phase_a_path)
    content_hash(selection)
    h.require(screen.implementation_hash() == selection["implementation_git_blob_hash"], "frozen implementation pin changed")
    h.require(selection == screen.selection_payload(ledger, args.phase_a_sha256, selection["implementation_git_blob_hash"]),
              "frozen selection/ledger/configuration mismatch")
    h.require(selection["selection_schema_version"] == screen.VERSION and selection["phase"] == "B-SCREEN", "invalid selection")
    h.require(selection["test_metrics_used"] is False and selection["lm_results_used"] is False,
              "selection contamination")
    h.require(selection["dataset"]["manifest_sha256"] == args.source_sha256, "selection/source mismatch")
    h.require(selection["safety_caps"] == {"flops": args.flops, "bytes": args.bytes}, "prescribed budget mismatch")
    h.require(selection["model_config_cpu_template"] == h.model_config("B", "cpu"), "model configuration changed")
    train_path = check_source(selection, source, source_path)
    pinned_paths = (selection_path, source_path, exposure_path, phase_a_path, train_path,
                    exposure_path.parent / "receipt.json", Path(__file__))
    input_hashes = {str(path.resolve()): h.file_hash(path) for path in pinned_paths}
    identity = h.runtime_identity()
    historical = selection["phase_a_identity"]
    runtime_blockers = []
    if identity["extension_hash"] != historical["extension_hash"]:
        runtime_blockers.append("current extension differs from pinned Phase A Linux extension")
    for package in ("python", "sentencepiece", "numpy", "regex"):
        if identity["versions"][package] != historical["versions"][package]:
            runtime_blockers.append(f"pinned tokenizer runtime mismatch: {package}")
    if identity["working_tree_dirty"]:
        runtime_blockers.append("reviewed preflight/harness changes are not committed")
    sample = check_exposure(exposure, receipt, args.selection_sha256, args.source_sha256)
    h.require(receipt["final_exposure_sha256"] == args.exposure_sha256, "exact-packing receipt/exposure hash mismatch")
    documents = load_frozen_documents(sample, train_path)
    selected = selected_phase_a_records(selection, ledger, phase_a_path)
    context = {"artifact_root": phase_a_path.resolve().parent, "context": selection["model_config_cpu_template"]["context"],
               "model_config": selection["model_config_cpu_template"]}
    records = []
    for index, row in enumerate(selected):
        started = time.monotonic()
        record = condition_report(row, sample, documents, flops_budget=args.flops, byte_budget=args.bytes, context=context)
        record["tokenizer_preflight_seconds"] = time.monotonic() - started
        record["preflight_identity"] = identity
        record["input_hashes"] = input_hashes
        publish(Path(args.output) / f"condition-{index:03d}.json", record)
        records.append(record)
        print(f"Completed {index + 1}/9: {row['tokenizer']} {row['vocab_budget']} {record['status']}", flush=True)
    h.require(len(records) == 9 and {(r["tokenizer"], r["vocab_budget"]) for r in records} == set(screen.CONDITIONS),
              "incomplete tokenizer preflight")
    h.require(all(set(r["regimes"]) == set(h.REGIMES) for r in records), "incomplete regime preflight")
    check_pins(input_hashes)
    h.require(h.runtime_identity() == identity, "runtime/code changed during preflight")
    for row in selected:
        h.require(h.artifact_hashes(context["artifact_root"] / row["artifact"]) == row["artifact_hashes"],
                  "artifact changed before publication")
    all_conditions_feasible = all(row["status"] == "feasible" for row in records)
    source_truncations = [{"id": d["id"], "position": d["position"], "dedup": d["dedup"]}
                          for d in sample["ordered_documents"] if d["dedup"].get("truncated_to_quota") is not False]
    gate_passed = all_conditions_feasible and not source_truncations and not runtime_blockers
    report = {
        "preflight_schema_version": VERSION, "status": STATUS if gate_passed else REJECTED,
        "mode": "tokenizer_only_schedule_simulation",
        "preflight_identity": identity, "runtime_blockers": runtime_blockers,
        "pinned_linux_runtime_verified": not any("pinned" in b for b in runtime_blockers),
        "flop_tolerance": 0.01,
        "flop_definition": "dense_matmul_forward_backward_v1; core + vocabulary output projection; excludes lookup/optimizer/norms",
        "experiments_started": False, "lm_model_constructed": False, "optimizer_constructed": False,
        "validation_data_opened": False, "test_data_opened": False, "authorized_for_lm_execution": False,
        "selection_sha256": args.selection_sha256, "source_manifest_sha256": args.source_sha256,
        "exposure_sha256": args.exposure_sha256, "phase_a_ledger_sha256": args.phase_a_sha256,
        "input_hashes": input_hashes, "exposure": {"documents": sample["selected_documents"],
            "normalized_utf8_bytes": sample["normalized_utf8_bytes"], "source_utf8_bytes": sample["source_utf8_bytes"],
            "ordered_exposure_hash": sample["ordered_exposure_hash"], "strata": len(sample["strata"])},
        "budgets": {"flops": args.flops, "bytes": args.bytes}, "conditions": records,
        "all_conditions_feasible": all_conditions_feasible,
        "gate_passed": gate_passed, "whole_frozen_records_preserved": True,
        "upstream_source_truncations_for_review": source_truncations,
    }
    return {**report, "content_sha256": h.digest(report)}


def check_pins(input_hashes):
    for path, expected in input_hashes.items():
        require_sha(path, expected, "preflight input")


def run(args):
    output = Path(args.output)
    h.require(not output.exists(), "preflight output exists; refusing to overwrite evidence")
    output.mkdir(parents=True)
    publish(output / "initial-rejection-receipt.json", {
        "status": REJECTED, "reason": "PREFLIGHT_NOT_COMPLETED", "final_preflight_sha256": None,
        "rule": "Only a complete frozen/ bundle is a passing preflight; interruption cannot imply launch approval.",
    })
    try:
        result = preflight(args)
        if not result["gate_passed"]:
            publish(output / "preflight-rejection.json", result)
            publish(output / "rejection-receipt.json", {
                "status": REJECTED, "reason": "FLOP_COVERAGE_BLOCKED" if not result["all_conditions_feasible"] else "PROVENANCE_REVIEW_REQUIRED",
                "final_preflight_sha256": None,
                "runtime_blockers": result["runtime_blockers"],
                "upstream_source_truncations_for_review": result["upstream_source_truncations_for_review"],
                "rejection_report_sha256": h.file_hash(output / "preflight-rejection.json"),
                "blocked_conditions": [{"tokenizer": row["tokenizer"], "vocab_budget": row["vocab_budget"],
                                        "completed_coverage_documents": row["regimes"]["flops"]["coverage_documents_completed"]}
                                       for row in result["conditions"] if row["status"] == "blocked"],
                "experiments_started": False, "authorized_for_lm_execution": False,
            })
            return h.read_json(output / "rejection-receipt.json")
        staging = output / ".publication-incomplete"
        staging.mkdir()
        publish(staging / "preflight.json", result)
        final_hash = h.file_hash(staging / "preflight.json")
        publish(staging / "receipt.json", {
            "status": STATUS, "final_preflight_sha256": final_hash, "conditions": len(result["conditions"]),
            "regimes": list(h.REGIMES), "experiments_started": False, "authorized_for_lm_execution": False,
        })
        h.require(not (output / "frozen").exists(), "duplicate preflight publication")
        os.rename(staging, output / "frozen")
        return h.read_json(output / "frozen" / "receipt.json")
    except BaseException as error:
        if not (output / "rejection-receipt.json").exists():
            publish(output / "rejection-receipt.json", {
                "status": REJECTED, "reason": "PREFLIGHT_ERROR", "detail": f"{type(error).__name__}: {error}",
                "final_preflight_sha256": None, "experiments_started": False, "authorized_for_lm_execution": False,
            })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--exposure", type=Path, required=True)
    parser.add_argument("--exposure-sha256", required=True)
    parser.add_argument("--phase-a", type=Path, required=True)
    parser.add_argument("--phase-a-sha256", required=True)
    parser.add_argument("--flops", type=float, required=True)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    h.require(math.isfinite(args.flops) and 0 < args.flops <= 1e11, "invalid/excessive FLOP budget")
    h.require(type(args.bytes) is int and 0 < args.bytes <= 1_000_000, "invalid/excessive byte budget")
    result = run(args)
    print(json.dumps(result, indent=2))
    if result["status"] != STATUS:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
