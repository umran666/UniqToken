"""Freeze and execute the user-declared nine-tokenizer, one-seed LM SCREENING gate.

This entry point never trains tokenizers, opens test data, or starts confirmation.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import time

from benchmarks import phase_a_migrate as lineage
from benchmarks import run_phase_a as stages
from benchmarks import run_research_experiments as h

VERSION = 1
POLICY = "user_declared_three_baselines_all_vocabularies_v1"
NAMES = ("sp_unigram", "boundary_bpe", "uniq_superbpe")
CONDITIONS = tuple((name, vocab) for vocab in h.VOCABS for name in NAMES)
METRICS = ("bytes_per_token", "tokens_per_unicode_character", "byte_fallback_percent")
RATIONALE = {
    "sp_unigram": "Canonical probabilistic baseline; retained by explicit user declaration.",
    "boundary_bpe": "Required historical comparison, NOT the current bytes-per-token winner.",
    "uniq_superbpe": "Proposed UniqToken method; retained by explicit user declaration.",
    "65536": "Required historical failure regime; cannot be dropped after seeing LM results.",
    "scope": "Fixed by the user after A-SCREEN and before any Phase B LM results; not the A-CONFIRM winner policy.",
    "metric_direction": "Higher bytes_per_token is better compression; lower tokens_per_unicode_character is better.",
    "excluded": "sp_bpe and uniq_unigram excluded by user declaration, not because they lost screening.",
}


def executable_paths(commit):
    for name in lineage.tree(commit):
        parts = Path(name).parts
        suffix = {"uniqtoken": ".py", "benchmarks": ".py", "crates": ".rs"}.get(parts[0])
        if (suffix and name.endswith(suffix) and not {"target", "legacy"}.intersection(parts)) or name in (
            "pyproject.toml", "crates/uniqtoken_core/Cargo.toml", "crates/uniqtoken_core/Cargo.lock"
        ):
            yield name


def verify_historical_code(identity):
    """Consume historical artifacts as inputs, not as newly trained/current results."""
    commit = identity["commit_hash"]
    h.require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "invalid Phase A commit")
    h.require(not identity["working_tree_dirty"], "dirty Phase A provenance")
    h.require(identity["extension_status"] == "installed" and identity["extension_hash"], "missing Phase A extension")
    h.require(lineage.source_hash(commit) == identity["source_hash"], "historical source hash mismatch")
    tree = lineage.tree(commit)
    for name in executable_paths(commit):
        actual = lineage.git("hash-object", f"--path={name}", "--", name).decode().strip()
        h.require(actual == tree[name][2], f"changed historical executable: {name}")


def implementation_hash():
    # Git's clean filters make the code pin portable across LF/CRLF checkouts.
    # The execution plan additionally records the ordinary on-disk source hash.
    files = []
    for directory, suffix in (("uniqtoken", ".py"), ("benchmarks", ".py"), ("crates", ".rs")):
        files.extend(p.relative_to(h.ROOT).as_posix() for p in (h.ROOT / directory).rglob("*" + suffix)
                     if not {"target", "legacy"}.intersection(p.parts))
    files.extend(("pyproject.toml", "crates/uniqtoken_core/Cargo.toml", "crates/uniqtoken_core/Cargo.lock"))
    return h.digest({name: lineage.git("hash-object", f"--path={name}", "--", name).decode().strip()
                     for name in sorted(files)})


def dataset_context(path):
    rows, texts, validation_rows, validation, source = stages.load_stage_source(path)
    train, info = stages.stratified_screen_training(rows, texts)
    pairs, _ = stages.partition_validation(validation_rows, validation)
    heldout = [text for _, text in pairs]
    raw = {h.digest(text): row["raw_utf8_bytes"] for row, text in zip(rows, texts)}
    byte_rows = {
        "train": [{h.BYTE_BUDGET_FIELD: len(text.encode("utf-8")), h.BYTE_AUDIT_FIELD: raw[h.digest(text)]}
                  for text in train],
        "validation": [{h.BYTE_BUDGET_FIELD: len(text.encode("utf-8")), h.BYTE_AUDIT_FIELD: row["raw_utf8_bytes"]}
                       for row, text in pairs],
    }
    training = {
        "scope": "representative_stratified_screen", "normalized_utf8_bytes": info["actual_normalized_utf8_bytes"],
        "source_utf8_bytes": info["source_utf8_bytes"], "documents": len(train),
        "assignment_hash": info["assignment_hash"], "screen_selection": info,
    }
    evaluation = {
        "partition_version": stages.VALIDATION_PARTITION_VERSION, "partition": "screening",
        **h.byte_totals(byte_rows["validation"]), "unicode_characters": sum(map(len, heldout)),
        "documents": len(heldout), "assignment_hash": h.digest([h.digest(text) for text in heldout]),
    }
    return {"train": train, "validation": heldout}, byte_rows, source, training, evaluation


def verify_phase_a(path, dataset_path):
    path = Path(path)
    ledger = h.read_json(path)
    docs, byte_rows, source, training, validation = dataset_context(dataset_path)
    meta = ledger["metadata"]
    h.require(meta["training"] == training and meta["validation"] == validation, "Phase A assignment mismatch")
    verify_historical_code(meta["identity"])
    grid = [(n, v) for n in h.COHORT for v in h.VOCABS]
    stages.validate_stage_ledger(ledger, meta["identity"], source, path.parent, "A-SCREEN", grid)
    for row in ledger["records"]:
        h.require(row["artifact_hashes"] and row["artifact"] == f"{row['tokenizer']}-{row['vocab_budget']}",
                  "invalid artifact assignment")
        fallback = row["validation"]["byte_fallback_tokens"]
        tokens = row["validation"]["tokens"]
        h.require(type(fallback) is int and 0 <= fallback <= tokens
                  and row["validation"]["byte_fallback_percent"] == 100.0 * fallback / tokens,
                  "invalid fallback metric")
    return ledger, docs, byte_rows


def selection_payload(ledger, ledger_hash, implementation_hash):
    meta = ledger["metadata"]
    h.require(meta["stage"] == "A-SCREEN" and meta["result_label"] == "SCREENING"
              and meta["status"] == "complete", "complete A-SCREEN required")
    rows = ledger["records"]
    h.require(len(rows) == 15 and {(r["tokenizer"], r["vocab_budget"]) for r in rows}
              == {(n, v) for n in h.COHORT for v in h.VOCABS}, "incomplete Phase A cohort")
    h.require(all("test" not in r for r in rows), "test metrics cannot enter selection")
    selected = []
    for name, vocab in CONDITIONS:
        row = next(r for r in rows if (r["tokenizer"], r["vocab_budget"]) == (name, vocab))
        selected.append({
            "tokenizer": name, "vocab_budget": vocab, "actual_vocab_size": row["actual_vocab_size"],
            "artifact": row["artifact"], "artifact_hashes": row["artifact_hashes"],
            "tokenizer_config": row["tokenizer_config"], "trained_commit": row["git_commit"],
            "extension_hash": row["extension_hash"], "migration": row.get("migration"),
            "training_assignment_hash": row["training_assignment_hash"],
            "validation_assignment_hash": row["validation_assignment_hash"],
            "phase_a_validation_metrics": {key: row["validation"][key] for key in METRICS},
        })
    body = {
        "selection_schema_version": VERSION, "policy": POLICY, "phase": "B-SCREEN",
        "result_label": "SCREENING", "test_metrics_used": False, "lm_results_used": False,
        "phase_a_ledger_sha256": ledger_hash, "phase_a_identity": meta["identity"],
        "dataset": meta["dataset"], "training": meta["training"], "validation": meta["validation"],
        "conditions": selected, "seeds": [0], "regimes": list(h.REGIMES), "lm_run_count": 18,
        "special_tokens": h.SPECIAL_IDS, "byte_tokens": 256, "rationale": RATIONALE,
        "selection_metric_fields": list(METRICS),
        "phase_a_validation_evidence": [
            {"tokenizer": r["tokenizer"], "vocab_budget": r["vocab_budget"],
             **{key: r["validation"][key] for key in METRICS}} for r in rows
        ],
        "implementation_git_blob_hash": implementation_hash,
        "model_config_cpu_template": h.model_config("B", "cpu"),
        "byte_budget_field": h.BYTE_BUDGET_FIELD, "byte_audit_field": h.BYTE_AUDIT_FIELD,
        "safety_caps": {"flops": 1e11, "bytes": 1_000_000},
    }
    return {**body, "content_sha256": h.digest(body)}


def freeze(args):
    ledger, _, _ = verify_phase_a(args.phase_a, args.dataset)
    selection = selection_payload(ledger, h.file_hash(args.phase_a), implementation_hash())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    h.write_new_json_atomic(output, selection)
    return {"selection": str(output.resolve()), "sha256": h.file_hash(output),
            "content_sha256": selection["content_sha256"], "tokenizer_conditions": 9, "lm_runs": 18}


def check_selection(path, expected_hash, ledger, ledger_path, identity):
    h.require(re.fullmatch(r"[0-9a-f]{64}", expected_hash or "") is not None, "explicit selection SHA-256 required")
    h.require(h.file_hash(path) == expected_hash, "selection changed or stale")
    expected = selection_payload(ledger, h.file_hash(ledger_path), implementation_hash())
    h.require(h.read_json(path) == expected, "selection/configuration/implementation mismatch")
    return expected


def check_runtime(identity, historical, device="cpu"):
    h.require(not identity["working_tree_dirty"], "commit the reviewed Phase B harness before execution")
    h.require(identity["extension_status"] == "installed" and identity["extension_hash"] == historical["extension_hash"],
              "Phase A tokenizer extension required (use the pinned Linux build)")
    for package in ("python", "sentencepiece", "numpy", "regex"):
        h.require(identity["versions"][package] == historical["versions"][package], f"tokenizer environment mismatch: {package}")
    # Torch may be a GPU build for LM screening. Its exact version is locked in the execution plan.
    if device.startswith("cuda"):
        import torch
        h.require(torch.cuda.is_available(), "requested CUDA is unavailable")
        if ":" in device:
            h.require(int(device.split(":")[1]) < torch.cuda.device_count(), "requested CUDA device does not exist")
        h.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"),
                  "set CUBLAS_WORKSPACE_CONFIG before deterministic CUDA screening")


def prepare(args, *, execution=False):
    ledger, docs, byte_rows = verify_phase_a(args.phase_a, args.dataset)
    identity = h.runtime_identity()
    selection = check_selection(args.selection, args.selection_sha256, ledger, args.phase_a, identity)
    h.require(math.isfinite(args.flops) and 0 < args.flops <= 1e11, "invalid/excessive screening FLOP budget")
    h.require(args.flops >= 100 * h.training_flops(max(h.VOCABS), h.SCREEN, 1),
              "screening FLOP budget too small for one-percent resolution")
    h.require(type(args.bytes) is int and 0 < args.bytes <= 1_000_000, "invalid/excessive screening byte budget")
    count = h.byte_schedule(docs["train"], args.bytes)
    h.require(args.device == "cpu" or re.fullmatch(r"cuda(?::\d+)?", args.device) is not None, "invalid LM device")
    if execution:
        check_runtime(identity, selection["phase_a_identity"], args.device)
    plan = {
        **identity, "identity": identity, "phase_b_schema_version": VERSION, "stage": "B-SCREEN",
        "result_label": "SCREENING", "status": "planned", "selection_sha256": args.selection_sha256,
        "selection_content_sha256": selection["content_sha256"], "selection": selection,
        "data_split": "document_disjoint_train_screening_validation_test_not_opened",
        "test_access": "forbidden_not_opened", "seeds": [0],
        "conditions": [[n, v, regime, 0] for n, v in CONDITIONS for regime in h.REGIMES],
        "model_config": h.model_config("B", args.device), "budgets": {"flops": args.flops, "bytes": args.bytes},
        "cuda_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if args.device.startswith("cuda") else None,
        "byte_matched_documents": count, "byte_matched_exposure": h.byte_totals(byte_rows["train"], count),
        "byte_matched_assignment_hash": h.digest([h.digest(docs["train"][i % len(docs["train"])]) for i in range(count)]),
    }
    return plan, ledger, docs, byte_rows


def validate_result(row, expected, plan, source, byte_rows):
    h.require(h.condition_key(row) == tuple(expected), "unexpected condition or seed")
    name, vocab, regime, _ = expected
    cfg = h.LMArchConfig(**plan["model_config"]["architecture"])
    required = {
        "model_kind": "causal_transformer", "actual_vocab_size": vocab, "result_label": "SCREENING",
        "git_commit": plan["identity"]["commit_hash"], "extension_hash": plan["identity"]["extension_hash"],
        "selection_sha256": plan["selection_sha256"], "model_config": plan["model_config"],
        "special_tokens": h.SPECIAL_IDS, "requested_budget": plan["budgets"][regime],
        "tokenizer_artifact_hash": h.digest(source["artifact_hashes"]),
        "dataset_manifest_hash": plan["selection"]["dataset"]["manifest_sha256"],
        "training_assignment_hash": plan["selection"]["training"]["assignment_hash"],
        "validation_assignment_hash": plan["selection"]["validation"]["assignment_hash"],
        **h.parameter_accounting(vocab, cfg, plan["model_config"]["context"]),
    }
    h.require(all(row.get(k) == v for k, v in required.items()), "LM provenance/configuration/parameter mismatch")
    h.require("test" not in row and "test_metrics" not in row, "screening cannot score test")
    targets, squares = row["training_target_tokens"], row["training_sequence_length_squared_sum"]
    h.require(type(targets) is int and type(squares) is int and targets > 0
              and targets <= squares <= targets * plan["model_config"]["context"] and row["training_steps"] > 0,
              "invalid/no-op training accounting")
    h.require(all(row.get(k) == v for k, v in h.flop_accounting(vocab, cfg, targets, squares).items()), "FLOP accounting mismatch")
    complete = row["completed_training_documents"]
    h.require(row["training_byte_scope"] == "complete_document_prefix", "ambiguous byte accounting")
    exposure = h.byte_totals(byte_rows["train"], complete)
    h.require(row["training_bytes"] == exposure and row["completed_document_bytes"] == exposure[h.BYTE_BUDGET_FIELD],
              "source/normalized exposure mismatch")
    if regime == "bytes":
        h.require(complete == plan["byte_matched_documents"] and exposure == plan["byte_matched_exposure"], "byte matching mismatch")
    else:
        h.require(0.99 * plan["budgets"][regime] <= row["actual_analytical_flops"] <= plan["budgets"][regime], "FLOP matching mismatch")
    metric = row["validation"]
    validation = plan["selection"]["validation"]
    expected_targets = source["validation"]["tokens"] + validation["documents"]
    h.require(metric["target_tokens_including_eos"] == expected_targets, "held-out token count mismatch")
    h.require(metric == h.nll_metrics(metric["total_nll_nats"], expected_targets, validation["normalized_utf8_bytes"],
                                    source_utf8_bytes=validation["source_utf8_bytes"]), "held-out CE/NLL/BPB mismatch")
    h.digest(row)  # Reject non-finite nested values.


def validate_ledger(payload, plan, ledger, byte_rows):
    h.require(payload["metadata"] == {**plan, "status": "complete"}, "stale/incomplete LM ledger")
    h.require(len(payload["records"]) == len(plan["conditions"]) == 18, "incomplete LM conditions")
    h.validate_ledger(payload, expected_commit=plan["identity"]["commit_hash"])
    for row, expected in zip(payload["records"], plan["conditions"]):
        source = next(r for r in ledger["records"] if (r["tokenizer"], r["vocab_budget"]) == tuple(expected[:2]))
        validate_result(row, expected, plan, source, byte_rows)
    return payload


def check_execution_lock(args, output, plan):
    h.require(h.file_hash(args.selection) == plan["selection_sha256"], "selection changed after execution started")
    h.require(h.read_json(output / "plan.json") == plan, "execution plan changed")
    h.require(h.read_json(output / "selection.json") == plan["selection"], "execution selection changed")
    h.require(h.file_hash(args.phase_a) == plan["selection"]["phase_a_ledger_sha256"], "Phase A ledger changed")
    h.require(h.file_hash(args.dataset) == plan["selection"]["dataset"]["manifest_sha256"], "dataset manifest changed")
    h.require(h.runtime_identity() == plan["identity"], "runtime changed during screening")
    if plan["model_config"]["device"].startswith("cuda"):
        h.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == plan["cuda_workspace_config"], "CUDA configuration changed")


def run(args):
    output = Path(args.output)
    h.require(not output.exists(), "output exists; Phase B cannot overwrite or silently resume")
    plan, ledger, docs, byte_rows = prepare(args, execution=True)
    output.mkdir(parents=True)
    h.write_new_json_atomic(output / "selection.json", plan["selection"])
    h.write_new_json_atomic(output / "plan.json", plan)
    records = []
    for index, condition in enumerate(plan["conditions"]):
        check_execution_lock(args, output, plan)
        name, vocab, regime, seed = condition
        source = next(r for r in ledger["records"] if (r["tokenizer"], r["vocab_budget"]) == (name, vocab))
        tok = h.load_tokenizer(source, Path(args.phase_a).parent)
        started = time.perf_counter()
        measured = h.train_lm(tok, docs, h.SCREEN, plan["model_config"]["context"], regime,
                              plan["budgets"][regime], seed, args.device, False, document_bytes=byte_rows)
        row = {
            **measured, "tokenizer": name, "vocab_budget": vocab, "actual_vocab_size": len(tok.vocab),
            "budget_regime": regime, "seed": seed, "model_kind": "causal_transformer", "result_label": "SCREENING",
            "model_config": plan["model_config"], "requested_budget": plan["budgets"][regime],
            "git_commit": plan["identity"]["commit_hash"], "extension_hash": plan["identity"]["extension_hash"],
            "selection_sha256": plan["selection_sha256"], "special_tokens": h.SPECIAL_IDS,
            "tokenizer_artifact_hash": h.digest(source["artifact_hashes"]),
            "dataset_manifest_hash": plan["selection"]["dataset"]["manifest_sha256"],
            "training_assignment_hash": plan["selection"]["training"]["assignment_hash"],
            "validation_assignment_hash": plan["selection"]["validation"]["assignment_hash"],
            "wall_clock_seconds": time.perf_counter() - started,
        }
        check_execution_lock(args, output, plan)
        h.require(h.artifact_hashes(Path(args.phase_a).parent / source["artifact"]) == source["artifact_hashes"],
                  "tokenizer artifact changed during condition")
        validate_result(row, condition, plan, source, byte_rows)
        h.write_new_json_atomic(output / f"condition-{index:03d}.json", {"status": "condition_complete", "record": row})
        records.append(row)
    result = {"metadata": {**plan, "status": "complete"}, "records": records}
    validate_ledger(result, plan, ledger, byte_rows)
    check_execution_lock(args, output, plan)
    for source in plan["selection"]["conditions"]:
        h.require(h.artifact_hashes(Path(args.phase_a).parent / source["artifact"]) == source["artifact_hashes"],
                  "tokenizer artifact changed before ledger publication")
    h.write_new_json_atomic(output / "ledger.json", result)
    return {"ledger": str(output / "ledger.json"), "conditions": len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "preflight", "run"):
        child = commands.add_parser(command)
        child.add_argument("--phase-a", type=Path, required=True)
        child.add_argument("--dataset", type=Path, required=True)
        if command in ("freeze", "run"):
            child.add_argument("--output", type=Path, required=True)
        if command != "freeze":
            child.add_argument("--selection", type=Path, required=True)
            child.add_argument("--selection-sha256", required=True)
            child.add_argument("--flops", type=float, required=True)
            child.add_argument("--bytes", type=int, required=True)
            child.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args)
    elif args.command == "run":
        result = run(args)
    else:
        plan, _, _, _ = prepare(args)
        blockers = []
        try:
            check_runtime(plan["identity"], plan["selection"]["phase_a_identity"], args.device)
        except ValueError as error:
            blockers.append(str(error))
        result = {"configuration_valid": True, "execution_ready": not blockers, "blockers": blockers,
                  "experiments_started": False, "plan": plan}
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
