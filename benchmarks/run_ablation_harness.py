"""Tokenizer-only objective ablation harness.

Compare tokenizer variants cheaply and objectively without training any language
model.  The harness inherits the frozen-dataset, provenance, exact-budget, and
leakage protections of the Phase A stages rather than duplicating them:

- train/validation splits come from ``stages.load_stage_source``, which never
  opens the declared test artifact;
- every condition retrains and re-validates a tokenizer under the shared exact
  vocabulary and special-token budgets (``h.validate_tokenizer``);
- deterministic seeding is applied exactly as Phase A applies it, so a rerun
  reproduces identical artifacts;
- the run receipt binds inputs, configuration, commit, source, extension, and
  artifacts; a ledger is published atomically only after every condition
  validates (fail-loud on incomplete runs).

No language-model path and no test-data path is reachable from this module's
entry point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re

import numpy as np

from benchmarks import run_phase_a as stages
from benchmarks import run_research_experiments as h

ABLATION_SCHEMA_VERSION = 1
RESULT_LABEL = "ABLATION"
STAGE_NAME = "ABLATION-TOKENIZER-ONLY"
SEED = 0
BASELINE = h.COHORT[0]
# Predeclared objective metrics and their direction.  Lower is better, so a
# candidate regression is an INCREASE over its matched-budget baseline.
METRICS = ("bytes_per_token", "tokens_per_unicode_character", "byte_fallback_percent")
VALIDATION_PARTITION_VERSION = stages.VALIDATION_PARTITION_VERSION


def _seeded_condition():
    random.seed(SEED)
    np.random.seed(SEED)


def _strata(rows, texts):
    """Group validation rows/documents by (domain, language) deterministically.

    Returns per-stratum document lists used only for metric computation; the
    raw texts themselves are never serialized into any plan, record, or ledger.
    """
    h.require(len(rows) == len(texts), "stratum row/text mismatch")
    strata = {}
    for row, text in zip(rows, texts):
        key = (row["domain"], row["language"])
        entry = strata.setdefault(
            key,
            {
                "domain": row["domain"],
                "language": row["language"],
                "documents": 0,
                "source_utf8_bytes": 0,
                "texts": [],
            },
        )
        entry["documents"] += 1
        entry["source_utf8_bytes"] += row["raw_utf8_bytes"]
        entry["texts"].append(text)
    return strata


def _stratum_key(row):
    return f"{row['domain']}::{row['language']}"


def _metrics(tokenizer, strata):
    """Whole-validation and per-stratum tokenizer metrics from one source of truth."""
    whole_texts = [text for entry in strata.values() for text in entry["texts"]]
    whole_source = sum(entry["source_utf8_bytes"] for entry in strata.values())
    whole = h.token_metrics(tokenizer, whole_texts, source_utf8_bytes=whole_source)
    per_stratum = {
        f"{entry['domain']}::{entry['language']}": {
            "domain": entry["domain"],
            "language": entry["language"],
            "documents": entry["documents"],
            **h.token_metrics(tokenizer, entry["texts"], source_utf8_bytes=entry["source_utf8_bytes"]),
        }
        for entry in strata.values()
    }
    return whole, per_stratum


def _build_plan(manifest_path, conditions):
    train_rows, train, validation_rows, validation, source = stages.load_stage_source(manifest_path)
    identity = h.runtime_identity()
    h.require(not identity["working_tree_dirty"], "commit the reviewed harness before research runs")
    strata = {}
    for row in validation_rows:
        key = _stratum_key(row)
        entry = strata.setdefault(key, {"domain": row["domain"], "language": row["language"], "documents": 0})
        entry["documents"] += 1
    return {
        **identity,
        "identity": identity,
        "ablation_schema_version": ABLATION_SCHEMA_VERSION,
        "purpose": "tokenizer_only_objective_ablation",
        "result_label": RESULT_LABEL,
        "stage": STAGE_NAME,
        "status": "planned",
        "dataset": source,
        "data_split": "document_disjoint_train_validation_test_not_opened",
        "test_access": "forbidden_not_opened",
        "training": {
            "scope": "full_frozen_training_corpus",
            "documents": len(train),
            "normalized_utf8_bytes": sum(len(text.encode("utf-8")) for text in train),
            "source_utf8_bytes": sum(row["raw_utf8_bytes"] for row in train_rows),
            "assignment_hash": h.digest([h.digest(text) for text in train]),
        },
        "validation": {
            "partition_version": VALIDATION_PARTITION_VERSION,
            "partition": "ablation",
            "documents": len(validation),
            "normalized_utf8_bytes": sum(len(text.encode("utf-8")) for text in validation),
            "source_utf8_bytes": sum(row["raw_utf8_bytes"] for row in validation_rows),
            "unicode_characters": sum(map(len, validation)),
            "assignment_hash": h.digest([h.digest(text) for text in validation]),
            "strata": {"domain/language": strata},
        },
        "conditions": [list(condition) for condition in conditions],
        "seed": SEED,
        "bootstrapping": "none",
        "objectives": {"metrics": list(METRICS), "directions": {metric: "lower_is_better" for metric in METRICS}},
        "language_model": {"initialized": False, "reachable": False},
    }


def _artifact_name(condition):
    name, budget = condition
    return f"{name}-{budget}"


def _validate_record(record, condition, plan, strata, output):
    name, budget = condition
    expected_index = plan["conditions"].index([name, budget])
    h.require((record.get("tokenizer"), record.get("vocab_budget")) == (name, budget), "condition mismatch")
    h.require(record.get("condition_index") == expected_index, "condition index mismatch")
    h.require(record.get("actual_vocab_size") == budget, "vocabulary target mismatch")
    h.require(record.get("seed") == SEED, "seed mismatch")
    h.require(record.get("model_kind") == "tokenizer_only", "model kind mismatch")
    h.require(record.get("result_label") == RESULT_LABEL, "result label mismatch")
    h.require(record.get("git_commit") == plan["identity"]["commit_hash"], "condition commit mismatch")
    h.require(record.get("extension_hash") == plan["identity"]["extension_hash"], "condition extension mismatch")
    h.require(record.get("dataset_manifest_hash") == plan["dataset"]["manifest_sha256"], "condition dataset mismatch")
    h.require(
        record.get("tokenizer_config") == h.tokenizer_configuration(name, budget), "tokenizer configuration mismatch"
    )
    h.require(
        record.get("training_assignment_hash") == plan["training"]["assignment_hash"], "training assignment mismatch"
    )
    h.require(
        record.get("validation_assignment_hash") == plan["validation"]["assignment_hash"],
        "validation assignment mismatch",
    )
    h.require(record.get("artifact") == _artifact_name(condition), "artifact name mismatch")
    h.require("test" not in record, "test metrics are forbidden in ablation records")
    tokenizer = h.load_tokenizer(record, output)
    h.validate_tokenizer(tokenizer, budget)
    whole, per_stratum = _metrics(tokenizer, strata)
    h.require(record["validation"] == whole, "validation metrics mismatch")
    h.require(record["strata"] == per_stratum, "stratum metrics mismatch")
    h.digest(record)
    return record


def _compare(records, conditions, metrics):
    """Matched-budget baseline/candidate comparison with regression flags."""
    baseline_by_budget = {record["vocab_budget"]: record for record in records if record["tokenizer"] == BASELINE}
    comparisons = []
    for condition, record in zip(conditions, records):
        row = {"condition": list(condition), "tokenizer": record["tokenizer"], "vocab_budget": record["vocab_budget"]}
        is_baseline = record["tokenizer"] == BASELINE
        row["is_baseline"] = is_baseline
        reference = record if is_baseline else baseline_by_budget.get(record["vocab_budget"])
        h.require(reference is not None, f"baseline missing for vocabulary budget {record['vocab_budget']}")
        regressions = []
        for metric in metrics:
            value = record["validation"][metric]
            baseline_value = reference["validation"][metric]
            # A literal zero baseline (e.g. no byte-fallback tokens emitted) cannot
            # divide; any positive value is then strictly worse, and equality maps
            # to the neutral ratio 1.0 so the check below stays regression = 1.0.
            if baseline_value:
                ratio = value / baseline_value
            else:
                ratio = 1.0 if value == 0 else None
            row[metric] = value
            row[f"{metric}_baseline"] = baseline_value
            row[f"{metric}_ratio"] = ratio
            regression = ratio is None or ratio > 1.0
            row[f"{metric}_regression"] = regression
            if regression:
                regressions.append(metric)
        row["regressions"] = regressions
        comparisons.append(row)
    return {
        "metrics": list(metrics),
        "directions": {metric: "lower_is_better" for metric in metrics},
        "rows": comparisons,
    }


def _conditions(plan):
    return [tuple(item) for item in plan["conditions"]]


def _validate_ledger(payload, plan, conditions):
    h.validate_ledger(payload, expected_commit=plan["identity"]["commit_hash"])
    records = payload.get("records", [])
    h.require(len(records) == len(conditions), "incomplete ablation ledger")
    for record, condition in zip(records, conditions):
        h.require((record.get("tokenizer"), record.get("vocab_budget")) == condition, "ledger condition mismatch")
        h.require(record.get("model_kind") == "tokenizer_only", "invalid ledger model kind")
        h.require("test" not in record, "test metrics are forbidden in ablation ledgers")
    h.require(
        payload.get("comparisons") == _compare(records, conditions, METRICS),
        "comparison table is not a deterministic function of the records",
    )
    return payload


def _check_lock(args, plan):
    h.require(h.file_hash(args.dataset) == plan["dataset"]["manifest_sha256"], "dataset changed during the run")


def resume_records(output, plan, strata):
    records = {}
    for path in sorted(output.glob("condition-*.json")):
        h.require(re.fullmatch(r"condition-\d{3}\.json", path.name), "invalid checkpoint filename")
        index = int(path.stem.removeprefix("condition-"))
        h.require(index < len(plan["conditions"]) and index not in records, "unexpected or duplicate checkpoint")
        envelope = h.read_json(path)
        h.require(envelope.get("status") == "condition_complete", "incomplete condition")
        condition = tuple(plan["conditions"][index])
        _validate_record(envelope["record"], condition, plan, strata, output)
        records[index] = envelope["record"]
    return records


def run(args):
    conditions = [(name, budget) for name in h.COHORT for budget in h.VOCABS]
    plan = _build_plan(args.dataset, conditions)
    output = Path(args.output)
    if output.exists():
        h.require(args.resume and output.is_dir(), "output exists; use --resume for an incomplete run")
        h.require(not (output / "ledger.json").exists(), "run already has a complete ledger")
        h.require(h.read_json(output / "plan.json") == plan, "resume plan mismatch")
    else:
        h.require(not args.resume, "resume output does not exist")
        output.mkdir(parents=True)
        h.write_new_json_atomic(output / "plan.json", plan)
    train_rows, train, validation_rows, validation, _ = stages.load_stage_source(args.dataset)
    strata = _strata(validation_rows, validation)
    records = resume_records(output, plan, strata) if output.exists() else {}
    for index, condition in enumerate(conditions):
        _check_lock(args, plan)
        if index in records:
            continue
        name, budget = condition
        _seeded_condition()
        artifact = output / _artifact_name(condition)
        tokenizer, elapsed = h.train_tokenizer(name, train, budget, artifact)
        whole, per_stratum = _metrics(tokenizer, strata)
        record = {
            "condition_index": index,
            "condition": list(condition),
            "tokenizer": name,
            "vocab_budget": budget,
            "actual_vocab_size": len(tokenizer.vocab),
            "seed": SEED,
            "model_kind": "tokenizer_only",
            "result_label": RESULT_LABEL,
            "git_commit": plan["identity"]["commit_hash"],
            "extension_hash": plan["identity"]["extension_hash"],
            "dataset_manifest_hash": plan["dataset"]["manifest_sha256"],
            "tokenizer_config": h.tokenizer_configuration(name, budget),
            "training_assignment_hash": plan["training"]["assignment_hash"],
            "validation_assignment_hash": plan["validation"]["assignment_hash"],
            "training_wall_clock_seconds": elapsed,
            "training_normalized_mb_per_sec": plan["training"]["normalized_utf8_bytes"] / (elapsed * 1_000_000),
            "learned_merges": tokenizer.merges,
            "artifact": _artifact_name(condition),
            "artifact_hashes": h.artifact_hashes(artifact),
            "training_documents": plan["training"]["documents"],
            "validation_documents": plan["validation"]["documents"],
            "validation": whole,
            "strata": per_stratum,
        }
        _validate_record(record, condition, plan, strata, output)
        h.write_new_json_atomic(
            output / f"condition-{index:03d}.json", {"status": "condition_complete", "record": record}
        )
        records[index] = record
    h.require(set(records) == set(range(len(conditions))), "incomplete ablation run")
    result = {
        "metadata": {**plan, "status": "complete"},
        "records": [records[index] for index in range(len(conditions))],
        "comparisons": _compare([records[i] for i in range(len(conditions))], conditions, METRICS),
    }
    _validate_ledger(result, plan, conditions)
    h.write_new_json_atomic(output / "ledger.json", result)
    return result


def report(args):
    """Compare an existing ablation ledger without opening any data splits."""
    ledger = h.read_json(Path(args.ledger))
    plan = ledger["metadata"]
    _validate_ledger(ledger, plan, _conditions(plan))
    return ledger["comparisons"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="Validate the frozen dataset and configuration")
    preflight.add_argument("--dataset", type=Path, required=True)
    run_parser = subparsers.add_parser("run", help="Run the tokenizer-only objective ablation")
    run_parser.add_argument("--dataset", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--resume", action="store_true")
    compare = subparsers.add_parser("compare", help="Compare an existing ablation ledger")
    compare.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        plan = _build_plan(args.dataset, [(name, budget) for name in h.COHORT for budget in h.VOCABS])
        result = {
            "configuration_valid": True,
            "experiments_started": False,
            "conditions": len(plan["conditions"]),
            "dataset_id": plan["dataset"]["dataset_id"],
            "test_access": plan["test_access"],
            "plan": plan,
        }
    elif args.command == "compare":
        result = report(args)
    else:
        result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
