"""Run a non-scientific tokenizer training-cost pilot on one fixed corpus prefix."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.run_research_experiments import (
    BYTE_BUDGET_FIELD,
    COHORT,
    artifact_hashes,
    digest,
    load_dataset,
    require,
    runtime_identity,
    tokenizer_configuration,
    train_tokenizer,
    write_new_json_atomic,
)

PILOT_SCHEMA_VERSION = 1
PILOT_VOCABS = (4096, 8192, 16384)


def fixed_corpus_prefix(texts, maximum_normalized_bytes):
    """Select a deterministic whole-document prefix shared by every pilot condition."""
    require(type(maximum_normalized_bytes) is int and maximum_normalized_bytes > 0, "invalid pilot corpus budget")
    selected, total = [], 0
    for text in texts:
        size = len(text.encode("utf-8"))
        if total + size > maximum_normalized_bytes:
            break
        selected.append(text)
        total += size
    require(selected and total > 0, "pilot budget does not contain one complete training document")
    return selected, total


def run(args):
    output = Path(args.output)
    require(not output.exists(), "pilot output already exists; never overwrite measurements")
    docs, dataset = load_dataset(args.dataset)
    identity = runtime_identity()
    require(not identity["working_tree_dirty"], "commit the pilot harness before measuring")
    texts, corpus_bytes = fixed_corpus_prefix(docs["train"], args.corpus_bytes)
    corpus_hash = digest(texts)
    output.mkdir(parents=True)
    metadata = {
        "pilot_schema_version": PILOT_SCHEMA_VERSION,
        "purpose": "tokenizer_training_cost_pilot",
        "scientific_phase_a": False,
        "identity": identity,
        "dataset_manifest_hash": dataset["manifest_hash"],
        "dataset_assignment_hash": dataset["assignment_hash"],
        "corpus_selection": "ordered_complete_document_prefix",
        "requested_maximum_normalized_utf8_bytes": args.corpus_bytes,
        BYTE_BUDGET_FIELD: corpus_bytes,
        "documents": len(texts),
        "corpus_hash": corpus_hash,
        "cohort": list(COHORT),
        "vocabulary_targets": list(PILOT_VOCABS),
    }
    write_new_json_atomic(output / "plan.json", metadata)
    records = []
    for name in COHORT:
        for budget in PILOT_VOCABS:
            artifact = f"{name}-{budget}"
            print(f"Cost pilot: {name} V={budget}", flush=True)
            tokenizer, elapsed = train_tokenizer(name, texts, budget, output / artifact)
            record = {
                "tokenizer": name,
                "vocab_budget": budget,
                "actual_vocab_size": len(tokenizer.vocab),
                "training_wall_clock_seconds": elapsed,
                "training_normalized_mb_per_sec": corpus_bytes / (elapsed * 1_000_000),
                BYTE_BUDGET_FIELD: corpus_bytes,
                "documents": len(texts),
                "corpus_hash": corpus_hash,
                "tokenizer_config": tokenizer_configuration(name, budget),
                "learned_merges": tokenizer.merges,
                "artifact": artifact,
                "artifact_hashes": artifact_hashes(output / artifact),
            }
            require(record["actual_vocab_size"] == budget, "pilot vocabulary target was not achieved")
            write_new_json_atomic(
                output / f"condition-{len(records):03d}.json",
                {"status": "pilot_measurement", "record": record},
            )
            records.append(record)
    require(len(records) == len(COHORT) * len(PILOT_VOCABS), "incomplete cost pilot")
    payload = {"metadata": metadata, "records": records}
    write_new_json_atomic(output / "ledger.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-bytes", type=int, default=1_000_000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
