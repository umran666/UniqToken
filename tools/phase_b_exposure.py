"""Tokenizer-blind construction of the reviewed, frozen Phase B exposure.

Only standard-library dependencies. No tokenizer, LM, evaluation or FLOP code.
An unsuccessful packing attempt is evidence, not an executable exposure manifest.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict, deque
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
POLICY = "phase_b_stratified_whole_document_exposure_v1"
SELECTION_SHA = "d3449786b636216057e01e1372b4805c1181c481ca521a74c9e29e07676a0ea4"
SOURCE_SHA = "2ad27746d9c139c8d7e814e89c977c9bd4410033d5c1b97b673ee1b893ece7ff"
EXACT_BYTES = 1_000_000
GROUPS = {
    "latin_english": (320000, ("en",)),
    "indic_cjk_arabic": (320000, ("ar", "bn", "fa", "gu", "hi", "ja", "kn", "ko", "ml", "mr", "ta", "te", "ur", "zh")),
    "cyrillic_african": (160000, ("am", "bg", "ru", "sw", "uk", "yo")),
    "code": (200000, ("c", "cpp", "go", "java", "javascript", "python", "rust", "sql", "typescript")),
}
SPACES = re.compile(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def publish(path, value):
    """Atomic, exclusive local-disk publication; never replace prior evidence."""
    path = Path(path)
    require(not path.exists(), "refusing to overwrite evidence")
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".exposure-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def normalize(text):
    return SPACES.sub(" ", unicodedata.normalize("NFKC", text))


def quotas():
    result = {}
    for domain, (budget, languages) in GROUPS.items():
        base, remainder = divmod(budget, len(languages))
        for index, language in enumerate(sorted(languages)):
            result[domain, language] = base + (index < remainder)
    return result


def input_context(selection_path, source_path):
    """Extract only provenance/training metadata, never selection metrics/models."""
    require(file_hash(selection_path) == SELECTION_SHA, "frozen selection SHA mismatch")
    source_sha = file_hash(source_path)
    selection = read_json(selection_path)
    source = read_json(source_path)
    regenerated = source_sha != SOURCE_SHA
    if regenerated:
        validate_regenerated_source(source, source_path, selection)
    require(selection["dataset"]["manifest_sha256"] == SOURCE_SHA, "selection dataset mismatch")
    require(source["schema_version"] == 2 and source["freeze"]["immutable"] is True, "unfrozen source")
    require(
        source["normalization"] == selection["dataset"]["normalization"] == "NFKC_unicode_spaces_v1",
        "normalization mismatch",
    )
    require(source["dataset_id"] == selection["dataset"]["dataset_id"], "dataset identity mismatch")
    require(source["freeze"]["source_revisions"] == selection["dataset"]["source_revisions"], "revision mismatch")
    for split, key in (
        ("train", "train_file_sha256"),
        ("validation", "validation_file_sha256"),
        ("test", "untouched_test_file_sha256"),
    ):
        if split != "train" or not regenerated:
            require(source["splits"][split]["sha256"] == selection["dataset"][key], "split provenance mismatch")
    training = copy.deepcopy(selection["training"])
    require(
        training["assignment_hash"] == training["screen_selection"]["assignment_hash"], "parent assignment mismatch"
    )
    if regenerated:
        training["_recompute_regenerated_parent"] = True
        load_parent(source_path, source, training)
    dataset = copy.deepcopy(selection["dataset"])
    dataset["manifest_sha256"] = source_sha
    dataset["train_file_sha256"] = source["splits"]["train"]["sha256"]
    return (
        source,
        training,
        {
            "selection_sha256": SELECTION_SHA,
            "source_manifest_sha256": source_sha,
            "historical_selection_source_sha256": SOURCE_SHA,
            "dataset": dataset,
            "parent_training_assignment_hash": training["assignment_hash"],
            "validation_assignment_hash": selection["validation"]["assignment_hash"],
        },
    )


def validate_regenerated_source(source, source_path, selection):
    receipt = read_json(Path(source_path).parent / "receipt.json")
    require(
        receipt["status"] == "whole_upstream_source_frozen" and receipt["manifest_sha256"] == file_hash(source_path),
        "unverified regenerated source",
    )
    repair = source["whole_record_regeneration"]
    require(
        repair["policy"] == "whole_upstream_prefix_exact_tail_v2"
        and repair["historical_source_manifest_sha256"] == SOURCE_SHA
        and repair["unchanged_selection_sha256"] == SELECTION_SHA,
        "invalid source regeneration lineage",
    )
    require(
        repair["all_selected_text_verified_against_upstream"] is True
        and repair["deduplication"] == "passed_existing_exact_and_minhash_lsh_train_evaluation_checks",
        "missing whole-record verification",
    )
    require(
        source["dataset_id"] == selection["dataset"]["dataset_id"]
        and source["normalization"] == selection["dataset"]["normalization"]
        and source["freeze"]["source_revisions"] == selection["dataset"]["source_revisions"],
        "regenerated source identity changed",
    )
    for split, field in (("validation", "validation_file_sha256"), ("test", "untouched_test_file_sha256")):
        require(source["splits"][split]["sha256"] == selection["dataset"][field], "evaluation split changed")


def load_parent(source_path, source, training):
    """Stream only the train artifact; reconstruct and verify the original pool."""
    root = Path(source_path).resolve().parent
    path = (root / source["splits"]["train"]["path"]).resolve()
    require(path.is_relative_to(root), "train path escapes frozen directory")
    require(file_hash(path) == source["splits"]["train"]["sha256"], "train file hash mismatch")
    group_info = training["screen_selection"]["groups"]
    limits = {(g["domain"], g["language"]): g["target_bytes"] for g in group_info}
    require(set(limits) == set(quotas()) and len(group_info) == len(limits), "parent strata mismatch")
    inventory = {s["local_path"]: s for s in source["freeze"]["source_files"]}
    used, counts = defaultdict(int), defaultdict(int)
    seen_ids, seen_texts = set(), set()
    documents = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            text = normalize(row["text"])
            length = len(text.encode("utf-8"))
            text_hash = digest(text)
            require(
                text
                and length == row["normalized_utf8_bytes"]
                and len(row["text"].encode("utf-8")) == row["raw_utf8_bytes"],
                "source byte mismatch",
            )
            require(row["id"] not in seen_ids and text_hash not in seen_texts, "duplicate training document")
            seen_ids.add(row["id"])
            seen_texts.add(text_hash)
            require(row["dedup"]["status"] == "accepted_after_exact_and_near_eval_check", "unverified dedup status")
            if "whole_record_regeneration" in source:
                require(row["dedup"].get("truncated_to_quota") is False, "truncated regenerated document")
                require(
                    hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                    == row["whole_record_provenance"]["upstream_text_sha256"],
                    "whole upstream text hash mismatch",
                )
            provenance = row["source"]
            original = inventory[provenance["local_path"]]
            # Stack shards describe per-record licensing; the already-hashed train
            # artifact retains each document's concrete license, not that summary.
            require(
                provenance["source_file_sha256"] == original["sha256"]
                and all(provenance[k] == original[k] for k in ("dataset", "revision", "release_variant", "url"))
                and isinstance(provenance.get("license"), str)
                and bool(provenance["license"].strip()),
                "source provenance mismatch",
            )
            key = row["domain"], row["language"]
            require(key in limits, "unapproved training stratum")
            if used[key] + length > limits[key]:
                continue
            used[key] += length
            counts[key] += 1
            documents.append(
                {
                    "id": row["id"],
                    "domain": key[0],
                    "language": key[1],
                    "train_row_index": index,
                    "normalized_text_hash": text_hash,
                    "normalized_utf8_bytes": length,
                    "source_utf8_bytes": row["raw_utf8_bytes"],
                    "source": provenance,
                    "dedup": row["dedup"],
                }
            )
    if training.pop("_recompute_regenerated_parent", False):
        for group in group_info:
            key = group["domain"], group["language"]
            group["actual_bytes"], group["documents"] = used[key], counts[key]
        assignment = digest([d["normalized_text_hash"] for d in documents])
        training["assignment_hash"] = training["screen_selection"]["assignment_hash"] = assignment
        training["documents"] = len(documents)
        training["normalized_utf8_bytes"] = sum(used.values())
        training["source_utf8_bytes"] = sum(d["source_utf8_bytes"] for d in documents)
    for group in group_info:
        key = group["domain"], group["language"]
        require(used[key] == group["actual_bytes"] and counts[key] == group["documents"], "parent group mismatch")
    require(
        digest([d["normalized_text_hash"] for d in documents]) == training["assignment_hash"],
        "parent assignment hash mismatch",
    )
    require(
        len(documents) == training["documents"]
        and sum(used.values()) == training["normalized_utf8_bytes"]
        and sum(d["source_utf8_bytes"] for d in documents) == training["source_utf8_bytes"],
        "parent accounting mismatch",
    )
    return documents, path


def ranked(document):
    fields = {k: document[k] for k in ("domain", "language", "id", "normalized_text_hash")}
    return {**document, "candidate_rank": digest({"policy": POLICY, "purpose": "candidate", **fields})}


def construct(documents, allocation):
    """Pure byte/metadata function. Output is deterministic even when packing fails."""
    groups = defaultdict(list)
    require(len({d["id"] for d in documents}) == len(documents), "duplicate candidate ID")
    require(len({d["normalized_text_hash"] for d in documents}) == len(documents), "duplicate candidate text")
    for document in documents:
        require(
            type(document["normalized_utf8_bytes"]) is int and document["normalized_utf8_bytes"] > 0,
            "invalid candidate length",
        )
        groups[document["domain"], document["language"]].append(ranked(document))
    require(set(groups) == set(allocation), "candidate strata mismatch")
    chosen, coverage, accounting = {}, {}, []
    for key in sorted(allocation):
        quota = allocation[key]
        candidates = sorted(groups[key], key=lambda d: (d["candidate_rank"], d["id"]))
        fitting = [d for d in candidates if d["normalized_utf8_bytes"] <= quota]
        first = (
            min(fitting, key=lambda d: (d["normalized_utf8_bytes"], d["candidate_rank"], d["id"])) if fitting else None
        )
        kept = [] if first is None else [first]
        used = 0 if first is None else first["normalized_utf8_bytes"]
        for document in candidates:
            if first is not None and document["id"] == first["id"]:
                continue
            if used + document["normalized_utf8_bytes"] <= quota:
                kept.append(document)
                used += document["normalized_utf8_bytes"]
        chosen[key] = kept
        if first is not None:
            coverage[key] = first
        minimum = (quota * 9 + 9) // 10
        accounting.append(
            {
                "domain": key[0],
                "language": key[1],
                "allocated_normalized_bytes": quota,
                "allocated_percent_of_cap": 100 * quota / sum(allocation.values()),
                "minimum_documents": 1,
                "minimum_normalized_bytes": minimum,
                "eligible_documents": len(candidates),
                "eligible_normalized_bytes": sum(d["normalized_utf8_bytes"] for d in candidates),
                "selected_documents": len(kept),
                "actual_normalized_bytes": used,
                "source_utf8_bytes": sum(d["source_utf8_bytes"] for d in kept),
                "unused_quota_bytes": quota - used,
                "skipped_documents": len(candidates) - len(kept),
                "early_exhaustion_allowed": True,
                "exhaustion_reason": (
                    "quota_filled"
                    if used == quota
                    else "no_whole_document_fits"
                    if len(kept) < len(candidates)
                    else "eligible_pool_exhausted"
                ),
                "coverage_document_id": None if first is None else first["id"],
                "coverage_order_rank": digest(
                    {"policy": POLICY, "purpose": "coverage_order", "domain": key[0], "language": key[1]}
                ),
                "order_rule": "R",
                "packing_status": "PASS" if kept and used >= minimum else "FAIL",
                "exact_quota_status": "PASS" if used == quota else "FAIL",
            }
        )
    macro_queues = {}
    for domain in sorted({key[0] for key in coverage}):
        macro_queues[domain] = deque(
            sorted(
                (k for k in coverage if k[0] == domain),
                key=lambda k: (
                    digest({"policy": POLICY, "purpose": "coverage_order", "domain": k[0], "language": k[1]}),
                    k[1],
                ),
            )
        )
    ordered = []
    while any(macro_queues.values()):
        for queue in macro_queues.values():
            if queue:
                key = queue.popleft()
                ordered.append({**coverage[key], "coverage_document": True})
    served = {k: coverage[k]["normalized_utf8_bytes"] if k in coverage else 0 for k in allocation}
    queues = {k: deque(chosen[k][1:] if k in coverage else chosen[k]) for k in allocation}
    while any(queues.values()):
        key = min((k for k in queues if queues[k]), key=lambda k: (Fraction(served[k], allocation[k]), k))
        document = queues[key].popleft()
        ordered.append({**document, "coverage_document": False})
        served[key] += document["normalized_utf8_bytes"]
    for position, document in enumerate(ordered):
        document["position"] = position
    total = sum(d["normalized_utf8_bytes"] for d in ordered)
    for group in accounting:
        entries = [d for d in ordered if (d["domain"], d["language"]) == (group["domain"], group["language"])]
        group["achieved_percent_of_actual"] = 100 * group["actual_normalized_bytes"] / total if total else 0
        group["ordered_document_ids"] = [d["id"] for d in entries]
        group["global_positions"] = [d["position"] for d in entries]
        group["last_global_position"] = entries[-1]["position"] if entries else None
        group["ordered_list_sha256"] = digest(
            [
                {k: d[k] for k in ("id", "normalized_text_hash", "normalized_utf8_bytes", "source_utf8_bytes")}
                for d in entries
            ]
        )
    return {
        "normalized_utf8_bytes": total,
        "source_utf8_bytes": sum(d["source_utf8_bytes"] for d in ordered),
        "selected_documents": len(ordered),
        "ordered_documents": ordered,
        "strata": accounting,
        "ordered_exposure_hash": digest(ordered),
    }


def persist_attempt(output, policy, context, sample, generator):
    """Publish failure evidence without allowing it to masquerade as a frozen exposure."""
    output = Path(output)
    require(not output.exists(), "duplicate exposure attempt: output exists")
    failures = []
    if sample["normalized_utf8_bytes"] != policy["required_normalized_bytes"]:
        failures.append("exact_normalized_byte_budget_not_achieved")
    if any(s["packing_status"] != "PASS" for s in sample["strata"]):
        failures.append("per_stratum_coverage_or_minimum_failed")
    passed = not failures
    output.mkdir(parents=True)
    publish(output / "policy.json", policy)
    policy_hash = file_hash(output / "policy.json")
    body = {
        "exposure_schema_version": 1,
        "status": "exposure_frozen" if passed else "exposure_rejected",
        "usable_for_preflight": passed,
        "policy_sha256": policy_hash,
        "policy": policy,
        "provenance": context,
        "generator": generator,
        "sample": sample,
        "failures": failures,
        "tokenizer_outputs_used_for_sampling": False,
        "token_counts_used_for_sampling": False,
        "flops_used_for_sampling": False,
        "validation_or_test_text_used_for_sampling": False,
        "selection_metrics_used": False,
        "lm_results_used": False,
    }
    manifest = {**body, "content_sha256": digest(body)}
    filename = "exposure.json" if passed else "rejected-exposure.json"
    publish(output / filename, manifest)
    manifest_hash = file_hash(output / filename)
    receipt = {
        "status": manifest["status"],
        "policy_sha256": policy_hash,
        "selection_sha256": context["selection_sha256"],
        "source_manifest_sha256": context["source_manifest_sha256"],
        "artifact": filename,
        "artifact_sha256": manifest_hash,
        "final_exposure_sha256": manifest_hash if passed else None,
        "ordered_exposure_hash": sample["ordered_exposure_hash"],
        "selected_documents": sample["selected_documents"],
        "normalized_utf8_bytes": sample["normalized_utf8_bytes"],
        "source_utf8_bytes": sample["source_utf8_bytes"],
        "required_normalized_bytes": policy["required_normalized_bytes"],
        "failures": failures,
        "tokenizer_preflight_run": False,
        "lm_experiments_run": False,
    }
    publish(output / "receipt.json", receipt)
    # Bind every stratum to the full candidate/final manifest hash, without self-reference.
    publish(
        output / "strata-report.json",
        {"manifest_sha256": manifest_hash, "manifest_status": manifest["status"], "strata": sample["strata"]},
    )
    require(
        read_json(output / filename) == manifest and file_hash(output / filename) == manifest_hash,
        "publication verification failed",
    )
    return receipt


def generate(args):
    require(not Path(args.output).exists(), "duplicate exposure attempt: output exists")
    policy_review_hash = file_hash(args.policy)
    source, training, context = input_context(args.selection, args.source)
    documents, train_path = load_parent(args.source, source, training)
    sample = construct(documents, quotas())
    require(sample == construct(list(reversed(documents)), quotas()), "nondeterministic exposure construction")
    require(
        file_hash(args.selection) == SELECTION_SHA
        and file_hash(args.source) == SOURCE_SHA
        and file_hash(train_path) == source["splits"]["train"]["sha256"]
        and file_hash(args.policy) == policy_review_hash,
        "input changed during generation",
    )
    policy = {
        "schema_version": 1,
        "policy_id": POLICY,
        "review_document_sha256": policy_review_hash,
        "review_document": Path(args.policy).name,
        "required_normalized_bytes": EXACT_BYTES,
        "acceptance_override": "Latest user instruction requires exactly 1000000, superseding draft 950000..1000000 acceptance only; selection/order unchanged.",
        "sampling_rule": "R: shortest coverage, hash-ranked one-pass whole-document packing, coverage macro round robin, rational weighted-byte service",
        "rank_digest": "sha256_sorted_ascii_json_v1",
        "normalization": "NFKC_unicode_spaces_v1",
        "stratum_minimum": "at_least_one_document_and_ceil_0.9_quota",
        "resampling_allowed": False,
        "quotas": [{"domain": k[0], "language": k[1], "normalized_bytes": v} for k, v in sorted(quotas().items())],
    }
    generator = {
        "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        ),
        "sampler_file": "tools/phase_b_exposure.py",
        "sampler_sha256": file_hash(__file__),
        "unicode_database_version": unicodedata.unidata_version,
        "scope": "standalone exposure construction only; no authorization to change frozen LM implementation pin",
    }
    return persist_attempt(args.output, policy, context, sample, generator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    receipt = generate(parser.parse_args())
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "exposure_frozen":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
