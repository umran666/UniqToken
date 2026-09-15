"""Offline regeneration gate for whole upstream records. Never trains a model.

Candidate order is the historical train order. Restore complete upstream records,
then remove an exact subset of the resulting byte excess within each stratum.
Only a verified, deduplicated exact selection can publish a new source manifest.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from tools import phase_b_exposure as base
from tools.phase_b_exact_exposure import subset_indices, MAX_TARGET, MAX_DOCUMENTS

POLICY = "whole_upstream_prefix_exact_tail_v2"
TAIL_CANDIDATES = 2048


def restored_record(row, text):
    base.require(isinstance(text, str) and bool(text.strip()), "invalid upstream text")
    truncated = row["dedup"].get("truncated_to_quota")
    base.require(type(truncated) is bool, "missing original truncation status")
    base.require(text.startswith(row["text"]) if truncated else text == row["text"],
                 "upstream text does not match historical record")
    new = copy.deepcopy(row)
    new["text"] = text
    new["raw_utf8_bytes"] = len(text.encode("utf-8"))
    new["normalized_utf8_bytes"] = len(base.normalize(text).encode("utf-8"))
    new["dedup"]["truncated_to_quota"] = False
    new["dedup"]["status"] = "pending_regenerated_corpus_dedup"
    new["whole_record_provenance"] = {
        "original_training_id": row["id"], "original_was_truncated": truncated,
        "upstream_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "original_text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
    }
    return new


def select_exact(rows, quota):
    """Remove exact excess; preserve retained input order, without slicing text."""
    excess = sum(r["normalized_utf8_bytes"] for r in rows) - quota
    base.require(excess >= 0, "whole-record candidate pool below quota")
    if not excess:
        return rows, []
    base.require(excess <= MAX_TARGET, "exact removal search exceeds declared bound")
    # Earlier documents have deterministic priority for retention.
    eligible = [i for i in reversed(range(len(rows))) if rows[i]["normalized_utf8_bytes"] <= excess]
    eligible = eligible[:MAX_DOCUMENTS]
    indices, _ = subset_indices([rows[i]["normalized_utf8_bytes"] for i in eligible], excess)
    base.require(indices is not None, "no exact whole-record subset in restored candidate pool")
    removed = {eligible[i] for i in indices}
    result = [r for i, r in enumerate(rows) if i not in removed]
    base.require(sum(r["normalized_utf8_bytes"] for r in result) == quota, "exact quota verification failed")
    return result, [rows[i]["id"] for i in sorted(removed)]


def upstream_rows(path, wanted):
    last = max(wanted)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index in wanted:
                    yield index, json.loads(line)["text"]
                if index >= last:
                    break
    else:
        import pyarrow.parquet as pq
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["content"], use_threads=False):
            for text in batch.column(0).to_pylist():
                if offset in wanted:
                    yield offset, text
                offset += 1
                if offset > last:
                    return


def select_whole_tail(rows, quota, additions):
    """Retain a whole-record prefix, solve its bounded residual over whole records."""
    prefix, candidates, used = [], [], 0
    for row in rows:
        size = row["normalized_utf8_bytes"]
        if quota - used > MAX_TARGET and used + size <= quota:
            prefix.append(row)
            used += size
        else:
            candidates.append(row)
    target = quota - used
    base.require(0 <= target <= MAX_TARGET, "whole-record tail target exceeds search bound")
    if not target:
        return prefix, {"tail_target": 0, "candidate_count": 0}
    seen = {r["whole_record_provenance"]["upstream_text_sha256"] for r in prefix}
    eligible = []

    def consider(row):
        digest = row["whole_record_provenance"]["upstream_text_sha256"]
        size = row["normalized_utf8_bytes"]
        if digest not in seen and 0 < size <= target:
            seen.add(digest)
            eligible.append(row)

    for row in candidates:
        consider(row)
    for row in additions():
        if len(eligible) >= TAIL_CANDIDATES:
            break
        consider(row)
    base.require(len(eligible) <= MAX_DOCUMENTS, "tail candidate bound exceeded")
    indices, dimensions = subset_indices([r["normalized_utf8_bytes"] for r in eligible], target)
    base.require(indices is not None, "no exact whole-record tail subset in pinned sources")
    selected = prefix + [eligible[i] for i in indices]
    base.require(sum(r["normalized_utf8_bytes"] for r in selected) == quota, "tail quota mismatch")
    return selected, {"tail_target": target, **dimensions,
                      "candidate_ids": [r["id"] for r in eligible]}


def regenerate(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base.publish(output / "initial-rejection-receipt.json", {
        "status": "rejected_until_complete", "policy": POLICY,
        "reason": "Only manifest.json plus a passing receipt constitutes a completed freeze.",
        "authorized_for_lm_execution": False,
    })
    try:
        from benchmarks import freeze_phase_a_dataset as freezer
        old_path = args.source.resolve()
        base.require(base.file_hash(old_path) == base.SOURCE_SHA, "historical manifest hash mismatch")
        base.require(base.file_hash(args.selection) == base.SELECTION_SHA, "selection changed")
        old = base.read_json(old_path)
        root = old_path.parent
        pins = {str(old_path): base.SOURCE_SHA, str(args.selection.resolve()): base.SELECTION_SHA}
        splits = {}
        for split, spec in old["splits"].items():
            path = (root / spec["path"]).resolve()
            base.require(path.is_relative_to(root) and base.file_hash(path) == spec["sha256"], "old split hash mismatch")
            pins[str(path)] = spec["sha256"]
            splits[split] = path
        rows = list(freezer.records_from_jsonl(splits["train"]))
        inventory = {s["local_path"]: s for s in old["freeze"]["source_files"]}
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[row["source"]["local_path"]].append(i)
        restored = [None] * len(rows)
        for number, (local, indices) in enumerate(groups.items(), 1):
            path = (root / local).resolve()
            expected = inventory[local]["sha256"]
            base.require(path.is_relative_to(root) and base.file_hash(path) == expected, "upstream file hash mismatch")
            pins[str(path)] = expected
            wanted = {}
            for i in indices:
                row = rows[i]
                base.require(row["source"]["source_file_sha256"] == expected, "record/source hash mismatch")
                wanted.setdefault(row["source"]["source_record"], []).append(i)
            for record_index, text in upstream_rows(path, wanted):
                for i in wanted[record_index]:
                    restored[i] = restored_record(rows[i], text)
            base.require(all(restored[i] is not None for i in indices), "missing upstream record")
            print(f"Verified upstream source {number}/{len(groups)}", flush=True)
        quotas = {(g["domain"], g["language"]): g["normalized_utf8_bytes"]
                  for g in old["freeze"]["selection"]["groups"] if g["split"] == "train"}
        selected, reports = [], []
        by_stratum = defaultdict(list)
        for row in restored:
            by_stratum[row["domain"], row["language"]].append(row)
        base.require(set(by_stratum) == set(quotas) and len(quotas) == 30, "stratum mismatch")
        for key, group in by_stratum.items():
            def additions():
                known = {(r["source"]["local_path"], r["source"]["source_record"]) for r in group}
                paths = list(dict.fromkeys(r["source"]["local_path"] for r in group))
                # Same cached shard order and upstream record order as the freezer.
                for local in paths:
                    last_historical = max(index for path, index in known if path == local)
                    source = {**inventory[local], "local_path_abs": str(root / local)}
                    if source["dataset"] == freezer.MADLAD:
                        iterator = freezer.madlad_records(source)
                    else:
                        license_entry = next(s for s in inventory.values() if s["local_path"].endswith("/licenses.json"))
                        license_path = root / license_entry["local_path"]
                        base.require(base.file_hash(license_path) == license_entry["sha256"], "license allowlist hash mismatch")
                        allowlist = freezer.stack_license_allowlist({"local_path_abs": str(license_path)})
                        iterator = freezer.stack_records(source, allowlist)
                    for text, provenance, index in iterator:
                        if index <= last_historical or not text.strip() or freezer.RESERVED_CORPUS_TEXT.search(text):
                            continue
                        row = freezer.source_record(text, identifier=f"whole:{source['dataset']}:{source['remote_path']}:{index}",
                            language=key[1], domain=key[0], source=provenance, source_record=index, truncated=False)
                        row = restored_record(row, text)
                        row["whole_record_provenance"]["original_training_id"] = None
                        yield row
            chosen, search = select_whole_tail(group, quotas[key], additions)
            base.require(all(not freezer.RESERVED_CORPUS_TEXT.search(r["text"]) for r in chosen),
                         "restored source contains reserved corpus text")
            selected.extend(chosen)
            reports.append({"domain": key[0], "language": key[1], "quota": quotas[key],
                            "selected_documents": len(chosen), "search": search})
            print(f"Exact whole-record quota packed: {key[0]}/{key[1]}", flush=True)
        base.publish(output / "packing.json", {"policy": POLICY, "strata": reports,
                     "restored_records": [r["whole_record_provenance"] | {"id": r["id"]}
                                          for r in restored if r["whole_record_provenance"]["original_was_truncated"]]})
        train = output / "train.jsonl"
        with train.open("xb") as stream:
            for row in selected:
                stream.write(freezer.json_line(row))
        print("Checking exact and near duplicates in regenerated train and against evaluation", flush=True)
        freezer.validate_dedup(train, [splits["validation"], splits["test"]])
        # Publication follows independent reread/validation; candidate text is never changed.
        verified = output / "verified-train.jsonl"
        with verified.open("xb") as stream:
            for row in freezer.records_from_jsonl(train):
                base.require(row["dedup"]["truncated_to_quota"] is False, "truncated candidate")
                row["dedup"]["status"] = "accepted_after_exact_and_near_eval_check"
                stream.write(freezer.json_line(row))
        for split in ("validation", "test"):
            shutil.copyfile(splits[split], output / f"{split}.jsonl")
        for path, expected in pins.items():
            base.require(base.file_hash(path) == expected, "input changed during freeze")
        # Hard links preserve pinned local bytes without another multi-GB download.
        for local, entry in inventory.items():
            original = (root / local).resolve()
            destination = output / local
            base.require(original.is_relative_to(root) and destination.resolve().is_relative_to(output),
                         "source inventory path escapes corpus")
            base.require(base.file_hash(original) == entry["sha256"], "inventory hash mismatch")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(original, destination)
        new_splits = {"train": verified, "validation": output / "validation.jsonl", "test": output / "test.jsonl"}
        manifest = freezer.build_manifest(output, old["freeze"]["source_files"], new_splits,
                                          revisions=old["freeze"]["source_revisions"])
        manifest["whole_record_regeneration"] = {
            "policy": POLICY, "historical_source_manifest_sha256": base.SOURCE_SHA,
            "unchanged_selection_sha256": base.SELECTION_SHA,
            "generator_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "generator_sha256": base.file_hash(__file__), "input_hashes": pins,
            "packing_sha256": base.file_hash(output / "packing.json"),
            "all_selected_text_verified_against_upstream": True,
            "deduplication": "passed_existing_exact_and_minhash_lsh_train_evaluation_checks",
        }
        base.publish(output / "manifest.json", manifest)
        base.publish(output / "receipt.json", {"status": "whole_upstream_source_frozen",
                     "manifest_sha256": base.file_hash(output / "manifest.json"),
                     "authorized_for_lm_execution": False})
    except BaseException as error:
        base.publish(output / "rejection-receipt.json", {"status": "source_regeneration_rejected",
                     "error": f"{type(error).__name__}: {error}", "authorized_for_lm_execution": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    regenerate(parser.parse_args())


if __name__ == "__main__":
    main()
