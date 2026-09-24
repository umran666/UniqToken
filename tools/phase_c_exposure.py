"""Freeze and independently verify the exact Phase C training exposure.

This module is deliberately tokenizer-blind.  It reads only the frozen training
and validation artifacts, never the test artifact, tokenizer outputs, or LM
results.  A usable manifest is published only after a fresh independent reread.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable

from benchmarks import run_phase_a as stages
from benchmarks import run_research_experiments as research
from tools import phase_b_exposure as base
from tools.phase_b_exact_exposure import MAX_DOCUMENTS, MAX_TARGET, subset_indices


POLICY = "phase_c_whole_document_proportional_exact_tail_v1"
SCHEMA_VERSION = 1
EXACT_BYTES = 300_000_000
SOURCE_SHA256 = "af11dc30fc96c683f92434d27c79fc45dd649dfb3787c6d9b02f81075aedd0c5"
PHASE_B_EXPOSURE_SHA256 = "68ce58aaf90754d8e5bc8b52a8de249fe3cdd4f5876148dd832cd45d61675e2e"
PHASE_B_LEDGER_SHA256 = "0c145550a2b10b266f83d35e4cc5edc335eb401e0e133ebd09d908402539fb33"
PHASE_B_REPORT_SHA256 = "1d43311cd2a0cc4c917bacce019928dc612c79d008819ed53aed4bb2270168ce"
PHASE_C_PROTOCOL_SHA256 = "2aac9a7d6d80f1c26a8716316d2458b4261c9b6219c18a5b78b49a55f0d6f335"


def source_quotas(manifest: dict) -> dict[tuple[str, str], int]:
    groups = manifest["freeze"]["selection"]["groups"]
    result = {
        (row["domain"], row["language"]): row["normalized_utf8_bytes"] for row in groups if row["split"] == "train"
    }
    base.require(
        len(result) == 30 and sum(result.values()) == 500_000_000,
        "source train quotas are not the frozen 30-stratum 500 MB partition",
    )
    return result


def proportional_quotas(source: dict[tuple[str, str], int], total: int = EXACT_BYTES) -> dict[tuple[str, str], int]:
    """Deterministic largest-remainder allocation, ties by stratum key."""
    denominator = sum(source.values())
    floors = {key: value * total // denominator for key, value in source.items()}
    remaining = total - sum(floors.values())
    order = sorted(source, key=lambda key: (-(source[key] * total % denominator), key))
    result = {key: floors[key] + (index < remaining) for index, key in enumerate(order)}
    base.require(sum(result.values()) == total and set(result) == set(source), "quota allocation failed")
    return result


def document(row: dict, index: int) -> dict:
    text = base.normalize(row["text"])
    normalized_bytes = len(text.encode("utf-8"))
    base.require(text and normalized_bytes == row["normalized_utf8_bytes"], "normalized byte mismatch")
    base.require(len(row["text"].encode("utf-8")) == row["raw_utf8_bytes"], "source byte mismatch")
    base.require(row["dedup"]["status"] == "accepted_after_exact_and_near_eval_check", "unverified dedup status")
    base.require(row["dedup"].get("truncated_to_quota") is False, "truncated source record")
    provenance = row.get("whole_record_provenance", {})
    base.require(
        hashlib.sha256(row["text"].encode("utf-8")).hexdigest() == provenance.get("upstream_text_sha256"),
        "whole-record upstream hash mismatch",
    )
    return {
        "id": row["id"],
        "domain": row["domain"],
        "language": row["language"],
        "train_row_index": index,
        "normalized_text_hash": base.digest(text),
        "normalized_utf8_bytes": normalized_bytes,
        "source_utf8_bytes": row["raw_utf8_bytes"],
        "source": row["source"],
        "dedup": row["dedup"],
        "upstream_text_sha256": provenance["upstream_text_sha256"],
    }


def load_training(source_path: Path, manifest: dict) -> tuple[list[dict], Path]:
    root = source_path.resolve().parent
    path = (root / manifest["splits"]["train"]["path"]).resolve()
    base.require(path.is_relative_to(root), "training path escapes source directory")
    base.require(base.file_hash(path) == manifest["splits"]["train"]["sha256"], "training file hash mismatch")
    rows, ids, hashes = [], set(), set()
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            item = document(json.loads(line), index)
            base.require(
                item["id"] not in ids and item["normalized_text_hash"] not in hashes, "duplicate source training record"
            )
            ids.add(item["id"])
            hashes.add(item["normalized_text_hash"])
            rows.append(item)
    base.require(sum(row["normalized_utf8_bytes"] for row in rows) == 500_000_000, "source training total changed")
    return rows, path


def phase_b_ids(exposure: dict) -> set[str]:
    sample = exposure.get("sample", {})
    rows = sample.get("ordered_documents", [])
    base.require(exposure.get("status") == "exposure_frozen" and len(rows) == 349, "invalid Phase B exposure")
    base.require(sample.get("normalized_utf8_bytes") == 1_000_000, "Phase B exposure byte mismatch")
    result = {row["id"] for row in rows}
    base.require(len(result) == len(rows), "duplicate Phase B exposure ID")
    return result


def select_stratum(rows: list[dict], quota: int) -> tuple[list[dict], dict]:
    """Keep an ordered prefix, then solve the bounded exact residual."""
    prefix, candidates, used = [], [], 0
    for row in rows:
        size = row["normalized_utf8_bytes"]
        if quota - used > MAX_TARGET and used + size <= quota:
            prefix.append(row)
            used += size
        else:
            candidates.append(row)
    target = quota - used
    base.require(0 <= target <= MAX_TARGET, "exact-tail target exceeds reviewed bound")
    all_eligible = [row for row in candidates if row["normalized_utf8_bytes"] <= target]
    # The reviewed solver has a fixed memory bound. Source-order truncation is
    # deterministic and affects only the exact-tail search pool, never text.
    eligible = all_eligible[:MAX_DOCUMENTS]
    if target:
        indices, dimensions = subset_indices([row["normalized_utf8_bytes"] for row in eligible], target)
        base.require(indices is not None, "no exact whole-document tail subset")
        selected = prefix + [eligible[index] for index in indices]
    else:
        dimensions, selected = {"candidate_count": 0, "residual_target": 0}, prefix
    selected.sort(key=lambda row: row["train_row_index"])
    base.require(sum(row["normalized_utf8_bytes"] for row in selected) == quota, "exact quota mismatch")
    return selected, {
        "prefix_documents": len(prefix),
        "tail_target": target,
        "eligible_tail_documents": len(all_eligible),
        "searched_tail_documents": len(eligible),
        **dimensions,
    }


def select_all(
    rows: list[dict], quotas: dict[tuple[str, str], int], excluded: set[str]
) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["id"] not in excluded:
            groups[row["domain"], row["language"]].append(row)
    base.require(set(groups) == set(quotas), "source/exposure stratum mismatch")
    selected, reports = [], []
    for key in sorted(quotas):
        chosen, search = select_stratum(groups[key], quotas[key])
        selected.extend(chosen)
        reports.append(
            {
                "domain": key[0],
                "language": key[1],
                "allocated_normalized_bytes": quotas[key],
                "eligible_documents": len(groups[key]),
                "selected_documents": len(chosen),
                "selected_normalized_bytes": sum(row["normalized_utf8_bytes"] for row in chosen),
                "selected_source_bytes": sum(row["source_utf8_bytes"] for row in chosen),
                "packing_status": "PASS",
                "search": search,
            }
        )
    selected.sort(key=lambda row: row["train_row_index"])
    base.require(len({row["id"] for row in selected}) == len(selected), "duplicate selected ID")
    base.require(not excluded.intersection(row["id"] for row in selected), "Phase B document reused")
    base.require(sum(row["normalized_utf8_bytes"] for row in selected) == EXACT_BYTES, "exposure total mismatch")
    return selected, reports


def validation_assignment(source_path: Path, manifest: dict, selected_hashes: set[str]) -> dict:
    """Freeze confirmation-only validation; never resolve or open test."""
    rows, texts = stages._load_rows(source_path, manifest, "validation")
    pairs, _ = stages.partition_validation(rows, texts)
    # partition_validation returns screening; confirmation is its exact complement.
    screening = {base.digest(text) for _, text in pairs}
    confirmation = [(row, text) for row, text in zip(rows, texts) if base.digest(text) not in screening]
    hashes = [base.digest(text) for _, text in confirmation]
    base.require(not selected_hashes.intersection(hashes), "training/confirmation-validation overlap")
    base.require(len(hashes) + len(screening) == len(texts), "validation partition accounting mismatch")
    body = {
        "partition_version": stages.VALIDATION_PARTITION_VERSION,
        "partition": "confirmation",
        "documents": len(confirmation),
        "assignment_hash": research.digest(hashes),
        "ordered_normalized_text_hashes": hashes,
        "normalized_utf8_bytes": sum(len(text.encode("utf-8")) for _, text in confirmation),
        "source_utf8_bytes": sum(row["raw_utf8_bytes"] for row, _ in confirmation),
        "screening_assignment_hash": research.digest([base.digest(text) for _, text in pairs]),
        "test_split_opened": False,
    }
    return {**body, "content_sha256": research.digest(body)}


def exposure_body(selected: list[dict], reports: list[dict], quotas: dict, pins: dict) -> dict:
    fields = (
        "id",
        "domain",
        "language",
        "train_row_index",
        "normalized_text_hash",
        "normalized_utf8_bytes",
        "source_utf8_bytes",
        "upstream_text_sha256",
    )
    ordered = [{**{field: row[field] for field in fields}, "position": index} for index, row in enumerate(selected)]
    body = {
        "exposure_schema_version": SCHEMA_VERSION,
        "status": "phase_c_exposure_frozen",
        "policy": POLICY,
        "tokenizer_blind": True,
        "lm_result_blind": True,
        "normalized_utf8_bytes": EXACT_BYTES,
        "source_utf8_bytes": sum(row["source_utf8_bytes"] for row in ordered),
        "selected_documents": len(ordered),
        "ordered_documents": ordered,
        "ordered_exposure_hash": research.digest(ordered),
        "strata": reports,
        "quotas": [
            {"domain": key[0], "language": key[1], "normalized_utf8_bytes": quotas[key]} for key in sorted(quotas)
        ],
        "provenance": pins,
        "test_split_opened": False,
    }
    return {**body, "content_sha256": research.digest(body)}


def verify_exposure(source_path: Path, manifest: dict, exposure: dict, excluded: set[str]) -> dict:
    """Independent source reread; does not call selection or subset solving."""
    rows, _ = load_training(source_path, manifest)
    originals = {row["id"]: row for row in rows}
    selected = exposure["ordered_documents"]
    base.require(len(selected) == exposure["selected_documents"], "selected count mismatch")
    base.require(
        [row["train_row_index"] for row in selected] == sorted(row["train_row_index"] for row in selected),
        "global source order changed",
    )
    base.require(not excluded.intersection(row["id"] for row in selected), "Phase B membership overlap")
    totals = defaultdict(int)
    for position, row in enumerate(selected):
        original = originals.get(row["id"])
        base.require(original is not None, "selected record absent from source")
        expected = {field: original[field] for field in row if field != "position"}
        base.require(row == {**expected, "position": position}, "selected record altered")
        totals[row["domain"], row["language"]] += row["normalized_utf8_bytes"]
    quotas = {(row["domain"], row["language"]): row["normalized_utf8_bytes"] for row in exposure["quotas"]}
    base.require(
        dict(totals) == quotas and sum(totals.values()) == EXACT_BYTES, "independent quota verification failed"
    )
    base.require(
        sum(row["source_utf8_bytes"] for row in selected) == exposure["source_utf8_bytes"],
        "independent source-byte verification failed",
    )
    base.require(research.digest(selected) == exposure["ordered_exposure_hash"], "ordered exposure hash mismatch")
    body = dict(exposure)
    content = body.pop("content_sha256")
    base.require(research.digest(body) == content, "exposure content hash mismatch")
    return {
        "status": "PASS",
        "fresh_source_reread": True,
        "membership": "PASS",
        "whole_documents": "PASS",
        "phase_b_exclusion": "PASS",
        "all_30_exact_quotas": "PASS",
        "global_source_order": "PASS",
        "normalized_utf8_bytes": EXACT_BYTES,
        "selected_documents": len(selected),
        "test_split_opened": False,
    }


def check_pin(path: Path, expected: str, label: str) -> None:
    base.require(base.file_hash(path) == expected, f"{label} SHA-256 mismatch")


def freeze(args) -> dict:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        source_path, phase_b_path = args.source.resolve(), args.phase_b_exposure.resolve()
        protocol, report, ledger = args.protocol.resolve(), args.phase_b_report.resolve(), args.phase_b_ledger.resolve()
        check_pin(source_path, SOURCE_SHA256, "source manifest")
        check_pin(phase_b_path, PHASE_B_EXPOSURE_SHA256, "Phase B exposure")
        check_pin(protocol, PHASE_C_PROTOCOL_SHA256, "Phase C protocol")
        check_pin(report, PHASE_B_REPORT_SHA256, "Phase B report")
        check_pin(ledger, PHASE_B_LEDGER_SHA256, "Phase B ledger")
        manifest, phase_b = base.read_json(source_path), base.read_json(phase_b_path)
        base.require(
            manifest["schema_version"] == 2 and manifest["freeze"]["immutable"] is True, "source manifest is not frozen"
        )
        source = source_quotas(manifest)
        quotas = proportional_quotas(source)
        excluded = phase_b_ids(phase_b)
        rows, train_path = load_training(source_path, manifest)
        selected, reports = select_all(rows, quotas, excluded)
        pins = {
            "source_manifest_sha256": SOURCE_SHA256,
            "source_train_sha256": manifest["splits"]["train"]["sha256"],
            "phase_b_exposure_sha256": PHASE_B_EXPOSURE_SHA256,
            "phase_b_ledger_sha256": PHASE_B_LEDGER_SHA256,
            "phase_b_report_sha256": PHASE_B_REPORT_SHA256,
            "phase_c_protocol_sha256": PHASE_C_PROTOCOL_SHA256,
            "generator_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "generator_sha256": base.file_hash(__file__),
        }
        exposure = exposure_body(selected, reports, quotas, pins)
        verification = verify_exposure(source_path, manifest, exposure, excluded)
        validation = validation_assignment(source_path, manifest, {row["normalized_text_hash"] for row in selected})
        # Recheck immutable inputs after all expensive work and before publication.
        check_pin(source_path, SOURCE_SHA256, "source manifest")
        check_pin(train_path, manifest["splits"]["train"]["sha256"], "source train")
        check_pin(phase_b_path, PHASE_B_EXPOSURE_SHA256, "Phase B exposure")
        frozen = output / "frozen"
        frozen.mkdir()
        base.publish(frozen / "exposure.json", exposure)
        base.publish(frozen / "verification.json", verification)
        base.publish(frozen / "confirmation-validation.json", validation)
        receipt = {
            "status": "phase_c_exposure_frozen",
            "authorized_for_phase_c_training": False,
            "exposure_sha256": base.file_hash(frozen / "exposure.json"),
            "verification_sha256": base.file_hash(frozen / "verification.json"),
            "confirmation_validation_sha256": base.file_hash(frozen / "confirmation-validation.json"),
            "normalized_utf8_bytes": EXACT_BYTES,
            "selected_documents": len(selected),
            "test_split_opened": False,
        }
        base.publish(output / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        base.publish(
            output / "rejection-receipt.json",
            {
                "status": "phase_c_exposure_rejected",
                "error": f"{type(error).__name__}: {error}",
                "authorized_for_phase_c_training": False,
                "test_split_opened": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--phase-b-exposure", type=Path, required=True)
    parser.add_argument("--phase-b-ledger", type=Path, required=True)
    parser.add_argument("--phase-b-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args), indent=2))


if __name__ == "__main__":
    main()
