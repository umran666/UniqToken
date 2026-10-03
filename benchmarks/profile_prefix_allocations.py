"""Untimed prefix-only allocation attribution for the 49 segmentation fixtures."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from benchmarks import compare_residual_native as c
from benchmarks import native_performance_report as r
from benchmarks import run_research_experiments as h


def run(source, native, build_commit, output):
    h.require(not output.exists(), "allocation output must be new")
    identity = h.runtime_identity()
    h.require(not identity["working_tree_dirty"], "commit attribution protocol before running")
    payload, _ = r.verified(source)
    output.mkdir(parents=True)
    records = []
    for row in payload["records"]:
        if row["surface"] != "segmentation" or row["round"] != 0:
            continue
        index = len(records)
        spec = {field: row[field] for field in (*r.FIELDS, "model") if field in row}
        spec.update(native=str(native.resolve()), variant="prefix")
        request, response = output / f"{index}-request.json", output / f"{index}.json"
        h.write_new_json(request, spec)
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.compare_residual_native",
                "--worker",
                str(request),
                "--operation",
                "allocations",
                "--output",
                str(response),
            ],
            capture_output=True,
            timeout=300,
            env={**os.environ, "RAYON_NUM_THREADS": "1", "PYTHONHASHSEED": "0"},
        )
        h.require(process.returncode == 0, process.stderr.decode("utf-8", errors="replace"))
        result = h.read_json(response)
        reference = row["measurements"]["prefix"]
        h.require(result["output_sha256"] == reference["output_sha256"], "prefix allocation stream mismatch")
        h.require(
            result["path_score_float_hex"] == reference["path_score_float_hex"], "prefix allocation score mismatch"
        )
        for field in ("fixture_sha256", "model_sha256", "normalized_input_bytes", "tokens"):
            h.require(result[field] == row[field], "prefix allocation fixture/count mismatch")
        records.append(result)
        print(f"{index} {row['vocab_budget']} {row['length']} {row['script']}", flush=True)
    h.require(len(records) == 49, "incomplete prefix allocation matrix")
    h.require(h.runtime_identity() == identity, "allocation attribution source/runtime changed")
    h.write_new_json(
        output / "results.json",
        {
            "identity": identity,
            "source_results_sha256": h.file_hash(source),
            "build_source_commit": build_commit,
            "records": records,
            "method": "fresh untimed System allocator workers, same release/feature flags as paired baseline and compact allocation workers",
        },
    )
    h.write_new_json(output / "manifest.json", {"status": "complete", "artifacts": h.artifact_hashes(output)})


def verified(path, source, payload):
    root = path.resolve().parent
    receipt = h.read_json(root / "manifest.json")
    h.require(receipt["status"] == "complete" and path.name in receipt["artifacts"], "incomplete prefix receipt")
    for relative, digest in receipt["artifacts"].items():
        target = (root / relative).resolve()
        h.require(target.is_relative_to(root) and target.is_file(), "prefix receipt path escape/missing file")
        h.require(h.file_hash(target) == digest, "prefix receipt mismatch")
    supplementary = h.read_json(path)
    h.require(not supplementary["identity"]["working_tree_dirty"], "uncommitted prefix attribution")
    h.require(supplementary["source_results_sha256"] == h.file_hash(source), "prefix attribution source mismatch")
    expected = {r.key(row): row for row in payload["records"] if row["surface"] == "segmentation" and row["round"] == 0}
    observed = {}
    binaries = set()
    for index, row in enumerate(supplementary["records"]):
        cell = r.key(row)
        h.require(
            cell in expected and cell not in observed and row["variant"] == "prefix",
            "unexpected/duplicate prefix allocation",
        )
        h.require(h.read_json(root / f"{index}.json") == row, "prefix response mismatch")
        reference = expected[cell]
        for field in ("fixture_sha256", "model_sha256", "tokens", "normalized_input_bytes"):
            h.require(row[field] == reference[field], "prefix allocation fixture/count mismatch")
        for field in ("output_sha256", "path_score_float_hex"):
            h.require(
                row[field] == reference["measurements"]["prefix"][field], "prefix allocation stream/score mismatch"
            )
        binaries.add(row["native_sha256"])
        observed[cell] = row
    h.require(
        set(observed) == set(expected) and len(binaries) == 1, "incomplete prefix allocation matrix/binary mismatch"
    )
    return supplementary, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--build-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.native, args.build_commit, args.output)


if __name__ == "__main__":
    main()
