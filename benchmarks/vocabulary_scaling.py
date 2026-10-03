"""Isolated tokenizer-only vocabulary scaling, with receipts for failed conditions."""

from __future__ import annotations

import argparse
import ctypes
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from benchmarks import byte_fallback_analysis as source
from benchmarks import run_research_experiments as h
from benchmarks.analyze_tokenizer_failures import csv_text
from benchmarks.token_density import density_metrics, project_record, validate_density_ledger
from benchmarks.ledger import SCHEMA_VERSION
from benchmarks.tokenizer_failure_metrics import analyze_condition

COHORT = ("sp_unigram", "boundary_bpe", "uniq_superbpe_r64")
BUDGETS = (8192, 16384, 32768, 65536, 131072)
RESERVE = 64


def peak_rss_bytes():
    """Process high-water RSS, including imports and input loading, not Python heap."""
    if os.name != "nt":
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak",
                "current",
                "peak_paged",
                "paged",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.peak)


def train_condition(name, texts, budget, directory):
    h.require(name in COHORT, "unsupported scaling tokenizer")
    if name != "uniq_superbpe_r64":
        tok, _ = h.train_tokenizer(name, texts, budget, directory)
        return tok, {
            "reserved_slots": 0,
            "learned_merges": tok.merges,
            "bpe_merge_rules": len(tok.model.merges) if name == "boundary_bpe" else None,
        }
    base = source.train_base(texts, budget - RESERVE)
    chunks = [c for t in texts for c in [*base.pre_tokenizer.pre_tokenize(base.normalizer.normalize(t)), h.SPECIALS[3]]]
    tok, _, merges = source.make_condition(base, chunks, budget, RESERVE, "baseline", 0, 0)
    tok.save(directory, save_binary=False)
    return h.ResearchTokenizer("uniq_superbpe", tok, tok.model.token_to_id, len(merges)), {
        "reserved_slots": RESERVE,
        "learned_merges": len(merges),
        "merge_records": merges,
    }


def worker(request, name, budget, output):
    specification = h.read_json(request)
    identity = h.runtime_identity()
    h.require(identity == specification["identity"], "worker source/runtime mismatch")
    pairs = {split: [(r["row"], r["text"]) for r in rows] for split, rows in specification["documents"].items()}
    output.mkdir(parents=True)
    started, cpu_started = time.perf_counter(), time.process_time()
    result = {
        "tokenizer": name,
        "vocab_budget": budget,
        "request_sha256": h.file_hash(request),
        "identity": identity,
        "status": "failed_training",
        "records": [],
        "process_peak_before_training_bytes": peak_rss_bytes(),
    }
    stage = "training"
    try:
        tok, allocation = train_condition(name, [t for _, t in pairs["train"]], budget, output / "model")
        result.update(
            training_seconds=time.perf_counter() - started,
            training_cpu_seconds=time.process_time() - cpu_started,
            training_process_peak_rss_bytes=peak_rss_bytes(),
            allocation=allocation,
        )
        stage = "validation"
        h.validate_tokenizer(tok, budget)
        result["actual_vocab_size"] = len(tok.vocab)
        records = analyze_condition(tok, pairs)
        for row in records:
            row["tokenizer"] = name
            row.update(
                density_metrics(
                    row["tokens"],
                    row["normalized_utf8_bytes"],
                    row["unicode_characters"],
                    row["token_length_bytes_histogram"],
                )
            )
        result.update(status="complete", records=records, model_hashes=h.artifact_hashes(output / "model"))
    except Exception as error:
        result.update(
            status=f"failed_{stage}",
            error_type=type(error).__name__,
            error=str(error),
            elapsed_until_failure_seconds=time.perf_counter() - started,
            process_peak_at_failure_bytes=peak_rss_bytes(),
        )
        if stage == "training" and any(
            term in str(error).lower() for term in ("vocabulary size too high", "exact vocabulary", "vocabulary budget")
        ):
            result["status"] = "budget_not_reached"
    h.require(h.runtime_identity() == identity, "worker changed source/runtime during measurement")
    h.write_new_json(output / "response.json", result)
    return result


def validate_results(payload):
    h.require(payload.get("schema_version") == 1, "unsupported scaling schema")
    config = payload["configuration"]
    h.require(config["cohort"] == list(COHORT) and config["budgets"] == list(BUDGETS), "predeclared matrix changed")
    expected = {(n, b) for n in config["cohort"] for b in config["budgets"]}
    seen = set()
    for condition in payload["conditions"]:
        key = (condition["tokenizer"], condition["vocab_budget"])
        h.require(key in expected and key not in seen, "unexpected/duplicate scaling condition")
        seen.add(key)
        h.require(
            condition["status"]
            in (
                "complete",
                "budget_not_reached",
                "failed_training",
                "failed_validation",
                "worker_failed",
                "resource_limit",
            ),
            "invalid status",
        )
        if condition["status"] != "complete":
            h.require(not condition.get("records"), "failed conditions must not emit favorable metrics")
            continue
        h.require(condition["actual_vocab_size"] == condition["vocab_budget"], "exact vocabulary mismatch")
        h.require(condition.get("model_hashes"), "validated model hashes required")
        h.require(condition["identity"] == payload["identity"], "condition measurement identity mismatch")
        for field in ("training_seconds", "training_cpu_seconds", "training_process_peak_rss_bytes"):
            h.require(math.isfinite(condition[field]) and condition[field] > 0, "invalid training cost")
        h.require(condition["records"], "complete condition has no diagnostics")
        for row in condition["records"]:
            h.require(
                row["tokenizer"] == condition["tokenizer"] and row["vocab_budget"] == condition["vocab_budget"],
                "condition metric identity mismatch",
            )
            h.require(row["split"] in ("train", "validation"), "test access forbidden")
            density_metrics(
                row["tokens"],
                row["normalized_utf8_bytes"],
                row["unicode_characters"],
                row["token_length_bytes_histogram"],
            )
        validate_density_ledger(
            {
                "density_schema_version": 1,
                "metadata": {
                    "ledger_schema_version": SCHEMA_VERSION,
                    **payload["identity"],
                    "data_split": "document_disjoint_frozen_diagnostic",
                },
                "records": [project_record(row) for row in condition["records"]],
            }
        )
    h.require(seen == expected, "incomplete matrix without failure receipts")
    return payload


def run(dataset, output, timeout=900):
    h.require(not output.exists(), "output must be new")
    h.require(not output.resolve().is_relative_to(dataset.resolve().parent), "output overlaps frozen source")
    h.require(timeout > 0, "positive per-condition timeout required")
    identity = h.runtime_identity()
    h.require(not identity["working_tree_dirty"], "commit protocol before running")
    pairs, provenance = source.assignments(dataset)
    output.mkdir(parents=True)
    request = output / "worker-input.json"
    documents = {
        split: [
            {
                "row": {
                    **{key: row[key] for key in ("id", "language", "domain")},
                    "raw_utf8_bytes": len(text.encode("utf-8")),
                },
                "text": text,
            }
            for row, text in records
        ]
        for split, records in pairs.items()
    }
    h.write_new_json(request, {"identity": identity, "documents": documents})
    payload = {
        "schema_version": 1,
        "identity": identity,
        "assignments": provenance,
        "configuration": {
            "cohort": list(COHORT),
            "budgets": list(BUDGETS),
            "merge_reserve": RESERVE,
            "worker_timeout_seconds": timeout,
            "randomness": "none",
            "seed": None,
            "rayon_threads": 1,
            "python_hash_seed": "0",
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "memory_method": "fresh-process high-water RSS through training and artifact serialization; includes imports and normalized excerpt inputs",
        "time_method": "one wall/CPU observation per fresh worker, training plus serialization; excludes imports and evaluation",
        "conditions": [],
    }
    env = {**os.environ, "RAYON_NUM_THREADS": "1", "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"}
    for budget in BUDGETS:
        for name in COHORT:
            print(f"Running {name} at {budget}", flush=True)
            directory = output / f"{name}-{budget}"
            log = output / f"{name}-{budget}.log"
            command = [
                sys.executable,
                "-m",
                "benchmarks.vocabulary_scaling",
                "--worker",
                str(request),
                "--tokenizer",
                name,
                "--budget",
                str(budget),
                "--output",
                str(directory),
            ]
            with log.open("wb") as stream:
                try:
                    process = subprocess.run(
                        command, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=timeout, check=False
                    )
                    response = directory / "response.json"
                    if process.returncode == 0 and response.is_file():
                        condition = h.read_json(response)
                    else:
                        condition = {"status": "worker_failed", "returncode": process.returncode, "records": []}
                except subprocess.TimeoutExpired:
                    condition = {"status": "resource_limit", "timeout_seconds": timeout, "records": []}
            condition.update(
                tokenizer=name, vocab_budget=budget, log_sha256=h.file_hash(log), artifact_directory=directory.name
            )
            payload["conditions"].append(condition)
            print(f"  {condition['status']}", flush=True)
    validate_results(payload)
    h.require(h.runtime_identity() == identity, "source/runtime changed during matrix")
    h.write_new_json(output / "results.json", payload)
    rows = [
        {
            **r,
            "training_seconds": c["training_seconds"],
            "training_peak_rss_bytes": c["training_process_peak_rss_bytes"],
        }
        for c in payload["conditions"]
        if c["status"] == "complete"
        for r in c["records"]
    ]
    (output / "metrics.csv").write_text(csv_text(rows), encoding="utf-8")
    from benchmarks.scaling_plots import plot_scaling, scaling_report

    plot_scaling(payload, output)
    (output / "REPORT.md").write_text(scaling_report(payload), encoding="utf-8")
    # The private worker request contains corpus excerpts; retain its hash, not its contents.
    payload_hashes = h.artifact_hashes(output)
    payload_hashes.pop("worker-input.json")
    h.write_new_json(
        output / "manifest.json",
        {"status": "complete", "artifacts": payload_hashes, "worker_input_sha256": h.file_hash(request)},
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--tokenizer", choices=COHORT)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.tokenizer, args.budget, args.output)
    else:
        h.require(args.dataset is not None, "--dataset is required")
        run(args.dataset, args.output, args.timeout)


if __name__ == "__main__":
    main()
