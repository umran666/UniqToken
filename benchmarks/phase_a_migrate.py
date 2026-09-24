"""Explicit, fail-closed migration of saved A-SCREEN tokenizer conditions."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from benchmarks import run_phase_a as stages
from benchmarks import run_research_experiments as research

VERSION = 1
POLICY = "screen_sp_boundary_metric_accessor_v1"
TRAINED_COMMIT = "996536ba0c3b27560bc6c6b4abfabb11d0f9ab74"
SUPPORTED = {"sp_unigram", "sp_bpe", "boundary_bpe"}
# Exact reviewed blobs, not arbitrary edits to an approved filename/function.
APPROVED_BLOBS = {
    "benchmarks/run_research_experiments.py": "2cc2adcde9689c2fc9a23751dfd1495de14d2223",
    "benchmarks/run_phase_a.py": "72e49b6bdadc4460e613d9594103979b2804204e",
}
SUPPORT_FILES = {
    "benchmarks/phase_a_migrate.py",
    "tests/test_phase_a_migration.py",
    "tests/test_research_experiments.py",
    "benchmarks/RESEARCH_PROTOCOL.md",
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=research.ROOT)


def tree(commit):
    entries = {}
    for line in git("ls-tree", "-r", "-z", commit).split(b"\0"):
        if line:
            info, path = line.split(b"\t")
            entries[path.decode()] = tuple(info.decode().split())
    return entries


def source_hash(commit):
    sources = {}
    for path in tree(commit):
        parts = Path(path).parts
        suffix = {"uniqtoken": ".py", "benchmarks": ".py", "crates": ".rs"}.get(parts[0])
        if (suffix and path.endswith(suffix) and not {"target", "legacy"}.intersection(parts)) or path in (
            "pyproject.toml",
            "crates/uniqtoken_core/Cargo.toml",
            "crates/uniqtoken_core/Cargo.lock",
        ):
            sources[path] = hashlib.sha256(git("show", f"{commit}:{path}")).hexdigest()
    return research.digest(sources)


def approve_code_change(trained, revalidated):
    research.require(trained == TRAINED_COMMIT, "unapproved trained commit")
    research.require(bool(re.fullmatch(r"[0-9a-f]{40}", revalidated)), "full revalidated commit required")
    before, after = tree(trained), tree(revalidated)
    for path in before.keys() | after.keys():
        if before.get(path) == after.get(path):
            continue
        if path in SUPPORT_FILES:
            research.require(path in after and after[path][0] == "100644", "invalid migration support file")
            continue
        research.require(
            path in APPROVED_BLOBS and after.get(path) == ("100644", "blob", APPROVED_BLOBS[path]),
            f"unapproved behavior/code change: {path}",
        )
    for path, blob in APPROVED_BLOBS.items():
        research.require(after.get(path) == ("100644", "blob", blob), f"approved implementation required: {path}")
    return {"policy": POLICY, "trained_commit": trained, "revalidated_commit": revalidated}


def validate_plans(old, current):
    research.require(
        old.get("stage") == current.get("stage") == "A-SCREEN", "migration only supports A-SCREEN; no Phase B/C"
    )
    research.require(old.get("result_label") == "SCREENING", "screening label required")
    research.require(
        {k: v for k, v in old.items() if k != "identity"} == {k: v for k, v in current.items() if k != "identity"},
        "dataset/configuration/assignment plan mismatch",
    )
    trained, active = old["identity"], current["identity"]
    research.require(
        not trained["working_tree_dirty"] and not active["working_tree_dirty"], "clean committed code required"
    )
    research.require(
        trained.get("extension_hash") and trained["extension_hash"] == active.get("extension_hash"),
        "extension mismatch",
    )
    research.require(trained.get("versions") == active.get("versions"), "runtime versions mismatch")
    proof = approve_code_change(trained["commit_hash"], active["commit_hash"])
    research.require(trained["source_hash"] == source_hash(trained["commit_hash"]), "original source hash mismatch")
    research.require(active["source_hash"] == source_hash(active["commit_hash"]), "current source hash mismatch")
    return proof


def validate_migrated(row, expected, identity, source, training, validation, output):
    m = row["migration"]
    research.require(
        m.get("schema_version") == VERSION and m.get("status") == "condition_revalidated",
        "invalid migration status/version",
    )
    research.require(m.get("revalidated_identity") == identity, "stale revalidation identity")
    timestamp = datetime.fromisoformat(m["revalidation_timestamp"])
    research.require(timestamp.tzinfo is not None, "timezone required")
    old = m["original_plan"]
    current = stages._stage_plan(
        "A-SCREEN", identity, source, training, validation, [(n, v) for n in research.COHORT for v in research.VOCABS]
    )
    proof = validate_plans(old, current)
    research.require(
        m.get("code_approval") == proof
        and m.get("trained_commit") == old["identity"]["commit_hash"]
        and m.get("revalidated_commit") == identity["commit_hash"],
        "migration commit evidence mismatch",
    )
    original_path = Path(output) / "originals" / m["original_condition_file"]
    research.require(re.fullmatch(r"condition-\d{3}\.json", m["original_condition_file"]), "invalid original path")
    research.require(
        research.file_hash(original_path) == m["original_condition_sha256"], "original condition hash mismatch"
    )
    plan_path = Path(output) / "originals" / "plan.json"
    research.require(
        research.file_hash(plan_path) == m["original_plan_sha256"] and research.read_json(plan_path) == old,
        "original plan evidence mismatch",
    )
    envelope = research.read_json(original_path)
    original = envelope["record"]
    research.require(
        envelope.get("status") == "condition_complete" and "migration" not in original, "no chained migration"
    )
    research.require(original["tokenizer"] in SUPPORTED, "tokenizer not approved for migration")
    metric = row["validation"]
    fallback, tokens = metric["byte_fallback_tokens"], metric["tokens"]
    research.require(
        type(fallback) is int
        and type(tokens) is int
        and tokens > 0
        and 0 <= fallback <= tokens
        and metric["byte_fallback_percent"] == 100.0 * fallback / tokens,
        "invalid revalidated fallback metric",
    )
    research.require(
        {k: v for k, v in row.items() if k not in ("migration", "validation")}
        == {k: v for k, v in original.items() if k != "validation"},
        "original training provenance changed",
    )
    stages._validate_condition(original, expected, old["identity"], source, training, validation, output)


def migrate(args):
    source_dir, output = Path(args.source).resolve(), Path(args.output).resolve()
    research.require(not output.exists(), "duplicate migration: output exists")
    research.require(not output.is_relative_to(source_dir), "output must be separate from source")
    old = research.read_json(source_dir / "plan.json")
    research.require(old.get("stage") == "A-SCREEN", "migration only supports A-SCREEN; no Phase B/C")
    stage_args = argparse.Namespace(stage="A-SCREEN", dataset=args.dataset)
    plan, _, validation_texts, conditions = stages.prepare_stage(stage_args)
    proof = validate_plans(old, plan)
    paths = sorted(source_dir.glob("condition-*.json"))
    research.require(paths, "no completed conditions")
    checked = []
    for path in paths:
        research.require(re.fullmatch(r"condition-\d{3}\.json", path.name), "invalid condition filename")
        index = int(path.stem.split("-")[1])
        research.require(index < len(conditions), "unexpected condition index")
        envelope = research.read_json(path)
        row = envelope["record"]
        research.require(
            row.get("artifact") == f"{conditions[index][0]}-{conditions[index][1]}",
            "condition artifact assignment mismatch",
        )
        research.require(
            (source_dir / row["artifact"]).is_dir()
            and (source_dir / row["artifact"]).resolve().is_relative_to(source_dir),
            "missing or escaping artifact",
        )
        research.require(
            envelope.get("status") == "condition_complete" and "migration" not in row, "no chained migration"
        )
        research.require(row["tokenizer"] in SUPPORTED, "tokenizer not approved for migration")
        stages._validate_condition(
            row, conditions[index], old["identity"], plan["dataset"], plan["training"], plan["validation"], source_dir
        )
        checked.append((path, index, envelope))
    if args.dry_run:
        return {
            "dry_run": True,
            "eligible_conditions": len(checked),
            "code_approval": proof,
            "metrics_recomputed": False,
            "output_written": False,
        }
    # Exclusive new directory; partial failures remain evidence and cannot be resumed.
    output.mkdir(parents=True)
    (output / "originals").mkdir()
    shutil.copyfile(source_dir / "plan.json", output / "originals" / "plan.json")
    for path, index, envelope in checked:
        original = envelope["record"]
        shutil.copyfile(path, output / "originals" / path.name)
        artifact = original["artifact"]
        shutil.copytree(source_dir / artifact, output / artifact)
        research.require(
            research.artifact_hashes(output / artifact) == original["artifact_hashes"], "copied artifact mismatch"
        )
        tokenizer = research.load_tokenizer(original, output)
        row = copy.deepcopy(original)
        row["validation"] = stages.validation_metric(
            tokenizer, validation_texts, plan["validation"]["source_utf8_bytes"]
        )
        row["migration"] = {
            "schema_version": VERSION,
            "status": "condition_revalidated",
            "trained_commit": original["git_commit"],
            "revalidated_commit": plan["identity"]["commit_hash"],
            "revalidated_identity": plan["identity"],
            "revalidation_timestamp": datetime.now(timezone.utc).isoformat(),
            "code_approval": proof,
            "original_plan": old,
            "original_plan_sha256": research.file_hash(output / "originals" / "plan.json"),
            "original_condition_file": path.name,
            "original_condition_sha256": research.file_hash(path),
        }
        stages._validate_condition(
            row, conditions[index], plan["identity"], plan["dataset"], plan["training"], plan["validation"], output
        )
        research.write_new_json_atomic(output / path.name, {"status": "condition_revalidated", "record": row})
    # Publish plan last: incomplete migrations cannot be passed to normal --resume.
    research.write_new_json_atomic(output / "plan.json", plan)
    return {"revalidated_conditions": len(checked), "output": str(output), "ledger_written": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(migrate(args), indent=2))


if __name__ == "__main__":
    main()
