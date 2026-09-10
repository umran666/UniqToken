"""Versioned provenance and validation for active benchmark JSON ledgers."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess

SCHEMA_VERSION = 3


def provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    return {"ledger_schema_version": SCHEMA_VERSION, "commit_hash": commit, "working_tree_dirty": dirty}


def validate_ledger(payload: dict, *, expected_commit: str | None = None) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("ledger must be a versioned object")
    metadata = payload.get("metadata", payload)
    if not isinstance(metadata, dict) or metadata.get("ledger_schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported ledger schema; expected {SCHEMA_VERSION}")
    commit = metadata.get("commit_hash")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("ledger requires a full Git commit_hash")
    if expected_commit is not None and commit != expected_commit:
        raise ValueError("ledger commit_hash does not match expected commit")
    if not isinstance(metadata.get("working_tree_dirty"), bool):
        raise ValueError("ledger must declare working_tree_dirty")
    if not str(metadata.get("data_split", "")).startswith("document_disjoint"):
        raise ValueError("ledger must declare document-disjoint evaluation")
    rows = payload.get("records", payload.get("entries"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("ledger must contain non-empty records or entries")
    for row in rows:
        if not isinstance(row, dict) or row.get("model_kind") not in (
            "causal_transformer",
            "tokenizer_only",
            "streaming_detokenizer",
        ):
            raise ValueError("every result requires a supported model_kind")
        if any("fertility" in key or key == "tokens_per_word" for key in row):
            raise ValueError("ambiguous cross-script fertility is not accepted; use tokens_per_unicode_character")
        budget = row.get("vocab_budget", row.get("target_vocab"))
        actual = row.get("actual_vocab_size", row.get("actual_vocab"))
        if budget is not None and row.get("category") != "external_pretrained":
            if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1 or actual != budget:
                raise ValueError("ledger vocabulary budget mismatch")
        if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
            raise ValueError("ledger metrics must be finite")
    return payload


def load_ledger(path: str | Path, *, expected_commit: str | None = None) -> dict:
    with open(path, encoding="utf-8") as stream:
        return validate_ledger(json.load(stream), expected_commit=expected_commit)


def write_ledger(path: str | Path, payload: dict) -> None:
    validate_ledger(payload)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
