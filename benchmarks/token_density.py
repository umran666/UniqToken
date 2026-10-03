"""Script-agnostic density and byte-length distributions from verified diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from benchmarks import run_research_experiments as h
from benchmarks.analyze_tokenizer_failures import csv_text
from benchmarks.ledger import provenance, reject_ambiguous_fertility, validate_ledger

SCHEMA = 1
LABELS = ("split", "scope", "domain", "language", "tokenizer", "vocab_budget")
COUNTS = ("tokens", "normalized_utf8_bytes", "unicode_characters")
DEFINITIONS = {
    "tokens_per_unicode_character": "emitted tokens / normalized Unicode code points (Python len); not graphemes",
    "tokens_per_normalized_utf8_byte": "emitted tokens / normalized UTF-8 source bytes",
    "bytes_per_token": "normalized UTF-8 source bytes / emitted tokens",
    "token_length_bytes_histogram": "exact source-byte contribution per token; each fallback leaf contributes one byte",
    "percentiles": "nearest rank: smallest length with cumulative count >= ceil(p * token_count)",
    "aggregation": "pool counts and histograms before computing ratios and percentiles; retain strata, languages and domains",
    "normalization": "NFKC_unicode_spaces_v1, before metaspace encoding; no whitespace-word denominator",
    "empty": "undefined ratios and percentiles are null",
}


def density_metrics(tokens, normalized_utf8_bytes, unicode_characters, histogram):
    for value in (tokens, normalized_utf8_bytes, unicode_characters):
        h.require(type(value) is int and value >= 0, "density counts must be nonnegative integers")
    h.require(isinstance(histogram, dict), "byte histogram required")
    lengths = {}
    for key, count in histogram.items():
        h.require(
            isinstance(key, str) and key.isdecimal() and str(int(key)) == key and int(key) > 0,
            "histogram keys must be canonical positive byte lengths",
        )
        h.require(type(count) is int and count > 0, "histogram frequencies must be positive integers")
        lengths[int(key)] = count
    h.require(sum(lengths.values()) == tokens, "histogram token total mismatch")
    h.require(sum(k * v for k, v in lengths.items()) == normalized_utf8_bytes, "histogram byte total mismatch")
    h.require((tokens == 0) == (unicode_characters == 0), "empty character/token mismatch")
    h.require(unicode_characters <= normalized_utf8_bytes <= 4 * unicode_characters, "invalid UTF-8/codepoint totals")

    def quantile(p):
        target, cumulative = math.ceil(p * tokens), 0
        for length, count in sorted(lengths.items()):
            cumulative += count
            if cumulative >= target:
                return length
        return None

    return {
        "tokens_per_unicode_character": tokens / unicode_characters if unicode_characters else None,
        "tokens_per_normalized_utf8_byte": tokens / normalized_utf8_bytes if normalized_utf8_bytes else None,
        "bytes_per_token": normalized_utf8_bytes / tokens if tokens else None,
        "token_length_bytes_histogram": {str(k): v for k, v in sorted(lengths.items())},
        "token_length_bytes_p50": quantile(0.5),
        "token_length_bytes_p95": quantile(0.95),
        "token_length_bytes_p99": quantile(0.99),
        "token_length_bytes_max": max(lengths, default=None),
    }


def project_record(row):
    h.require(isinstance(row, dict), "density record must be an object")
    required = (*LABELS, *COUNTS, "token_length_bytes_histogram")
    h.require(all(key in row for key in required), "density record is missing required canonical fields")
    reject_ambiguous_fertility(row)
    metrics = density_metrics(*(row[k] for k in COUNTS), row["token_length_bytes_histogram"])
    for key, value in metrics.items():
        if key in row:
            h.require(row[key] == value, f"inconsistent density metric: {key}")
    return {
        **{k: row[k] for k in LABELS + COUNTS},
        "model_kind": "tokenizer_only",
        "actual_vocab_size": row["vocab_budget"],
        **metrics,
    }


def validate_density_ledger(payload):
    h.require(payload.get("density_schema_version") == SCHEMA, "unsupported density schema")
    validate_ledger(payload)
    groups, seen = defaultdict(list), set()
    for row in payload["records"]:
        expected_row = project_record(row)
        h.require(
            all(key in row and row[key] == value for key, value in expected_row.items()), "incomplete density record"
        )
        key = tuple(row[k] for k in LABELS)
        h.require(key not in seen, "duplicate density record")
        seen.add(key)
        h.require(row["split"] in ("train", "validation"), "held-out test diagnostics forbidden")
        scope = row["scope"]
        h.require(scope in ("aggregate", "stratum", "language", "domain"), "unknown density scope")
        h.require(bool(row["language"]) == (scope in ("stratum", "language")), "invalid language scope")
        h.require(bool(row["domain"]) == (scope in ("stratum", "domain")), "invalid domain scope")
        groups[(row["split"], row["tokenizer"], row["vocab_budget"])].append(row)
    for rows in groups.values():
        strata = [r for r in rows if r["scope"] == "stratum"]
        h.require(bool(strata), "stratum details required")
        expected = {("aggregate", None, None)}
        expected.update(("language", None, r["language"]) for r in strata)
        expected.update(("domain", r["domain"], None) for r in strata)
        actual = {(r["scope"], r["domain"], r["language"]) for r in rows if r["scope"] != "stratum"}
        h.require(actual == expected, "aggregate/language/domain coverage mismatch")
        for row in rows:
            if row["scope"] == "stratum":
                continue
            selected = [
                r
                for r in strata
                if (row["language"] is None or row["language"] == r["language"])
                and (row["domain"] is None or row["domain"] == r["domain"])
            ]
            for name in COUNTS:
                h.require(row[name] == sum(r[name] for r in selected), f"pooled count mismatch: {name}")
            hist = Counter()
            for r in selected:
                hist.update(r["token_length_bytes_histogram"])
            h.require(dict(hist) == row["token_length_bytes_histogram"], "pooled histogram mismatch")
    return payload


def export_diagnostics(source, output):
    h.require(not output.exists(), "output must be new")
    receipt = h.read_json(source.parent / "manifest.json")
    h.require(receipt.get("status") == "complete", "complete diagnostic receipt required")
    h.require(source.name in receipt["artifacts"], "source is not receipted")
    h.require(h.file_hash(source) == receipt["artifacts"][source.name], "source hash mismatch")
    original = h.read_json(source)
    h.require(h.digest(original) == receipt["results_content_sha256"], "source content hash mismatch")
    h.require(
        original["schema_version"] == 1 and original["result_label"] == "DIAGNOSTIC_DESCRIPTIVE",
        "expected frozen tokenizer diagnostic schema",
    )
    h.require(original["test_access"] == "forbidden_not_opened_or_hashed", "test access forbidden")
    h.require(original["language_model"] == {"initialized": False, "trained": False}, "tokenizer-only input required")
    identity = provenance()
    h.require(not identity["working_tree_dirty"], "commit source before exporting evidence")
    payload = {
        "density_schema_version": SCHEMA,
        "metadata": {**identity, "data_split": "document_disjoint_frozen_screening_validation"},
        "metric_definitions": DEFINITIONS,
        "source_receipt": {
            "results_sha256": h.file_hash(source),
            "manifest_sha256": h.file_hash(source.parent / "manifest.json"),
            "original_identity": original["identity"],
            "inputs": original["inputs"],
            "models": original["models"],
        },
        "records": [project_record(row) for row in original["records"]],
    }
    validate_density_ledger(payload)
    h.require(provenance() == identity, "source changed during export")
    output.mkdir(parents=True)
    h.write_new_json(output / "results.json", payload)
    for scope in ("aggregate", "stratum", "language", "domain"):
        (output / f"{scope}.csv").write_text(
            csv_text([r for r in payload["records"] if r["scope"] == scope]), encoding="utf-8"
        )
    h.write_new_json(output / "manifest.json", {"status": "complete", "artifacts": h.artifact_hashes(output)})
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", type=Path, required=True, help="Receipted analyze_tokenizer_failures results.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = export_diagnostics(args.analysis, args.output)
    print(json.dumps({"status": "complete", "records": len(payload["records"])}))


if __name__ == "__main__":
    main()
