"""Performance Regression Gate (Issue #106).

Compares a canonical benchmark run (issue #105 ``results.json``) against a
committed, versioned baseline and separates hard correctness failures from
statistical performance signals:

- Tokenizer parity failures are gated authoritatively by the deterministic
  exact-parity gate (a separate CI job; this module never gates timing on its
  own). Any parity failure is a hard failure there.
- Performance signals remain advisory, including with --enforce: canonical
  schema 1.0 supplies observed throughput extrema, not calibrated confidence
  intervals, and supplies no repeated memory uncertainty. Enforcement requires
  separately reviewed noise evidence and a measurement protocol that supports it.

Usage:
    # Record a baseline from a canonical benchmark receipt:
    python -m benchmarks.canonical_gate --write-baseline \
        --results benchmarks/canonical/issue105/results.json \
        --baseline benchmarks/canonical/gate-baseline.json

    # Compare a fresh run against the committed baseline:
    python -m benchmarks.canonical_gate \
        --results perf-gate/current/results.json \
        --baseline benchmarks/canonical/gate-baseline.json \
        --out perf-gate/comparison.json [--enforce]

Exit codes: 0 pass/advisory/soft-skip, 2 hard usage or input errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GATE_SCHEMA_VERSION = "1.0"
DEFAULT_BASELINE = Path("benchmarks/canonical/gate-baseline.json")
DEFAULT_THROUGHPUT_DROP_PCT = 5.0
DEFAULT_MEMORY_RISE_PCT = 10.0
DEFAULT_MAX_AGE_DAYS = 30

FINGERPRINT_KEYS = (
    "platform",
    "machine",
    "processor",
    "physical_cpus",
    "logical_cpus",
    "total_ram_gb",
    "build_mode",
    "python_version",
    "worker_count",
    "warmup",
    "repetitions",
    "iterations_per_repetition",
    "baselines",
    "artifact_hashes",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def to_iso(moment: datetime) -> str:
    """Format a UTC datetime as an ISO timestamp with second precision."""
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(timestamp: Optional[str]) -> Optional[datetime]:
    """Parse an ISO timestamp, returning None for missing or invalid values."""
    if not isinstance(timestamp, str) or not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return parsed if parsed.utcoffset() is not None else None
    except ValueError:
        return None


def load_json(path: Path) -> Any:
    """Read and decode a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def cell_key(tokenizer: str, workload: str) -> str:
    """Return the comparison key for a tokenizer and workload pair."""
    return f"{tokenizer}|{workload}"


def positive_number(value: Any) -> bool:
    """Reject booleans, strings, zero and nonfinite measurements."""
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def validate_records(records: Any, keyed: bool = False) -> None:
    """Validate the measured cells before indexing or performing arithmetic."""
    if not isinstance(records, dict if keyed else list) or not records:
        raise ValueError("records must be a nonempty workload collection")
    seen = set()
    for key, row in records.items() if keyed else enumerate(records):
        if not isinstance(row, dict):
            raise ValueError("each workload record must be an object")
        if keyed:
            identity = key
            if not isinstance(key, str) or key.count("|") != 1:
                raise ValueError("baseline workload keys must be tokenizer|workload")
        else:
            for name in ("tokenizer", "workload", "fixture"):
                if not isinstance(row.get(name), str) or not row[name] or "|" in row[name]:
                    raise ValueError(f"missing or invalid {name}")
            identity = cell_key(row["tokenizer"], row["workload"])
        if identity in seen:
            raise ValueError(f"duplicate workload {identity}")
        seen.add(identity)
        if type(row.get("batch_size")) is not int or row["batch_size"] < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(row.get("fixture"), str) or not row["fixture"]:
            raise ValueError("fixture must be a nonempty string")
        for name in ("throughput_mb_s", "peak_rss_mb"):
            if not positive_number(row.get(name)):
                raise ValueError(f"{name} must be positive and finite")
        bounds = row.get("throughput_ci_95_mb_s")
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not all(positive_number(value) for value in bounds)
            or not bounds[0] <= row["throughput_mb_s"] <= bounds[1]
        ):
            raise ValueError("throughput range must be finite, ordered, and contain throughput")


def validate_results(results: Any) -> None:
    """Check the canonical input shape without treating missing evidence as proof."""
    if not isinstance(results, dict) or not isinstance(results.get("metadata"), dict):
        raise ValueError("canonical results must contain metadata and records")
    if results["metadata"].get("schema_version") != "1.0":
        raise ValueError("canonical results must carry schema_version 1.0")
    validate_records(results.get("records"))


def fingerprint_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract hardware metadata, rounding total RAM to whole gigabytes."""
    meta = results.get("metadata", {})
    fingerprint = {key: meta.get(key) for key in FINGERPRINT_KEYS}
    ram = fingerprint.get("total_ram_gb")
    if positive_number(ram):
        # Total RAM reporting fluctuates between runs; compare at GB granularity.
        fingerprint["total_ram_gb"] = round(ram)
    return fingerprint


def fingerprint_differences(baseline_fp: Dict[str, Any], current_fp: Dict[str, Any]) -> List[str]:
    """Return hardware fingerprint keys whose baseline and current values differ."""
    return [
        key
        for key in FINGERPRINT_KEYS
        if baseline_fp.get(key) is None or current_fp.get(key) is None or baseline_fp[key] != current_fp[key]
    ]


# ---------------------------------------------------------------------------
# Baseline construction
# ---------------------------------------------------------------------------


def build_baseline(
    results: Dict[str, Any],
    source_path: Path,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    throughput_drop_pct: float = DEFAULT_THROUGHPUT_DROP_PCT,
    memory_rise_pct: float = DEFAULT_MEMORY_RISE_PCT,
) -> Dict[str, Any]:
    """Builds a versioned baseline artifact from a canonical results.json."""
    validate_results(results)
    if type(max_age_days) is not int or max_age_days < 1:
        raise ValueError("max_age_days must be positive")
    if not positive_number(throughput_drop_pct) or not positive_number(memory_rise_pct):
        raise ValueError("threshold percentages must be positive and finite")
    meta = results["metadata"]
    records = results.get("records", [])
    if meta.get("schema_version") != "1.0" or not records:
        raise ValueError("canonical results.json must carry schema_version 1.0 and workload records")

    now = utc_now()
    measured_at = parse_iso(meta.get("timestamp_utc"))
    anchor = measured_at or now
    baseline: Dict[str, Any] = {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate_issue": 106,
        "source": {
            "path": source_path.as_posix(),
            "sha256": sha256_file(source_path),
            "git_commit": meta.get("git_commit"),
            "measured_at": meta.get("timestamp_utc"),
            "recorded_at": to_iso(now),
        },
        "expiry": {
            "anchor": "source.measured_at when available, else source.recorded_at",
            "max_age_days": max_age_days,
            "expires_at": to_iso(anchor + timedelta(days=max_age_days)),
        },
        "fingerprint": fingerprint_from_results(results),
        "thresholds": {
            "throughput_drop_pct": throughput_drop_pct,
            "memory_rise_pct": memory_rise_pct,
            "metrics": ["throughput_mb_s (primary, decimal MB/s per #105)", "peak_rss_mb"],
        },
        "fixture_hashes": meta.get("fixture_hashes", {}),
        "records": {},
    }
    for record in records:
        key = cell_key(record["tokenizer"], record["workload"])
        baseline["records"][key] = {
            "throughput_mb_s": record["throughput_mb_s"],
            "throughput_ci_95_mb_s": record["throughput_ci_95_mb_s"],
            "peak_rss_mb": record["peak_rss_mb"],
            "batch_size": record.get("batch_size"),
            "fixture": record.get("fixture"),
            "repetitions": record.get("repetitions"),
            "iterations_per_rep": record.get("iterations_per_rep"),
        }
    return baseline


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def throughput_verdict(
    baseline_cell: Dict[str, Any],
    current_cell: Dict[str, Any],
    drop_threshold_pct: float,
) -> Dict[str, Any]:
    """Classify throughput drops using the threshold and uncertainty intervals.

    A drop is significant only when it exceeds the threshold and the current
    interval's upper bound is below the baseline interval's lower bound.
    """
    base_value = baseline_cell.get("throughput_mb_s") or 0.0
    current_value = current_cell.get("throughput_mb_s") or 0.0
    drop_pct = ((base_value - current_value) / base_value * 100.0) if base_value > 0 else 0.0
    breach = drop_pct > drop_threshold_pct

    base_ci = baseline_cell.get("throughput_ci_95_mb_s") or []
    current_ci = current_cell.get("throughput_ci_95_mb_s") or []
    significant: Optional[bool] = None
    if len(base_ci) == 2 and len(current_ci) == 2:
        # Schema 1.0 calls these CI bounds, but they are observed min/max.
        # Non-overlap is a diagnostic heuristic, not statistical significance.
        significant = current_ci[1] < base_ci[0]
    verdict = "pass"
    if breach:
        verdict = "outside-observed-range" if significant else "nominal-regression"
    return {
        "baseline_mb_s": base_value,
        "current_mb_s": current_value,
        "drop_pct": round(drop_pct, 3),
        "breaches_threshold": breach,
        "observed_range_overlap": (not significant) if significant is not None else None,
        "verdict": verdict,
    }


def memory_verdict(
    baseline_cell: Dict[str, Any],
    current_cell: Dict[str, Any],
    rise_threshold_pct: float,
) -> Dict[str, Any]:
    """Report peak RSS growth and whether it exceeds the percentage threshold."""
    base_value = baseline_cell.get("peak_rss_mb") or 0.0
    current_value = current_cell.get("peak_rss_mb") or 0.0
    rise_pct = ((current_value - base_value) / base_value * 100.0) if base_value > 0 else 0.0
    return {
        "baseline_mb": base_value,
        "current_mb": current_value,
        "rise_pct": round(rise_pct, 3),
        "breaches_threshold": rise_pct > rise_threshold_pct,
        "verdict": "regression" if rise_pct > rise_threshold_pct else "pass",
    }


def evaluate(
    baseline_path: Path,
    results_path: Path,
    enforce: bool = False,
    now: Optional[datetime] = None,
) -> Tuple[Dict[str, Any], int]:
    """Evaluates a canonical run against a baseline.

    Returns the machine-readable comparison artifact and the process exit code.
    Schema 1.0 cannot justify performance enforcement, even when requested.
    """
    now = now or utc_now()
    artifact: Dict[str, Any] = {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate_issue": 106,
        "generated_at_utc": to_iso(now),
        "enforced": enforce,
        "enforcement_eligible": False,
        "enforcement_reason": "schema 1.0 has observed ranges, not calibrated throughput or memory uncertainty",
        "baseline": {"path": baseline_path.as_posix()},
        "results": {"path": results_path.as_posix()},
    }

    def finish(status: str, exit_code: int, note: str) -> Tuple[Dict[str, Any], int]:
        """Finalize the comparison artifact's status and return it with the exit code."""
        artifact["status"] = status
        artifact["merge_blocking"] = exit_code != 0
        artifact["note"] = note
        return artifact, exit_code

    try:
        results = load_json(results_path)
        validate_results(results)
    except (OSError, ValueError) as error:
        return finish("results-unavailable", 2, f"cannot read current results.json: {error}")

    if not baseline_path.is_file():
        return finish(
            "baseline-missing", 0, "no committed baseline; run in advisory mode and record one from a trusted host"
        )

    try:
        baseline = load_json(baseline_path)
    except (OSError, ValueError) as error:
        return finish("baseline-missing", 0, f"baseline unreadable: {error}")

    if not isinstance(baseline, dict) or baseline.get("schema_version") != GATE_SCHEMA_VERSION:
        return finish("baseline-schema-mismatch", 0, "baseline schema_version is incompatible; regenerate the baseline")

    try:
        validate_records(baseline.get("records"), keyed=True)
        for name in ("thresholds", "source", "expiry", "fingerprint", "fixture_hashes"):
            if not isinstance(baseline.get(name), dict):
                raise ValueError(f"baseline {name} must be an object")
        thresholds = baseline["thresholds"]
        drop_threshold = thresholds.get("throughput_drop_pct")
        memory_threshold = thresholds.get("memory_rise_pct")
        if not positive_number(drop_threshold) or not positive_number(memory_threshold):
            raise ValueError("threshold percentages must be positive and finite")
    except ValueError as error:
        return finish("baseline-invalid", 0, str(error))
    artifact["thresholds"] = {"throughput_drop_pct": drop_threshold, "memory_rise_pct": memory_threshold}

    source = baseline.get("source", {})
    expiry = baseline.get("expiry", {})
    artifact["baseline"].update(
        {
            "recorded_at": source.get("recorded_at"),
            "measured_at": source.get("measured_at"),
            "git_commit": source.get("git_commit"),
            "sha256": source.get("sha256"),
            "expires_at": expiry.get("expires_at"),
        }
    )

    # Baseline provenance: fail-loud but never merge-blocking on a stale receipt.
    artifact["baseline"]["provenance_verified"] = False
    if not isinstance(source.get("path"), str) or not source["path"]:
        return finish("baseline-provenance-unavailable", 0, "baseline source receipt path is invalid")
    source_path = Path(source["path"])
    if not source.get("sha256") or not source_path.is_file():
        return finish("baseline-provenance-unavailable", 0, "baseline source receipt or SHA-256 is unavailable")
    try:
        provenance_ok = sha256_file(source_path) == source["sha256"]
        artifact["baseline"]["provenance_verified"] = provenance_ok
        if not provenance_ok:
            return finish(
                "baseline-provenance-mismatch",
                0,
                f"baseline source {source_path} changed since recording; refresh the baseline",
            )
        source_results = load_json(source_path)
        validate_results(source_results)
        source_cells = {cell_key(row["tokenizer"], row["workload"]): row for row in source_results["records"]}
        source_meta = source_results["metadata"]
        source_fingerprint = fingerprint_from_results(source_results)
        if (
            set(source_cells) != set(baseline["records"])
            or any(
                any(row.get(name) != source_cells[key].get(name) for name in row)
                for key, row in baseline["records"].items()
            )
            or any(value != source_fingerprint.get(key) for key, value in baseline["fingerprint"].items())
            or baseline["fixture_hashes"] != source_meta.get("fixture_hashes")
            or source.get("measured_at") != source_meta.get("timestamp_utc")
            or source.get("git_commit") != source_meta.get("git_commit")
        ):
            artifact["baseline"]["provenance_verified"] = False
            return finish(
                "baseline-provenance-mismatch", 0, "baseline records or metadata differ from the source receipt"
            )
    except (OSError, ValueError) as error:
        return finish("baseline-provenance-unavailable", 0, str(error))

    anchor = parse_iso(source.get("measured_at")) or parse_iso(source.get("recorded_at"))
    expires_at = parse_iso(expiry.get("expires_at"))
    if anchor is None or expires_at is None or anchor > now or expires_at <= anchor:
        return finish("baseline-invalid", 0, "baseline timestamps must define a valid, nonfuture measurement lifetime")
    artifact["baseline"]["age_days"] = round((now - anchor).total_seconds() / 86400, 2)
    if now > expires_at:
        return finish("baseline-expired", 0, f"baseline expired at {expiry.get('expires_at')}; refresh the baseline")

    baseline_fp = baseline.get("fingerprint", {})
    current_fp = fingerprint_from_results(results)
    differences = fingerprint_differences(baseline_fp, current_fp)
    artifact["fingerprint"] = {
        "baseline": baseline_fp,
        "current": current_fp,
        "controlled_match": not differences,
        "differences": differences,
    }
    if differences:
        return finish(
            "hardware-mismatch",
            0,
            "runner fingerprint differs from the baseline host; comparison is advisory-only on uncontrolled hardware: "
            + ", ".join(differences),
        )

    baseline_fixtures = baseline.get("fixture_hashes", {})
    current_fixtures = results.get("metadata", {}).get("fixture_hashes", {})
    if not baseline_fixtures or not current_fixtures or baseline_fixtures != current_fixtures:
        return finish(
            "fixture-drift",
            0,
            "canonical fixture hashes differ from the baseline; the #105 fixtures are versioned and must not drift",
        )

    baseline_records = baseline.get("records", {})
    current_records = results.get("records", [])
    current_index = {cell_key(r["tokenizer"], r["workload"]): r for r in current_records}
    if set(baseline_records) != set(current_index) or any(
        any(baseline_records[key][name] != current_index[key][name] for name in ("fixture", "batch_size"))
        for key in set(baseline_records) & set(current_index)
    ):
        return finish(
            "workload-mismatch", 0, "workload identities, fixtures and batch sizes must match the complete baseline"
        )
    for rows, meta in ((baseline_records.values(), baseline_fp), (current_records, results["metadata"])):
        if (
            type(meta.get("repetitions")) is not int
            or meta["repetitions"] < 3
            or type(meta.get("iterations_per_repetition")) is not int
            or meta["iterations_per_repetition"] < 1
            or any(
                type(row.get("repetitions")) is not int
                or row["repetitions"] != meta["repetitions"]
                or type(row.get("iterations_per_rep")) is not int
                or row["iterations_per_rep"] != meta["iterations_per_repetition"]
                for row in rows
            )
        ):
            return finish(
                "insufficient-evidence", 0, "matching repeated measurement counts are required for every cell"
            )

    cells: List[Dict[str, Any]] = []
    significant_throughput: List[str] = []
    nominal_throughput: List[str] = []
    memory_regressions: List[str] = []
    unmatched: List[str] = []

    for key in sorted(set(baseline_records) | set(current_index)):
        baseline_cell = baseline_records.get(key)
        current_cell = current_index.get(key)
        if not baseline_cell or not current_cell:
            unmatched.append(key)
            continue
        throughput = throughput_verdict(baseline_cell, current_cell, drop_threshold)
        memory = memory_verdict(baseline_cell, current_cell, memory_threshold)
        if throughput["breaches_threshold"]:
            (significant_throughput if throughput["observed_range_overlap"] is False else nominal_throughput).append(
                key
            )
        if memory["breaches_threshold"]:
            memory_regressions.append(key)
        cells.append({"key": key, "throughput": throughput, "memory": memory})

    artifact["summary"] = {
        "cells_compared": len(cells),
        "throughput_regressions_outside_range": significant_throughput,
        "throughput_regressions_nominal": nominal_throughput,
        "memory_regressions": memory_regressions,
        "unmatched_workloads": unmatched,
    }
    artifact["cells"] = cells

    if significant_throughput or memory_regressions:
        return finish(
            "advisory",
            0,
            "performance changes require investigation; schema 1.0 cannot support statistical enforcement",
        )
    if nominal_throughput:
        return finish(
            "advisory",
            0,
            "nominal drops with overlapping observed ranges; investigate repeated runs: "
            + ", ".join(nominal_throughput),
        )
    return finish("pass", 0, "no nominal throughput or memory threshold breach; no statistical enforcement is implied")


def format_summary(artifact: Dict[str, Any]) -> str:
    """Render a comparison artifact as a concise status and regression summary."""
    lines = [f"Performance gate status: {artifact['status']} (enforced={artifact['enforced']})"]
    summary = artifact.get("summary")
    if summary:
        lines.append(
            f"cells={summary['cells_compared']} "
            f"outside-range-throughput={len(summary['throughput_regressions_outside_range'])} "
            f"nominal-throughput={len(summary['throughput_regressions_nominal'])} "
            f"memory={len(summary['memory_regressions'])} "
            f"unmatched={len(summary['unmatched_workloads'])}"
        )
    fp = artifact.get("fingerprint")
    if fp and not fp.get("controlled_match"):
        lines.append("runner fingerprint differences: " + ", ".join(fp.get("differences", [])))
    if artifact.get("note"):
        lines.append(f"note: {artifact['note']}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """Run baseline recording or comparison from CLI arguments and return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="canonical benchmark results.json to evaluate")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help="versioned baseline artifact")
    parser.add_argument("--out", type=Path, default=None, help="where to write the comparison artifact")
    parser.add_argument("--enforce", action="store_true", help="request enforcement (schema 1.0 remains advisory)")
    parser.add_argument(
        "--write-baseline", action="store_true", help="record a new baseline from --results instead of comparing"
    )
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    parser.add_argument("--throughput-drop-pct", type=float, default=DEFAULT_THROUGHPUT_DROP_PCT)
    parser.add_argument("--memory-rise-pct", type=float, default=DEFAULT_MEMORY_RISE_PCT)
    args = parser.parse_args(argv)

    if args.write_baseline:
        try:
            results = load_json(args.results)
            baseline = build_baseline(
                results,
                args.results,
                max_age_days=args.max_age_days,
                throughput_drop_pct=args.throughput_drop_pct,
                memory_rise_pct=args.memory_rise_pct,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(f"baseline recording failed: {error}", file=sys.stderr)
            return 2
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(
            f"Baseline recorded: {args.baseline} ({len(baseline['records'])} workload cells, expires {baseline['expiry']['expires_at']})"
        )
        return 0

    artifact, exit_code = evaluate(args.baseline, args.results, enforce=args.enforce)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(format_summary(artifact))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
