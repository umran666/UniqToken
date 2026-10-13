"""Performance Regression Gate (Issue #106).

Compares a canonical benchmark run (issue #105 ``results.json``) against a
committed, versioned baseline and separates hard correctness failures from
statistical performance signals:

- Tokenizer parity failures are gated authoritatively by the deterministic
  exact-parity gate (a separate CI job; this module never gates timing on its
  own). Any parity failure is a hard failure there.
- Throughput regressions are only *failed* when repeated measurements establish
  statistical meaning: the nominal drop must exceed the configured threshold
  AND the current run's uncertainty interval must not overlap the baseline's.
  On unsuitable or unverified CI hardware the gate stays advisory by design so
  that a noisy shared runner cannot block merges on a single timing sample.
- Memory regressions follow the same controlled-environment rule: they fail
  only under enforcement on fingerprint-matched hardware.

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

Exit codes: 0 pass/advisory/soft-skip, 1 enforced significant regression,
2 hard usage or input errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(timestamp: Optional[str]) -> Optional[datetime]:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def cell_key(tokenizer: str, workload: str) -> str:
    return f"{tokenizer}|{workload}"


def fingerprint_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
    meta = results.get("metadata", {})
    fingerprint = {key: meta.get(key) for key in FINGERPRINT_KEYS}
    ram = fingerprint.get("total_ram_gb")
    if isinstance(ram, (int, float)):
        # Total RAM reporting fluctuates between runs; compare at GB granularity.
        fingerprint["total_ram_gb"] = round(ram)
    return fingerprint


def fingerprint_differences(baseline_fp: Dict[str, Any], current_fp: Dict[str, Any]) -> List[str]:
    return [key for key in FINGERPRINT_KEYS if baseline_fp.get(key) != current_fp.get(key)]


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
    meta = results.get("metadata", {})
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
    base_value = baseline_cell.get("throughput_mb_s") or 0.0
    current_value = current_cell.get("throughput_mb_s") or 0.0
    drop_pct = ((base_value - current_value) / base_value * 100.0) if base_value > 0 else 0.0
    breach = drop_pct > drop_threshold_pct

    base_ci = baseline_cell.get("throughput_ci_95_mb_s") or []
    current_ci = current_cell.get("throughput_ci_95_mb_s") or []
    significant: Optional[bool] = None
    if len(base_ci) == 2 and len(current_ci) == 2:
        # Repeated-measurement intervals must not overlap at all for the
        # regression to count as statistically meaningful.
        significant = current_ci[1] < base_ci[0]
    verdict = "pass"
    if breach:
        verdict = "significant-regression" if significant else "nominal-regression"
    return {
        "baseline_mb_s": base_value,
        "current_mb_s": current_value,
        "drop_pct": round(drop_pct, 3),
        "breaches_threshold": breach,
        "uncertainty_overlap": (not significant) if significant is not None else None,
        "verdict": verdict,
    }


def memory_verdict(
    baseline_cell: Dict[str, Any],
    current_cell: Dict[str, Any],
    rise_threshold_pct: float,
) -> Dict[str, Any]:
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
    Merge-blocking requires ``enforce``; every other outcome is advisory.
    """
    now = now or utc_now()
    artifact: Dict[str, Any] = {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate_issue": 106,
        "generated_at_utc": to_iso(now),
        "enforced": enforce,
        "baseline": {"path": baseline_path.as_posix()},
        "results": {"path": results_path.as_posix()},
    }

    def finish(status: str, exit_code: int, note: str) -> Tuple[Dict[str, Any], int]:
        artifact["status"] = status
        artifact["merge_blocking"] = exit_code == 1
        artifact["note"] = note
        return artifact, exit_code

    try:
        results = load_json(results_path)
    except (OSError, json.JSONDecodeError) as error:
        return finish("results-unavailable", 2, f"cannot read current results.json: {error}")
    if not isinstance(results, dict) or not results.get("records"):
        return finish("results-unavailable", 2, "current results.json has no workload records")

    if not baseline_path.is_file():
        return finish(
            "baseline-missing", 0, "no committed baseline; run in advisory mode and record one from a trusted host"
        )

    try:
        baseline = load_json(baseline_path)
    except (OSError, json.JSONDecodeError) as error:
        return finish("baseline-missing", 0, f"baseline unreadable: {error}")

    if baseline.get("schema_version") != GATE_SCHEMA_VERSION:
        return finish("baseline-schema-mismatch", 0, "baseline schema_version is incompatible; regenerate the baseline")

    thresholds = baseline.get("thresholds", {})
    drop_threshold = float(thresholds.get("throughput_drop_pct", DEFAULT_THROUGHPUT_DROP_PCT))
    memory_threshold = float(thresholds.get("memory_rise_pct", DEFAULT_MEMORY_RISE_PCT))
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
    source_path = Path(source.get("path", ""))
    if source.get("sha256") and source_path.is_file():
        provenance_ok = sha256_file(source_path) == source["sha256"]
        artifact["baseline"]["provenance_verified"] = provenance_ok
        if not provenance_ok:
            return finish(
                "baseline-provenance-mismatch",
                0,
                f"baseline source {source_path} changed since recording; refresh the baseline",
            )

    anchor = parse_iso(source.get("measured_at")) or parse_iso(source.get("recorded_at"))
    expires_at = parse_iso(expiry.get("expires_at"))
    if anchor and expires_at:
        artifact["baseline"]["age_days"] = round((now - anchor).total_seconds() / 86400, 2)
        if now > expires_at:
            return finish(
                "baseline-expired",
                0,
                f"baseline expired at {expiry.get('expires_at')}; re-record from a trusted host before trusting comparisons",
            )

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
    if baseline_fixtures and current_fixtures and baseline_fixtures != current_fixtures:
        return finish(
            "fixture-drift",
            0,
            "canonical fixture hashes differ from the baseline; the #105 fixtures are versioned and must not drift",
        )

    baseline_records = baseline.get("records", {})
    current_records = results.get("records", [])
    current_index = {cell_key(r["tokenizer"], r["workload"]): r for r in current_records}

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
            (significant_throughput if throughput["uncertainty_overlap"] is False else nominal_throughput).append(key)
        if memory["breaches_threshold"]:
            memory_regressions.append(key)
        cells.append({"key": key, "throughput": throughput, "memory": memory})

    artifact["summary"] = {
        "cells_compared": len(cells),
        "throughput_regressions_significant": significant_throughput,
        "throughput_regressions_nominal": nominal_throughput,
        "memory_regressions": memory_regressions,
        "unmatched_workloads": unmatched,
    }
    artifact["cells"] = cells

    if significant_throughput or memory_regressions:
        if enforce:
            return finish(
                "fail",
                1,
                "statistically meaningful regressions under a controlled, fingerprint-matched runner: "
                + ", ".join(significant_throughput + memory_regressions),
            )
        return finish(
            "advisory",
            0,
            "regressions detected without enforcement or without statistical meaning; see cells for detail",
        )
    if nominal_throughput:
        return finish(
            "advisory",
            0,
            "nominal drops within benchmark noise tolerance (uncertainty intervals overlap): "
            + ", ".join(nominal_throughput),
        )
    return finish("pass", 0, "no throughput or memory regression beyond thresholds and noise tolerance")


def format_summary(artifact: Dict[str, Any]) -> str:
    lines = [f"Performance gate status: {artifact['status']} (enforced={artifact['enforced']})"]
    summary = artifact.get("summary")
    if summary:
        lines.append(
            f"cells={summary['cells_compared']} "
            f"significant-throughput={len(summary['throughput_regressions_significant'])} "
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="canonical benchmark results.json to evaluate")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help="versioned baseline artifact")
    parser.add_argument("--out", type=Path, default=None, help="where to write the comparison artifact")
    parser.add_argument("--enforce", action="store_true", help="allow merge-blocking failures on controlled hardware")
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
