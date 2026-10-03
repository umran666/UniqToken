"""Validate paired native receipts and export all cells, including regressions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics

from benchmarks import compare_residual_native as c
from benchmarks import profile_residual_native as p
from benchmarks import run_research_experiments as h

FIELDS = ("surface", "vocab_budget", "length", "script", "batch_size", "output", "cache", "characters")


def key(row):
    return tuple(row.get(field) for field in FIELDS)


def verified(path):
    root = path.resolve().parent
    receipt = h.read_json(root / "manifest.json")
    h.require(receipt.get("status") == "complete", "incomplete comparison receipt")
    h.require(path.name in receipt["artifacts"], "comparison results missing from receipt")
    h.require("manifest.json" not in receipt["artifacts"], "receipt cannot include itself")
    for relative, digest in receipt["artifacts"].items():
        target = (root / relative).resolve()
        h.require(target.is_relative_to(root) and target.is_file(), "comparison receipt path escape/missing file")
        h.require(h.file_hash(target) == digest, f"comparison receipt mismatch: {relative}")
    payload = h.read_json(path)
    validate(payload)
    paired = {(key(row), row["round"]): row for row in payload["records"]}
    allocations = {(key(row), row["variant"]): row for row in payload["allocations"]}
    for index, specification in enumerate(c.specs({b: Path(str(b)) for b in (8192, 16384, 32768)}, Path("stress"))):
        cell = key(specification)
        for round_number in range(c.ROUNDS):
            relative = f"{index}-paired-{round_number}.json"
            h.require(relative in receipt["artifacts"], "paired response absent from receipt")
            h.require(
                h.read_json(root / relative) == paired[(cell, round_number)], "paired response differs from results"
            )
        for name in ("baseline", "compact"):
            relative = f"{index}-allocations-{name}.json"
            h.require(relative in receipt["artifacts"], "allocation response absent from receipt")
            h.require(
                h.read_json(root / relative) == allocations[(cell, name)], "allocation response differs from results"
            )
    return payload, receipt


def validate(payload):
    h.require(not payload["identity"]["working_tree_dirty"], "uncommitted comparison")
    h.require(payload["configuration"]["rounds"] == c.ROUNDS, "unexpected round protocol")
    h.require(payload["configuration"]["repetitions"] == p.REPETITIONS, "unexpected repetitions")
    expected = {key(row) for row in c.specs({b: Path(str(b)) for b in (8192, 16384, 32768)}, Path("stress"))}
    observed, allocations, invariant, binaries = {}, {}, {}, {}
    for row in payload["records"]:
        cell = key(row)
        identifier = (cell, row["round"])
        h.require(
            cell in expected and row["round"] in range(c.ROUNDS) and identifier not in observed,
            "unexpected/duplicate paired cell",
        )
        observed[identifier] = row
        h.require(set(row["measurements"]) == {"baseline", "prefix", "compact"}, "incomplete variants")
        stream = None
        for name, measurement in row["measurements"].items():
            samples = measurement["wall_ns_samples"]
            h.require(
                len(samples) == p.REPETITIONS and all(math.isfinite(x) and x > 0 for x in samples), "invalid timings"
            )
            h.require(measurement["latency_p50_ns"] == statistics.median(samples), "median mismatch")
            h.require(
                measurement["normalized_mb_per_second"]
                == row["normalized_input_bytes"] * 1000 / statistics.median(samples),
                "throughput mismatch",
            )
            current = (
                row["fixture_sha256"],
                row["model_sha256"],
                row["normalized_input_bytes"],
                row["tokens"],
                measurement["output_sha256"],
                measurement["path_score_float_hex"],
            )
            h.require(stream is None or stream == current, "cross-variant output/score/count mismatch")
            stream = current
            if name in binaries:
                h.require(binaries[name] == measurement["native_sha256"], "binary changed between cells")
            binaries[name] = measurement["native_sha256"]
            if name != "baseline":
                ratios = [a / b for a, b in zip(row["measurements"]["baseline"]["wall_ns_samples"], samples)]
                comparison = row["comparisons"][name]
                h.require(comparison["paired_ratios"] == ratios, "paired ratios mismatch")
                h.require(comparison["speedup_median"] == statistics.median(ratios), "paired median mismatch")
                h.require(comparison["bootstrap_95_interval"] == c.interval(ratios), "paired interval mismatch")
        h.require(len({m["error_sha256"] for m in row["measurements"].values()}) == 1, "cross-variant error mismatch")
        h.require(cell not in invariant or invariant[cell] == stream, "cross-round parity mismatch")
        invariant[cell] = stream
    for row in payload["allocations"]:
        cell, name = key(row), row["variant"]
        identifier = (cell, name)
        h.require(
            cell in expected and name in ("baseline", "compact") and identifier not in allocations,
            "unexpected/duplicate allocation cell",
        )
        allocations[identifier] = row
        current = (
            row["fixture_sha256"],
            row["model_sha256"],
            row["normalized_input_bytes"],
            row["tokens"],
            row["output_sha256"],
            row.get("path_score_float_hex"),
        )
        h.require(invariant[cell] == current, "allocation/timing stream mismatch")
        counts = row["rust_allocations"]
        h.require(all(type(x) is int and x >= 0 for x in counts.values()), "invalid allocation counters")
        h.require(
            counts["requests"] > 0 and counts["peak_live_bytes"] >= counts["live_before_bytes"],
            "invalid allocation peak",
        )
    h.require(
        set(observed) == {(cell, round_number) for cell in expected for round_number in range(c.ROUNDS)},
        "incomplete timing matrix",
    )
    h.require(
        set(allocations) == {(cell, name) for cell in expected for name in ("baseline", "compact")},
        "incomplete allocation matrix",
    )
    return payload


def classify(comparisons):
    h.require(len(comparisons) == c.ROUNDS, "repeatability requires both independent rounds")
    if all(row["bootstrap_95_interval"][0] > 1 for row in comparisons):
        return "repeatable_gain"
    if all(row["bootstrap_95_interval"][1] < 1 for row in comparisons):
        return "repeatable_degradation"
    return "inconclusive_or_mixed"


def analyze(payload):
    groups = {}
    allocations = {(key(row), row["variant"]): row for row in payload["allocations"]}
    for row in payload["records"]:
        groups.setdefault(key(row), []).append(row)
    rows = []
    for cell, rounds in groups.items():
        rounds.sort(key=lambda row: row["round"])
        base = {field: rounds[0].get(field) for field in FIELDS}
        for name in ("baseline", "prefix", "compact"):
            measurements = [row["measurements"][name] for row in rounds]
            samples = [sample for measurement in measurements for sample in measurement["wall_ns_samples"]]
            median = statistics.median(samples)
            result = {
                **base,
                "variant": name,
                "normalized_input_bytes": rounds[0]["normalized_input_bytes"],
                "tokens": rounds[0]["tokens"],
                "latency_p50_ns": median,
                "latency_p95_ns": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
                "normalized_mb_per_second": rounds[0]["normalized_input_bytes"] * 1000 / median,
                "tokens_per_second": rounds[0]["tokens"] * 1e9 / median,
                "process_cpu_percent_rounds": [measurement["process_cpu_percent"] for measurement in measurements],
                "cpu_observation_wall_ns_rounds": [
                    measurement["cpu_observation_wall_ns"] for measurement in measurements
                ],
                "fixture_sha256": rounds[0]["fixture_sha256"],
                "model_sha256": rounds[0]["model_sha256"],
                "output_sha256": measurements[0]["output_sha256"],
                "native_sha256": measurements[0]["native_sha256"],
            }
            if name != "baseline":
                comparisons = [row["comparisons"][name] for row in rounds]
                ratios = [value for comparison in comparisons for value in comparison["paired_ratios"]]
                result.update(
                    speedup_median=statistics.median(ratios),
                    classification=classify(comparisons),
                    speedup_rounds=[comparison["speedup_median"] for comparison in comparisons],
                    bootstrap_95_intervals_rounds=[comparison["bootstrap_95_interval"] for comparison in comparisons],
                    interval_envelope=[
                        min(x["bootstrap_95_interval"][0] for x in comparisons),
                        max(x["bootstrap_95_interval"][1] for x in comparisons),
                    ],
                )
            if name != "prefix":
                allocation = allocations[(cell, name)]
                result.update(allocation["rust_allocations"])
                result["temporary_peak_live_bytes"] = result["peak_live_bytes"] - result["live_before_bytes"]
                result["process_peak_rss_bytes"] = allocation["process_peak_rss_bytes"]
                result["process_peak_before_measurement_bytes"] = allocation["process_peak_before_measurement_bytes"]
            rows.append(result)
    return rows


def plots(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#666666", "prefix": "#4477AA", "compact": "#228833"}
    figure, axes = plt.subplots(3, 5, figsize=(19, 10), squeeze=False)
    for i, length in enumerate(p.LENGTHS):
        for j, script in enumerate(p.SCRIPTS):
            axis = axes[i][j]
            for variant in colors:
                selected = sorted(
                    [
                        row
                        for row in rows
                        if row["surface"] == "segmentation"
                        and row["length"] == length
                        and row["script"] == script
                        and row["variant"] == variant
                    ],
                    key=lambda row: row["vocab_budget"],
                )
                axis.plot(
                    [row["vocab_budget"] / 1024 for row in selected],
                    [row["normalized_mb_per_second"] for row in selected],
                    "o-",
                    color=colors[variant],
                    label=variant,
                )
            axis.set(title=f"{length} / {script}", xlabel="Vocabulary (Ki entries)", ylabel="Normalized MB/s")
            axis.grid(alpha=0.2)
    axes[0][0].legend()
    figure.suptitle("Cold raw segmentation: fixed script, length and vocabulary; medians over two paired rounds")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output / "segmentation-throughput.png", dpi=130)
    plt.close(figure)
    for metric, filename, ylabel in (
        ("normalized_mb_per_second", "batch-throughput.png", "Normalized MB/s"),
        ("requests", "batch-allocations.png", "Rust allocation requests"),
        ("temporary_peak_live_bytes", "batch-temporary-memory.png", "Temporary Rust peak requested bytes"),
        ("process_peak_rss_bytes", "batch-process-memory.png", "Process high-water RSS bytes"),
    ):
        figure, axes = plt.subplots(3, 4, figsize=(17, 10), squeeze=False)
        for i, length in enumerate(p.LENGTHS):
            for j, (api, cache) in enumerate((api, cache) for api in ("strings", "ids") for cache in ("warm", "cold")):
                axis = axes[i][j]
                for variant in colors if metric == "normalized_mb_per_second" else ("baseline", "compact"):
                    selected = sorted(
                        [
                            row
                            for row in rows
                            if row["surface"] == "batch"
                            and row["length"] == length
                            and row["output"] == api
                            and row["cache"] == cache
                            and row["variant"] == variant
                        ],
                        key=lambda row: row["batch_size"],
                    )
                    axis.plot(
                        [row["batch_size"] for row in selected],
                        [row[metric] for row in selected],
                        "o-",
                        color=colors[variant],
                        label=variant,
                    )
                axis.set(
                    title=f"{length} / {api} / {cache}",
                    xlabel="Batch size",
                    ylabel=ylabel,
                    xscale="log",
                    xticks=p.BATCHES,
                )
                axis.set_xticklabels(p.BATCHES)
                axis.grid(alpha=0.2)
        axes[0][0].legend()
        figure.suptitle("Fused batch matrix: one Rayon worker; long words bypass cache in warm and cold cases")
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        figure.savefig(output / filename, dpi=130)
        plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric, ylabel in zip(
        axes,
        ("latency_p50_ns", "requested_bytes", "temporary_peak_live_bytes"),
        ("Latency ns", "Rust requested allocation bytes", "Temporary Rust peak requested bytes"),
    ):
        for variant in ("baseline", "compact"):
            selected = sorted(
                [row for row in rows if row["length"] == "stress" and row["variant"] == variant],
                key=lambda row: row["characters"],
            )
            axis.loglog(
                [row["characters"] for row in selected],
                [row[metric] for row in selected],
                "o-",
                color=colors[variant],
                label=variant,
            )
        axis.set(xlabel="Characters (all 16 prefixes present)", ylabel=ylabel)
        axis.grid(alpha=0.2)
    axes[0].legend()
    figure.suptitle("Dense prefix stress: maximum token length 16; observed scaling plus explicit O(n L) bound")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output / "dense-prefix-scaling.png", dpi=130)
    plt.close(figure)


def report(rows, payload):
    lines = [
        "# Native encode and segmentation comparison",
        "",
        "Two independent fresh-worker rounds, eleven paired repetitions per round, one Rayon worker.",
        "Primary throughput is normalized decimal input MB/s. All token/ID/offset/error/decode and selected-path score hashes agree.",
        "",
        "## Repeatability gates",
        "",
    ]
    for variant in ("prefix", "compact"):
        for surface in ("segmentation", "batch"):
            selected = [row for row in rows if row["variant"] == variant and row["surface"] == surface]
            counts = {
                label: sum(row["classification"] == label for row in selected)
                for label in ("repeatable_gain", "repeatable_degradation", "inconclusive_or_mixed")
            }
            lines.append(f"- {variant}, {surface}: {counts}")
    regressions = [
        row
        for row in rows
        if row["variant"] == "compact"
        and row["surface"] == "batch"
        and row["batch_size"] == 1
        and row["classification"] == "repeatable_degradation"
    ]
    lines += [
        "",
        f"Batch-size-one rejection gate: {'FAIL' if regressions else 'PASS'}; repeatable degraded cells: {len(regressions)}.",
        "",
        "## Every Cell",
        "",
        "`metrics.csv` contains all 97 cells for each variant, with p50/p95 latency, MB/s, parity-identical tokens/s, both CPU observation windows, and both round intervals. Raw requests/responses and hashes are in the source bundle.",
        "",
        "A repeatable gain requires both round interval lower bounds above 1. A repeatable degradation requires both upper bounds below 1. Other cells are inconclusive or mixed; they are never silently discarded. The interval envelope shown in CSV is the union of the two round intervals, not a new confidence interval.",
        "",
        "## Memory and Complexity",
        "",
        "Every batch size has isolated allocation/requested-byte/live-peak and process high-water RSS observations. Rust counts exclude Python and C++ heaps. Process RSS includes imports, model construction and validation; its high-water subtraction does not isolate live scratch. Prefix-only memory is not directly measured; its purpose is timing attribution while retaining the old lattice.",
        "",
        "The compact unpruned algorithm takes O(n L) time and O(n) scratch plus O(n) selected output. At most four fallback tokens are emitted per source character. With no maximum subword length, L can equal n, yielding quadratic time but linear scratch. The old dense lattice stores O(n L) owned edges and up to O(n L squared) copied prefix bytes. Pruned decoding retains the old lattice, edge sorting and tie behavior. No additional cache, buffer pooling, unbounded retention, trie representation change or SIMD is introduced.",
        "",
        "## Limits",
        "",
        "CPU uses pooled process CPU/wall over eleven blocks, including loop overhead and cache clears; Windows accounting remains quantized. These are fixed local fixtures and two process rounds, not a general hardware/corpus claim. Warm short/medium batches exercise the unchanged cache/output path. Long words exceed its 1024-byte cache cutoff in both cache scenarios. No corpus, held-out test or downstream LM is used.",
        "",
        "## Build Sources",
        "",
    ]
    lines += [f"- {name}: `{commit}`" for name, commit in payload["build_sources"].items()]
    lines += [
        "",
        f"Compiler: `{payload['compiler'].splitlines()[0]}`",
        "",
        "Release opt-level 3, LTO enabled, one codegen unit; allocation instrumentation disabled for all primary timings.",
        "",
    ]
    return "\n".join(lines)


def export(path, output):
    h.require(
        not output.exists() and not output.resolve().is_relative_to(path.resolve().parent),
        "output must be new and disjoint",
    )
    identity = h.runtime_identity()
    h.require(not identity["working_tree_dirty"], "commit reporting source before export")
    payload, receipt = verified(path)
    rows = analyze(payload)
    output.mkdir(parents=True)
    h.write_new_json(
        output / "summary.json", {"schema_version": 1, "source_results_sha256": h.file_hash(path), "records": rows}
    )
    (output / "metrics.csv").write_text(p.profile_csv(rows), encoding="utf-8")
    (output / "REPORT.md").write_text(report(rows, payload), encoding="utf-8")
    plots(rows, output)
    h.require(h.runtime_identity() == identity, "reporting source/runtime changed")
    h.write_new_json(
        output / "manifest.json",
        {
            "status": "complete",
            "source_manifest_sha256": h.file_hash(path.parent / "manifest.json"),
            "source_results_sha256": h.file_hash(path),
            "source_artifacts": receipt["artifacts"],
            "export_identity": identity,
            "artifacts": h.artifact_hashes(output),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)


if __name__ == "__main__":
    main()
