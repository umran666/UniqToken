"""Read-only multilingual diagnostics of frozen Phase A tokenizer artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.metadata
import json
from pathlib import Path

from benchmarks import run_phase_a as stages
from benchmarks import run_research_experiments as h
from benchmarks.tokenizer_failure_metrics import analyze_condition

SCHEMA_VERSION = 1
COHORT = ("sp_unigram", "boundary_bpe", "uniq_superbpe")
LABELS = {"sp_unigram": "SPM-Unigram", "boundary_bpe": "Boundary-BPE", "uniq_superbpe": "UT-SuperBPE"}
PLOT_METRICS = (
    ("bytes_per_token", "Normalized bytes per token"),
    ("tokens_per_unicode_character", "Tokens per Unicode character"),
    ("byte_fallback_percent", "Byte fallback (%)"),
    ("token_length_bytes_p95", "Token source length p95 (bytes)"),
    ("vocabulary_utilization_percent", "Usable vocabulary observed (%)"),
    ("rare_token_percent", "Rare in diagnostic training (%)"),
    ("cross_word_merge_rate_percent", "Cross-word merges / initial boundaries (%)"),
    ("whitespace_tokens_per_run", "Whitespace token intersections / run"),
    ("punctuation_tokens_per_run", "Punctuation token intersections / run"),
)
DEFINITIONS = {
    "bytes_per_token": "normalized UTF-8 bytes / emitted tokens; higher is more compression",
    "tokens_per_unicode_character": "emitted tokens / normalized Python Unicode characters",
    "byte_fallback_percent": "100 * canonical byte-token emissions / all emissions",
    "token_length_bytes": "exact decoded source-byte contribution per token; fallback contributes one byte, not its spelling length",
    "vocabulary_utilization_percent": "100 * distinct observed IDs / (vocabulary size - four controls); byte IDs remain eligible",
    "rare_token_percent": "100 * emissions with pooled diagnostic-training count <= rare_threshold / emissions; not full tokenizer-training rarity when capped",
    "cross_word_merge_rate_percent": "100 * observed SuperBPE pair applications / initial adjacent token boundaries, summed within documents; other engines have zero cross-word application events",
    "cross_word_token_percent": "100 * emitted tokens intersecting at least two whitespace-delimited non-whitespace fields / emissions; not morphological word segmentation",
    "fragmentation": "maximal whitespace (str.isspace) or Unicode category P runs; tokens_per_run is token-byte-span intersections / runs; split_run_percent is 100 * runs touched by >1 token / runs",
    "aggregation": "pool counts, vocabulary-ID unions and length histograms before ratios/quantiles; never average stratum rates",
    "nulls": "undefined zero-denominator ratios and absent validation coverage are null, never zero evidence",
}


def assignment_info(pairs):
    return {
        "documents": len(pairs),
        "source_utf8_bytes": sum(row["raw_utf8_bytes"] for row, _ in pairs),
        "normalized_utf8_bytes": sum(len(text.encode("utf-8")) for _, text in pairs),
        "unicode_characters": sum(len(text) for _, text in pairs),
        "assignment_hash": h.digest([h.digest(text) for _, text in pairs]),
        "document_ids_hash": h.digest([row["id"] for row, _ in pairs]),
        "strata": [
            {"domain": domain, "language": language, "documents": count}
            for (domain, language), count in sorted(
                Counter((row["domain"], row["language"]) for row, _ in pairs).items()
            )
        ],
    }


def select_training(pairs, cap):
    h.require(type(cap) is int and cap >= 0, "training document cap must be a non-negative integer")
    seen: Counter[tuple[str, str]] = Counter()
    selected = []
    for row, text in pairs:
        key = (row["domain"], row["language"])
        if not cap or seen[key] < cap:
            selected.append((row, text))
            seen[key] += 1
    return selected


def guard_split_paths(dataset_path, manifest):
    """Reject aliases to held-out test before the shared loader opens any split."""
    base = Path(dataset_path).resolve().parent
    paths = {split: (base / entry["path"]).resolve() for split, entry in manifest["splits"].items()}
    h.require(set(paths) == {"train", "validation", "test"}, "three frozen splits required")
    h.require(all(path.is_relative_to(base) for path in paths.values()), "split path escapes dataset directory")
    h.require(len(set(paths.values())) == 3, "split paths alias one another, including held-out test")
    return paths


def prepare_context(dataset_path, phase_a_path, *, training_cap=32, expected_training_strata=30):
    """Reconstruct historical assignments without opening or hashing the test file."""
    dataset_path, phase_a_path = Path(dataset_path), Path(phase_a_path)
    ledger = h.read_json(phase_a_path)
    meta = ledger["metadata"]
    h.require(meta.get("stage") == "A-SCREEN" and meta.get("result_label") == "SCREENING", "complete A-SCREEN required")
    guard_split_paths(dataset_path, h.read_json(dataset_path))
    train_rows, train, validation_rows, validation, source = stages.load_stage_source(dataset_path)
    h.require(
        not {row["id"] for row in train_rows}.intersection(row["id"] for row in validation_rows),
        "train/validation document ID leakage",
    )
    h.require(meta.get("dataset") == source, "Phase A dataset provenance mismatch")
    screen_train, screen_info = stages.stratified_screen_training(train_rows, train)
    screening, _ = stages.partition_validation(validation_rows, validation)
    h.require(
        meta["training"]["assignment_hash"] == screen_info["assignment_hash"], "Phase A training assignment mismatch"
    )
    h.require(meta["training"]["documents"] == len(screen_train), "Phase A training document count mismatch")
    h.require(
        meta["training"]["normalized_utf8_bytes"] == screen_info["actual_normalized_utf8_bytes"],
        "Phase A training byte count mismatch",
    )
    validation_info = assignment_info(screening)
    for name in ("assignment_hash", "documents", "normalized_utf8_bytes", "source_utf8_bytes", "unicode_characters"):
        h.require(meta["validation"][name] == validation_info[name], f"Phase A validation {name} mismatch")
    h.require(meta["validation"]["partition"] == "screening", "screening validation required")
    h.require(
        meta["validation"]["partition_version"] == stages.VALIDATION_PARTITION_VERSION, "partition version mismatch"
    )
    stages.validate_stage_ledger(
        ledger,
        meta["identity"],
        source,
        phase_a_path.parent,
        "A-SCREEN",
        [(name, budget) for name in h.COHORT for budget in h.VOCABS],
    )
    by_text = {text: row for row, text in zip(train_rows, train)}
    frozen_training = [(by_text[text], text) for text in screen_train]
    inventory = sorted({(row["domain"], row["language"]) for row, _ in frozen_training})
    h.require(len(inventory) == expected_training_strata, "unexpected frozen training stratum count")
    selected = select_training(frozen_training, training_cap)
    h.require(
        {(row["domain"], row["language"]) for row, _ in selected} == set(inventory),
        "training diagnostic lost a stratum",
    )
    assignments = {"train": selected, "validation": screening}
    validation_languages = {row["language"] for row, _ in screening}
    coverage = [
        {
            "training_domain": domain,
            "language": language,
            "training_diagnostic_available": True,
            "screening_validation_available": language in validation_languages,
            "validation_source_domains": sorted({row["domain"] for row, _ in screening if row["language"] == language}),
        }
        for domain, language in inventory
    ]
    return (
        ledger,
        assignments,
        {
            "dataset": source,
            "phase_a_ledger_sha256": h.file_hash(phase_a_path),
            "phase_a_identity": meta["identity"],
            "frozen_tokenizer_training": meta["training"],
            "frozen_screening_validation": meta["validation"],
            "diagnostic_assignments": {split: assignment_info(pairs) for split, pairs in assignments.items()},
            "coverage": coverage,
        },
    )


def check_legacy_metrics(records, original):
    metric = next(row for row in records if row["split"] == "validation" and row["scope"] == "aggregate")
    for name in (
        "tokens",
        "bytes_per_token",
        "tokens_per_unicode_character",
        "byte_fallback_tokens",
        "byte_fallback_percent",
    ):
        h.require(metric[name] == original["validation"][name], f"frozen Phase A validation metric drift: {name}")


def csv_text(records):
    import io

    stream = io.StringIO(newline="")
    fields = list(records[0])
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in record.items()
            }
        )
    return stream.getvalue()


def write_text(path, text):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def plot_results(records, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    colors = ("#2378a8", "#be6b24", "#27875e")
    budgets = sorted({row["vocab_budget"] for row in records})
    paths = []
    for split in ("train", "validation"):
        aggregate = [row for row in records if row["split"] == split and row["scope"] == "aggregate"]
        fig, axes = plt.subplots(3, 3, figsize=(16, 13), constrained_layout=True)
        for ax, (metric, label) in zip(axes.flat, PLOT_METRICS):
            for name, color in zip(COHORT, colors):
                points = sorted((row["vocab_budget"], row[metric]) for row in aggregate if row["tokenizer"] == name)
                ax.plot(
                    [budget / 1024 for budget, _ in points],
                    [value for _, value in points],
                    marker="o",
                    label=LABELS[name],
                    color=color,
                )
            ax.set(title=label, xlabel="Vocabulary (Ki entries)", xticks=[budget / 1024 for budget in budgets])
            ax.grid(axis="y", alpha=0.2)
        axes.flat[0].legend()
        fig.suptitle(f"{split.capitalize()} diagnostics: pooled counts; descriptive evidence", fontsize=14)
        path = output / f"aggregate_{split}.png"
        fig.savefig(path, dpi=130, metadata={"Software": "UniqToken failure analysis v1"})
        plt.close(fig)
        paths.append(path)
        for budget in budgets:
            rows = [
                row
                for row in records
                if row["split"] == split and row["scope"] == "language" and row["vocab_budget"] == budget
            ]
            languages = sorted({row["language"] for row in rows})
            fig, axes = plt.subplots(3, 3, figsize=(22, 17), constrained_layout=True)
            for ax, (metric, label) in zip(axes.flat, PLOT_METRICS):
                for index, (name, color) in enumerate(zip(COHORT, colors)):
                    lookup = {row["language"]: row[metric] for row in rows if row["tokenizer"] == name}
                    points = [(position, lookup.get(language)) for position, language in enumerate(languages)]
                    points = [(position, value) for position, value in points if value is not None]
                    ax.bar(
                        [position + (index - 1) * 0.25 for position, _ in points],
                        [value for _, value in points],
                        width=0.25,
                        color=color,
                        label=LABELS[name],
                    )
                ax.set(title=label, xticks=range(len(languages)), xticklabels=languages)
                ax.tick_params(axis="x", labelrotation=90)
                ax.grid(axis="y", alpha=0.2)
            axes.flat[0].legend()
            fig.suptitle(f"{split.capitalize()} languages, V={budget}: missing validation is not zero", fontsize=14)
            path = output / f"languages_{split}_{budget}.png"
            fig.savefig(path, dpi=130, metadata={"Software": "UniqToken failure analysis v1"})
            plt.close(fig)
            paths.append(path)
    return paths


def report_text(payload):
    config, inputs, records = payload["configuration"], payload["inputs"], payload["records"]
    coverage = inputs["coverage"]
    missing = [
        f"{row['training_domain']}/{row['language']}" for row in coverage if not row["screening_validation_available"]
    ]
    lines = [
        "# Multilingual tokenizer failure diagnostics",
        "",
        "Descriptive tokenizer-only evidence. These associations do not establish causes or a globally best tokenizer. No LM was initialized or trained; held-out test data was not opened or hashed. The exploratory 16K byte-matched Phase B result is not confirmation.",
        "",
        "## Scope and provenance",
        "",
        f"- Analysis source commit: `{payload['identity']['commit_hash']}`; source hash: `{payload['identity']['source_hash']}`.",
        f"- Frozen dataset manifest SHA-256: `{inputs['dataset']['manifest_sha256']}`.",
        f"- Frozen Phase A ledger SHA-256: `{inputs['phase_a_ledger_sha256']}`.",
        f"- Training diagnostics: ordered whole-document prefixes, cap {config['max_training_documents_per_stratum']} per stratum (0 means all), across {len(coverage)} frozen training strata.",
        "- Validation: complete original screening half, using the frozen normalized-document SHA-256 parity assignment. Confirmation validation is excluded from scoring.",
        f"- Missing screening-validation languages: {', '.join(missing) or 'none'}. These have training diagnostics only; no substitute validation corpus was created.",
        "- Validation domain labels remain FLORES source labels; they are not relabeled as training web/code domains.",
        "",
        "| Split | Documents | Normalized bytes | Unicode characters | Strata |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, info in inputs["diagnostic_assignments"].items():
        lines.append(
            f"| {split} | {info['documents']} | {info['normalized_utf8_bytes']} | {info['unicode_characters']} | {len(info['strata'])} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate observations",
            "",
            "Aggregate ratios pool counts; languages and domains remain separate in the CSVs and plots.",
            "",
            "| Split | Vocab | Tokenizer | Bytes/token | Tokens/character | Byte fallback % | Observed vocab % | Rare % | Cross-word merge % |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in records:
        if row["scope"] == "aggregate":
            values = [
                row[key]
                for key in (
                    "bytes_per_token",
                    "tokens_per_unicode_character",
                    "byte_fallback_percent",
                    "vocabulary_utilization_percent",
                    "rare_token_percent",
                    "cross_word_merge_rate_percent",
                )
            ]
            lines.append(
                f"| {row['split']} | {row['vocab_budget']} | {LABELS[row['tokenizer']]} | "
                + " | ".join(f"{value:.4f}" if value is not None else "NA" for value in values)
                + " |"
            )
    heldout = [row for row in records if row["split"] == "validation" and row["scope"] == "stratum"]
    lookup = {(row["tokenizer"], row["vocab_budget"], row["domain"], row["language"]): row for row in heldout}
    differences = []
    for row in heldout:
        if row["tokenizer"] == "uniq_superbpe":
            reference = lookup[("sp_unigram", row["vocab_budget"], row["domain"], row["language"])]
            differences.append((row["byte_fallback_percent"] - reference["byte_fallback_percent"], row, reference))
    lines.extend(
        [
            "",
            "## Largest observed failure signals",
            "",
            "The following are measured on matching screening-validation documents and vocabulary budgets. They describe this cohort and assignment only.",
            "",
            "| Vocab | Language | UT-SuperBPE byte fallback % | SPM-Unigram byte fallback % | Difference (percentage points) |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for difference, row, reference in sorted(
        differences, key=lambda item: (-item[0], item[1]["vocab_budget"], item[1]["language"])
    )[:8]:
        lines.append(
            f"| {row['vocab_budget']} | {row['language']} | {row['byte_fallback_percent']:.4f} | {reference['byte_fallback_percent']:.4f} | {difference:+.4f} |"
        )
    lines.extend(
        [
            "",
            "| Vocab | Tokenizer | Language | Punctuation tokens/run | Whitespace tokens/run |",
            "| ---: | --- | --- | ---: | ---: |",
        ]
    )
    for row in sorted(
        heldout,
        key=lambda item: (
            -(item["punctuation_tokens_per_run"] or 0),
            item["vocab_budget"],
            item["tokenizer"],
            item["language"],
        ),
    )[:8]:
        values = [row["punctuation_tokens_per_run"], row["whitespace_tokens_per_run"]]
        lines.append(
            f"| {row['vocab_budget']} | {LABELS[row['tokenizer']]} | {row['language']} | "
            + " | ".join(f"{value:.4f}" if value is not None else "NA" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Plausible hypotheses, not causal conclusions",
            "",
            "- Where fallback percentages differ, investigate byte-fallback edge scores and vocabulary coverage in a controlled matched-candidate ablation (#86). This diagnostic does not isolate those mechanisms.",
            "- Where language-specific vocabulary utilization and rare-token emissions differ, investigate training allocation and merge distribution (#88). Sample size, scripts and source-domain differences are alternative explanations.",
            "- Where punctuation or whitespace runs touch many tokens, inspect boundary rules with fixed normalization and vocabulary (#89). Multi-byte fallback can itself split one punctuation character; token/run ratios do not prove a boundary-rule defect.",
            "",
            "## Metric definitions and limitations",
            "",
        ]
    )
    lines.extend(f"- `{name}`: {definition}." for name, definition in DEFINITIONS.items())
    lines.extend(
        [
            f"- Rare threshold: <= {config['rare_threshold']} occurrences in the pooled diagnostic-training sample. Counts of zero are included and reported separately. With a nonzero document cap this is sample rarity, not the learned model's complete training frequency.",
            "- Vocabulary utilization and tail frequencies depend on sample exposure. Training observations are in-sample; they cannot establish held-out code behavior. Missing validation coverage remains an evidence gap for #94.",
            "- A cross-word event is an application in the observed SuperBPE pass; emitted cross-field tokens are reported separately. These are not estimates of a causal compression benefit.",
            "- UTF-8 token source contributions exactly reconstruct each normalized document. Byte fallback contributes one byte even when multiple bytes represent one Unicode character. No raw-offset non-overlap assumption is imposed on public Token spans.",
            "- Plots and tables use the same recorded counts. No significance tests, downstream advantage, algorithm change, or Phase A/B/C confirmation is claimed.",
            "",
            "See `results.json` for configuration, definitions, model hashes, assignments and all records; `strata.csv`, `languages.csv`, `domains.csv`, `aggregate.csv`, `coverage.csv`, and `token_lengths.csv` for separate tables. PNGs show aggregate and per-language diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(payload, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    h.write_new_json(output / "results.json", payload)
    for scope, filename in (
        ("stratum", "strata.csv"),
        ("language", "languages.csv"),
        ("domain", "domains.csv"),
        ("aggregate", "aggregate.csv"),
    ):
        write_text(output / filename, csv_text([row for row in payload["records"] if row["scope"] == scope]))
    write_text(output / "coverage.csv", csv_text(payload["inputs"]["coverage"]))
    lengths = []
    for row in payload["records"]:
        for length, count in row["token_length_bytes_histogram"].items():
            lengths.append(
                {
                    **{key: row[key] for key in ("split", "scope", "domain", "language", "tokenizer", "vocab_budget")},
                    "source_bytes": int(length),
                    "tokens": count,
                }
            )
    write_text(output / "token_lengths.csv", csv_text(lengths))
    write_text(output / "REPORT.md", report_text(payload))
    if payload["configuration"]["plots"]:
        plot_results(payload["records"], output)
    files = {path.name: h.file_hash(path) for path in sorted(output.iterdir()) if path.is_file()}
    h.write_new_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "artifacts": files,
            "results_content_sha256": h.digest(payload),
        },
    )


def run(args):
    output = Path(args.output).resolve()
    dataset, phase_a = Path(args.dataset).resolve(), Path(args.phase_a).resolve()
    h.require(not output.exists(), "diagnostic output already exists")
    h.require(
        not output.is_relative_to(dataset.parent) and not output.is_relative_to(phase_a.parent),
        "output must be separate from frozen inputs",
    )
    h.require(
        type(args.rare_threshold) is int and args.rare_threshold >= 0, "rare threshold must be a non-negative integer"
    )
    h.require(
        type(args.max_training_documents_per_stratum) is int and args.max_training_documents_per_stratum >= 0,
        "training document cap must be a non-negative integer",
    )
    identity = h.runtime_identity()
    h.require(not identity["working_tree_dirty"], "commit the diagnostic code before research runs")
    identity["versions"]["matplotlib"] = importlib.metadata.version("matplotlib") if not args.no_plots else None
    ledger, assignments, inputs = prepare_context(
        dataset, phase_a, training_cap=args.max_training_documents_per_stratum
    )
    selected = [
        row
        for name in COHORT
        for budget in args.vocab_budget
        for row in ledger["records"]
        if (row["tokenizer"], row["vocab_budget"]) == (name, budget)
    ]
    h.require(len(selected) == len(COHORT) * len(args.vocab_budget), "diagnostic cohort incomplete")
    records, models = [], []
    for original in selected:
        name, budget = original["tokenizer"], original["vocab_budget"]
        print(f"Analyzing {name} V={budget}: training and screening validation", flush=True)
        tokenizer = h.load_tokenizer(original, phase_a.parent)
        measured = analyze_condition(tokenizer, assignments, rare_threshold=args.rare_threshold)
        check_legacy_metrics(measured, original)
        records.extend(measured)
        models.append(
            {
                key: original[key]
                for key in (
                    "tokenizer",
                    "vocab_budget",
                    "artifact",
                    "artifact_hashes",
                    "git_commit",
                    "extension_hash",
                    "training_assignment_hash",
                    "validation_assignment_hash",
                    "tokenizer_config",
                )
            }
        )
        h.require(
            h.artifact_hashes(phase_a.parent / original["artifact"]) == original["artifact_hashes"],
            "frozen model changed during diagnostics",
        )
        print(f"Completed {name} V={budget}; legacy validation metrics match", flush=True)
    h.require(
        h.file_hash(dataset) == inputs["dataset"]["manifest_sha256"], "dataset manifest changed during diagnostics"
    )
    manifest = h.read_json(dataset)
    split_paths = guard_split_paths(dataset, manifest)
    for split in ("train", "validation"):
        h.require(
            h.file_hash(split_paths[split]) == manifest["splits"][split]["sha256"],
            f"{split} data changed during diagnostics",
        )
    h.require(h.file_hash(phase_a) == inputs["phase_a_ledger_sha256"], "Phase A ledger changed during diagnostics")
    current = h.runtime_identity()
    h.require(
        all(
            current[key] == identity[key]
            for key in ("commit_hash", "source_hash", "extension_hash", "working_tree_dirty")
        ),
        "analysis source/runtime changed during diagnostics",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "result_label": "DIAGNOSTIC_DESCRIPTIVE",
        "test_access": "forbidden_not_opened_or_hashed",
        "language_model": {"initialized": False, "trained": False},
        "identity": identity,
        "configuration": {
            "cohort": list(COHORT),
            "vocab_budgets": list(args.vocab_budget),
            "max_training_documents_per_stratum": args.max_training_documents_per_stratum,
            "training_selection": "ordered_whole_document_stratum_prefix_v1",
            "validation_partition": "screening",
            "rare_threshold": args.rare_threshold,
            "dropout_prob": 0.0,
            "boundary_bpe_reader": "validated_immutable_maps_v1",
            "plots": not args.no_plots,
        },
        "metric_definitions": DEFINITIONS,
        "inputs": inputs,
        "models": models,
        "records": records,
    }
    write_outputs(payload, output)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Frozen Phase A dataset manifest; test path is never opened"
    )
    parser.add_argument(
        "--phase-a", type=Path, required=True, help="Completed A-SCREEN ledger and unchanged model directories"
    )
    parser.add_argument("--output", type=Path, required=True, help="New directory, separate from all frozen inputs")
    parser.add_argument("--vocab-budget", type=int, nargs="+", choices=h.VOCABS, default=list(h.VOCABS))
    parser.add_argument(
        "--max-training-documents-per-stratum",
        type=int,
        default=32,
        help="Whole-document prefix cap per frozen training stratum; 0 uses all",
    )
    parser.add_argument("--rare-threshold", type=int, default=5)
    parser.add_argument(
        "--no-plots", action="store_true", help="Tables-only run; normal runs produce aggregate/language PNGs"
    )
    args = parser.parse_args()
    h.require(len(set(args.vocab_budget)) == len(args.vocab_budget), "duplicate vocabulary budgets")
    payload = run(args)
    print(
        json.dumps(
            {"status": "complete", "records": len(payload["records"]), "output": str(args.output)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
