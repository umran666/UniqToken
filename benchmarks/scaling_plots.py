"""Plots and descriptive reporting for the predeclared vocabulary matrix."""

from __future__ import annotations

METRICS = (
    ("bytes_per_token", "Normalized bytes/token"),
    ("byte_fallback_percent", "Byte fallback (%)"),
    ("tokens_per_unicode_character", "Tokens/Unicode code point"),
    ("vocabulary_utilization_percent", "Observed usable vocabulary (%)"),
    ("training_seconds", "Training + serialization (s)"),
    ("training_peak_rss_bytes", "Process peak RSS (bytes)"),
)


def metric_rows(payload):
    return [
        {
            **row,
            "training_seconds": condition["training_seconds"],
            "training_peak_rss_bytes": condition["training_process_peak_rss_bytes"],
        }
        for condition in payload["conditions"]
        if condition["status"] == "complete"
        for row in condition["records"]
    ]


def plot_scaling(payload, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = metric_rows(payload)
    cohort = payload["configuration"]["cohort"]
    budgets = payload["configuration"]["budgets"]
    colors = ("#2378a8", "#be6b24", "#27875e")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, METRICS):
        for name, color in zip(cohort, colors):
            lookup = {
                r["vocab_budget"]: r[metric]
                for r in rows
                if r["tokenizer"] == name and r["split"] == "validation" and r["scope"] == "aggregate"
            }
            ax.plot([b / 1024 for b in budgets], [lookup.get(b) for b in budgets], marker="o", label=name, color=color)
        ax.set(title=label, xlabel="Vocabulary (Ki entries)", xticks=[b / 1024 for b in budgets])
        ax.grid(axis="y", alpha=0.2)
    axes.flat[0].legend()
    fig.suptitle("Fixed-corpus tokenizer scaling; gaps are failed or blocked conditions")
    fig.savefig(output / "aggregate.png", dpi=130)
    plt.close(fig)
    languages = sorted({r["language"] for r in rows if r["split"] == "validation" and r["scope"] == "language"})
    for metric, label in METRICS[:4]:
        fig, axes = plt.subplots(5, 4, figsize=(20, 21), constrained_layout=True)
        for ax, language in zip(axes.flat, languages):
            for name, color in zip(cohort, colors):
                lookup = {
                    r["vocab_budget"]: r[metric]
                    for r in rows
                    if r["tokenizer"] == name
                    and r["split"] == "validation"
                    and r["scope"] == "language"
                    and r["language"] == language
                }
                ax.plot(
                    [b / 1024 for b in budgets], [lookup.get(b) for b in budgets], marker="o", label=name, color=color
                )
            ax.set(title=language, xlabel="Vocabulary (Ki entries)", xticks=[b / 1024 for b in budgets])
            ax.grid(axis="y", alpha=0.2)
        for ax in list(axes.flat)[len(languages) :]:
            ax.set_visible(False)
        axes.flat[0].legend()
        fig.suptitle(label + ": observed validation languages")
        fig.savefig(output / f"languages_{metric}.png", dpi=100)
        plt.close(fig)


def scaling_report(payload):
    lines = [
        "# Vocabulary budget scaling",
        "",
        "Tokenizer-only descriptive matrix on one frozen diagnostic train/validation selection.",
        "No held-out test access, LM run, or modification of frozen Phase A/B/C artifacts.",
        "UT-SuperBPE uses 64 reserved merge slots at every budget (uniq_superbpe_r64); this is explicitly a diagnostic variant, not a replay of the Phase A training configuration.",
        "All budgets include four controls and 256 byte leaves. Underfilled conditions have failure receipts and no favorable metric imputation.",
        "",
        "| Tokenizer | Budget | Status | Training s | Peak RSS MiB |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for condition in payload["conditions"]:
        elapsed = condition.get("training_seconds")
        peak = condition.get("training_process_peak_rss_bytes")
        elapsed_text = "NA" if elapsed is None else f"{elapsed:.4f}"
        peak_text = "NA" if peak is None else f"{peak / 1024**2:.2f}"
        lines.append(
            f"| {condition['tokenizer']} | {condition['vocab_budget']} | {condition['status']} | {elapsed_text} | {peak_text} |"
        )
        if condition["status"] != "complete":
            lines.extend(
                [
                    "",
                    f"Failure receipt for {condition['tokenizer']} at {condition['vocab_budget']}: "
                    + condition.get("error", condition["status"]),
                    "",
                ]
            )
    lines.extend(
        [
            "",
            "## Measurement scope",
            "",
            payload["time_method"] + ".",
            payload["memory_method"] + ". The pre-training process high-water RSS is recorded separately.",
            "Each condition runs in a new process, with one Rayon worker, Python hash seed zero and deterministic trainer configuration. There is one timing observation per condition; no statistical confidence interval is claimed.",
            "The driver runs conditions sequentially. Model validation and all train/validation diagnostics precede a complete receipt.",
            "Density uses normalized Unicode code points and UTF-8 bytes. Token lengths are exact normalized source-byte contributions. Metrics, histograms and utilization retain aggregate/domain/language/stratum detail.",
            "The plots leave gaps for failed conditions. Logs retain the original failure. Models and output hashes are included in the final manifest.",
            "Validation has FLORES domain labels. Training domains and English/code languages without validation cannot be certified from this matrix.",
            "Scaling curves describe this bounded corpus and trainer configuration. They do not establish a universal scaling law or a global winner.",
            "",
        ]
    )
    return "\n".join(lines)
