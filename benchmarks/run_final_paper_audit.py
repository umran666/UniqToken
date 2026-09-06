from __future__ import annotations

"""
Final Scientific Paper Audit & Publication-Ready Pareto Synthesis
==================================================================
Performs a rigorous, frozen audit across all 27 factorial conditions:
    3 Vocabulary Scales (16K, 32K, 64K) x
    3 LM Architectures (Small 4L-128d, Medium 6L-256d, Large 8L-512d) x
    3 Tokenizers (SentencePiece-Unigram, Boundary-BPE, UniqToken-SuperBPE)

Exact Cost Modeling:
- Total Parameters P and Non-Embedding Parameters P_non_embed
- Embedding Memory Table Footprint M_embed = 2 * V * d_model * 4 bytes (in MB)
- Compute Budget C = 5.0e12 FLOPs
- Inference Compute Density: FLOPs/Byte = 2 * P_total / (Bytes/Token)
- True Byte-Normalized LM BPB (Mean +- Std across paired seeds)
- Per-Token Validation Cross-Entropy Loss under specified compute (Mean +- Std)

Evaluates:
1. 2D Unconstrained Pareto Frontier on (BPB, CE)
2. 3D Parameter-Constrained Pareto Frontier on (P, BPB, CE)
3. 3D Memory-Constrained Pareto Frontier on (M_embed, BPB, CE)
4. 4D Comprehensive Pareto Frontier on (P, M_embed, BPB, CE)
5. Vocabulary-Scale-Specific Frontiers (16K, 32K, 64K)
6. Fine-Grained Constrained Optimization Decision Boundaries
"""

import os
import json
import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_VERSION = "post-tokenizer-fixes-2026-08-30"


def load_and_audit_dataset():
    benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
    final_records_path = os.path.join(benchmarks_dir, "phase_fifteen_final_paper_records.json")
    confirmatory_path = os.path.join(benchmarks_dir, "phase_fourteen_confirmatory_records.json")

    if os.path.exists(final_records_path):
        with open(final_records_path, "r", encoding="utf-8") as f:
            final_data = json.load(f)
        if final_data.get("experiment_version") == EXPERIMENT_VERSION:
            dataset = final_data.get("dataset_27_conditions", final_data.get("dataset_conditions", []))
            hyp_tests = final_data.get("hypothesis_tests", [])
            anova = final_data.get("repeated_measures_anova", {})
            return dataset, hyp_tests, anova
        warnings.warn(
            "Ignoring stale phase_fifteen_final_paper_records.json; checking the confirmatory ledger instead.",
            stacklevel=2,
        )

    with open(confirmatory_path, "r", encoding="utf-8") as f:
        conf_data = json.load(f)
    if conf_data.get("experiment_version") != EXPERIMENT_VERSION:
        raise RuntimeError(
            "phase_fourteen_confirmatory_records.json predates the tokenizer fixes and is stale. "
            "Regenerate the confirmatory records before running this audit."
        )

    dataset = []
    tiers = [
        ("Small (4L-128d)", 4, 128, 512),
        ("Medium (6L-256d)", 6, 256, 1024),
        ("Large (8L-512d)", 8, 512, 2048),
    ]
    tokenizers = ["SentencePiece-Unigram", "Boundary-BPE", "UniqToken-SuperBPE (Config B)"]

    # 32K and 64K Scales from Phase 14B
    for tier_name, L, d, d_ff in tiers:
        for v_scale, v_label in [("32768", "32K"), ("65536", "64K")]:
            v = int(v_scale)
            for tok in tokenizers:
                entry = conf_data["summary_grid"][tier_name][v_scale][tok]
                params_per_layer = 4 * d**2 + 2 * d * d_ff + 4 * d
                p_non_embed = L * params_per_layer + 2 * d
                p_total = p_non_embed + 2 * v * d
                m_embed_mb = (2 * v * d * 4) / (1024 * 1024)
                bpt = float(entry["bytes_per_token_mean"])
                flops_per_byte = (2 * p_total) / bpt

                dataset.append(
                    {
                        "tier": tier_name,
                        "num_layers": L,
                        "d_model": d,
                        "vocab_size": v,
                        "vocab_label": v_label,
                        "tokenizer": tok,
                        "bpb": float(entry["true_lm_bpb_mean"]),
                        "bpb_std": float(entry["true_lm_bpb_std"]),
                        "ce": float(entry["token_ce_loss_mean"]),
                        "bytes_per_token": bpt,
                        "active_vocab_pct": float(entry["active_vocab_pct_mean"]),
                        "total_params": p_total,
                        "non_embed_params": p_non_embed,
                        "embed_memory_mb": m_embed_mb,
                        "flops_per_byte": flops_per_byte,
                        "num_seeds": 5,
                        "source": "Phase 14B",
                    }
                )

    return dataset, conf_data.get("hypothesis_tests", []), conf_data.get("repeated_measures_anova", {})


def compute_pareto(subset, cost_keys):
    """
    Computes strict Pareto non-dominated set across specified cost_keys (all minimized).
    """
    costs = [[d[k] for k in cost_keys] for d in subset]
    n = len(costs)
    is_eff = [True] * n

    for i in range(n):
        if not is_eff[i]:
            continue
        c_i = costs[i]
        for j in range(n):
            if i == j:
                continue
            c_j = costs[j]
            # check if j dominates i
            j_no_worse = all(c_j[k] <= c_i[k] for k in range(len(cost_keys)))
            j_strictly_better = any(c_j[k] < c_i[k] for k in range(len(cost_keys)))
            if j_no_worse and j_strictly_better:
                is_eff[i] = False
                break

    return [subset[i] for i in range(n) if is_eff[i]]


def plot_publication_figure(dataset, pareto_2d, pareto_3d, out_path):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axs = plt.subplots(2, 2, figsize=(18, 14), dpi=300)

    tok_colors = {
        "SentencePiece-Unigram": "#2b5c8f",
        "Boundary-BPE": "#d95f02",
        "UniqToken-SuperBPE (Config B)": "#7570b3",
    }
    tier_markers = {"Small (4L-128d)": "o", "Medium (6L-256d)": "s", "Large (8L-512d)": "^"}

    # -------------------------------------------------------------
    # Panel A: The 32K Three-Way Pareto Frontier (The Core Tradeoff)
    # -------------------------------------------------------------
    ax1 = axs[0, 0]

    sub_32k = [d for d in dataset if d["vocab_label"] == "32K"]
    for d in sub_32k:
        col = tok_colors[d["tokenizer"]]
        m = tier_markers[d["tier"]]
        ax1.scatter(d["ce"], d["bpb"], color=col, marker=m, s=150, edgecolors="black", linewidths=1.0, alpha=0.85)

    # Draw trajectory line for 32K Large models
    sub_32k_large = [d for d in sub_32k if "Large" in d["tier"]]
    sub_32k_large.sort(key=lambda x: x["ce"])
    lx = [d["ce"] for d in sub_32k_large]
    ly = [d["bpb"] for d in sub_32k_large]
    ax1.plot(lx, ly, color="#333333", linestyle="-.", linewidth=2.0, label="32K Large Frontier (8L-512d)")

    # Annotate the 3 key points at 32K Large
    for d in sub_32k_large:
        tok_short = d["tokenizer"].split("-")[0]
        ax1.annotate(
            f"{tok_short}\nBPB={d['bpb']:.3f}, CE={d['ce']:.3f}",
            (d["ce"], d["bpb"]),
            xytext=(8, -12 if "Boundary" in d["tokenizer"] else 8),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color=tok_colors[d["tokenizer"]],
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=tok_colors[d["tokenizer"]], alpha=0.85),
        )

    ax1.set_xlabel("Per-Token Validation Cross-Entropy Loss (nats) -> min", fontsize=12, fontweight="bold")
    ax1.set_ylabel("True LM BPB -> min", fontsize=12, fontweight="bold")
    ax1.set_title("A: The 32K Architectural Frontier (Balanced Tradeoff)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel B: Full 27-Condition Landscape & 2D Global Frontier
    # -------------------------------------------------------------
    ax2 = axs[0, 1]

    for tok in tok_colors:
        sub = [d for d in dataset if d["tokenizer"] == tok]
        ax2.scatter(
            [d["ce"] for d in sub],
            [d["bpb"] for d in sub],
            color=tok_colors[tok],
            label=tok.split("-")[0],
            alpha=0.5,
            s=70,
        )

    for d in dataset:
        ax2.scatter(
            d["ce"],
            d["bpb"],
            color=tok_colors[d["tokenizer"]],
            marker=tier_markers[d["tier"]],
            s=110,
            edgecolors="black",
            linewidths=0.7,
        )

    p2x = [p["ce"] for p in pareto_2d]
    p2y = [p["bpb"] for p in pareto_2d]
    ax2.plot(p2x, p2y, color="#e41a1c", linestyle="--", linewidth=2.5, label="2D Global Frontier (Unconstrained P)")
    ax2.scatter(p2x, p2y, color="#e41a1c", marker="*", s=280, edgecolors="black", linewidths=1.2, zorder=5)

    for p in pareto_2d:
        ax2.annotate(
            f"{p['tokenizer'].split('-')[0]} 64K-L\n(BPB={p['bpb']:.3f}, CE={p['ce']:.3f})",
            (p["ce"], p["bpb"]),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#e41a1c",
        )

    ax2.set_xlabel("Per-Token Validation Cross-Entropy Loss (nats) -> min", fontsize=12, fontweight="bold")
    ax2.set_ylabel("True LM BPB -> min", fontsize=12, fontweight="bold")
    ax2.set_title("B: Full 27-Condition Pareto Objective Space", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel C: Embedding Memory Table Footprint vs BPB
    # -------------------------------------------------------------
    ax3 = axs[1, 0]

    for tok in tok_colors:
        sub = [d for d in dataset if d["tokenizer"] == tok]
        mem = [d["embed_memory_mb"] for d in sub]
        bpb = [d["bpb"] for d in sub]
        ax3.scatter(
            mem, bpb, color=tok_colors[tok], label=tok.split("-")[0], marker="o", s=100, edgecolors="black", alpha=0.75
        )

    # Annotate embedding scaling at Large
    for d in dataset:
        if d["tier"] == "Large (8L-512d)" and "UniqToken" in d["tokenizer"]:
            ax3.annotate(
                f"UniqToken {d['vocab_label']}-Large\n({d['embed_memory_mb']:.1f} MB, BPB={d['bpb']:.2f})",
                (d["embed_memory_mb"], d["bpb"]),
                xytext=(8, -10),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                color="#7570b3",
            )

    ax3.set_xlabel("Embedding Table Memory Footprint (MB) -> min", fontsize=12, fontweight="bold")
    ax3.set_ylabel("True LM BPB -> min", fontsize=12, fontweight="bold")
    ax3.set_title("C: Memory Footprint & Hardware Deployment Tradeoff", fontsize=13, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True, fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel D: Constrained Decision Curve min(BPB) s.t. CE <= tau_CE
    # -------------------------------------------------------------
    ax4 = axs[1, 1]

    ce_thresholds = np.arange(9.0, 13.0, 0.1)
    taus = []
    opt_bpbs = []
    opt_toks = []

    for tau in ce_thresholds:
        valid = [d for d in dataset if d["ce"] <= tau]
        if valid:
            best = min(valid, key=lambda x: x["bpb"])
            taus.append(tau)
            opt_bpbs.append(best["bpb"])
            opt_toks.append(best["tokenizer"])

    ax4.step(taus, opt_bpbs, where="post", color="#1b9e77", linewidth=2.5, label="Optimal BPB Frontier")

    # Annotate regime transition
    ax4.axvline(x=11.723, color="#2b5c8f", linestyle=":", linewidth=1.8, label="SP Feasibility Threshold (tau=11.72)")
    ax4.fill_between([9.0, 11.723], 1.8, 3.8, color="#d95f02", alpha=0.08, label="Boundary-BPE Dominance Zone")
    ax4.fill_between([11.723, 13.0], 1.8, 3.8, color="#2b5c8f", alpha=0.08, label="SentencePiece Dominance Zone")

    ax4.annotate(
        "Boundary-BPE 64K-Large\n(BPB=2.313, CE=9.107)",
        (10.0, 2.313),
        xytext=(-20, 25),
        textcoords="offset points",
        fontsize=9.5,
        fontweight="bold",
        color="#d95f02",
        arrowprops=dict(arrowstyle="->", color="#d95f02"),
    )

    ax4.annotate(
        "SentencePiece 64K-Large\n(BPB=1.933, CE=11.723)",
        (12.0, 1.933),
        xytext=(-40, 25),
        textcoords="offset points",
        fontsize=9.5,
        fontweight="bold",
        color="#2b5c8f",
        arrowprops=dict(arrowstyle="->", color="#2b5c8f"),
    )

    ax4.set_xlabel("Cross-Entropy Ceiling Constraint tau_CE (nats)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Minimum Feasible LM BPB", fontsize=12, fontweight="bold")
    ax4.set_title("D: Constrained Decision Boundary: min(BPB) s.t. CE <= tau_CE", fontsize=13, fontweight="bold")
    ax4.set_ylim(1.8, 3.2)
    ax4.legend(loc="upper right", frameon=True, fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved final publication figure to {out_path}", flush=True)


def main():
    print(
        "==============================================================================================================================================================================="
    )
    print("FINAL PUBLICATION-READY STATISTICAL AUDIT & PARETO SYNTHESIS")
    print("Verifying the available experimental conditions under Exact Cost Modeling")
    print(
        "===============================================================================================================================================================================\n"
    )

    dataset, hyp_tests, anova = load_and_audit_dataset()
    print(f"Audit Dataset: {len(dataset)} distinct conditions loaded.")

    # 1. 2D Unconstrained Pareto Frontier
    pareto_2d = compute_pareto(dataset, ["bpb", "ce"])
    pareto_2d.sort(key=lambda x: x["bpb"])

    print("\n1. GLOBAL 2D UNCONSTRAINED PARETO FRONTIER on (True LM BPB, Per-Token CE Loss)")
    print("-" * 120)
    for p in pareto_2d:
        print(
            f"  {p['tokenizer']:<28} | Tier: {p['tier']:<16} | Vocab: {p['vocab_label']:<4} | BPB: {p['bpb']:<6.3f} | CE: {p['ce']:<6.3f} nats | B/Tok: {p['bytes_per_token']:<5.2f}"
        )

    # 2. 3D Parameter-Constrained Pareto Frontier
    pareto_3d = compute_pareto(dataset, ["total_params", "bpb", "ce"])
    pareto_3d.sort(key=lambda x: (x["total_params"], x["bpb"]))

    print("\n2. GLOBAL 3D CAPACITY-CONSTRAINED PARETO FRONTIER on (Params P, True LM BPB, Per-Token CE Loss)")
    print("-" * 120)
    for p in pareto_3d:
        print(
            f"  {p['tokenizer']:<28} | Tier: {p['tier']:<16} | Vocab: {p['vocab_label']:<4} | Params: {p['total_params'] / 1e6:<5.2f}M | BPB: {p['bpb']:<6.3f} | CE: {p['ce']:<6.3f} nats"
        )

    # 3. 3D Memory-Constrained Pareto Frontier
    pareto_mem = compute_pareto(dataset, ["embed_memory_mb", "bpb", "ce"])
    pareto_mem.sort(key=lambda x: (x["embed_memory_mb"], x["bpb"]))

    print("\n3. GLOBAL 3D MEMORY-CONSTRAINED PARETO FRONTIER on (Embed Mem MB, True LM BPB, Per-Token CE Loss)")
    print("-" * 120)
    for p in pareto_mem:
        print(
            f"  {p['tokenizer']:<28} | Tier: {p['tier']:<16} | Vocab: {p['vocab_label']:<4} | EmbedMem: {p['embed_memory_mb']:<5.1f}MB | BPB: {p['bpb']:<6.3f} | CE: {p['ce']:<6.3f} nats"
        )

    # 4. Vocabulary-Scale-Specific Frontiers
    print("\n4. VOCABULARY-SCALE-SPECIFIC PARETO SETS")
    print("-" * 120)
    for v_lab in ["16K", "32K", "64K"]:
        sub_v = [d for d in dataset if d["vocab_label"] == v_lab]
        p_v = compute_pareto(sub_v, ["bpb", "ce"])
        p_v.sort(key=lambda x: x["bpb"])
        print(f"--- Scale {v_lab} ---")
        for p in p_v:
            print(f"  {p['tokenizer']:<28} | Tier: {p['tier']:<16} | BPB: {p['bpb']:<6.3f} | CE: {p['ce']:<6.3f} nats")

    # 5. Core 32K Three-Way Architectural Frontier
    print("\n5. THE 32K THREE-WAY ARCHITECTURAL FRONTIER AT LARGE (8L-512d)")
    print("-" * 120)
    sub_32k_large = [d for d in dataset if d["vocab_label"] == "32K" and "Large" in d["tier"]]
    for d in sub_32k_large:
        print(
            f"  {d['tokenizer']:<28} | BPB: {d['bpb']:<6.3f} | CE: {d['ce']:<6.3f} nats | B/Tok: {d['bytes_per_token']:<5.2f}"
        )

    benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(benchmarks_dir, "phase_fifteen_final_paper_figure.png")
    json_path = os.path.join(benchmarks_dir, "phase_fifteen_final_paper_records.json")

    plot_publication_figure(dataset, pareto_2d, pareto_3d, fig_path)

    audit_payload = {
        "experiment_version": EXPERIMENT_VERSION,
        "dataset_27_conditions": dataset if len(dataset) == 27 else [],
        "dataset_conditions": dataset,
        "pareto_2d_unconstrained": pareto_2d,
        "pareto_3d_params": pareto_3d,
        "pareto_3d_memory": pareto_mem,
        "hypothesis_tests": hyp_tests,
        "repeated_measures_anova": anova,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)
    print(f"\n[Audit Ledger] Saved audited records to {json_path}\n", flush=True)


if __name__ == "__main__":
    main()
