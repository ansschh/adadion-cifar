#!/usr/bin/env python3
"""
Generate plots for the feedback items:
  1. Communication overhead bar chart
  2. Rank-performance trade-off (from ablation data)
  3. Compression ratio comparison
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "adamw": "#4C72B0", "muon": "#DD8452", "dion": "#55A868",
    "dion2": "#C44E52", "adadion": "#8172B3",
}
LABELS = {
    "adamw": "AdamW", "muon": "Muon", "dion": "Dion",
    "dion2": "Dion2", "adadion": "AdaDion V2",
}
OPT_ORDER = ["adadion", "dion", "dion2", "muon", "adamw"]

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
                      "legend.fontsize": 10, "figure.dpi": 150, "savefig.bbox": "tight"})


def plot_communication_overhead(output_dir):
    with open("results/distributed/communication_analysis.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Communication Cost Per Step (Estimated, DDP Ring All-Reduce)", fontweight="bold")

    for ax_idx, model_name in enumerate(["resnet18", "vit_small"]):
        ax = axes[ax_idx]
        model_data = data[model_name]
        opts = [o for o in OPT_ORDER if o in model_data["optimizers"]]
        names = [LABELS[o] for o in opts]
        colors = [COLORS[o] for o in opts]
        mb = [model_data["optimizers"][o]["megabytes_per_step"] for o in opts]
        comp = [model_data["optimizers"][o]["compression_ratio"] for o in opts]

        bars = ax.bar(names, mb, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, m, c in zip(bars, mb, comp):
            label = f"{m:.1f} MB\n({c:.1f}x)" if c > 1.01 else f"{m:.1f} MB"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    label, ha="center", va="bottom", fontsize=9)

        title = "ResNet-18" if model_name == "resnet18" else "ViT-Small"
        ax.set_title(f"{title} ({model_data['total_params']//1000}K params)")
        ax.set_ylabel("MB per step")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "communication_overhead.png"))
    plt.close()
    print("  Saved communication_overhead.png")


def plot_compression_ratio(output_dir):
    with open("results/distributed/communication_analysis.json") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(OPT_ORDER))
    width = 0.35
    r18 = [data["resnet18"]["optimizers"][o]["compression_ratio"] for o in OPT_ORDER]
    vit = [data["vit_small"]["optimizers"][o]["compression_ratio"] for o in OPT_ORDER]

    bars1 = ax.bar(x - width/2, r18, width, label="ResNet-18", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width/2, vit, width, label="ViT-Small", color="#DD8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[o] for o in OPT_ORDER])
    ax.set_ylabel("Compression Ratio (higher = less communication)")
    ax.set_title("Communication Compression Ratio vs Full All-Reduce", fontweight="bold")
    ax.legend()
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            if bar.get_height() > 1.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"{bar.get_height():.1f}x", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compression_ratio.png"))
    plt.close()
    print("  Saved compression_ratio.png")


def plot_rank_performance_tradeoff(output_dir):
    with open("results/ablation_results.json") as f:
        ablation = json.load(f)

    rf_runs = [(r["opt_config"]["init_rank_fraction"], r["best_val_acc"])
               for r in ablation if r.get("ablation_name", "").startswith("adadion_rf") and "error" not in r]
    rf_runs.sort()

    if not rf_runs:
        print("  No rank fraction data found, skipping")
        return

    fractions, accs = zip(*rf_runs)

    # Compute compression ratios for each rank fraction on resnet18
    # For a representative layer: 512x512 conv (flattened to 512x4608)
    # Compression = full / (m*r + n*r) where r = rf * min(m,n)
    m, n = 512, 4608  # largest conv layer flattened
    compressions = []
    for rf in fractions:
        r = max(1, int(rf * min(m, n)))
        full = m * n
        compressed = m * r + n * r
        compressions.append(full / compressed)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_acc = "#8172B3"
    color_comp = "#55A868"

    ax1.plot(fractions, accs, color=color_acc, marker="s", linewidth=2, markersize=10, label="Val Accuracy")
    ax1.set_xlabel("Init Rank Fraction")
    ax1.set_ylabel("Validation Accuracy (%)", color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(min(accs) - 0.2, max(accs) + 0.2)

    best_idx = np.argmax(accs)
    ax1.annotate(f"{accs[best_idx]:.2f}%", (fractions[best_idx], accs[best_idx]),
                 textcoords="offset points", xytext=(0, 12), ha="center", fontweight="bold", color=color_acc)

    ax2 = ax1.twinx()
    ax2.plot(fractions, compressions, color=color_comp, marker="o", linewidth=2, markersize=8,
             linestyle="--", label="Compression Ratio")
    ax2.set_ylabel("Compression Ratio (512x4608 layer)", color=color_comp)
    ax2.tick_params(axis="y", labelcolor=color_comp)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.set_title("Rank Fraction: Accuracy vs Communication Compression", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rank_performance_tradeoff.png"))
    plt.close()
    print("  Saved rank_performance_tradeoff.png")


if __name__ == "__main__":
    out = "results/plots"
    os.makedirs(out, exist_ok=True)
    plot_communication_overhead(out)
    plot_compression_ratio(out)
    plot_rank_performance_tradeoff(out)
    print(f"\nAll new plots saved to {out}/")
