#!/usr/bin/env python3
"""
Generate publication-quality plots from final benchmark results.

Plots:
  1. ResNet-18 validation accuracy bar chart (mean +/- std, 3 seeds)
  2. ResNet-18 validation loss bar chart
  3. ResNet-18 training loss curves (per-epoch, all optimizers)
  4. ResNet-18 validation accuracy curves (per-epoch)
  5. ResNet-18 validation loss curves (per-epoch)
  6. ViT-Small comparison bar chart
  7. ViT-Small AdaDion LR sweep
  8. Ablation summary heatmap
  9. Training throughput & memory comparison
  10. Convergence speed (epochs to reach target accuracy)
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

COLORS = {
    "adamw": "#4C72B0",
    "muon": "#DD8452",
    "dion": "#55A868",
    "dion2": "#C44E52",
    "adadion": "#8172B3",
}
LABELS = {
    "adamw": "AdamW",
    "muon": "Muon",
    "dion": "Dion",
    "dion2": "Dion2",
    "adadion": "AdaDion V2",
}
OPT_ORDER = ["adadion", "dion", "dion2", "muon", "adamw"]

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def load_epoch_data(results_dir, run_name):
    path = os.path.join(results_dir, run_name, "epoch_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def load_final_results(results_dir):
    path = os.path.join(results_dir, "final_results.json")
    with open(path) as f:
        return json.load(f)


def plot_resnet_bars(results_dir, plot_dir):
    """Bar charts for ResNet-18 val accuracy and val loss (3 seeds)."""
    results = load_final_results(results_dir)

    acc_data = defaultdict(list)
    loss_data = defaultdict(list)
    time_data = defaultdict(list)
    mem_data = defaultdict(list)

    for r in results:
        if r.get("model") != "resnet18" or "error" in r:
            continue
        opt = r["optimizer"]
        acc_data[opt].append(r["best_val_acc"])
        loss_data[opt].append(r["final_val_loss"])
        time_data[opt].append(r["total_train_time_sec"])
        mem_data[opt].append(r["peak_gpu_mem_mb"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ResNet-18 CIFAR-10 — Optimizer Comparison (100 epochs, 3 seeds)", fontweight="bold", fontsize=14)

    # Val accuracy
    ax = axes[0]
    names = [LABELS[o] for o in OPT_ORDER if o in acc_data]
    means = [np.mean(acc_data[o]) for o in OPT_ORDER if o in acc_data]
    stds = [np.std(acc_data[o]) for o in OPT_ORDER if o in acc_data]
    colors = [COLORS[o] for o in OPT_ORDER if o in acc_data]

    bars = ax.bar(names, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title("Validation Accuracy")
    ax.set_ylim(bottom=min(means) - 1.5)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.05,
                f"{m:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Val loss
    ax = axes[1]
    means_l = [np.mean(loss_data[o]) for o in OPT_ORDER if o in loss_data]
    stds_l = [np.std(loss_data[o]) for o in OPT_ORDER if o in loss_data]
    bars = ax.bar(names, means_l, yerr=stds_l, capsize=6, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Final Validation Loss")
    ax.set_title("Validation Loss (lower is better)")
    for bar, m in zip(bars, means_l):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{m:.4f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "resnet18_comparison_bars.png"))
    plt.close()
    print("  Saved resnet18_comparison_bars.png")


def plot_resnet_curves(results_dir, plot_dir):
    """Training/validation curves for ResNet-18 (mean across seeds)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("ResNet-18 CIFAR-10 — Training Curves (mean over 3 seeds, shaded = std)", fontweight="bold")

    for opt in OPT_ORDER:
        epoch_by_ep = defaultdict(lambda: {"train_loss": [], "val_loss": [], "val_acc": []})
        for seed in [42, 123, 456]:
            run_name = f"resnet18_{opt}_seed{seed}"
            data = load_epoch_data(results_dir, run_name)
            for em in data:
                ep = em["epoch"]
                epoch_by_ep[ep]["train_loss"].append(em["train_loss"])
                epoch_by_ep[ep]["val_loss"].append(em["val_loss"])
                epoch_by_ep[ep]["val_acc"].append(em["val_acc"])

        if not epoch_by_ep:
            continue

        epochs = sorted(epoch_by_ep.keys())
        color = COLORS[opt]
        label = LABELS[opt]

        for ax_idx, (metric, title, ylabel) in enumerate([
            ("train_loss", "Training Loss", "Loss"),
            ("val_loss", "Validation Loss", "Loss"),
            ("val_acc", "Validation Accuracy", "Accuracy (%)"),
        ]):
            ax = axes[ax_idx]
            means = np.array([np.mean(epoch_by_ep[e][metric]) for e in epochs])
            stds = np.array([np.std(epoch_by_ep[e][metric]) for e in epochs])
            ax.plot(epochs, means, color=color, label=label, linewidth=2)
            ax.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.12)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "resnet18_training_curves.png"))
    plt.close()
    print("  Saved resnet18_training_curves.png")


def plot_vit_comparison(results_dir, plot_dir):
    """ViT-Small comparison bar chart."""
    results = load_final_results(results_dir)

    vit_runs = {}
    for r in results:
        name = r.get("run_name", "")
        if name.startswith("vit_") and name.endswith("_final") and "error" not in r:
            opt = name.replace("vit_", "").replace("_final", "")
            vit_runs[opt] = r

    if not vit_runs:
        print("  No ViT final runs found, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ViT-Small CIFAR-10 — Optimizer Comparison (100 epochs)", fontweight="bold")

    opts = [o for o in OPT_ORDER if o in vit_runs]
    names = [LABELS[o] for o in opts]
    colors = [COLORS[o] for o in opts]

    # Accuracy
    accs = [vit_runs[o]["best_val_acc"] for o in opts]
    bars = axes[0].bar(names, accs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Best Validation Accuracy (%)")
    axes[0].set_title("Validation Accuracy")
    axes[0].set_ylim(bottom=min(accs) - 3)
    for bar, a in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f"{a:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Loss
    losses = [vit_runs[o]["final_val_loss"] for o in opts]
    bars = axes[1].bar(names, losses, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Final Validation Loss")
    axes[1].set_title("Validation Loss (lower is better)")
    for bar, l in zip(bars, losses):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{l:.4f}", ha="center", va="bottom", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "vit_small_comparison_bars.png"))
    plt.close()
    print("  Saved vit_small_comparison_bars.png")


def plot_vit_lr_sweep(results_dir, plot_dir):
    """ViT-Small AdaDion LR sweep."""
    results = load_final_results(results_dir)

    lr_runs = []
    for r in results:
        name = r.get("run_name", "")
        if name.startswith("vit_adadion_lr") and "error" not in r:
            lr = r["opt_config"]["lr"]
            lr_runs.append((lr, r["best_val_acc"], r["final_val_loss"]))

    if not lr_runs:
        return

    lr_runs.sort()
    lrs, accs, losses = zip(*lr_runs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ViT-Small — AdaDion V2 Learning Rate Sweep", fontweight="bold")

    color = COLORS["adadion"]
    axes[0].plot(range(len(lrs)), accs, color=color, marker="s", linewidth=2, markersize=10)
    axes[0].set_xticks(range(len(lrs)))
    axes[0].set_xticklabels([f"{lr:.0e}" for lr in lrs])
    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("Best Val Accuracy (%)")
    axes[0].set_title("Accuracy vs LR")
    axes[0].grid(True, alpha=0.3)
    best_idx = np.argmax(accs)
    axes[0].annotate(f"{accs[best_idx]:.2f}%", (best_idx, accs[best_idx]),
                     textcoords="offset points", xytext=(0, 12), ha="center", fontweight="bold")

    axes[1].plot(range(len(lrs)), losses, color=color, marker="s", linewidth=2, markersize=10)
    axes[1].set_xticks(range(len(lrs)))
    axes[1].set_xticklabels([f"{lr:.0e}" for lr in lrs])
    axes[1].set_xlabel("Learning Rate")
    axes[1].set_ylabel("Final Val Loss")
    axes[1].set_title("Loss vs LR")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "vit_adadion_lr_sweep.png"))
    plt.close()
    print("  Saved vit_adadion_lr_sweep.png")


def plot_throughput_memory(results_dir, plot_dir):
    """Training time and memory comparison."""
    results = load_final_results(results_dir)

    # ResNet-18 seed42
    resnet_runs = {}
    for r in results:
        name = r.get("run_name", "")
        if "seed42" in name and "resnet18" in name and "error" not in r:
            opt = r["optimizer"]
            resnet_runs[opt] = r

    if not resnet_runs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ResNet-18 — Throughput & Memory (seed 42)", fontweight="bold")

    opts = [o for o in OPT_ORDER if o in resnet_runs]
    names = [LABELS[o] for o in opts]
    colors = [COLORS[o] for o in opts]

    # Time
    times = [resnet_runs[o]["total_train_time_sec"] for o in opts]
    bars = axes[0].bar(names, times, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Training Time (seconds)")
    axes[0].set_title("Total Training Time")
    for bar, t in zip(bars, times):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     f"{t:.0f}s", ha="center", va="bottom", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Throughput (samples/sec)
    throughputs = [resnet_runs[o].get("avg_throughput_samples_sec", 0) for o in opts]
    bars = axes[1].bar(names, throughputs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Throughput (samples/sec)")
    axes[1].set_title("Average Throughput")
    for bar, t in zip(bars, throughputs):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                     f"{t:.0f}", ha="center", va="bottom", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "throughput_memory.png"))
    plt.close()
    print("  Saved throughput_memory.png")


def plot_convergence_speed(results_dir, plot_dir):
    """Epochs to reach target accuracy thresholds."""
    thresholds = [85, 90, 92, 93, 94, 95]

    fig, ax = plt.subplots(figsize=(10, 6))

    for opt in OPT_ORDER:
        epochs_to = {t: [] for t in thresholds}
        for seed in [42, 123, 456]:
            run_name = f"resnet18_{opt}_seed{seed}"
            data = load_epoch_data(results_dir, run_name)
            for t in thresholds:
                for em in data:
                    if em["val_acc"] >= t:
                        epochs_to[t].append(em["epoch"])
                        break

        t_vals, e_means, e_stds = [], [], []
        for t in thresholds:
            if epochs_to[t]:
                t_vals.append(t)
                e_means.append(np.mean(epochs_to[t]))
                e_stds.append(np.std(epochs_to[t]))

        if t_vals:
            ax.errorbar(t_vals, e_means, yerr=e_stds, color=COLORS[opt],
                        label=LABELS[opt], marker="o", linewidth=2, capsize=4, markersize=7)

    ax.set_xlabel("Target Validation Accuracy (%)")
    ax.set_ylabel("Epochs to Reach Target")
    ax.set_title("Convergence Speed — ResNet-18 CIFAR-10", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "convergence_speed.png"))
    plt.close()
    print("  Saved convergence_speed.png")


def plot_ablation_summary(ablation_dir, plot_dir):
    """Ablation results summary bar chart."""
    path = os.path.join(ablation_dir, "ablation_results.json")
    if not os.path.exists(path):
        print("  No ablation results found, skipping")
        return

    with open(path) as f:
        results = json.load(f)

    # Group by phase
    phases = {
        "Baselines": [r for r in results if r.get("ablation_name", "").startswith("baseline")],
        "LR Sweep": [r for r in results if r.get("ablation_name", "").startswith("adadion_lr")],
        "Adaptive Rank": [r for r in results if "adaptive" in r.get("ablation_name", "")],
        "Grad Clip": [r for r in results if "clip" in r.get("ablation_name", "")],
        "Weight Decay": [r for r in results if r.get("ablation_name", "").startswith("adadion_wd")],
        "Rank Fraction": [r for r in results if r.get("ablation_name", "").startswith("adadion_rf")],
    }

    fig, ax = plt.subplots(figsize=(16, 7))
    all_names = []
    all_accs = []
    all_colors = []
    group_boundaries = []

    for phase_name, runs in phases.items():
        if not runs:
            continue
        group_boundaries.append((len(all_names), phase_name))
        for r in runs:
            if "error" in r:
                continue
            name = r.get("ablation_name", "?")
            # Shorten names
            name = name.replace("baseline_", "").replace("adadion_", "")
            all_names.append(name)
            all_accs.append(r.get("best_val_acc", 0))
            # Color: baselines get their own color, adadion runs are purple
            if "adamw" in r.get("ablation_name", ""):
                all_colors.append(COLORS["adamw"])
            elif "muon" in r.get("ablation_name", ""):
                all_colors.append(COLORS["muon"])
            elif "dion" in r.get("ablation_name", "") and "adadion" not in r.get("ablation_name", ""):
                all_colors.append(COLORS["dion"])
            else:
                all_colors.append(COLORS["adadion"])

    bars = ax.bar(range(len(all_names)), all_accs, color=all_colors, alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title("AdaDion V2 Ablation Study — All Runs", fontweight="bold")
    ax.set_ylim(bottom=min(all_accs) - 1)
    ax.axhline(y=max(all_accs), color="red", linestyle="--", alpha=0.5, label=f"Best: {max(all_accs):.2f}%")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add phase labels
    for start_idx, phase_name in group_boundaries:
        ax.axvline(x=start_idx - 0.5, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "ablation_summary.png"))
    plt.close()
    print("  Saved ablation_summary.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results/final")
    parser.add_argument("--ablation_dir", default="./results/ablation")
    parser.add_argument("--plot_dir", default="./results/plots")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    print("Generating plots...")
    plot_resnet_bars(args.results_dir, args.plot_dir)
    plot_resnet_curves(args.results_dir, args.plot_dir)
    plot_vit_comparison(args.results_dir, args.plot_dir)
    plot_vit_lr_sweep(args.results_dir, args.plot_dir)
    plot_throughput_memory(args.results_dir, args.plot_dir)
    plot_convergence_speed(args.results_dir, args.plot_dir)
    plot_ablation_summary(args.ablation_dir, args.plot_dir)
    print(f"\nAll plots saved to {args.plot_dir}/")


if __name__ == "__main__":
    main()
