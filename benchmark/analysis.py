#!/usr/bin/env python3
"""
Post-run analysis and visualization for CIFAR-10 optimizer benchmark.

Generates:
  - Training/validation curves (loss, accuracy) per optimizer
  - Comparison bar charts (best val acc, convergence speed, memory, throughput)
  - LR sweep heatmaps
  - Statistical summary tables with mean/std across seeds
  - Optimizer-specific diagnostics (rank evolution for AdaDion)
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Consistent styling
COLORS = {
    "adamw": "#1f77b4",
    "muon": "#ff7f0e",
    "dion": "#2ca02c",
    "dion2": "#d62728",
    "adadion": "#9467bd",
}
LABELS = {
    "adamw": "AdamW",
    "muon": "Muon",
    "dion": "Dion",
    "dion2": "Dion2",
    "adadion": "AdaDion V2",
}


def load_run_data(results_dir: str, run_name: str) -> dict:
    """Load all metrics for a single run."""
    run_dir = os.path.join(results_dir, run_name)
    data = {}
    for fname in ["step_metrics.json", "epoch_metrics.json", "summary.json",
                   "config.json", "optimizer_metrics.json"]:
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data[fname.replace(".json", "")] = json.load(f)
    return data


def load_all_results(results_dir: str) -> dict:
    """Load all_results.json and per-run data."""
    results_path = os.path.join(results_dir, "all_results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results found at {results_path}")
    with open(results_path) as f:
        all_results = json.load(f)

    # Group by (model, optimizer)
    grouped = defaultdict(list)
    for r in all_results:
        if "error" in r:
            continue
        key = (r.get("model", "resnet18"), r.get("optimizer", "unknown"))
        grouped[key].append(r)

    return {"all_results": all_results, "grouped": grouped}


def plot_training_curves(results_dir: str, output_dir: str, model_name: str = "resnet18"):
    """Plot training loss and validation accuracy curves for each optimizer."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=16, fontweight="bold")

    for opt_name in ["adamw", "muon", "dion", "dion2", "adadion"]:
        # Collect across seeds
        epoch_data = defaultdict(lambda: defaultdict(list))

        for seed in [42, 123, 456]:
            run_name = f"{model_name}_{opt_name}_seed{seed}"
            data = load_run_data(results_dir, run_name)
            if "epoch_metrics" not in data:
                continue
            for em in data["epoch_metrics"]:
                ep = em["epoch"]
                epoch_data["train_loss"][ep].append(em["train_loss"])
                epoch_data["train_acc"][ep].append(em["train_acc"])
                epoch_data["val_loss"][ep].append(em["val_loss"])
                epoch_data["val_acc"][ep].append(em["val_acc"])

        if not epoch_data["train_loss"]:
            continue

        epochs = sorted(epoch_data["train_loss"].keys())
        color = COLORS.get(opt_name, "gray")
        label = LABELS.get(opt_name, opt_name)

        for ax_idx, (metric, title, ylabel) in enumerate([
            ("train_loss", "Training Loss", "Loss"),
            ("val_loss", "Validation Loss", "Loss"),
            ("train_acc", "Training Accuracy", "Accuracy (%)"),
            ("val_acc", "Validation Accuracy", "Accuracy (%)"),
        ]):
            ax = axes[ax_idx // 2][ax_idx % 2]
            means = [np.mean(epoch_data[metric][e]) for e in epochs]
            stds = [np.std(epoch_data[metric][e]) for e in epochs]
            means, stds = np.array(means), np.array(stds)

            ax.plot(epochs, means, color=color, label=label, linewidth=1.5)
            ax.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.15)
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_curves_{model_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_bars(results_dir: str, output_dir: str):
    """Bar charts comparing best val acc, training time, memory, throughput."""
    data = load_all_results(results_dir)
    grouped = data["grouped"]

    models = sorted(set(k[0] for k in grouped.keys()))

    for model_name in models:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Optimizer Comparison — {model_name}", fontsize=14, fontweight="bold")

        opt_names = []
        best_accs_mean, best_accs_std = [], []
        times_mean, times_std = [], []
        mem_mean, mem_std = [], []
        throughput_mean, throughput_std = [], []

        for opt_name in ["adamw", "muon", "dion", "dion2", "adadion"]:
            key = (model_name, opt_name)
            if key not in grouped:
                continue

            runs = grouped[key]
            opt_names.append(LABELS.get(opt_name, opt_name))
            accs = [r["best_val_acc"] for r in runs]
            best_accs_mean.append(np.mean(accs))
            best_accs_std.append(np.std(accs))

            t = [r.get("total_train_time_sec", 0) for r in runs]
            times_mean.append(np.mean(t))
            times_std.append(np.std(t))

            m = [r.get("peak_gpu_mem_mb", 0) for r in runs]
            mem_mean.append(np.mean(m))
            mem_std.append(np.std(m))

            tp = [r.get("avg_throughput_samples_sec", 0) for r in runs]
            throughput_mean.append(np.mean(tp))
            throughput_std.append(np.std(tp))

        colors = [COLORS.get(n.lower().replace(" v2", "").replace(" ", ""), "gray") for n in opt_names]

        # Best val accuracy
        ax = axes[0]
        bars = ax.bar(opt_names, best_accs_mean, yerr=best_accs_std, capsize=5, color=colors, alpha=0.85)
        ax.set_ylabel("Best Val Accuracy (%)")
        ax.set_title("Best Validation Accuracy")
        ax.set_ylim(bottom=min(best_accs_mean) - 3 if best_accs_mean else 0)
        for bar, val in zip(bars, best_accs_mean):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

        # Training time
        ax = axes[1]
        ax.bar(opt_names, times_mean, yerr=times_std, capsize=5, color=colors, alpha=0.85)
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Total Training Time")

        # Peak GPU memory
        ax = axes[2]
        ax.bar(opt_names, mem_mean, yerr=mem_std, capsize=5, color=colors, alpha=0.85)
        ax.set_ylabel("Memory (MB)")
        ax.set_title("Peak GPU Memory")

        # Throughput
        ax = axes[3]
        ax.bar(opt_names, throughput_mean, yerr=throughput_std, capsize=5, color=colors, alpha=0.85)
        ax.set_ylabel("Samples/sec")
        ax.set_title("Avg Throughput")

        for ax in axes:
            ax.tick_params(axis="x", rotation=30)
            ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_{model_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()


def plot_convergence_speed(results_dir: str, output_dir: str, model_name: str = "resnet18"):
    """Plot epochs to reach various accuracy thresholds."""
    thresholds = [80, 85, 90, 92, 93, 94, 95]

    fig, ax = plt.subplots(figsize=(10, 6))

    for opt_name in ["adamw", "muon", "dion", "dion2", "adadion"]:
        epochs_to_threshold = {t: [] for t in thresholds}

        for seed in [42, 123, 456]:
            run_name = f"{model_name}_{opt_name}_seed{seed}"
            data = load_run_data(results_dir, run_name)
            if "epoch_metrics" not in data:
                continue

            for t in thresholds:
                reached = None
                for em in data["epoch_metrics"]:
                    if em["val_acc"] >= t:
                        reached = em["epoch"]
                        break
                if reached is not None:
                    epochs_to_threshold[t].append(reached)

        color = COLORS.get(opt_name, "gray")
        label = LABELS.get(opt_name, opt_name)

        t_vals = []
        e_means = []
        e_stds = []
        for t in thresholds:
            if epochs_to_threshold[t]:
                t_vals.append(t)
                e_means.append(np.mean(epochs_to_threshold[t]))
                e_stds.append(np.std(epochs_to_threshold[t]))

        if t_vals:
            ax.errorbar(t_vals, e_means, yerr=e_stds, color=color, label=label,
                        marker="o", linewidth=1.5, capsize=4)

    ax.set_xlabel("Target Validation Accuracy (%)", fontsize=12)
    ax.set_ylabel("Epochs to Reach Target", fontsize=12)
    ax.set_title(f"Convergence Speed — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"convergence_{model_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_lr_sweep_results(results_dir: str, output_dir: str, optimizer_name: str):
    """Plot LR sweep results for a single optimizer."""
    data = load_all_results(results_dir)
    sweep_runs = [r for r in data["all_results"]
                  if r.get("optimizer") == optimizer_name and "lr" in r and "error" not in r]

    if not sweep_runs:
        return

    # Group by LR
    by_lr = defaultdict(list)
    for r in sweep_runs:
        by_lr[r["lr"]].append(r["best_val_acc"])

    lrs = sorted(by_lr.keys())
    means = [np.mean(by_lr[lr]) for lr in lrs]
    stds = [np.std(by_lr[lr]) for lr in lrs]

    fig, ax = plt.subplots(figsize=(8, 5))
    color = COLORS.get(optimizer_name, "gray")
    ax.errorbar(range(len(lrs)), means, yerr=stds, color=color,
                marker="s", linewidth=2, capsize=5, markersize=8)
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], rotation=45)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Best Val Accuracy (%)")
    ax.set_title(f"LR Sweep — {LABELS.get(optimizer_name, optimizer_name)}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"lr_sweep_{optimizer_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_optimizer_diagnostics(results_dir: str, output_dir: str, model_name: str = "resnet18"):
    """Plot AdaDion-specific diagnostics (rank evolution)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"AdaDion V2 Diagnostics — {model_name}", fontsize=14, fontweight="bold")

    for seed in [42, 123, 456]:
        run_name = f"{model_name}_adadion_seed{seed}"
        data = load_run_data(results_dir, run_name)
        if "optimizer_metrics" not in data:
            continue

        opt_m = data["optimizer_metrics"]
        steps = [m["step"] for m in opt_m]

        # Plot rank and effective rank for first available layer
        rank_keys = [k for k in opt_m[0].keys() if k != "step" and "rank" not in k.lower()]
        erank_keys = [k for k in opt_m[0].keys() if "erank" in k.lower() or "effective" in k.lower()]

        for k in list(opt_m[0].keys()):
            if k == "step":
                continue
            vals = [m.get(k, 0) for m in opt_m]
            if "erank" in k.lower() or "effective" in k.lower():
                axes[1].plot(steps, vals, alpha=0.6, label=f"seed{seed}:{k[:30]}")
            else:
                axes[0].plot(steps, vals, alpha=0.6, label=f"seed{seed}:{k[:30]}")

    axes[0].set_title("Rank Evolution")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Rank")
    axes[0].grid(True, alpha=0.3)
    if axes[0].get_legend_handles_labels()[1]:
        axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_title("Effective Rank Evolution")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Effective Rank")
    axes[1].grid(True, alpha=0.3)
    if axes[1].get_legend_handles_labels()[1]:
        axes[1].legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"adadion_diagnostics_{model_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def generate_summary_table(results_dir: str, output_dir: str):
    """Generate a comprehensive summary table as text and CSV."""
    data = load_all_results(results_dir)
    grouped = data["grouped"]

    lines = []
    csv_lines = ["model,optimizer,best_val_acc_mean,best_val_acc_std,final_val_acc_mean,"
                 "final_val_acc_std,train_time_mean,peak_mem_mean,throughput_mean"]

    header = (f"{'Model':<12} {'Optimizer':<12} {'Best Val Acc':>14} {'Final Val Acc':>14} "
              f"{'Time (s)':>10} {'Mem (MB)':>10} {'Thru (s/s)':>12}")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("=" * len(header))

    for model_name in sorted(set(k[0] for k in grouped.keys())):
        for opt_name in ["adamw", "muon", "dion", "dion2", "adadion"]:
            key = (model_name, opt_name)
            if key not in grouped:
                continue
            runs = grouped[key]
            best = [r["best_val_acc"] for r in runs]
            final = [r.get("final_val_acc", 0) for r in runs]
            time_s = [r.get("total_train_time_sec", 0) for r in runs]
            mem = [r.get("peak_gpu_mem_mb", 0) for r in runs]
            thr = [r.get("avg_throughput_samples_sec", 0) for r in runs]

            lines.append(
                f"{model_name:<12} {LABELS.get(opt_name, opt_name):<12} "
                f"{np.mean(best):>6.2f}±{np.std(best):>4.2f}% "
                f"{np.mean(final):>6.2f}±{np.std(final):>4.2f}% "
                f"{np.mean(time_s):>10.0f} "
                f"{np.mean(mem):>10.0f} "
                f"{np.mean(thr):>12.0f}"
            )
            csv_lines.append(
                f"{model_name},{opt_name},{np.mean(best):.2f},{np.std(best):.2f},"
                f"{np.mean(final):.2f},{np.std(final):.2f},"
                f"{np.mean(time_s):.0f},{np.mean(mem):.0f},{np.mean(thr):.0f}"
            )
        lines.append("-" * len(header))

    lines.append("=" * len(header))

    table_text = "\n".join(lines)
    print(table_text)

    with open(os.path.join(output_dir, "summary_table.txt"), "w") as f:
        f.write(table_text)
    with open(os.path.join(output_dir, "summary_table.csv"), "w") as f:
        f.write("\n".join(csv_lines))


def generate_all_plots(results_dir: str):
    """Generate all analysis plots."""
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    data = load_all_results(results_dir)
    models = sorted(set(k[0] for k in data["grouped"].keys()))

    for model_name in models:
        print(f"Generating plots for {model_name}...")
        try:
            plot_training_curves(results_dir, output_dir, model_name)
        except Exception as e:
            print(f"  Training curves failed: {e}")
        try:
            plot_convergence_speed(results_dir, output_dir, model_name)
        except Exception as e:
            print(f"  Convergence plot failed: {e}")
        try:
            plot_optimizer_diagnostics(results_dir, output_dir, model_name)
        except Exception as e:
            print(f"  Diagnostics plot failed: {e}")

    try:
        plot_comparison_bars(results_dir, output_dir)
    except Exception as e:
        print(f"Comparison bars failed: {e}")

    # LR sweep plots for each optimizer
    for opt_name in ["adamw", "muon", "dion", "dion2", "adadion"]:
        try:
            plot_lr_sweep_results(results_dir, output_dir, opt_name)
        except Exception as e:
            pass

    generate_summary_table(results_dir, output_dir)
    print(f"\nAll plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory containing benchmark results")
    args = parser.parse_args()
    generate_all_plots(args.results_dir)


if __name__ == "__main__":
    main()
