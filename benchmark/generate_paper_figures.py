#!/usr/bin/env python3
"""
Generate paper figures matching NeurIPS/ICML academic style.

Style reference: Balles et al. (ICML 2018), Daxberger et al. (NeurIPS 2021).
Key: raw+smooth lines, dashed grid, all spines, log-scale loss, saturated colors,
LaTeX math in labels, confidence bands with visible alpha.
"""

import json, os, glob
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Academic ML paper style ──
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "axes.linewidth": 0.8,
})

# Saturated colors matching academic papers
COLORS = {
    "adadion": "#6A3D9A",  # deep purple
    "dion":    "#33A02C",  # strong green
    "dion2":   "#E31A1C",  # strong red
    "muon":    "#FF7F00",  # strong orange
    "adamw":   "#1F78B4",  # strong blue
}
LABELS = {"adadion": "AdaDion V2", "dion": "Dion", "dion2": "Dion2",
          "muon": "Muon", "adamw": "AdamW"}
ORDER = ["adadion", "dion", "dion2", "muon", "adamw"]
BAND_ALPHA = 0.18

def load_final():
    with open("results/final/final_results.json") as f: return json.load(f)

def load_epochs(run):
    p = f"results/final/{run}/epoch_metrics.json"
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return []

OUT = "results/paper_figures"

# ═══════════════════════════════════════════════════════════════
# Fig 1: ResNet-18 bars (accuracy + loss)
# ═══════════════════════════════════════════════════════════════
def fig1():
    R = load_final()
    acc, loss = defaultdict(list), defaultdict(list)
    for r in R:
        if r.get("model") != "resnet18": continue
        acc[r["optimizer"]].append(r["best_val_acc"])
        loss[r["optimizer"]].append(r["final_val_loss"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    opts = [o for o in ORDER if o in acc]
    x = np.arange(len(opts))
    names = [LABELS[o] for o in opts]
    colors = [COLORS[o] for o in opts]
    w = 0.5

    # Accuracy
    m = [np.mean(acc[o]) for o in opts]
    s = [np.std(acc[o]) for o in opts]
    a1.bar(x, m, w, color=colors, edgecolor=[mcolors.to_rgba(c, 0.9) for c in colors],
           linewidth=0.8, zorder=3)
    a1.errorbar(x, m, yerr=s, fmt="none", ecolor="black", capsize=3.5, capthick=1.0,
                elinewidth=1.0, zorder=4)
    for i in range(len(opts)):
        a1.text(x[i], m[i] + s[i] + 0.04, f"{m[i]:.2f}", ha="center", va="bottom",
                fontsize=8.5, color="#222")
    a1.set_xticks(x); a1.set_xticklabels(names, fontsize=9)
    a1.set_ylabel("Validation accuracy (%)")
    a1.set_ylim(min(m) - 0.8, max(m) + 0.6)

    # Loss
    ml = [np.mean(loss[o]) for o in opts]
    sl = [np.std(loss[o]) for o in opts]
    a2.bar(x, ml, w, color=colors, edgecolor=[mcolors.to_rgba(c, 0.9) for c in colors],
           linewidth=0.8, zorder=3)
    a2.errorbar(x, ml, yerr=sl, fmt="none", ecolor="black", capsize=3.5, capthick=1.0,
                elinewidth=1.0, zorder=4)
    for i in range(len(opts)):
        a2.text(x[i], ml[i] + sl[i] + 0.002, f"{ml[i]:.4f}", ha="center", va="bottom",
                fontsize=8, color="#222")
    a2.set_xticks(x); a2.set_xticklabels(names, fontsize=9)
    a2.set_ylabel("Validation loss")
    a2.set_ylim(0, max(ml) + 0.04)

    plt.tight_layout()
    fig.savefig(f"{OUT}/resnet18_bars.pdf", format="pdf"); plt.close()
    print("  resnet18_bars.pdf")


# ═══════════════════════════════════════════════════════════════
# Fig 2: Training curves (raw faint + smooth bold, confidence bands)
# ═══════════════════════════════════════════════════════════════
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    from scipy.ndimage import uniform_filter1d

    for opt in ORDER:
        by_ep = defaultdict(lambda: {"train_loss": [], "val_loss": [], "val_acc": []})
        for seed in [42, 123, 456]:
            for em in load_epochs(f"resnet18_{opt}_seed{seed}"):
                by_ep[em["epoch"]]["train_loss"].append(em["train_loss"])
                by_ep[em["epoch"]]["val_loss"].append(em["val_loss"])
                by_ep[em["epoch"]]["val_acc"].append(em["val_acc"])
        if not by_ep: continue
        eps = sorted(by_ep.keys())
        c = COLORS[opt]

        for ai, (k, yl) in enumerate([
            ("train_loss", "Training loss"),
            ("val_loss", "Validation loss"),
            ("val_acc", "Validation accuracy (%)"),
        ]):
            ax = axes[ai]
            raw = np.array([np.mean(by_ep[e][k]) for e in eps])
            std = np.array([np.std(by_ep[e][k]) for e in eps])
            sm = uniform_filter1d(raw, size=5)

            # Faint raw line
            ax.plot(eps, raw, color=c, alpha=0.15, linewidth=0.8)
            # Bold smooth line
            ax.plot(eps, sm, color=c, linewidth=2.0, label=LABELS[opt])
            # Confidence band
            ax.fill_between(eps, raw - std, raw + std, color=c, alpha=BAND_ALPHA)

            ax.set_xlabel("Epoch")
            ax.set_ylabel(yl)

    # Log scale for loss plots
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")

    for ax in axes:
        ax.legend(fontsize=7.5, loc="best", framealpha=0.8, edgecolor="#ccc", fancybox=False)

    plt.tight_layout()
    fig.savefig(f"{OUT}/training_curves.pdf", format="pdf"); plt.close()
    print("  training_curves.pdf")


# ═══════════════════════════════════════════════════════════════
# Fig 3: ViT-Small bars
# ═══════════════════════════════════════════════════════════════
def fig3():
    R = load_final()
    vit = {r["optimizer"]: r for r in R if r.get("model") == "vit_small"}
    if not vit: return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    opts = [o for o in ORDER if o in vit]
    x = np.arange(len(opts))
    names = [LABELS[o] for o in opts]
    colors = [COLORS[o] for o in opts]
    w = 0.5

    ac = [vit[o]["best_val_acc"] for o in opts]
    a1.bar(x, ac, w, color=colors, edgecolor=[mcolors.to_rgba(c, 0.9) for c in colors],
           linewidth=0.8, zorder=3)
    for i in range(len(opts)):
        a1.text(x[i], ac[i] + 0.15, f"{ac[i]:.1f}", ha="center", va="bottom", fontsize=8.5, color="#222")
    a1.set_xticks(x); a1.set_xticklabels(names)
    a1.set_ylabel("Validation accuracy (%)")
    a1.set_ylim(min(ac) - 2, max(ac) + 1.5)

    lo = [vit[o]["final_val_loss"] for o in opts]
    a2.bar(x, lo, w, color=colors, edgecolor=[mcolors.to_rgba(c, 0.9) for c in colors],
           linewidth=0.8, zorder=3)
    for i in range(len(opts)):
        a2.text(x[i], lo[i] + 0.008, f"{lo[i]:.3f}", ha="center", va="bottom", fontsize=8, color="#222")
    a2.set_xticks(x); a2.set_xticklabels(names)
    a2.set_ylabel("Validation loss")
    a2.set_ylim(0, max(lo) + 0.08)

    plt.tight_layout()
    fig.savefig(f"{OUT}/vit_bars.pdf", format="pdf"); plt.close()
    print("  vit_bars.pdf")


# ═══════════════════════════════════════════════════════════════
# Fig 4: Convergence speed
# ═══════════════════════════════════════════════════════════════
def fig4():
    thresholds = [85, 88, 90, 92, 93, 94, 95]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for opt in ORDER:
        ep_to = {t: [] for t in thresholds}
        for seed in [42, 123, 456]:
            data = load_epochs(f"resnet18_{opt}_seed{seed}")
            for t in thresholds:
                for em in data:
                    if em["val_acc"] >= t:
                        ep_to[t].append(em["epoch"]); break
        tv, em_, es = [], [], []
        for t in thresholds:
            if ep_to[t]:
                tv.append(t); em_.append(np.mean(ep_to[t])); es.append(np.std(ep_to[t]))
        if tv:
            ax.errorbar(tv, em_, yerr=es, color=COLORS[opt], label=LABELS[opt],
                        marker="o", linewidth=1.8, capsize=3, markersize=5, markeredgewidth=0)
    ax.set_xlabel("Target validation accuracy (%)")
    ax.set_ylabel("Epochs to reach target")
    ax.legend(framealpha=0.8, edgecolor="#ccc", fancybox=False)
    plt.tight_layout()
    fig.savefig(f"{OUT}/convergence.pdf", format="pdf"); plt.close()
    print("  convergence.pdf")


# ═══════════════════════════════════════════════════════════════
# Fig 5: Throughput
# ═══════════════════════════════════════════════════════════════
def fig5():
    R = load_final()
    r18 = {r["optimizer"]: r for r in R if r.get("model") == "resnet18" and r.get("seed") == 42}
    if not r18: return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    opts = [o for o in ORDER if o in r18]
    x = np.arange(len(opts))
    names = [LABELS[o] for o in opts]
    colors = [COLORS[o] for o in opts]
    w = 0.5

    times = [r18[o]["total_train_time_sec"] for o in opts]
    a1.bar(x, times, w, color=colors, edgecolor=[mcolors.to_rgba(c, 0.9) for c in colors], linewidth=0.8, zorder=3)
    for i in range(len(opts)):
        a1.text(x[i], times[i] + max(times)*0.02, f"{times[i]:.0f}s", ha="center", va="bottom", fontsize=8, color="#222")
    a1.set_xticks(x); a1.set_xticklabels(names)
    a1.set_ylabel("Training time (s)")
    a1.set_ylim(0, max(times) * 1.15)

    thru = [r18[o].get("avg_throughput_samples_sec", 0) for o in opts]
    a2.bar(x, thru, w, color=colors, edgecolor=[mcolors.to_rgba(c, 0.9) for c in colors], linewidth=0.8, zorder=3)
    for i in range(len(opts)):
        a2.text(x[i], thru[i] + max(thru)*0.02, f"{thru[i]:.0f}", ha="center", va="bottom", fontsize=8, color="#222")
    a2.set_xticks(x); a2.set_xticklabels(names)
    a2.set_ylabel("Throughput (samples/s)")
    a2.set_ylim(0, max(thru) * 1.15)

    plt.tight_layout()
    fig.savefig(f"{OUT}/throughput.pdf", format="pdf"); plt.close()
    print("  throughput.pdf")


# ═══════════════════════════════════════════════════════════════
# Fig 6: Communication overhead
# ═══════════════════════════════════════════════════════════════
def fig6():
    with open("results/distributed/communication_analysis.json") as f: data = json.load(f)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, mn, title in [(a1, "resnet18", "ResNet-18"), (a2, "vit_small", "ViT-Small")]:
        md = data[mn]
        opts = [o for o in ORDER if o in md["optimizers"]]
        x = np.arange(len(opts))
        names = [LABELS[o] for o in opts]
        colors = [COLORS[o] for o in opts]
        mb = [md["optimizers"][o]["megabytes_per_step"] for o in opts]
        comp = [md["optimizers"][o]["compression_ratio"] for o in opts]
        w = 0.5

        ax.bar(x, mb, w, color=colors, edgecolor=[mcolors.to_rgba(c, 0.9) for c in colors], linewidth=0.8, zorder=3)
        for i in range(len(opts)):
            label = f"{mb[i]:.0f}" if comp[i] < 1.05 else f"{mb[i]:.0f} ({comp[i]:.1f}x)"
            ax.text(x[i], mb[i] + max(mb)*0.02, label, ha="center", va="bottom", fontsize=7, color="#222")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("MB per step"); ax.set_title(title)
        ax.set_ylim(0, max(mb) * 1.2)

    plt.tight_layout()
    fig.savefig(f"{OUT}/communication.pdf", format="pdf"); plt.close()
    print("  communication.pdf")


# ═══════════════════════════════════════════════════════════════
# Fig 7: Rank scale sweep
# ═══════════════════════════════════════════════════════════════
def fig7():
    rs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    accs = [95.81, 95.76, 96.20, 96.22, 96.29, 95.99]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    c = COLORS["adadion"]
    ax.plot(rs, accs, color=c, marker="s", linewidth=2.0, markersize=7, markeredgewidth=0)
    ax.set_xlabel(r"Rank scale factor $\gamma$")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(min(accs) - 0.15, max(accs) + 0.2)
    bi = np.argmax(accs)
    ax.annotate(f"{accs[bi]:.2f}", (rs[bi], accs[bi]),
                textcoords="offset points", xytext=(10, 8), fontsize=9, color=c)
    plt.tight_layout()
    fig.savefig(f"{OUT}/rank_scale_sweep.pdf", format="pdf"); plt.close()
    print("  rank_scale_sweep.pdf")


# ═══════════════════════════════════════════════════════════════
# Fig 8: Compression ratio
# ═══════════════════════════════════════════════════════════════
def fig8():
    with open("results/distributed/communication_analysis.json") as f: data = json.load(f)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(ORDER)); w = 0.32
    r18 = [data["resnet18"]["optimizers"][o]["compression_ratio"] for o in ORDER]
    vit = [data["vit_small"]["optimizers"][o]["compression_ratio"] for o in ORDER]

    ax.bar(x - w/2, r18, w * 0.85, color="#1F78B4", edgecolor="#1F78B4", alpha=0.75, label="ResNet-18", zorder=3)
    ax.bar(x + w/2, vit, w * 0.85, color="#FF7F00", edgecolor="#FF7F00", alpha=0.75, label="ViT-Small", zorder=3)
    for i in range(len(ORDER)):
        for v, xp in [(r18[i], x[i] - w/2), (vit[i], x[i] + w/2)]:
            if v > 1.05:
                ax.text(xp, v + 0.06, f"{v:.1f}x", ha="center", va="bottom", fontsize=7.5, color="#222")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[o] for o in ORDER])
    ax.set_ylabel("Compression ratio")
    ax.axhline(1, color="#666", ls=":", lw=0.8)
    ax.legend(framealpha=0.8, edgecolor="#ccc", fancybox=False)
    plt.tight_layout()
    fig.savefig(f"{OUT}/compression_ratio.pdf", format="pdf"); plt.close()
    print("  compression_ratio.pdf")


# ═══════════════════════════════════════════════════════════════
# Fig 9: ViT LR sweep
# ═══════════════════════════════════════════════════════════════
def fig9():
    pts = [(0.001, 90.41), (0.002, 91.75), (0.005, 92.25), (0.01, 91.15), (0.02, 87.63)]
    lrs, accs = zip(*pts)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(range(len(lrs)), accs, color=COLORS["adadion"], marker="s", linewidth=2.0,
            markersize=7, markeredgewidth=0)
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([str(lr) for lr in lrs])
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Validation accuracy (%)")
    bi = np.argmax(accs)
    ax.annotate(f"{accs[bi]:.2f}", (bi, accs[bi]),
                textcoords="offset points", xytext=(10, 8), fontsize=9, color=COLORS["adadion"])
    plt.tight_layout()
    fig.savefig(f"{OUT}/vit_lr_sweep.pdf", format="pdf"); plt.close()
    print("  vit_lr_sweep.pdf")


# ═══════════════════════════════════════════════════════════════
# Fig 10: WRN scaling (CIFAR-10 + CIFAR-100 side by side)
# ═══════════════════════════════════════════════════════════════
def fig10():
    widths = ["wrn-28-2", "wrn-28-4", "wrn-28-10"]
    plabels = {"wrn-28-2": "1.5M", "wrn-28-4": "5.9M", "wrn-28-10": "36.5M"}
    opts_here = ["adadion", "dion", "muon", "adamw"]

    c10, c100 = {}, {}
    for s in glob.glob("results/wide_resnet/**/summary.json", recursive=True):
        with open(s) as f: r = json.load(f)
        c10[(r["model"], r["optimizer"])] = r
    for s in glob.glob("results/cifar100_scaling/**/summary.json", recursive=True):
        with open(s) as f: r = json.load(f)
        c100[(r["model"], r["optimizer"])] = r

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, runs, title in [(a1, c10, "CIFAR-10"), (a2, c100, "CIFAR-100")]:
        if not runs: continue
        for opt in opts_here:
            vals = [runs.get((w, opt), {}).get("best_val_acc") for w in widths]
            vx = [i for i, v in enumerate(vals) if v is not None]
            vy = [v for v in vals if v is not None]
            ax.plot(vx, vy, color=COLORS[opt], marker="o", linewidth=2.0,
                    markersize=7, markeredgewidth=0, label=LABELS[opt])
        ax.set_xticks(range(len(widths)))
        ax.set_xticklabels([f"{w}\n({plabels[w]})" for w in widths], fontsize=8)
        ax.set_xlabel("Model (parameters)")
        ax.set_ylabel("Validation accuracy (%)")
        ax.set_title(title)
        ax.legend(fontsize=7.5, framealpha=0.8, edgecolor="#ccc", fancybox=False)

    plt.tight_layout()
    fig.savefig(f"{OUT}/wrn_scaling.pdf", format="pdf"); plt.close()
    print("  wrn_scaling.pdf")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("Generating paper figures...")
    fig1()
    try: fig2()
    except ImportError: print("  SKIP fig2 (needs scipy)")
    fig3(); fig4(); fig5(); fig6(); fig7(); fig8(); fig9(); fig10()
    print(f"\nDone. All PDFs in {OUT}/")
