#!/usr/bin/env python3
"""
Plot rank dynamics: effective rank and actual rank vs steps for different gamma values.

Generates two side-by-side plots:
  Left: steps vs observed effective rank (erank), legend = gamma
  Right: steps vs actual rank used by AdaDion, legend = gamma

Style: academic ML (log scale where appropriate, dashed grid, raw+smooth, saturated colors).
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

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
    "axes.linewidth": 0.8,
})

# Colors: gradient from cool to warm as gamma increases
GAMMA_COLORS = {
    "0.5": "#1F78B4",   # blue
    "1.0": "#33A02C",   # green
    "1.5": "#FF7F00",   # orange
    "2.0": "#E31A1C",   # red
    "2.5": "#6A3D9A",   # purple
    "3.0": "#A65628",   # brown
}


def plot_rank_dynamics(data_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(data_path) as f:
        all_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for gamma_str in sorted(all_data.keys(), key=float):
        steps_data = all_data[gamma_str]
        gamma = float(gamma_str)

        steps = np.array([d["step"] for d in steps_data])
        eranks = np.array([d.get("erank") for d in steps_data], dtype=float)
        ranks = np.array([d.get("rank") for d in steps_data], dtype=float)

        # Filter NaN
        valid_e = ~np.isnan(eranks)
        valid_r = ~np.isnan(ranks)

        color = GAMMA_COLORS.get(gamma_str, "#333333")
        label = f"$\\gamma = {gamma}$"

        # Effective rank plot
        if valid_e.sum() > 10:
            s_e, e_e = steps[valid_e], eranks[valid_e]
            smooth_e = uniform_filter1d(e_e, size=50)
            ax1.plot(s_e, e_e, color=color, alpha=0.1, linewidth=0.5)
            ax1.plot(s_e, smooth_e, color=color, linewidth=1.8, label=label)

        # Actual rank plot
        if valid_r.sum() > 10:
            s_r, r_r = steps[valid_r], ranks[valid_r]
            smooth_r = uniform_filter1d(r_r, size=50)
            ax2.plot(s_r, r_r, color=color, alpha=0.1, linewidth=0.5)
            ax2.plot(s_r, smooth_r, color=color, linewidth=1.8, label=label)

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Effective rank (erank)")
    ax1.legend(framealpha=0.8, edgecolor="#ccc", fancybox=False)

    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Actual rank $r_t$")
    ax2.legend(framealpha=0.8, edgecolor="#ccc", fancybox=False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "rank_dynamics.pdf")
    fig.savefig(out_path, format="pdf")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/rank_dynamics/all_dynamics.json")
    parser.add_argument("--output_dir", default="results/paper_figures")
    args = parser.parse_args()
    plot_rank_dynamics(args.data, args.output_dir)
