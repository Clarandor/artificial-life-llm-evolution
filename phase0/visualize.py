"""
Phase 0.2: Visualization — Group Selection + Prey Hunt
========================================================
Plots:
  1. Fitness curve (composite + prey captures)
  2. Prey capture emergence
  3. Tribe competition (tribe avg fitness over time)
  4. PCA of message vectors
  5. Message variance evolution
"""

import numpy as np
from typing import List, Optional
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.decomposition import PCA
    HAS_SKL = True
except ImportError:
    HAS_SKL = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_fitness_curve(generation_log: List[dict], save=True, show=False):
    if not HAS_MPL:
        print("[skip] matplotlib not available")
        return
    gens     = [e["generation"]   for e in generation_log]
    mean_fit = [e["mean_fitness"] for e in generation_log]
    max_fit  = [e["max_fitness"]  for e in generation_log]
    min_fit  = [e["min_fitness"]  for e in generation_log]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Phase 0.2: Composite Fitness (food + 3×prey)", fontsize=14, fontweight="bold")
    ax.fill_between(gens, min_fit, max_fit, alpha=0.15, color="steelblue")
    ax.plot(gens, mean_fit, color="steelblue", linewidth=2, label="mean fitness")
    ax.plot(gens, max_fit,  color="darkorange", linewidth=1, linestyle="--", label="max fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Composite Fitness")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "fitness_curve.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()


def plot_prey_captures(generation_log: List[dict], save=True, show=False):
    if not HAS_MPL:
        return
    gens = [e["generation"] for e in generation_log]
    mean_prey = [e.get("mean_prey_cap", 0) for e in generation_log]
    max_prey  = [e.get("max_prey_cap", 0)  for e in generation_log]
    total_prey = [e.get("total_prey_caps", 0) for e in generation_log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Phase 0.2: Cooperative Prey Captures", fontsize=14, fontweight="bold")

    ax1.plot(gens, mean_prey, color="purple", linewidth=2, label="mean prey/agent")
    ax1.plot(gens, max_prey,  color="orchid", linewidth=1, linestyle="--", label="max prey/agent")
    ax1.set_ylabel("Prey Captures per Agent")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Rolling average for total
    window = min(20, len(total_prey))
    if window > 1:
        rolling = np.convolve(total_prey, np.ones(window)/window, mode='valid')
        ax2.plot(gens[:len(rolling)], rolling, color="crimson", linewidth=2, label=f"rolling avg (w={window})")
    ax2.bar(gens, total_prey, color="salmon", alpha=0.4, label="total captures/gen")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Total Prey Captures")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "prey_captures.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()


def plot_tribe_competition(generation_log: List[dict], save=True, show=False):
    """Plot tribe average fitness over time to show group competition dynamics."""
    if not HAS_MPL:
        return

    # Collect all tribe IDs
    all_tribes = set()
    for entry in generation_log:
        ta = entry.get("tribe_avg", {})
        all_tribes.update(ta.keys())

    if not all_tribes:
        print("[skip] No tribe data found")
        return

    tribe_ids = sorted(all_tribes)
    gens = [e["generation"] for e in generation_log]
    colors = plt.cm.tab10(np.linspace(0, 1, len(tribe_ids)))

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Phase 0.2: Tribe Competition (Group Selection)", fontsize=14, fontweight="bold")

    for tid, color in zip(tribe_ids, colors):
        vals = [e.get("tribe_avg", {}).get(tid, np.nan) for e in generation_log]
        ax.plot(gens, vals, color=color, linewidth=1.2, alpha=0.7, label=f"Tribe {tid}")

    # Overall mean
    mean_fit = [e["mean_fitness"] for e in generation_log]
    ax.plot(gens, mean_fit, color="black", linewidth=2.5, linestyle="--", label="Population mean")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Average Fitness")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "tribe_competition.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()


def plot_message_pca(generation_log: List[dict], save=True, show=False):
    if not HAS_MPL or not HAS_SKL:
        print("[skip] matplotlib/sklearn not available")
        return

    n = len(generation_log)
    sample_gens = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
    sample_gens = [g for g in sample_gens if 0 <= g < n]

    all_msgs, all_colors = [], []
    color_vals = np.linspace(0, 1, len(sample_gens))

    for cv, gi in zip(color_vals, sample_gens):
        msgs = generation_log[gi].get("sample_messages")
        if msgs is None or len(msgs) == 0:
            continue
        all_msgs.append(msgs)
        all_colors.extend([cv] * len(msgs))

    if not all_msgs:
        return

    all_msgs = np.vstack(all_msgs)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_msgs)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(coords[:,0], coords[:,1], c=all_colors, cmap="viridis", alpha=0.7, s=20)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Generation (normalized)")
    ax.set_title("PCA of Agent Messages (4D → 2D)", fontsize=13)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "message_pca.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()


def plot_message_variance(generation_log: List[dict], save=True, show=False):
    if not HAS_MPL:
        return

    gens, variances = [], []
    for entry in generation_log:
        msgs = entry.get("sample_messages")
        if msgs is not None and len(msgs) > 1:
            gens.append(entry["generation"])
            variances.append(float(np.var(msgs, axis=0).mean()))

    if not gens:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, variances, color="crimson", linewidth=2)
    ax.set_title("Mean Message Variance Over Generations", fontsize=13)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Variance (MSG_DIM=4)")
    ax.grid(alpha=0.3)

    if len(variances) > 10:
        v = np.array(variances)
        # Annotate final variance
        ax.annotate(f"final={v[-1]:.4f}", xy=(gens[-1], v[-1]),
                    fontsize=9, ha="right", va="bottom", color="crimson")
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "message_variance.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()


def generate_all_plots(generation_log: List[dict], world=None):
    print("\nGenerating plots...")
    plot_fitness_curve(generation_log)
    plot_prey_captures(generation_log)
    plot_tribe_competition(generation_log)
    plot_message_pca(generation_log)
    plot_message_variance(generation_log)
    print("All plots saved to results/")
