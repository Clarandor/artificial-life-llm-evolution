"""
Phase 0: Visualization & Observation Metrics
=============================================
1. Fitness curve over generations
2. Population size curve
3. PCA of message vectors (semantic drift tracking)
4. Message-context correlation heatmap
"""

import numpy as np
from typing import List, Optional
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
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


def plot_fitness_curve(generation_log: List[dict], save: bool = True, show: bool = False):
    if not HAS_MPL:
        print("[skip] matplotlib not available — skipping fitness curve plot")
        return
    """Plot mean/max/min fitness across generations."""
    gens        = [e["generation"]   for e in generation_log]
    mean_fit    = [e["mean_fitness"] for e in generation_log]
    max_fit     = [e["max_fitness"]  for e in generation_log]
    min_fit     = [e["min_fitness"]  for e in generation_log]
    population  = [e["population"]   for e in generation_log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Phase 0: Evolution Progress", fontsize=14, fontweight="bold")

    # Fitness
    ax1.fill_between(gens, min_fit, max_fit, alpha=0.2, color="steelblue", label="min-max range")
    ax1.plot(gens, mean_fit, color="steelblue", linewidth=2, label="mean fitness")
    ax1.plot(gens, max_fit,  color="darkorange", linewidth=1, linestyle="--", label="max fitness")
    ax1.set_ylabel("Food Collected (fitness)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Population
    ax2.plot(gens, population, color="seagreen", linewidth=2)
    ax2.set_ylabel("Population Size")
    ax2.set_xlabel("Generation")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "fitness_curve.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close()


def plot_message_pca(
    generation_log: List[dict],
    sample_gens: Optional[List[int]] = None,
    save: bool = True,
    show: bool = False,
):
    if not HAS_MPL or not HAS_SKL:
        print("[skip] matplotlib/sklearn not available — skipping PCA plot")
        return
    """
    PCA of message vectors sampled at different generations.
    Each point = one agent's message vector; color = generation.
    """
    if sample_gens is None:
        n = len(generation_log)
        sample_gens = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        sample_gens = [g for g in sample_gens if g < n]

    all_msgs, all_colors = [], []
    cmap = plt.cm.viridis
    color_vals = np.linspace(0, 1, len(sample_gens))

    for color_val, gen_idx in zip(color_vals, sample_gens):
        entry = generation_log[gen_idx]
        msgs  = entry.get("sample_messages")
        if msgs is None or len(msgs) == 0:
            continue
        all_msgs.append(msgs)
        all_colors.extend([color_val] * len(msgs))

    if not all_msgs:
        print("No message samples found in generation log.")
        return

    all_msgs = np.vstack(all_msgs)  # (N, MSG_DIM)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_msgs)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=all_colors, cmap="viridis", alpha=0.7, s=20)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Generation (normalized)")
    ax.set_title("PCA of Agent Message Vectors Across Generations", fontsize=13)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "message_pca.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close()


def plot_message_variance(generation_log: List[dict], save: bool = True, show: bool = False):
    if not HAS_MPL:
        print("[skip] matplotlib not available — skipping variance plot")
        return
    """
    Track how message vector variance changes over generations.
    Rising variance = diverging signals (semantic differentiation).
    Stable low variance = convergence or collapse.
    """
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
    ax.set_title("Mean Message Vector Variance Over Generations", fontsize=13)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Variance (across MSG_DIM)")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, "message_variance.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close()


def render_grid_snapshot(world, save: bool = True, show: bool = False, tag: str = ""):
    """Render a snapshot of the grid world (food + agent positions)."""
    if not HAS_MPL:
        print("[skip] matplotlib not available — skipping grid snapshot")
        return
    import matplotlib.patches as patches

    g = world.grid_size
    fig, ax = plt.subplots(figsize=(7, 7))

    # Food heatmap
    ax.imshow(world.grid.T, origin="lower", cmap="YlGn", vmin=0, vmax=1, alpha=0.6)

    # Agent dots
    if world.agents:
        xs = [a.x for a in world.agents]
        ys = [a.y for a in world.agents]
        energies = np.array([a.energy for a in world.agents])
        energies_norm = np.clip(energies / 80.0, 0, 1)
        ax.scatter(xs, ys, c=energies_norm, cmap="RdYlGn", s=30, alpha=0.9,
                   vmin=0, vmax=1, zorder=3)

    ax.set_title(f"Grid World — Step {world.step_count} {tag}", fontsize=12)
    ax.set_xlim(0, g)
    ax.set_ylim(0, g)
    ax.set_aspect("equal")

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, f"grid_snapshot_{world.step_count:06d}.png")
        plt.savefig(path, dpi=100)
    if show:
        plt.show()
    plt.close()


def generate_all_plots(generation_log: List[dict], world=None):
    """Convenience: generate all standard plots."""
    print("\nGenerating plots...")
    plot_fitness_curve(generation_log)
    plot_message_pca(generation_log)
    plot_message_variance(generation_log)
    if world is not None:
        render_grid_snapshot(world)
    print("All plots saved to results/")
