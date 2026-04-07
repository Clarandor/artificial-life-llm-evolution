"""
Phase 2: Visualization — Hybrid GA + REINFORCE
==================================================
Plots:
  1. Fitness curve
  2. Prey captures
  3. Attention entropy (KEY METRIC — should decrease with REINFORCE)
  4. REINFORCE gradient norm (NEW — shows learning signal strength)
  5. Tribe competition
  6. Hidden state PCA
"""

import numpy as np
from typing import List
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


def plot_fitness_curve(generation_log: List[dict], save=True):
    if not HAS_MPL:
        return
    gens     = [e["generation"]   for e in generation_log]
    mean_fit = [e["mean_fitness"] for e in generation_log]
    max_fit  = [e["max_fitness"]  for e in generation_log]
    min_fit  = [e["min_fitness"]  for e in generation_log]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Phase 2: Fitness (Hybrid GA + REINFORCE)", fontsize=14, fontweight="bold")
    ax.fill_between(gens, min_fit, max_fit, alpha=0.15, color="steelblue")
    ax.plot(gens, mean_fit, color="steelblue", linewidth=2, label="mean fitness")
    ax.plot(gens, max_fit,  color="darkorange", linewidth=1, linestyle="--", label="max fitness")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (food + 3x prey)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "fitness_curve.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()


def plot_prey_captures(generation_log: List[dict], save=True):
    if not HAS_MPL:
        return
    gens = [e["generation"] for e in generation_log]
    total_prey = [e.get("total_prey_caps", 0) for e in generation_log]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Phase 2: Cooperative Prey Captures (GA + REINFORCE)", fontsize=14, fontweight="bold")

    window = min(20, len(total_prey))
    if window > 1:
        rolling = np.convolve(total_prey, np.ones(window)/window, mode='valid')
        ax.plot(gens[:len(rolling)], rolling, color="crimson", linewidth=2, label=f"rolling avg (w={window})")
    ax.bar(gens, total_prey, color="salmon", alpha=0.4, label="captures/gen")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Total Prey Captures")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "prey_captures.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()


def plot_attention_entropy(generation_log: List[dict], save=True):
    if not HAS_MPL:
        return

    gens = [e["generation"] for e in generation_log]
    entropies = [e.get("mean_attn_entropy", 0) for e in generation_log]
    max_h = np.log(8)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Phase 2: Attention Entropy (REINFORCE should lower this)", fontsize=14, fontweight="bold")

    ax.plot(gens, entropies, color="teal", linewidth=2, label="mean attention entropy")
    ax.axhline(y=max_h, color="gray", linestyle=":", alpha=0.5, label=f"max entropy (ln8={max_h:.2f})")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)

    if len(entropies) > 20:
        rolling = np.convolve(entropies, np.ones(20)/20, mode='valid')
        ax.plot(gens[:len(rolling)], rolling, color="darkgreen", linewidth=2,
                linestyle="--", label="rolling avg (w=20)")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Attention Entropy (nats)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    if len(entropies) > 10:
        ax.annotate(f"final={entropies[-1]:.3f}", xy=(gens[-1], entropies[-1]),
                    fontsize=9, ha="right", va="bottom", color="teal")

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "attention_entropy.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()


def plot_gradient_norm(generation_log: List[dict], save=True):
    """NEW: Plot REINFORCE gradient norm over generations."""
    if not HAS_MPL:
        return

    gens = [e["generation"] for e in generation_log]
    grad_norms = [e.get("mean_grad_norm", 0) for e in generation_log]
    lr_vals = [e.get("rl_lr", 0) for e in generation_log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[2, 1])
    fig.suptitle("Phase 2: REINFORCE Gradient Signal", fontsize=14, fontweight="bold")

    # Gradient norm
    ax1.plot(gens, grad_norms, color="purple", linewidth=1.5, alpha=0.7, label="mean grad norm")
    if len(grad_norms) > 20:
        rolling = np.convolve(grad_norms, np.ones(20)/20, mode='valid')
        ax1.plot(gens[:len(rolling)], rolling, color="darkviolet", linewidth=2,
                 linestyle="--", label="rolling avg (w=20)")
    ax1.set_ylabel("Gradient Norm")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Learning rate
    ax2.plot(gens, lr_vals, color="olive", linewidth=2, label="REINFORCE lr")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Learning Rate")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "gradient_norm.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()


def plot_tribe_competition(generation_log: List[dict], save=True):
    if not HAS_MPL:
        return
    all_tribes = set()
    for entry in generation_log:
        all_tribes.update(entry.get("tribe_avg", {}).keys())
    if not all_tribes:
        return

    tribe_ids = sorted(all_tribes)
    gens = [e["generation"] for e in generation_log]
    colors = plt.cm.tab10(np.linspace(0, 1, len(tribe_ids)))

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Phase 2: Tribe Competition (GA + REINFORCE)", fontsize=14, fontweight="bold")

    for tid, color in zip(tribe_ids, colors):
        vals = [e.get("tribe_avg", {}).get(tid, np.nan) for e in generation_log]
        ax.plot(gens, vals, color=color, linewidth=1.2, alpha=0.7, label=f"Tribe {tid}")

    mean_fit = [e["mean_fitness"] for e in generation_log]
    ax.plot(gens, mean_fit, color="black", linewidth=2.5, linestyle="--", label="Pop mean")

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


def plot_hidden_pca(generation_log: List[dict], save=True):
    if not HAS_MPL or not HAS_SKL:
        return

    n = len(generation_log)
    sample_gens = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
    sample_gens = [g for g in sample_gens if 0 <= g < n]

    all_h, all_colors = [], []
    color_vals = np.linspace(0, 1, len(sample_gens))

    for cv, gi in zip(color_vals, sample_gens):
        h = generation_log[gi].get("sample_hiddens")
        if h is None or len(h) == 0:
            continue
        all_h.append(h)
        all_colors.extend([cv] * len(h))

    if not all_h:
        return

    all_h = np.vstack(all_h)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_h)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(coords[:,0], coords[:,1], c=all_colors, cmap="viridis", alpha=0.7, s=20)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Generation (normalized)")
    ax.set_title("PCA of Agent Hidden States — Hybrid GA+REINFORCE (32D→2D)", fontsize=13)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "hidden_pca.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()


def generate_all_plots(generation_log: List[dict], world=None):
    print("\nGenerating plots...")
    plot_fitness_curve(generation_log)
    plot_prey_captures(generation_log)
    plot_attention_entropy(generation_log)
    plot_gradient_norm(generation_log)
    plot_tribe_competition(generation_log)
    plot_hidden_pca(generation_log)
    print("All plots saved to results/")
