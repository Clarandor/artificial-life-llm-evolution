"""
Entry point: Run Phase 2 — Hybrid GA + REINFORCE experiment.

Phase 2: GA optimizes behavior weights, REINFORCE optimizes attention weights.
This is the critical test: can gradient-based optimization succeed where GA failed?

Key hypothesis: Attention entropy should DECREASE over generations as REINFORCE
learns to focus attention on relevant neighbors (prey-adjacent, tribe-mates).

Usage:
    python run_phase2.py                        # default 300 generations
    python run_phase2.py --generations 500
    python run_phase2.py --seed 0
    python run_phase2.py --no-plots
"""

import argparse
import json
import os
import numpy as np

from phase2.evolution import HybridEvolutionEngine
from phase2.environment import N_TRIBES, TRIBE_SIZE, NUM_PREDATORS, NUM_PREY, NEIGHBOR_RADIUS
from phase2.agent import HybridAttentionMLP, HIDDEN_DIM, ATT_DIM

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: Hybrid GA + REINFORCE")
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--grid-size",   type=int, default=32)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--no-plots",    action="store_true")
    p.add_argument("--save-log",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    pop = N_TRIBES * TRIBE_SIZE
    mlp = HybridAttentionMLP()

    print("=" * 60)
    print("  Artificial Life × LLM Evolution — Phase 2")
    print("  Hybrid GA + REINFORCE")
    print("=" * 60)
    print(f"  Generations : {args.generations}")
    print(f"  Population  : {pop} ({N_TRIBES} tribes × {TRIBE_SIZE})")
    print(f"  Grid size   : {args.grid_size}×{args.grid_size}")
    print(f"  Seed        : {args.seed}")
    print(f"  Architecture:")
    print(f"    - Encoder: obs(16) → hidden({HIDDEN_DIM})")
    print(f"    - Attention: Q/K({ATT_DIM}D), NO V projection")
    print(f"    - Decoder: (hidden+context={HIDDEN_DIM+ATT_DIM}) → action(6)")
    print(f"    - Total params: {mlp.param_count}")
    print(f"  Optimization:")
    print(f"    - GA (behavior): {mlp.ga_param_count} params — W_enc, W_dec, W_act")
    print(f"    - REINFORCE (attention): {mlp.rl_param_count} params — W_q, W_k")
    print("=" * 60 + "\n")

    engine = HybridEvolutionEngine(
        grid_size=args.grid_size,
        seed=args.seed,
        verbose=True,
    )

    log = engine.run(generations=args.generations)

    # Summary
    final = log[-1]
    best_gen = max(log, key=lambda e: e["mean_fitness"])
    total_prey_all = sum(e.get("total_prey_caps", 0) for e in log)

    first_100 = log[:100]
    last_100 = log[-100:]
    early_prey = np.mean([e.get("total_prey_caps", 0) for e in first_100])
    late_prey = np.mean([e.get("total_prey_caps", 0) for e in last_100])
    early_entropy = np.mean([e.get("mean_attn_entropy", 0) for e in first_100])
    late_entropy = np.mean([e.get("mean_attn_entropy", 0) for e in last_100])
    early_grad = np.mean([e.get("mean_grad_norm", 0) for e in first_100])
    late_grad = np.mean([e.get("mean_grad_norm", 0) for e in last_100])

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Final mean fitness      : {final['mean_fitness']:.3f}")
    print(f"  Final max fitness       : {final['max_fitness']:.3f}")
    print(f"  Total prey captures     : {total_prey_all}")
    print(f"  Prey/gen (gen 0-99)     : {early_prey:.1f}")
    print(f"  Prey/gen (last 100)     : {late_prey:.1f}")
    print(f"  Prey change             : {late_prey - early_prey:+.1f} ({(late_prey/max(early_prey,0.1)-1)*100:+.0f}%)")
    print(f"  Best generation         : {best_gen['generation']} (mean={best_gen['mean_fitness']:.3f})")
    print(f"  Attn entropy (gen 0-99) : {early_entropy:.3f}")
    print(f"  Attn entropy (last 100) : {late_entropy:.3f}")
    print(f"  Entropy change          : {late_entropy - early_entropy:+.3f}")
    print(f"  Grad norm (gen 0-99)    : {early_grad:.4f}")
    print(f"  Grad norm (last 100)    : {late_grad:.4f}")
    print("=" * 60)

    # Save log
    if args.save_log:
        log_path = os.path.join(RESULTS_DIR, "generation_log.json")
        def jsonify(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, dict):
                return {str(kk): float(vv) if isinstance(vv, (np.floating, float)) else vv
                        for kk, vv in v.items()}
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            return v
        serializable = [
            {k: jsonify(v) for k, v in entry.items() if k != "sample_hiddens"}
            for entry in log
        ]
        with open(log_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nLog saved to {log_path}")

    if not args.no_plots:
        try:
            from phase2.visualize import generate_all_plots
            generate_all_plots(log, world=engine.world)
        except Exception as e:
            print(f"\n[Warning] Plot generation failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
