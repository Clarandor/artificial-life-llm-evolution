"""
Phase 2.1: Supervised Attention Learning
Entry point
"""

import argparse
import json
import os
import numpy as np

from phase2_1.evolution import SupervisedEvolutionEngine
from phase2_1.environment import N_TRIBES, TRIBE_SIZE, NUM_PREDATORS, NUM_PREY, NEIGHBOR_RADIUS
from phase2_1.agent import SupervisedAttentionMLP, HIDDEN_DIM, ATT_DIM

RESULTS_DIR = "results/phase2_1"
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2.1: Supervised Attention")
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--grid-size",   type=int, default=32)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--save-log",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    pop = N_TRIBES * TRIBE_SIZE
    mlp = SupervisedAttentionMLP()

    print("=" * 60)
    print("  Artificial Life × LLM Evolution — Phase 2.1")
    print("  Supervised Attention Learning")
    print("=" * 60)
    print(f"  Generations : {args.generations}")
    print(f"  Population  : {pop} ({N_TRIBES} tribes × {TRIBE_SIZE})")
    print(f"  Grid size   : {args.grid_size}×{args.grid_size}")
    print(f"  Seed        : {args.seed}")
    print(f"  Architecture:")
    print(f"    - Encoder: obs(16) → hidden({HIDDEN_DIM})")
    print(f"    - Attention: Q/K({ATT_DIM}D), supervised target")
    print(f"    - Total params: {mlp.param_count}")
    print(f"  Optimization: GA (all weights) + attention supervision")
    print("=" * 60 + "\n")

    engine = SupervisedEvolutionEngine(
        grid_size=args.grid_size,
        seed=args.seed,
        verbose=True,
    )

    log = engine.run(generations=args.generations)

    # Summary
    final = log[-1]
    best_gen = max(log, key=lambda e: e["mean_raw_fitness"])
    total_prey_all = sum(e.get("total_prey_caps", 0) for e in log)

    first_100 = log[:100]
    last_100 = log[-100:]
    early_prey = np.mean([e.get("total_prey_caps", 0) for e in first_100])
    late_prey = np.mean([e.get("total_prey_caps", 0) for e in last_100])
    early_entropy = np.mean([e.get("mean_attn_entropy", 0) for e in first_100])
    late_entropy = np.mean([e.get("mean_attn_entropy", 0) for e in last_100])
    early_loss = np.mean([e.get("mean_attn_loss", 0) for e in first_100])
    late_loss = np.mean([e.get("mean_attn_loss", 0) for e in last_100])

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Final raw fitness       : {final['mean_raw_fitness']:.3f}")
    print(f"  Total prey captures     : {total_prey_all}")
    print(f"  Prey/gen (gen 0-99)     : {early_prey:.1f}")
    print(f"  Prey/gen (last 100)     : {late_prey:.1f}")
    print(f"  Prey change             : {late_prey - early_prey:+.1f} ({(late_prey/max(early_prey,0.1)-1)*100:+.0f}%)")
    print(f"  Best generation         : {best_gen['generation']} (raw={best_gen['mean_raw_fitness']:.3f})")
    print(f"  Attn entropy (gen 0-99) : {early_entropy:.3f}")
    print(f"  Attn entropy (last 100) : {late_entropy:.3f}")
    print(f"  Entropy change          : {late_entropy - early_entropy:+.3f}")
    print(f"  Attn loss (gen 0-99)    : {early_loss:.4f}")
    print(f"  Attn loss (last 100)    : {late_loss:.4f}")
    print(f"  Loss change             : {late_loss - early_loss:+.4f}")
    print("=" * 60)

    if args.save_log:
        log_path = os.path.join(RESULTS_DIR, "generation_log.json")
        def jsonify(v):
            if isinstance(v, np.ndarray): return v.tolist()
            if isinstance(v, dict): return {str(kk): float(vv) if isinstance(vv, (np.floating, float)) else vv for kk, vv in v.items()}
            if isinstance(v, (np.floating,)): return float(v)
            if isinstance(v, (np.integer,)): return int(v)
            return v
        serializable = [{k: jsonify(v) for k, v in entry.items()} for entry in log]
        with open(log_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nLog saved to {log_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
