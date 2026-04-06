"""
Entry point: Run Phase 1 evolution experiment.

Phase 1: Attention-based coordination (no explicit messaging).
Agents attend over neighbor hidden states instead of sending/receiving
messages. Tests whether reducing bilateral protocol to unilateral
reading makes coordination evolvable by GA.

Usage:
    python run_phase1.py                        # default 300 generations
    python run_phase1.py --generations 500
    python run_phase1.py --seed 0
    python run_phase1.py --no-plots
"""

import argparse
import json
import os
import numpy as np

from phase1.evolution import EvolutionEngine
from phase1.environment import N_TRIBES, TRIBE_SIZE, NUM_PREDATORS, NUM_PREY, NEIGHBOR_RADIUS
from phase1.agent import AttentionMLP, HIDDEN_DIM, ATT_DIM

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1: Attention-based Coordination")
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--grid-size",   type=int, default=32)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--no-plots",    action="store_true")
    p.add_argument("--save-log",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    pop = N_TRIBES * TRIBE_SIZE

    print("=" * 60)
    print("  Artificial Life × LLM Evolution — Phase 1")
    print("  Attention-based Coordination (no messaging)")
    print("=" * 60)
    print(f"  Generations : {args.generations}")
    print(f"  Population  : {pop} ({N_TRIBES} tribes × {TRIBE_SIZE})")
    print(f"  Grid size   : {args.grid_size}×{args.grid_size}")
    print(f"  Seed        : {args.seed}")
    print(f"  Architecture:")
    print(f"    - Encoder: obs(16) → hidden({HIDDEN_DIM})")
    print(f"    - Attention: Q/K/V({ATT_DIM}D) over ≤8 neighbors (radius {NEIGHBOR_RADIUS})")
    print(f"    - Decoder: (hidden+context) → action(6)")
    print(f"    - Params ≈ {AttentionMLP().param_count}")
    print(f"  Selection: 2-level (tribe + tournament), NO reward shaping")
    print("=" * 60 + "\n")

    engine = EvolutionEngine(
        grid_size=args.grid_size,
        seed=args.seed,
        verbose=True,
    )

    log = engine.run(generations=args.generations)

    # Summary
    final = log[-1]
    best_gen = max(log, key=lambda e: e["mean_fitness"])
    total_prey_all = sum(e.get("total_prey_caps", 0) for e in log)

    # First vs last 100 generations comparison
    first_100 = log[:100]
    last_100 = log[-100:]
    early_prey = np.mean([e.get("total_prey_caps", 0) for e in first_100])
    late_prey = np.mean([e.get("total_prey_caps", 0) for e in last_100])
    early_entropy = np.mean([e.get("mean_attn_entropy", 0) for e in first_100])
    late_entropy = np.mean([e.get("mean_attn_entropy", 0) for e in last_100])

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Final mean fitness      : {final['mean_fitness']:.3f}")
    print(f"  Final max fitness       : {final['max_fitness']:.3f}")
    print(f"  Total prey captures     : {total_prey_all}")
    print(f"  Prey/gen (gen 0-99)     : {early_prey:.1f}")
    print(f"  Prey/gen (last 100)     : {late_prey:.1f}")
    print(f"  Best generation         : {best_gen['generation']} (mean={best_gen['mean_fitness']:.3f})")
    print(f"  Attn entropy (gen 0-99) : {early_entropy:.3f}")
    print(f"  Attn entropy (last 100) : {late_entropy:.3f}")
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
            from phase1.visualize import generate_all_plots
            generate_all_plots(log, world=engine.world)
        except Exception as e:
            print(f"\n[Warning] Plot generation failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
