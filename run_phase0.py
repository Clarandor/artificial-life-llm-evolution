"""
Entry point: Run Phase 0.2 evolution experiment.

Usage:
    python run_phase0.py                        # default 300 generations
    python run_phase0.py --generations 500
    python run_phase0.py --seed 0
    python run_phase0.py --no-plots
"""

import argparse
import json
import os
import numpy as np

from phase0.evolution import EvolutionEngine
from phase0.environment import N_TRIBES, TRIBE_SIZE, NUM_PREDATORS, NUM_PREY, MSG_DIM
from phase0.agent import MLP, HIDDEN_DIM
from phase0.visualize import generate_all_plots

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0.2: Group Selection + Prey Hunt")
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
    print("  Artificial Life × LLM Evolution — Phase 0.2")
    print("  Group Selection + Prey Hunt + Kin Clustering")
    print("=" * 60)
    print(f"  Generations : {args.generations}")
    print(f"  Population  : {pop} ({N_TRIBES} tribes × {TRIBE_SIZE})")
    print(f"  Grid size   : {args.grid_size}×{args.grid_size}")
    print(f"  Seed        : {args.seed}")
    print(f"  Mechanisms:")
    print(f"    - Tribes with group selection (proportional slot allocation)")
    print(f"    - Prey hunt: {NUM_PREY} moving prey, need 2+ agents to capture")
    print(f"    - Predators: {NUM_PREDATORS} (speed=2)")
    print(f"    - Kin clustering: children ±3 cells from tribe center")
    print(f"    - MSG_DIM={MSG_DIM} | Hidden={HIDDEN_DIM} | params≈{MLP().param_count}")
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
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Final mean fitness  : {final['mean_fitness']:.3f}")
    print(f"  Final max fitness   : {final['max_fitness']:.3f}")
    print(f"  Final mean prey/agt : {final['mean_prey_cap']:.3f}")
    print(f"  Total prey captures : {total_prey_all}")
    print(f"  Best generation     : {best_gen['generation']} (mean={best_gen['mean_fitness']:.3f})")
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
            {k: jsonify(v) for k, v in entry.items() if k != "sample_messages"}
            for entry in log
        ]
        with open(log_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nLog saved to {log_path}")

    # Plots
    if not args.no_plots:
        try:
            generate_all_plots(log, world=engine.world)
        except Exception as e:
            print(f"\n[Warning] Plot generation failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
