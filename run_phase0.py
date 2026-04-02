"""
Entry point: Run Phase 0 evolution experiment.

Usage:
    python run_phase0.py                        # default 200 generations
    python run_phase0.py --generations 500      # longer run
    python run_phase0.py --population 200       # larger population
    python run_phase0.py --seed 0               # reproducible run
    python run_phase0.py --no-plots             # skip visualization
"""

import argparse
import json
import os
import numpy as np

from phase0.evolution import EvolutionEngine
from phase0.visualize import generate_all_plots

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0: Artificial Life Evolution")
    p.add_argument("--generations", type=int,   default=200,  help="Number of generations")
    p.add_argument("--population",  type=int,   default=100,  help="Population size")
    p.add_argument("--grid-size",   type=int,   default=32,   help="Grid side length")
    p.add_argument("--seed",        type=int,   default=42,   help="Random seed")
    p.add_argument("--no-plots",    action="store_true",       help="Skip plot generation")
    p.add_argument("--save-log",    action="store_true",       help="Save generation log to JSON")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Artificial Life × LLM Evolution — Phase 0")
    print("  Vector Communication + Genetic Algorithm")
    print("=" * 60)
    print(f"  Generations : {args.generations}")
    print(f"  Population  : {args.population}")
    print(f"  Grid size   : {args.grid_size}×{args.grid_size}")
    print(f"  Seed        : {args.seed}")
    print("=" * 60 + "\n")

    engine = EvolutionEngine(
        population_size=args.population,
        grid_size=args.grid_size,
        seed=args.seed,
        verbose=True,
    )

    log = engine.run(generations=args.generations)

    # ── Summary stats ──────────────────────────────────────────────────────────
    final = log[-1]
    best_gen = max(log, key=lambda e: e["mean_fitness"])
    print("\n" + "=" * 60)
    print("  Final Generation Summary")
    print("=" * 60)
    print(f"  Final mean fitness : {final['mean_fitness']:.3f}")
    print(f"  Final max fitness  : {final['max_fitness']:.3f}")
    print(f"  Final population   : {final['population']}")
    print(f"  Best generation    : {best_gen['generation']} (mean={best_gen['mean_fitness']:.3f})")
    print("=" * 60)

    # ── Save log ───────────────────────────────────────────────────────────────
    if args.save_log:
        log_path = os.path.join(RESULTS_DIR, "generation_log.json")
        serializable_log = [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in entry.items() if k != "sample_messages"}
            for entry in log
        ]
        with open(log_path, "w") as f:
            json.dump(serializable_log, f, indent=2)
        print(f"\nGeneration log saved to {log_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if not args.no_plots:
        try:
            generate_all_plots(log, world=engine.world)
        except ImportError as e:
            print(f"\n[Warning] Could not generate plots: {e}")
            print("Install visualization deps: pip install matplotlib scikit-learn")

    print("\nDone. Check results/ for outputs.")


if __name__ == "__main__":
    main()
