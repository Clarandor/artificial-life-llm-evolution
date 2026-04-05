"""
Entry point: Run Phase 0.3 evolution experiment.

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
from phase0.environment import N_TRIBES, TRIBE_SIZE, NUM_PREDATORS, NUM_PREY, MSG_DIM, FIXED_ENCODING
from phase0.agent import MLP, HIDDEN_DIM
from phase0.visualize import generate_all_plots

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0.3: Reward Shaping + Curriculum Decay")
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--grid-size",   type=int, default=32)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--no-plots",    action="store_true")
    p.add_argument("--save-log",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    pop = N_TRIBES * TRIBE_SIZE

    phase_name = "0.4c (Fixed Encoding Diagnostic)" if FIXED_ENCODING else "0.4a (Receiver Shaping)"
    print("=" * 60)
    print(f"  Artificial Life × LLM Evolution — Phase {phase_name}")
    print("=" * 60)
    print(f"  Generations : {args.generations}")
    print(f"  Population  : {pop} ({N_TRIBES} tribes × {TRIBE_SIZE})")
    print(f"  Grid size   : {args.grid_size}×{args.grid_size}")
    print(f"  Seed        : {args.seed}")
    print(f"  FIXED_ENCODING: {FIXED_ENCODING}")
    print(f"  Mechanisms:")
    if FIXED_ENCODING:
        print(f"    - FIXED msg[:2] = nearest prey direction (no sender evolution)")
    else:
        print(f"    - Signal-Action Alignment (sender: msg[:2] ↔ own move)")
    print(f"    - Receiver Shaping (receiver: move ↔ neighbor msg[:2])")
    print(f"    - Approach Prey reward (moving closer to nearest prey)")
    print(f"    - Curriculum: full shaping 0-100, decay 100-200, natural 200+")
    print(f"    - Group selection + prey hunt + kin clustering (from 0.2)")
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

    # Pre/post shaping comparison
    pre_shaping = [e for e in log if e["generation"] < 100]
    post_shaping = [e for e in log if e["generation"] >= 200]
    pre_prey = np.mean([e.get("total_prey_caps", 0) for e in pre_shaping]) if pre_shaping else 0
    post_prey = np.mean([e.get("total_prey_caps", 0) for e in post_shaping]) if post_shaping else 0

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Final mean fitness    : {final['mean_fitness']:.3f}")
    print(f"  Final max fitness     : {final['max_fitness']:.3f}")
    print(f"  Total prey captures   : {total_prey_all}")
    print(f"  Prey/gen (gen 0-99)   : {pre_prey:.1f}  (with shaping)")
    print(f"  Prey/gen (gen 200-299): {post_prey:.1f}  (pure natural)")
    print(f"  Best generation       : {best_gen['generation']} (mean={best_gen['mean_fitness']:.3f})")

    # Receiver score summary
    pre_recv = np.mean([e.get("mean_receiver", 0) for e in pre_shaping]) if pre_shaping else 0
    post_recv = np.mean([e.get("mean_receiver", 0) for e in post_shaping]) if post_shaping else 0
    print(f"  Receiver μ (gen 0-99) : {pre_recv:.2f}  (with shaping)")
    print(f"  Receiver μ (gen 200+) : {post_recv:.2f}  (pure natural)")
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

    if not args.no_plots:
        try:
            generate_all_plots(log, world=engine.world)
        except Exception as e:
            print(f"\n[Warning] Plot generation failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
