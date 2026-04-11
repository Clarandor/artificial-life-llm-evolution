"""
Run Phase 5: CoordinationWorld with Emergent Communication
===========================================================
"""

import argparse
import numpy as np
import os

from phase5.evolution import Evolution

def parse_args():
    p = argparse.ArgumentParser(description="Phase 5: CoordinationWorld + Communication")
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--save-log", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("  Artificial Life × LLM Evolution — Phase 5")
    print("  CoordinationWorld + Emergent Communication")
    print("=" * 60)
    print(f"  Generations : {args.generations}")
    print(f"  Seed        : {args.seed}")
    print()
    
    evo = Evolution(seed=args.seed)
    log = evo.run(generations=args.generations, save_log=args.save_log)
    
    # Summary
    total_big = sum(g.get('large_prey_captured', 0) for g in log)
    total_small = sum(g.get('small_prey_captured', 0) for g in log)
    early = log[:50]
    late = log[-50:]
    
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Large prey captures : {total_big}")
    print(f"  Small prey captures : {total_small}")
    print(f"  Coordination rate   : {total_big / max(sum(g.get('large_prey_attempts', 1) for g in log), 1):.2%}")
    print(f"  Mean raw fitness (gen 0-49): {np.mean([g['mean_raw_fitness'] for g in early]):.3f}")
    print(f"  Mean raw fitness (last 50):  {np.mean([g['mean_raw_fitness'] for g in late]):.3f}")
    print(f"  Attn entropy (gen 0-49):    {np.mean([g['mean_attn_entropy'] for g in early]):.3f}")
    print(f"  Attn entropy (last 50):     {np.mean([g['mean_attn_entropy'] for g in late]):.3f}")
    print(f"  Signals sent (gen 0-49):   {np.mean([g.get('mean_signals_sent', 0) for g in early]):.2f}")
    print(f"  Signals sent (last 50):    {np.mean([g.get('mean_signals_sent', 0) for g in late]):.2f}")
    print("\n  Log saved to results/phase5/generation_log.json")
    print("  Done.")

if __name__ == "__main__":
    main()
