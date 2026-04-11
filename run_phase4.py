"""
Run Phase 4: LSTM Recurrent Agents
"""

import argparse
import numpy as np
import os

from phase4.evolution import Evolution

def parse_args():
    p = argparse.ArgumentParser(description="Phase 4: LSTM Recurrent Agents")
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--save-log", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("  Artificial Life × LLM Evolution — Phase 4")
    print("  LSTM Recurrent Agents + Supervised Attention")
    print("=" * 60)
    print(f"  Generations : {args.generations}")
    print(f"  Seed        : {args.seed}")
    print()
    
    evo = Evolution(seed=args.seed)
    
    log = evo.run(generations=args.generations, save_log=args.save_log)
    
    # Summary
    total_prey = sum(g["total_prey_caps"] for g in log)
    early = log[:50]
    late = log[-50:]
    
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Total prey captures : {total_prey}")
    print(f"  Mean raw fitness (gen 0-49): {np.mean([g['mean_raw_fitness'] for g in early]):.3f}")
    print(f"  Mean raw fitness (last 50):  {np.mean([g['mean_raw_fitness'] for g in late]):.3f}")
    print(f"  Attn entropy (gen 0-49):    {np.mean([g['mean_attn_entropy'] for g in early]):.3f}")
    print(f"  Attn entropy (last 50):      {np.mean([g['mean_attn_entropy'] for g in late]):.3f}")
    print(f"  Attn loss (gen 0-49):        {np.mean([g['mean_attn_loss'] for g in early]):.4f}")
    print(f"  Attn loss (last 50):         {np.mean([g['mean_attn_loss'] for g in late]):.4f}")
    print("\n  Log saved to results/phase4/generation_log.json")
    print("  Done.")

if __name__ == "__main__":
    main()
