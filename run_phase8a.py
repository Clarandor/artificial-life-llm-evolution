"""
Run Phase 8A: Gated Recursive Attention
========================================
300 generations with seed=42

Compares to Phase 7:
  - Phase 7: Recursive Attention (unconditional) → 0.70% coordination rate
  - Phase 8A: Gated Recursive Attention → target: >0.70% coordination rate

Hypothesis: Gating Level-2 attention to "prey nearby" only prevents
attention budget theft, restoring coordination performance.
"""

import argparse
import numpy as np
import sys
import os

# Ensure phase8a is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase8a.evolution import Evolution

def main():
    p = argparse.ArgumentParser(description="Phase 8A: Gated Recursive Attention")
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    
    print("=" * 70)
    print("  Phase 8A: Gated Recursive Attention")
    print("  Gate: activates Level-2 only when prey nearby (threshold=0.5)")
    print("  Hypothesis: Avoid 'attention budget theft' from Phase 7")
    print("=" * 70)
    
    evo = Evolution(seed=args.seed)
    log = evo.run(generations=args.generations)
    
    early = log[:50]
    late = log[-50:]
    
    total_big = sum(g['large_prey_captured'] for g in log)
    total_failed = sum(g['failed_attacks'] for g in log)
    coord_rate = total_big / max(total_failed + total_big, 1)
    
    print("\n" + "=" * 70)
    print("  Phase 8A Summary")
    print("=" * 70)
    print(f"  Total large prey captured : {total_big}")
    print(f"  Total failed attacks      : {total_failed}")
    print(f"  Coordination rate         : {coord_rate:.2%}")
    print(f"  --- vs Phase 7: 0.70% ---")
    if coord_rate > 0.007:
        print(f"  ✅ Gated attention IMPROVED coordination (+{coord_rate - 0.007:.2%})")
    else:
        print(f"  ⚠️  Gated attention did NOT improve coordination")
    print()
    print(f"  Fitness early: {np.mean([g['mean_raw_fitness'] for g in early]):.2f}")
    print(f"  Fitness late:  {np.mean([g['mean_raw_fitness'] for g in late]):.2f}")
    print(f"  Attn entropy early: {np.mean([g['mean_attn_entropy'] for g in early]):.3f}")
    print(f"  Attn entropy late:  {np.mean([g['mean_attn_entropy'] for g in late]):.3f}")
    print()
    print(f"  Gate value early:  {np.mean([g['mean_gate_value'] for g in early]):.3f}")
    print(f"  Gate value late:   {np.mean([g['mean_gate_value'] for g in late]):.3f}")
    print(f"  Rec usage early:   {np.mean([g['recursive_usage_rate'] for g in early]):.3f}")
    print(f"  Rec usage late:    {np.mean([g['recursive_usage_rate'] for g in late]):.3f}")
    print(f"  Prox when active (late):  {np.mean([g['prey_prox_when_active'] for g in late]):.3f}")
    print(f"  Prox when inactive (late): {np.mean([g['prey_prox_when_inactive'] for g in late]):.3f}")
    print()
    print(f"  Signals sent early: {np.mean([g['mean_signals_sent'] for g in early]):.2f}")
    print(f"  Signals sent late:  {np.mean([g['mean_signals_sent'] for g in late]):.2f}")
    print(f"\n  Results saved to: results/phase8a/generation_log.json")

if __name__ == "__main__":
    main()
