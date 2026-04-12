"""
Run Phase 8B: Temporal Alignment
================================
Key innovation: Temporal alignment of recursive attention + reduced weight
  - rec_attn_output = rec_ctx * 0.3 + aligned_rec_ctx * 0.5
  - Hypothesis: fixes Phase 7 coordination decline by reducing temporal lag
  - Baseline: Phase 7 had 0.70% coordination success rate
"""
import argparse
import numpy as np
from phase8b.evolution import Evolution

def main():
    p = argparse.ArgumentParser(description="Phase 8B: Temporal Alignment")
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    
    print("=" * 60)
    print("  Phase 8B: Temporal Alignment")
    print("  REC_WEIGHT=0.3  ALIGN_WEIGHT=0.5")
    print("  Hypothesis B: fix decision latency in recursive attention")
    print("=" * 60)
    
    evo = Evolution(seed=args.seed)
    log = evo.run(generations=args.generations)
    
    early = log[:50]
    late = log[-50:]
    
    total_big = sum(g['large_prey_captured'] for g in log)
    total_failed = sum(g['failed_attacks'] for g in log)
    coord_rate = total_big / max(total_failed + total_big, 1)
    
    print("\n" + "=" * 60)
    print("  Phase 8B Summary")
    print("=" * 60)
    print(f"  Large prey captured : {total_big}")
    print(f"  Failed attacks      : {total_failed}")
    print(f"  Coordination rate   : {coord_rate:.2%}")
    print(f"  vs Phase 7 (0.70%)  : {'✓ IMPROVED' if coord_rate > 0.007 else '✗ not improved'}")
    print(f"  vs Phase 6 (1.08%)  : {'✓ IMPROVED' if coord_rate > 0.0108 else '✗ below Phase 6'}")
    print(f"  Fitness early: {np.mean([g['mean_raw_fitness'] for g in early]):.2f}")
    print(f"  Fitness late:  {np.mean([g['mean_raw_fitness'] for g in late]):.2f}")
    print(f"  Attn entropy early: {np.mean([g['mean_attn_entropy'] for g in early]):.3f}")
    print(f"  Attn entropy late:  {np.mean([g['mean_attn_entropy'] for g in late]):.3f}")
    print(f"  Temporal align usage: {np.mean([g['temporal_align_usage'] for g in late]):.3f}")
    print(f"  Signals sent late:   {np.mean([g['mean_signals_sent'] for g in late]):.2f}")
    print(f"  Rec attn usage late: {np.mean([g['recursive_attn_usage'] for g in late]):.3f}")

if __name__ == "__main__":
    main()
