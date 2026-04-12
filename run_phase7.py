"""
Run Phase 7: Recursive Attention (Theory of Mind)
"""
import argparse
import numpy as np
from phase7.evolution import Evolution

def main():
    p = argparse.ArgumentParser(description="Phase 7: Recursive Attention")
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    
    print("=" * 60)
    print("  Phase 7: Recursive Attention (Theory of Mind)")
    print("  Level 2: 'What is my neighbor attending to?'")
    print("=" * 60)
    
    evo = Evolution(seed=args.seed)
    log = evo.run(generations=args.generations)
    
    early = log[:50]
    late = log[-50:]
    
    print("\n" + "=" * 60)
    print("  Phase 7 Summary")
    print("=" * 60)
    total_big = sum(g['large_prey_captured'] for g in log)
    total_failed = sum(g['failed_attacks'] for g in log)
    print(f"  Large prey captured : {total_big}")
    print(f"  Failed attacks      : {total_failed}")
    print(f"  Coordination rate   : {total_big/max(total_failed+total_big,1):.2%}")
    print(f"  Rec attention usage: {np.mean([g['recursive_attn_usage'] for g in late]):.3f}")
    print(f"  Fitness early: {np.mean([g['mean_raw_fitness'] for g in early]):.2f}")
    print(f"  Fitness late:  {np.mean([g['mean_raw_fitness'] for g in late]):.2f}")
    print(f"  Attn entropy early: {np.mean([g['mean_attn_entropy'] for g in early]):.3f}")
    print(f"  Attn entropy late:  {np.mean([g['mean_attn_entropy'] for g in late]):.3f}")
    print(f"  Signals sent early: {np.mean([g['mean_signals_sent'] for g in early]):.2f}")
    print(f"  Signals sent late:   {np.mean([g['mean_signals_sent'] for g in late]):.2f}")

if __name__ == "__main__":
    main()
