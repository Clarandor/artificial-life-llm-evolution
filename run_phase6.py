"""
Run Phase 6: Strong Coordination Incentive
"""
import argparse
import numpy as np
from phase6.evolution import Evolution

def main():
    import argparse
    p = argparse.ArgumentParser(description="Phase 6: Strong Coordination")
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--save-log", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    
    print("=" * 60)
    print("  Phase 6: Strong Coordination Incentive")
    print("  No small prey — coordination = survival")
    print("=" * 60)
    
    evo = Evolution(seed=args.seed)
    log = evo.run(generations=args.generations, save_log=args.save_log)
    
    early = log[:50]
    late = log[-50:]
    
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    total_big = sum(g['large_prey_captured'] for g in log)
    total_failed = sum(g['failed_attacks'] for g in log)
    print(f"  Large prey captured : {total_big}")
    print(f"  Failed attacks      : {total_failed}")
    print(f"  Coordination rate   : {total_big/max(total_failed+total_big,1):.2%}")
    print(f"  Fitness (gen 0-49): {np.mean([g['mean_raw_fitness'] for g in early]):.2f}")
    print(f"  Fitness (last 50):  {np.mean([g['mean_raw_fitness'] for g in late]):.2f}")
    print(f"  Attn entropy early: {np.mean([g['mean_attn_entropy'] for g in early]):.3f}")
    print(f"  Attn entropy late:  {np.mean([g['mean_attn_entropy'] for g in late]):.3f}")
    print(f"  Signals sent early: {np.mean([g['mean_signals_sent'] for g in early]):.2f}")
    print(f"  Signals sent late:   {np.mean([g['mean_signals_sent'] for g in late]):.2f}")

if __name__ == "__main__":
    main()
