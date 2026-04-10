"""
Phase 3: Analyze Consciousness Metrics
Entry point for computing consciousness-related metrics from Phase 2/2.1 results.
"""

import argparse
import json
import os
import numpy as np

from phase3.consciousness_metrics import (
    compute_all_consciousness_metrics,
    analyze_consciousness_emergence,
    compute_phi_simplified,
)

RESULTS_DIR = "results"


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3: Consciousness Metrics Analysis")
    p.add_argument("--input", type=str, default="results/generation_log.json",
                   help="Path to generation log from Phase 2/2.1")
    p.add_argument("--output", type=str, default="results/phase3_metrics.json",
                   help="Output path for metrics")
    return p.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  Phase 3: Consciousness Metrics Analysis")
    print("=" * 60)
    
    # Load generation log
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    with open(args.input) as f:
        log = json.load(f)
    
    print(f"\nLoaded {len(log)} generations from {args.input}")
    
    # Compute metrics
    print("\nComputing consciousness metrics...")
    metrics = compute_all_consciousness_metrics(log)
    
    # Analyze emergence
    print("Analyzing emergence patterns...")
    analysis = analyze_consciousness_emergence(metrics)
    
    # Print summary
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    
    for metric_name, result in analysis.items():
        print(f"\n{metric_name.upper()}:")
        print(f"  Early mean: {result['early_mean']:.4f}")
        print(f"  Late mean:  {result['late_mean']:.4f}")
        print(f"  Trend:      {result['trend']:+.4f}")
        print(f"  Max rate:   {result['max_rate']:+.4f} at gen {result['max_rate_gen']}")
        print(f"  Emerged:    {'✅ YES' if result['emerged'] else '❌ NO'}")
    
    # Save results
    output_data = {
        "metrics": {k: [float(v) if v is not None else None for v in vals] 
                   for k, vals in metrics.items()},
        "analysis": {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else 
                        bool(vv) if isinstance(vv, np.bool_) else vv 
                        for kk, vv in vals.items()} 
                    for k, vals in analysis.items()},
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print("\nDone.")


if __name__ == "__main__":
    main()
