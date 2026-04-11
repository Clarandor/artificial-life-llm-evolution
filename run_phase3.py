"""
Phase 3: Full Consciousness Metrics Analysis
=============================================
Analyze emergence of consciousness-related properties in evolved populations.
"""

import argparse
import json
import os
import numpy as np

from phase3.consciousness_metrics import compute_all_metrics, analyze_emergence


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3: Consciousness Metrics")
    p.add_argument("--input", type=str, default="results/phase2_1/generation_log.json")
    p.add_argument("--output", type=str, default="results/phase3_full.json")
    p.add_argument("--no-phi", action="store_true", help="Skip Φ (slow)")
    return p.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  Phase 3: Consciousness Metrics Analysis (v2)")
    print("=" * 60)
    
    with open(args.input) as f:
        log = json.load(f)
    
    print(f"\nLoaded {len(log)} generations from {args.input}")
    
    # Check if sample_hiddens are available
    has_hiddens = sum(1 for g in log if g.get("sample_hiddens") is not None)
    print(f"Generations with hidden states: {has_hiddens}/{len(log)}")
    
    # Compute all metrics
    print("\nComputing metrics...")
    metrics = compute_all_metrics(log, compute_phi_flag=not args.no_phi)
    
    # Analyze emergence
    print("Analyzing emergence patterns...")
    analysis = analyze_emergence(metrics, window_size=30)
    
    # Print summary
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    
    for name, r in analysis.items():
        emoji = "✅" if r["emerged"] else "❌"
        label = {
            "phi": "Φ (信息整合度)",
            "gwt_cross_similarity": "GWT 跨智能体相似度",
            "gwt_influence_var": "GWT 影响力方差",
            "gwt_entropy": "GWT 注意力熵",
            "selfother_within": "自/他 部落内相似度",
            "selfother_between": "自/他 部落间相似度",
            "selfother_ratio": "自/他 区分比率",
            "selfother_diversity": "自/他 表示多样性",
        }.get(name, name)
        
        print(f"\n{label} {emoji}")
        print(f"  Early: {r['early_mean']:.4f}  Late: {r['late_mean']:.4f}")
        print(f"  Trend: {r['trend']:+.4f}  Slope: {r['slope_per_gen']:.6f}/gen")
        print(f"  Rel change: {r['rel_change']:.1%}  Emerged: {'YES' if r['emerged'] else 'NO'}")
    
    # Save
    def convert(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    output = {
        "input": args.input,
        "generations": len(log),
        "metrics": {k: [convert(v) for v in vals] for k, vals in metrics.items()},
        "analysis": {k: {kk: convert(vv) for kk, vv in vals.items()} 
                    for k, vals in analysis.items()},
    }
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    
    print(f"\nResults saved to {args.output}")
    print("\nDone.")


if __name__ == "__main__":
    main()
