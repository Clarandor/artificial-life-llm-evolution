"""
Phase 3: Consciousness Metrics (v2)
=====================================
Numerically stable implementations of:
1. Φ (Integrated Information) - IIT theory, using SVD log-det
2. Global Workspace Broadcast - GWT theory
3. Self/Other Distinction - Representational Similarity Analysis
4. Population Diversity - within/between tribe variance

v2 fixes:
- Φ: SVD-based log-determinant to avoid overflow
- Self/Other: RSA-based analysis without needing separate weight matrices
- Added population diversity and hidden state structure metrics
"""

import numpy as np
from typing import List, Tuple, Optional


# ============================================================================
# Helpers
# ============================================================================

def _log_det_svd(cov: np.ndarray) -> float:
    """Compute log(det(cov)) via SVD — numerically stable."""
    S = np.linalg.svd(cov, compute_uv=False)
    S = np.maximum(S, 1e-10)  # clamp near-zero singular values
    return np.sum(np.log(S))


def _gaussian_entropy(cov: np.ndarray) -> float:
    """Entropy of multivariate Gaussian: H = 0.5 * log det(2πe Σ) = 0.5*(d + d*ln(2π) + logdet(Σ))"""
    d = cov.shape[0]
    return 0.5 * (d * (1 + np.log(2 * np.pi)) + _log_det_svd(cov))


# ============================================================================
# 1. Φ (Integrated Information) - IIT
# ============================================================================

def compute_phi(
    hiddens: np.ndarray,
    n_partitions: int = 10,
) -> float:
    """
    Compute simplified Φ using average over random bipartitions.
    
    Φ ≈ H(whole) - average(H(partition_A) + H(partition_B))
    
    Higher Φ = more integrated (less reducible to parts).
    
    Args:
        hiddens: (N, D) agent hidden states
        n_partitions: number of random bipartitions to average over
    
    Returns:
        Φ value (non-negative)
    """
    N, D = hiddens.shape
    if N < 4 or D < 2:
        return 0.0
    
    # Standardize
    h = (hiddens - hiddens.mean(axis=0)) / (hiddens.std(axis=0) + 1e-8)
    
    # Whole-system entropy
    cov_all = np.cov(h.T)
    if cov_all.ndim == 0:
        return 0.0
    H_all = _gaussian_entropy(cov_all)
    
    # Average over random bipartitions
    rng = np.random.RandomState(42)
    phi_sum = 0.0
    valid = 0
    
    for _ in range(n_partitions):
        perm = rng.permutation(N)
        mid = N // 2
        
        h_a = h[perm[:mid]]
        h_b = h[perm[mid:]]
        
        cov_a = np.cov(h_a.T)
        cov_b = np.cov(h_b.T)
        
        if cov_a.ndim == 0 or cov_b.ndim == 0:
            continue
        
        H_a = _gaussian_entropy(cov_a)
        H_b = _gaussian_entropy(cov_b)
        
        phi_sum += max(0.0, H_all - (H_a + H_b))
        valid += 1
    
    return phi_sum / max(valid, 1)


# ============================================================================
# 2. GWT: Global Workspace Metrics
# ============================================================================

def compute_gwt_metrics(
    hiddens: np.ndarray,
    attention_weights: Optional[List[np.ndarray]] = None,
) -> dict:
    """
    Global Workspace Theory metrics from hidden states.
    
    Even without attention weights, we can measure:
    - information_integration: how correlated are agents' hidden states?
    - broadcast_uniformity: how evenly distributed is information?
    """
    N, D = hiddens.shape
    
    # Pairwise correlation structure
    h = (hiddens - hiddens.mean(axis=0)) / (hiddens.std(axis=0) + 1e-8)
    
    # Cross-agent similarity matrix
    gram = h @ h.T  # (N, N)
    gram_diag = np.diag(np.diag(gram))
    
    # Mean off-diagonal similarity (information sharing)
    off_diag_mask = ~np.eye(N, dtype=bool)
    mean_sharing = gram[off_diag_mask].mean()
    std_sharing = gram[off_diag_mask].std()
    
    # Variance of row norms (agents with more "influence" = higher norm)
    row_norms = np.linalg.norm(gram, axis=1)
    influence_variance = np.var(row_norms)
    
    result = {
        "mean_cross_agent_similarity": float(mean_sharing),
        "similarity_std": float(std_sharing),
        "influence_variance": float(influence_variance),
    }
    
    # If attention weights are available, compute attention-specific metrics
    if attention_weights:
        entropies = []
        for aw in attention_weights:
            if len(aw) > 1:
                ent = -np.sum(aw * np.log(aw + 1e-10))
                entropies.append(ent)
        if entropies:
            result["mean_attn_entropy"] = float(np.mean(entropies))
            result["entropy_variance"] = float(np.var(entropies))
    
    return result


# ============================================================================
# 3. Self/Other Distinction (RSA-based)
# ============================================================================

def compute_self_other_distinction(
    hiddens: np.ndarray,
    tribe_ids: Optional[np.ndarray] = None,
) -> dict:
    """
    Self/Other distinction using Representational Similarity Analysis.
    
    Key metrics:
    - within_tribe_similarity: how similar are same-tribe agents?
    - between_tribe_similarity: how similar are different-tribe agents?
    - distinction_ratio: within / between (>1 = more similar to own tribe)
    - self_consistency: how stable is an agent's hidden state over time?
    
    If no tribe_ids, use first 5 agents as "self" and rest as "other".
    """
    N, D = hiddens.shape
    h = (hiddens - hiddens.mean(axis=0)) / (hiddens.std(axis=0) + 1e-8)
    
    # Default: no tribe info, use positional split
    if tribe_ids is None:
        tribe_ids = np.array([0] * (N // 2) + [1] * (N - N // 2))
    
    unique_tribes = np.unique(tribe_ids)
    if len(unique_tribes) < 2:
        return {"within_tribe_sim": 0.0, "between_tribe_sim": 0.0, "distinction_ratio": 1.0}
    
    gram = h @ h.T  # (N, N)
    
    within_sims = []
    between_sims = []
    
    for t in unique_tribes:
        members = np.where(tribe_ids == t)[0]
        if len(members) < 2:
            continue
        # Within-tribe pairwise similarity
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                within_sims.append(gram[members[i], members[j]])
        # Between-tribe
        others = np.where(tribe_ids != t)[0]
        for i in range(len(members)):
            between_sims.extend(gram[members[i], others])
    
    if not within_sims or not between_sims:
        return {"within_tribe_sim": 0.0, "between_tribe_sim": 0.0, "distinction_ratio": 1.0}
    
    within = np.mean(within_sims)
    between = np.mean(between_sims)
    
    # Distinction ratio: >1 means agents are more similar to their own tribe
    ratio = within / (between + 1e-10)
    
    # Representational diversity: std of agent norms
    agent_norms = np.linalg.norm(h, axis=1)
    
    return {
        "within_tribe_sim": float(within),
        "between_tribe_sim": float(between),
        "distinction_ratio": float(ratio),
        "representational_diversity": float(np.std(agent_norms)),
        "mean_agent_norm": float(np.mean(agent_norms)),
    }


# ============================================================================
# 4. Combined Analysis
# ============================================================================

def compute_all_metrics(
    generation_log: List[dict],
    compute_phi_flag: bool = True,
) -> dict:
    """
    Compute all metrics from generation log.
    
    Expects each generation entry to have:
    - sample_hiddens: (N, D) array or list of arrays
    - mean_attn_entropy: float
    """
    metrics = {
        "phi": [],
        "gwt_cross_similarity": [],
        "gwt_influence_var": [],
        "gwt_entropy": [],
        "selfother_within": [],
        "selfother_between": [],
        "selfother_ratio": [],
        "selfother_diversity": [],
    }
    
    for gen_data in generation_log:
        hiddens = gen_data.get("sample_hiddens")
        
        # GWT entropy (always available)
        metrics["gwt_entropy"].append(gen_data.get("mean_attn_entropy", 0))
        
        if hiddens is None:
            for k in ["phi", "gwt_cross_similarity", "gwt_influence_var",
                       "selfother_within", "selfother_between", "selfother_ratio",
                       "selfother_diversity"]:
                metrics[k].append(None)
            continue
        
        h = np.array(hiddens)
        
        # Φ (expensive, optional)
        if compute_phi_flag:
            metrics["phi"].append(compute_phi(h))
        else:
            metrics["phi"].append(None)
        
        # GWT
        gwt = compute_gwt_metrics(h)
        metrics["gwt_cross_similarity"].append(gwt["mean_cross_agent_similarity"])
        metrics["gwt_influence_var"].append(gwt["influence_variance"])
        
        # Self/Other
        so = compute_self_other_distinction(h)
        metrics["selfother_within"].append(so["within_tribe_sim"])
        metrics["selfother_between"].append(so["between_tribe_sim"])
        metrics["selfother_ratio"].append(so["distinction_ratio"])
        metrics["selfother_diversity"].append(so["representational_diversity"])
    
    return metrics


def analyze_emergence(metrics: dict, window_size: int = 30) -> dict:
    """
    Detect emergence: significant sustained trend in a metric.
    """
    results = {}
    
    for name, values in metrics.items():
        clean = [v for v in values if v is not None and np.isfinite(v)]
        if len(clean) < window_size * 2:
            continue
        
        early = np.mean(clean[:window_size])
        late = np.mean(clean[-window_size:])
        trend = late - early
        
        # Linear regression slope
        x = np.arange(len(clean))
        slope = np.polyfit(x, clean, 1)[0]
        
        # Phase transition: max |change| in window
        windows = [np.mean(clean[i:i+window_size]) - np.mean(clean[max(0,i-window_size):i])
                   for i in range(window_size, len(clean))]
        max_rate = max(abs(w) for w in windows) if windows else 0
        max_rate_idx = int(np.argmax([abs(w) for w in windows])) if windows else 0
        
        # Significance: |trend| > 10% of |early| AND |slope| meaningful
        rel_change = abs(trend) / (abs(early) + 1e-10)
        
        results[name] = {
            "early_mean": float(early),
            "late_mean": float(late),
            "trend": float(trend),
            "slope_per_gen": float(slope),
            "rel_change": float(rel_change),
            "max_rate": float(max_rate),
            "max_rate_gen": max_rate_idx,
            "emerged": rel_change > 0.1 and abs(slope) > 1e-5,
        }
    
    return results
