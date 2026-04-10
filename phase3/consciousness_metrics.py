"""
Phase 3: Consciousness Metrics
==============================
Measure consciousness-related metrics in evolved agent populations:
1. Φ (Integrated Information) - IIT theory
2. Global Workspace Broadcast - GWT theory
3. Self/Other Distinction - Self-model theory

These metrics can be computed from the existing simulation data
without requiring successful coordination.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# 1. Φ (Integrated Information) - Simplified IIT
# ============================================================================

def compute_phi_simplified(
    hiddens: np.ndarray,
    partition_strategy: str = "min_cut"
) -> float:
    """
    Compute simplified Φ (integrated information).
    
    Φ measures how much the whole system is more than the sum of its parts.
    Simplified version: use mutual information between partitions.
    
    Args:
        hiddens: (N, D) array of agent hidden states
        partition_strategy: how to partition the system
    
    Returns:
        Φ value (higher = more integrated)
    """
    N, D = hiddens.shape
    if N < 2 or D < 2:
        return 0.0
    
    # Normalize
    h = (hiddens - hiddens.mean(axis=0)) / (hiddens.std(axis=0) + 1e-8)
    
    # Compute covariance matrix
    cov = np.cov(h.T)
    
    # Total entropy (approximated by log determinant)
    total_entropy = 0.5 * np.log(np.linalg.det(cov) + 1e-8)
    
    # Partition: split agents into two groups
    if partition_strategy == "min_cut":
        # Simple partition: first half vs second half
        mid = N // 2
        
        # Entropy of each partition
        cov_a = np.cov(h[:mid].T) if mid > 0 else np.eye(D)
        cov_b = np.cov(h[mid:].T) if (N - mid) > 0 else np.eye(D)
        
        entropy_a = 0.5 * np.log(np.linalg.det(cov_a) + 1e-8)
        entropy_b = 0.5 * np.log(np.linalg.det(cov_b) + 1e-8)
        
        # Φ = Total - sum of parts
        phi = total_entropy - (entropy_a + entropy_b)
    else:
        # Alternative: use mean field approximation
        phi = total_entropy - np.sum(np.log(np.diag(cov) + 1e-8))
    
    return max(0.0, phi)


# ============================================================================
# 2. Global Workspace Broadcast - GWT
# ============================================================================

def compute_gwt_broadcast(
    hiddens: np.ndarray,
    attention_weights: List[np.ndarray],
    threshold: float = 0.3
) -> dict:
    """
    Measure Global Workspace Broadcast properties.
    
    GWT: Information becomes globally available when it enters the workspace.
    Proxy: Measure how widely attention is distributed.
    
    Args:
        hiddens: (N, D) agent hidden states
        attention_weights: list of attention distributions per agent
        threshold: threshold for "broadcast" detection
    
    Returns:
        dict with broadcast metrics
    """
    N = len(hiddens)
    
    # Broadcast range: how many agents receive significant attention
    broadcast_ranges = []
    for aw in attention_weights:
        if len(aw) > 0:
            # Count neighbors receiving > threshold attention
            broadcast_count = np.sum(aw > threshold)
            broadcast_ranges.append(broadcast_count / len(aw))
    
    # Broadcast speed: how quickly information spreads
    # Proxy: variance of attention entropy across agents
    attn_entropies = []
    for aw in attention_weights:
        if len(aw) > 0:
            entropy = -np.sum(aw * np.log(aw + 1e-10))
            attn_entropies.append(entropy)
    
    return {
        "mean_broadcast_range": np.mean(broadcast_ranges) if broadcast_ranges else 0.0,
        "broadcast_variance": np.var(broadcast_ranges) if broadcast_ranges else 0.0,
        "mean_attn_entropy": np.mean(attn_entropies) if attn_entropies else 0.0,
        "entropy_variance": np.var(attn_entropies) if attn_entropies else 0.0,
    }


# ============================================================================
# 3. Self/Other Distinction
# ============================================================================

def compute_self_other_distinction(
    hiddens: np.ndarray,
    self_weights: np.ndarray,
    other_weights: np.ndarray,
) -> dict:
    """
    Measure self/other distinction in hidden representations.
    
    Theory: Conscious agents should have distinct representations
    for self-state vs other-states.
    
    Args:
        hiddens: (N, D) agent hidden states
        self_weights: (D,) weights for self-representation
        other_weights: (D,) weights for other-representation
    
    Returns:
        dict with distinction metrics
    """
    N, D = hiddens.shape
    
    # Self-activation: how much each hidden activates self-representation
    self_activations = hiddens @ self_weights  # (N,)
    
    # Other-activation: how much each hidden activates other-representation
    other_activations = hiddens @ other_weights  # (N,)
    
    # Distinction: correlation between self and other activations
    # Lower correlation = better distinction
    correlation = np.corrcoef(self_activations, other_activations)[0, 1]
    
    # Self-preference: do agents activate self more than others?
    self_preference = np.mean(self_activations) - np.mean(other_activations)
    
    return {
        "self_other_correlation": correlation,
        "self_preference": self_preference,
        "self_activation_mean": np.mean(self_activations),
        "self_activation_std": np.std(self_activations),
        "other_activation_mean": np.mean(other_activations),
        "other_activation_std": np.std(other_activations),
    }


def extract_self_other_weights(
    batch_weights: dict,
    method: str = "attention"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract self/other weights from trained network.
    
    Methods:
    - "attention": W_q represents self, W_k represents others
    - "decoder": First half of W_dec for self, second for others
    """
    if method == "attention":
        # Use attention weights as proxy
        # W_q: how agent queries (self-perspective)
        # W_k: how agent is queried by others (other-perspective)
        W_q = batch_weights.get('W_q', np.zeros((1, 4, 32)))[0]  # (4, 32)
        W_k = batch_weights.get('W_k', np.zeros((1, 4, 32)))[0]  # (4, 32)
        
        # Average over attention dimensions
        self_weights = W_q.mean(axis=0)  # (32,)
        other_weights = W_k.mean(axis=0)  # (32,)
    else:
        # Use decoder weights
        W_dec = batch_weights.get('W_dec', np.zeros((1, 32, 36)))[0]  # (32, 36)
        self_weights = W_dec[:, :32].mean(axis=0)  # First 32 dims
        other_weights = W_dec[:, 32:].mean(axis=0) if W_dec.shape[1] > 32 else np.zeros(32)
    
    return self_weights, other_weights


# ============================================================================
# 4. Combined Metrics Computation
# ============================================================================

def compute_all_consciousness_metrics(
    generation_log: List[dict],
    batch_weights: Optional[dict] = None,
) -> dict:
    """
    Compute all consciousness metrics from generation log.
    
    Args:
        generation_log: list of generation data
        batch_weights: trained network weights (optional)
    
    Returns:
        dict with all metrics over time
    """
    metrics = {
        "phi": [],
        "gwt_broadcast_range": [],
        "gwt_entropy": [],
        "self_other_corr": [],
        "self_preference": [],
    }
    
    for gen_data in generation_log:
        # Get sample hiddens if available
        hiddens = gen_data.get("sample_hiddens")
        if hiddens is not None:
            h = np.array(hiddens)
            
            # Φ
            phi = compute_phi_simplified(h)
            metrics["phi"].append(phi)
            
            # Self/Other (if weights available)
            if batch_weights is not None:
                self_w, other_w = extract_self_other_weights(batch_weights)
                so_metrics = compute_self_other_distinction(h, self_w, other_w)
                metrics["self_other_corr"].append(so_metrics["self_other_correlation"])
                metrics["self_preference"].append(so_metrics["self_preference"])
        else:
            metrics["phi"].append(None)
            metrics["self_other_corr"].append(None)
            metrics["self_preference"].append(None)
        
        # GWT metrics (would need attention weights from log)
        # For now, use entropy as proxy
        entropy = gen_data.get("mean_attn_entropy", 0)
        metrics["gwt_entropy"].append(entropy)
    
    return metrics


# ============================================================================
# 5. Analysis and Visualization
# ============================================================================

def analyze_consciousness_emergence(
    metrics: dict,
    window_size: int = 20
) -> dict:
    """
    Analyze whether consciousness metrics show emergence patterns.
    
    Emergence indicators:
    - Sudden jump (phase transition)
    - Sustained increase
    - Correlation with fitness
    """
    results = {}
    
    for metric_name, values in metrics.items():
        # Filter out None values
        clean_values = [v for v in values if v is not None]
        if len(clean_values) < window_size * 2:
            continue
        
        # Early vs late comparison
        early = np.mean(clean_values[:window_size])
        late = np.mean(clean_values[-window_size:])
        
        # Trend
        trend = late - early
        
        # Phase transition detection (max rate of change)
        if len(clean_values) > window_size:
            rates = []
            for i in range(len(clean_values) - window_size):
                rate = (clean_values[i + window_size] - clean_values[i]) / window_size
                rates.append(rate)
            max_rate_idx = np.argmax(np.abs(rates))
            max_rate = rates[max_rate_idx]
        else:
            max_rate = 0
            max_rate_idx = 0
        
        results[metric_name] = {
            "early_mean": early,
            "late_mean": late,
            "trend": trend,
            "max_rate": max_rate,
            "max_rate_gen": max_rate_idx,
            "emerged": abs(trend) > 0.1 * abs(early) if early != 0 else False,
        }
    
    return results
