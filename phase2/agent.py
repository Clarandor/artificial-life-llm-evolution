"""
Phase 2: Hybrid GA + REINFORCE Agent
========================================
Same factored attention architecture as Phase 1.1 (ATT_DIM=4, no V).

Key difference: attend() also returns raw attention logits (pre-softmax scores)
so REINFORCE can compute ∇log π(attention | state).

REINFORCE gradient for attention:
  - attention distribution α = softmax(scores) IS the policy π
  - "action" = the attention weights themselves (continuous)
  - For continuous attention, we use the score function estimator:
    ∇_θ J ≈ Σ_t (R_t - b) · ∇_θ log p(attn_t | h_self, neighbors; θ)
  - Since attn = softmax(Q·K^T / √d), the gradient flows through Q and K.

We compute the gradient analytically:
  ∂L/∂scores = (R - b) · (attn - attn_target)  [where attn_target = e_i for attended]
  ∂scores/∂W_q and ∂scores/∂W_k via chain rule

Parameter budget: same as Phase 1.1 ≈ 2,182 params
  GA optimizes:  W_enc(544) + W_dec(1,184) + W_act(198) = 1,926 params
  REINFORCE:     W_q(128) + W_k(128) = 256 params
"""

import numpy as np
from typing import List, Tuple, Optional

OBS_DIM    = 16
HIDDEN_DIM = 32
ATT_DIM    = 4
DEC_DIM    = 32
ACTION_DIM = 6     # up/down/left/right/collect/attack
MAX_NEIGHBORS = 8


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-8)


class HybridAttentionMLP:
    """Factored attention agent with REINFORCE-compatible forward pass."""

    # Weights managed by GA (behavior)
    GA_WEIGHT_NAMES = ['W_enc', 'b_enc', 'W_dec', 'b_dec', 'W_act', 'b_act']
    # Weights managed by REINFORCE (attention)
    RL_WEIGHT_NAMES = ['W_q', 'W_k']
    # All weights
    WEIGHT_NAMES = GA_WEIGHT_NAMES + RL_WEIGHT_NAMES

    def __init__(self, weights: List[np.ndarray] = None, seed: int = None):
        rng = np.random.default_rng(seed)
        if weights is None:
            self.W_enc = rng.normal(0, np.sqrt(2/OBS_DIM),    (HIDDEN_DIM, OBS_DIM)).astype(np.float32)
            self.b_enc = np.zeros(HIDDEN_DIM, dtype=np.float32)
            self.W_q   = rng.normal(0, np.sqrt(2/HIDDEN_DIM), (ATT_DIM, HIDDEN_DIM)).astype(np.float32)
            self.W_k   = rng.normal(0, np.sqrt(2/HIDDEN_DIM), (ATT_DIM, HIDDEN_DIM)).astype(np.float32)
            self.W_dec = rng.normal(0, np.sqrt(2/(HIDDEN_DIM+ATT_DIM)), (DEC_DIM, HIDDEN_DIM + ATT_DIM)).astype(np.float32)
            self.b_dec = np.zeros(DEC_DIM, dtype=np.float32)
            self.W_act = rng.normal(0, np.sqrt(2/DEC_DIM),    (ACTION_DIM, DEC_DIM)).astype(np.float32)
            self.b_act = np.zeros(ACTION_DIM, dtype=np.float32)
        else:
            (self.W_enc, self.b_enc, self.W_dec, self.b_dec,
             self.W_act, self.b_act, self.W_q, self.W_k) = weights

    def encode(self, obs: np.ndarray) -> np.ndarray:
        return relu(self.W_enc @ obs + self.b_enc)

    def attend(self, h_self: np.ndarray, neighbor_hiddens: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns: (context, attn_weights, raw_scores)
        raw_scores needed for REINFORCE gradient.
        """
        K = neighbor_hiddens.shape[0]
        if K == 0:
            return (np.zeros(ATT_DIM, dtype=np.float32),
                    np.array([], dtype=np.float32),
                    np.array([], dtype=np.float32))

        q = self.W_q @ h_self                         # (ATT_DIM,)
        keys = (self.W_k @ neighbor_hiddens.T).T      # (K, ATT_DIM)
        vals = neighbor_hiddens[:, :ATT_DIM]           # (K, ATT_DIM)

        scores = keys @ q / np.sqrt(ATT_DIM)          # (K,)
        attn = softmax(scores)                         # (K,)
        context = attn @ vals                          # (ATT_DIM,)
        return context, attn, scores

    def decide(self, h_self: np.ndarray, context: np.ndarray) -> int:
        combined = np.concatenate([h_self, context])
        h_dec = relu(self.W_dec @ combined + self.b_dec)
        logits = self.W_act @ h_dec + self.b_act
        probs = softmax(logits)
        return int(np.random.choice(ACTION_DIM, p=probs))

    def get_weights(self) -> List[np.ndarray]:
        return [getattr(self, name) for name in self.WEIGHT_NAMES]

    def get_ga_weights(self) -> List[np.ndarray]:
        return [getattr(self, name) for name in self.GA_WEIGHT_NAMES]

    def get_rl_weights(self) -> List[np.ndarray]:
        return [getattr(self, name) for name in self.RL_WEIGHT_NAMES]

    @property
    def param_count(self) -> int:
        return sum(w.size for w in self.get_weights())

    @property
    def ga_param_count(self) -> int:
        return sum(w.size for w in self.get_ga_weights())

    @property
    def rl_param_count(self) -> int:
        return sum(w.size for w in self.get_rl_weights())
