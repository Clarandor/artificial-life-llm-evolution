"""
Phase 1.1: Factored Attention MLP Agent
==========================================
Key changes from Phase 1:
  - ATT_DIM: 16 → 4  (QK projection dimension)
  - V projection: REMOVED — context = attn @ h_neighbors[:, :ATT_DIM]
    (directly reads the first 4 dims of neighbor hidden states)
  - Decoder input: 32 + 4 = 36 (was 32 + 16 = 48)

Parameter budget:
  W_enc(32×16) + b_enc(32) = 544
  W_q(4×32) = 128
  W_k(4×32) = 128
  W_dec(32×36) + b_dec(32) = 1,184
  W_act(6×32) + b_act(6) = 198
  ────────────────────────────
  Total ≈ 2,182 params (vs Phase 1's 3,846)

Hypothesis: With only 256 attention params (vs 1,536), GA can explore
the QK space effectively and find selective attention patterns.
"""

import numpy as np
from typing import List, Tuple, Optional

OBS_DIM    = 16
HIDDEN_DIM = 32
ATT_DIM    = 4     # reduced from 16 → 4
DEC_DIM    = 32
ACTION_DIM = 6     # up/down/left/right/collect/attack
MAX_NEIGHBORS = 8


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-8)


class FactoredAttentionMLP:
    """Factored attention: small QK dim, no V projection."""

    WEIGHT_NAMES = [
        'W_enc', 'b_enc',     # encoder
        'W_q', 'W_k',         # attention (no W_v!)
        'W_dec', 'b_dec',     # decoder
        'W_act', 'b_act',     # action head
    ]

    def __init__(self, weights: List[np.ndarray] = None, seed: int = None):
        rng = np.random.default_rng(seed)
        if weights is None:
            self.W_enc = rng.normal(0, np.sqrt(2/OBS_DIM),    (HIDDEN_DIM, OBS_DIM)).astype(np.float32)
            self.b_enc = np.zeros(HIDDEN_DIM, dtype=np.float32)
            self.W_q   = rng.normal(0, np.sqrt(2/HIDDEN_DIM), (ATT_DIM, HIDDEN_DIM)).astype(np.float32)
            self.W_k   = rng.normal(0, np.sqrt(2/HIDDEN_DIM), (ATT_DIM, HIDDEN_DIM)).astype(np.float32)
            # No W_v — context reads neighbor hiddens directly
            self.W_dec = rng.normal(0, np.sqrt(2/(HIDDEN_DIM+ATT_DIM)), (DEC_DIM, HIDDEN_DIM + ATT_DIM)).astype(np.float32)
            self.b_dec = np.zeros(DEC_DIM, dtype=np.float32)
            self.W_act = rng.normal(0, np.sqrt(2/DEC_DIM),    (ACTION_DIM, DEC_DIM)).astype(np.float32)
            self.b_act = np.zeros(ACTION_DIM, dtype=np.float32)
        else:
            (self.W_enc, self.b_enc,
             self.W_q, self.W_k,
             self.W_dec, self.b_dec,
             self.W_act, self.b_act) = weights

    def encode(self, obs: np.ndarray) -> np.ndarray:
        return relu(self.W_enc @ obs + self.b_enc)

    def attend(self, h_self: np.ndarray, neighbor_hiddens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Factored attention — no V projection.
        Context = weighted sum of neighbor hiddens' first ATT_DIM dims.
        """
        K = neighbor_hiddens.shape[0]
        if K == 0:
            return np.zeros(ATT_DIM, dtype=np.float32), np.array([], dtype=np.float32)

        q = self.W_q @ h_self                         # (ATT_DIM,)
        keys = (self.W_k @ neighbor_hiddens.T).T      # (K, ATT_DIM)
        # No V projection — use raw hidden dims
        vals = neighbor_hiddens[:, :ATT_DIM]           # (K, ATT_DIM)

        scores = keys @ q / np.sqrt(ATT_DIM)          # (K,)
        attn = softmax(scores)                         # (K,)
        context = attn @ vals                          # (ATT_DIM,)
        return context, attn

    def decide(self, h_self: np.ndarray, context: np.ndarray) -> int:
        combined = np.concatenate([h_self, context])   # (HIDDEN_DIM + ATT_DIM,)
        h_dec = relu(self.W_dec @ combined + self.b_dec)
        logits = self.W_act @ h_dec + self.b_act
        probs = softmax(logits)
        return int(np.random.choice(ACTION_DIM, p=probs))

    def forward(self, obs: np.ndarray, neighbor_hiddens: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        h = self.encode(obs)
        context, attn = self.attend(h, neighbor_hiddens)
        action = self.decide(h, context)
        return action, h, attn

    def get_weights(self) -> List[np.ndarray]:
        return [getattr(self, name) for name in self.WEIGHT_NAMES]

    @classmethod
    def from_weights(cls, weights: List[np.ndarray]) -> "FactoredAttentionMLP":
        return cls(weights=weights)

    def mutate(self, sigma: float = 0.01, rng: np.random.Generator = None) -> "FactoredAttentionMLP":
        if rng is None:
            rng = np.random.default_rng()
        new_weights = [w + rng.normal(0, sigma, w.shape).astype(np.float32) for w in self.get_weights()]
        return FactoredAttentionMLP(weights=new_weights)

    @property
    def param_count(self) -> int:
        return sum(w.size for w in self.get_weights())
