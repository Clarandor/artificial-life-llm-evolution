"""
Phase 1: Attention-based MLP Agent
=====================================
Architecture:
    1. Encoder:  obs(16) → W_enc(32×16) + b_enc → h_self (32), ReLU
    2. Attention: single-head over K nearest neighbor hidden states
       Q = W_q(16×32) · h_self            → (16,)
       K = W_k(16×32) · h_neighbor        → (K, 16)
       V = W_v(16×32) · h_neighbor        → (K, 16)
       attn = softmax(Q · K^T / sqrt(16)) → (K,)
       context = attn · V                 → (16,)
    3. Decoder:  concat(h_self(32), context(16)) = 48 → W_dec(32×48) + b_dec → ReLU
                 → W_act(6×32) + b_act → softmax → action

No message head. Communication is implicit via attention over neighbor states.
Observable: attention weights reveal "who looks at whom".

Total params: ~3,400 per agent.
"""

import numpy as np
from typing import List, Tuple, Optional

OBS_DIM    = 16
HIDDEN_DIM = 32
ATT_DIM    = 16    # attention key/query/value dimension
DEC_DIM    = 32    # decoder hidden dim
ACTION_DIM = 6     # up/down/left/right/collect/attack
MAX_NEIGHBORS = 8  # max neighbors to attend over


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-8)


class AttentionMLP:
    """Single agent with attention over neighbor hidden states."""

    WEIGHT_NAMES = [
        'W_enc', 'b_enc',     # encoder
        'W_q', 'W_k', 'W_v', # attention projections (no bias for compactness)
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
            self.W_v   = rng.normal(0, np.sqrt(2/HIDDEN_DIM), (ATT_DIM, HIDDEN_DIM)).astype(np.float32)
            self.W_dec = rng.normal(0, np.sqrt(2/(HIDDEN_DIM+ATT_DIM)), (DEC_DIM, HIDDEN_DIM + ATT_DIM)).astype(np.float32)
            self.b_dec = np.zeros(DEC_DIM, dtype=np.float32)
            self.W_act = rng.normal(0, np.sqrt(2/DEC_DIM),    (ACTION_DIM, DEC_DIM)).astype(np.float32)
            self.b_act = np.zeros(ACTION_DIM, dtype=np.float32)
        else:
            (self.W_enc, self.b_enc,
             self.W_q, self.W_k, self.W_v,
             self.W_dec, self.b_dec,
             self.W_act, self.b_act) = weights

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Encode observation into hidden state."""
        return relu(self.W_enc @ obs + self.b_enc)

    def attend(self, h_self: np.ndarray, neighbor_hiddens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Attend over neighbor hidden states.
        Args:
            h_self: (HIDDEN_DIM,) — this agent's hidden state
            neighbor_hiddens: (K, HIDDEN_DIM) — neighbor hidden states
        Returns:
            context: (ATT_DIM,) — weighted sum of neighbor values
            attn_weights: (K,) — attention distribution
        """
        K = neighbor_hiddens.shape[0]
        if K == 0:
            return np.zeros(ATT_DIM, dtype=np.float32), np.array([], dtype=np.float32)

        q = self.W_q @ h_self                         # (ATT_DIM,)
        keys = (self.W_k @ neighbor_hiddens.T).T      # (K, ATT_DIM)
        vals = (self.W_v @ neighbor_hiddens.T).T      # (K, ATT_DIM)

        scores = keys @ q / np.sqrt(ATT_DIM)          # (K,)
        attn = softmax(scores)                         # (K,)
        context = attn @ vals                          # (ATT_DIM,)
        return context, attn

    def decide(self, h_self: np.ndarray, context: np.ndarray) -> int:
        """Decide action from hidden state + attention context."""
        combined = np.concatenate([h_self, context])   # (HIDDEN_DIM + ATT_DIM,)
        h_dec = relu(self.W_dec @ combined + self.b_dec)
        logits = self.W_act @ h_dec + self.b_act
        probs = softmax(logits)
        return int(np.random.choice(ACTION_DIM, p=probs))

    def forward(self, obs: np.ndarray, neighbor_hiddens: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Full forward pass.
        Returns: (action, hidden_state, attention_weights)
        """
        h = self.encode(obs)
        context, attn = self.attend(h, neighbor_hiddens)
        action = self.decide(h, context)
        return action, h, attn

    def get_weights(self) -> List[np.ndarray]:
        return [getattr(self, name) for name in self.WEIGHT_NAMES]

    @classmethod
    def from_weights(cls, weights: List[np.ndarray]) -> "AttentionMLP":
        return cls(weights=weights)

    def mutate(self, sigma: float = 0.01, rng: np.random.Generator = None) -> "AttentionMLP":
        if rng is None:
            rng = np.random.default_rng()
        new_weights = [w + rng.normal(0, sigma, w.shape).astype(np.float32) for w in self.get_weights()]
        return AttentionMLP(weights=new_weights)

    @property
    def param_count(self) -> int:
        return sum(w.size for w in self.get_weights())
