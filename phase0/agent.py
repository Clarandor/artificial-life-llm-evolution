"""
Phase 0.2: MLP Agent — Prey Hunt + Group Selection
====================================================
Architecture:
    Input:   OBS_DIM(16) + MSG_DIM(4) = 20
    Hidden1: 32, ReLU
    Hidden2: 32, ReLU
    Output:
        action_head: 6  (up/down/left/right/collect/attack), softmax
        message_head: 4, tanh

Total params: ~2,100 per agent.
"""

import numpy as np
from typing import List, Tuple

INPUT_DIM  = 20   # OBS_DIM(16) + MSG_DIM(4)
HIDDEN_DIM = 32
ACTION_DIM = 6    # ← 5→6: added attack
MSG_DIM    = 4


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class MLP:
    def __init__(self, weights: List[np.ndarray] = None, seed: int = None):
        rng = np.random.default_rng(seed)
        if weights is None:
            self.W1 = rng.normal(0, np.sqrt(2/INPUT_DIM),  (HIDDEN_DIM, INPUT_DIM )).astype(np.float32)
            self.b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
            self.W2 = rng.normal(0, np.sqrt(2/HIDDEN_DIM), (HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32)
            self.b2 = np.zeros(HIDDEN_DIM, dtype=np.float32)
            self.Wa = rng.normal(0, np.sqrt(2/HIDDEN_DIM), (ACTION_DIM, HIDDEN_DIM)).astype(np.float32)
            self.ba = np.zeros(ACTION_DIM, dtype=np.float32)
            self.Wm = rng.normal(0, np.sqrt(2/HIDDEN_DIM), (MSG_DIM,   HIDDEN_DIM)).astype(np.float32)
            self.bm = np.zeros(MSG_DIM,    dtype=np.float32)
        else:
            self.W1, self.b1, self.W2, self.b2, self.Wa, self.ba, self.Wm, self.bm = weights

    def forward(self, obs: np.ndarray) -> Tuple[int, np.ndarray]:
        h1 = relu(self.W1 @ obs + self.b1)
        h2 = relu(self.W2 @ h1 + self.b2)
        action_logits = self.Wa @ h2 + self.ba
        action_probs  = softmax(action_logits)
        action = int(np.random.choice(ACTION_DIM, p=action_probs))
        message = tanh(self.Wm @ h2 + self.bm)
        return action, message

    def get_weights(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.Wa, self.ba, self.Wm, self.bm]

    @classmethod
    def from_weights(cls, weights: List[np.ndarray]) -> "MLP":
        return cls(weights=weights)

    def mutate(self, sigma: float = 0.01, rng: np.random.Generator = None) -> "MLP":
        if rng is None:
            rng = np.random.default_rng()
        new_weights = [w + rng.normal(0, sigma, w.shape).astype(np.float32) for w in self.get_weights()]
        return MLP(weights=new_weights)

    @property
    def param_count(self) -> int:
        return sum(w.size for w in self.get_weights())
