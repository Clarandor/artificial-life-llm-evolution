"""
Phase 0: MLP Agent with dual output heads
==========================================
Architecture:
    Input:   OBS_DIM(10) + MSG_DIM(16) = 26
    Hidden1: 64, ReLU
    Hidden2: 64, ReLU
    Output:
        action_head: 5  (up/down/left/right/collect), softmax
        message_head: 16, tanh

Total params: ~6,400 per agent.
"""

import numpy as np
from typing import List, Tuple

# Dimensions (must match environment.py)
INPUT_DIM  = 26   # OBS_DIM + MSG_DIM
HIDDEN_DIM = 64
ACTION_DIM = 5
MSG_DIM    = 16


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class MLP:
    """
    Minimal 2-hidden-layer MLP with dual output heads.
    Weights stored as plain numpy arrays for easy GA mutation.
    """

    def __init__(self, weights: List[np.ndarray] = None, seed: int = None):
        rng = np.random.default_rng(seed)
        if weights is None:
            # Xavier-ish initialization
            self.W1 = rng.normal(0, np.sqrt(2 / INPUT_DIM),  (HIDDEN_DIM, INPUT_DIM)).astype(np.float32)
            self.b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
            self.W2 = rng.normal(0, np.sqrt(2 / HIDDEN_DIM), (HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32)
            self.b2 = np.zeros(HIDDEN_DIM, dtype=np.float32)
            # Action head
            self.Wa = rng.normal(0, np.sqrt(2 / HIDDEN_DIM), (ACTION_DIM, HIDDEN_DIM)).astype(np.float32)
            self.ba = np.zeros(ACTION_DIM, dtype=np.float32)
            # Message head
            self.Wm = rng.normal(0, np.sqrt(2 / HIDDEN_DIM), (MSG_DIM, HIDDEN_DIM)).astype(np.float32)
            self.bm = np.zeros(MSG_DIM, dtype=np.float32)
        else:
            self.W1, self.b1, self.W2, self.b2, self.Wa, self.ba, self.Wm, self.bm = weights

    def forward(self, obs: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Forward pass.
        Returns:
            action (int): sampled action index
            message (np.ndarray): outgoing message vector shape (MSG_DIM,)
        """
        h1 = relu(self.W1 @ obs + self.b1)
        h2 = relu(self.W2 @ h1 + self.b2)

        action_logits = self.Wa @ h2 + self.ba
        action_probs  = softmax(action_logits)
        action        = int(np.random.choice(ACTION_DIM, p=action_probs))

        message = tanh(self.Wm @ h2 + self.bm)

        return action, message

    def get_weights(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.Wa, self.ba, self.Wm, self.bm]

    @classmethod
    def from_weights(cls, weights: List[np.ndarray]) -> "MLP":
        return cls(weights=weights)

    def mutate(self, sigma: float = 0.01, rng: np.random.Generator = None) -> "MLP":
        """Return a new MLP with Gaussian noise added to all weights."""
        if rng is None:
            rng = np.random.default_rng()
        new_weights = [
            w + rng.normal(0, sigma, w.shape).astype(np.float32)
            for w in self.get_weights()
        ]
        return MLP(weights=new_weights)

    def crossover(self, other: "MLP", rng: np.random.Generator = None) -> "MLP":
        """Uniform crossover between two parents (optional, not used in Phase 0 baseline)."""
        if rng is None:
            rng = np.random.default_rng()
        child_weights = []
        for w1, w2 in zip(self.get_weights(), other.get_weights()):
            mask = rng.random(w1.shape) > 0.5
            child_weights.append(np.where(mask, w1, w2).astype(np.float32))
        return MLP(weights=child_weights)

    @property
    def param_count(self) -> int:
        return sum(w.size for w in self.get_weights())
