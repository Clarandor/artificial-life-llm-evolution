"""
Phase 2.1: Supervised Attention Learning
=========================================
Instead of REINFORCE, use supervised learning to train attention
to focus on neighbors that are:
1. Closest to prey
2. Same tribe (for coordination)
3. Have high fitness (successful agents)

Key insight: REINFORCE failed because gradient vanished.
Supervised learning provides stable gradient signal.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

OBS_DIM    = 16
HIDDEN_DIM = 32
ATT_DIM    = 4
DEC_DIM    = 32
ACTION_DIM = 6
MAX_NEIGHBORS = 8


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-8)


class SupervisedAttentionMLP:
    """Attention agent with supervised attention target."""

    WEIGHT_NAMES = ['W_enc', 'b_enc', 'W_q', 'W_k', 'W_dec', 'b_dec', 'W_act', 'b_act']

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
            for name, w in zip(self.WEIGHT_NAMES, weights):
                setattr(self, name, w)

    def encode(self, obs: np.ndarray) -> np.ndarray:
        return relu(self.W_enc @ obs + self.b_enc)

    def attend(self, h_self: np.ndarray, neighbor_hiddens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns: (context, attn_weights)"""
        K = neighbor_hiddens.shape[0]
        if K == 0:
            return np.zeros(ATT_DIM, dtype=np.float32), np.array([], dtype=np.float32)

        q = self.W_q @ h_self
        keys = (self.W_k @ neighbor_hiddens.T).T
        vals = neighbor_hiddens[:, :ATT_DIM]

        scores = keys @ q / np.sqrt(ATT_DIM)
        attn = softmax(scores)
        context = attn @ vals
        return context, attn

    def decide(self, h_self: np.ndarray, context: np.ndarray) -> int:
        combined = np.concatenate([h_self, context])
        h_dec = relu(self.W_dec @ combined + self.b_dec)
        logits = self.W_act @ h_dec + self.b_act
        probs = softmax(logits)
        return int(np.random.choice(ACTION_DIM, p=probs))

    def get_weights(self) -> List[np.ndarray]:
        return [getattr(self, name) for name in self.WEIGHT_NAMES]

    @property
    def param_count(self) -> int:
        return sum(w.size for w in self.get_weights())


def compute_attention_target(
    agent_pos: Tuple[int, int],
    neighbor_positions: List[Tuple[int, int]],
    prey_positions: List[Tuple[int, int]],
    neighbor_tribes: List[int],
    agent_tribe: int,
    neighbor_fitnesses: List[float],
    grid_size: int = 32,
) -> np.ndarray:
    """
    Compute target attention distribution based on task-relevant factors.
    
    Target is a weighted combination of:
    1. Prey proximity: neighbors closer to prey get higher weight
    2. Same tribe: tribe members get bonus
    3. Fitness: successful neighbors get bonus
    """
    K = len(neighbor_positions)
    if K == 0:
        return np.array([], dtype=np.float32)
    
    scores = np.zeros(K, dtype=np.float32)
    
    for i, (npos, ntribe, nfit) in enumerate(zip(neighbor_positions, neighbor_tribes, neighbor_fitnesses)):
        # Prey proximity score
        if prey_positions:
            prey_dists = [abs(npos[0]-p[0]) + abs(npos[1]-p[1]) for p in prey_positions]
            min_prey_dist = min(prey_dists)
            # Closer to prey = higher score
            scores[i] += 2.0 / (min_prey_dist + 1)
        
        # Same tribe bonus
        if ntribe == agent_tribe:
            scores[i] += 1.0
        
        # Fitness bonus (normalized)
        scores[i] += 0.1 * min(nfit, 10) / 10.0
    
    # Softmax to get distribution
    return softmax(scores)
