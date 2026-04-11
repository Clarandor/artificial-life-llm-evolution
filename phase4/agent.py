"""
Phase 4: Recurrent Agent (LSTM)
==================================
"""

import numpy as np
from typing import List, Tuple


HIDDEN_DIM = 32
ATT_DIM = 4
MAX_NEIGHBORS = 8
OBS_DIM = 16
ACTION_DIM = 6
GRID_SIZE = 32


# ── LSTM Cell ──────────────────────────────────────────────────────────────

class LSTMCell:
    def __init__(self, input_dim: int, hidden_dim: int, rng: np.random.RandomState):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        total_w = 4 * hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W = rng.normal(0, scale, (total_w, input_dim + hidden_dim)).astype(np.float32)
        self.b = rng.normal(0, 0.01, (total_w,)).astype(np.float32)

    @property
    def num_params(self) -> int:
        return self.W.size + self.b.size

    def set_params(self, flat: np.ndarray):
        n = 4 * self.hidden_dim
        self.W = flat[:n * (self.input_dim + self.hidden_dim)].reshape(n, self.input_dim + self.hidden_dim)
        self.b = flat[n * (self.input_dim + self.hidden_dim):]

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W.flatten(), self.b])

    def forward(self, x: np.ndarray, h: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        combined = np.concatenate([x, h])
        gates = self.W @ combined + self.b
        D = self.hidden_dim
        i = 1.0 / (1.0 + np.exp(-gates[:D]))
        f = 1.0 / (1.0 + np.exp(-gates[D:2*D]))
        o = 1.0 / (1.0 + np.exp(-gates[2*D:3*D]))
        g = np.tanh(gates[3*D:])
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        return h_new.astype(np.float32), c_new.astype(np.float32)


# ── Attention QK ─────────────────────────────────────────────────────────

class AttentionQK:
    def __init__(self, hidden_dim: int, att_dim: int, rng: np.random.RandomState):
        self.hidden_dim = hidden_dim
        self.att_dim = att_dim
        self.W_q = rng.normal(0, 0.01, (att_dim, hidden_dim)).astype(np.float32)
        self.W_k = rng.normal(0, 0.01, (att_dim, hidden_dim)).astype(np.float32)

    @property
    def num_params(self) -> int:
        return self.W_q.size + self.W_k.size

    def set_params(self, flat: np.ndarray):
        mid = self.att_dim * self.hidden_dim
        self.W_q = flat[:mid].reshape(self.att_dim, self.hidden_dim)
        self.W_k = flat[mid:2*mid].reshape(self.att_dim, self.hidden_dim)

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W_q.flatten(), self.W_k.flatten()])

    def attend(self, h_self: np.ndarray, h_neighbors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single-head attention. h_neighbors: (K, hidden_dim) or list of (hidden_dim,)"""
        if isinstance(h_neighbors, list):
            if not h_neighbors:
                return np.zeros(self.att_dim, dtype=np.float32), np.array([], dtype=np.float32)
            h_neighbors = np.stack(h_neighbors)
        K = len(h_neighbors)
        if K == 0:
            return np.zeros(self.att_dim, dtype=np.float32), np.array([], dtype=np.float32)
        
        q = self.W_q @ h_self  # (att_dim,)
        k = h_neighbors @ self.W_k.T  # (K, att_dim)
        scores = (k @ q) / np.sqrt(self.att_dim)  # (K,)
        weights = self._softmax(scores)
        context = np.tanh(np.einsum('k,kd->d', weights, h_neighbors[:, :self.att_dim]))
        return context.astype(np.float32), weights.astype(np.float32)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        exp = np.exp(x)
        return exp / (exp.sum() + 1e-10)


# ── Agent ────────────────────────────────────────────────────────────────

class Agent:
    """LSTM agent with attention. State + weights stored as plain fields (not dataclass)."""
    
    _next_id = 0
    
    def __init__(self, tribe_id: int, rng: np.random.RandomState, tribe_templates: dict = None):
        self.id = Agent._next_id
        Agent._next_id += 1
        self.tribe_id = tribe_id
        
        if tribe_templates is not None:
            # Share LSTM/Attention templates within tribe
            lstm_t = tribe_templates[tribe_id]['lstm']
            attn_t = tribe_templates[tribe_id]['attn']
        else:
            # Create new
            lstm_t = LSTMCell(OBS_DIM, HIDDEN_DIM, rng)
            attn_t = AttentionQK(HIDDEN_DIM, ATT_DIM, rng)
        
        self.lstm = lstm_t
        self.attn = attn_t
        
        # Decoder: concat(h, ctx) -> hidden_dim
        self.dec_W = rng.normal(0, 0.01, (HIDDEN_DIM, HIDDEN_DIM + ATT_DIM)).astype(np.float32)
        self.dec_b = rng.normal(0, 0.01, (HIDDEN_DIM,)).astype(np.float32)
        # Action head
        self.act_W = rng.normal(0, 0.01, (ACTION_DIM, HIDDEN_DIM)).astype(np.float32)
        self.act_b = rng.normal(0, 0.01, (ACTION_DIM,)).astype(np.float32)
        
        self.x = rng.uniform(0, GRID_SIZE)
        self.y = rng.uniform(0, GRID_SIZE)
        self.energy = 200.0  # Increased from 40 for LSTM stability
        self.alive = True
        self.food_collected = 0.0
        self.prey_captured = 0
        self.age = 0
        
        # LSTM state
        self.h = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.c = np.zeros(HIDDEN_DIM, dtype=np.float32)
    
    @property
    def hidden(self) -> np.ndarray:
        return self.h
    
    def reset_hidden(self):
        self.h = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.c = np.zeros(HIDDEN_DIM, dtype=np.float32)
    
    def encode(self, obs: np.ndarray):
        self.h, self.c = self.lstm.forward(obs.astype(np.float32), self.h, self.c)
    
    def decide(self, neighbor_hiddens: List[np.ndarray]) -> np.ndarray:
        ctx, _ = self.attn.attend(self.h, neighbor_hiddens)
        dec_in = np.concatenate([self.h, ctx])
        decoded = np.tanh(self.dec_W @ dec_in + self.dec_b)
        action = self.act_W @ decoded + self.act_b
        return np.array([
            np.clip(action[0], -1, 1),
            np.clip(action[1], -1, 1),
            np.clip(action[2], 0, 1),
            float(action[3] > 0),
            float(action[4] > 0),
            float(action[5]),
        ], dtype=np.float32)
    
    def set_weights(self, lstm_p, attn_p, dec_p, act_p):
        self.lstm.set_params(lstm_p)
        self.attn.set_params(attn_p)
        self.dec_W = dec_p[:-HIDDEN_DIM].reshape(HIDDEN_DIM, HIDDEN_DIM + ATT_DIM)
        self.dec_b = dec_p[-HIDDEN_DIM:]
        self.act_W = act_p[:-ACTION_DIM].reshape(ACTION_DIM, HIDDEN_DIM)
        self.act_b = act_p[-ACTION_DIM:]

    def clone_weights_from(self, other: 'Agent'):
        """Copy weights from another agent."""
        self.lstm.set_params(other.lstm.get_params())
        self.attn.set_params(other.attn.get_params())
        self.dec_W = other.dec_W.copy()
        self.dec_b = other.dec_b.copy()
        self.act_W = other.act_W.copy()
        self.act_b = other.act_b.copy()
    
    @staticmethod
    def count_params() -> dict:
        rng = np.random.RandomState(0)
        lstm = LSTMCell(OBS_DIM, HIDDEN_DIM, rng)
        attn = AttentionQK(HIDDEN_DIM, ATT_DIM, rng)
        dec = rng.normal(0, 0.01, (HIDDEN_DIM, HIDDEN_DIM + ATT_DIM))
        act = rng.normal(0, 0.01, (ACTION_DIM, HIDDEN_DIM))
        return {
            "lstm": lstm.num_params,
            "attn": attn.num_params,
            "decoder": dec.size + HIDDEN_DIM,
            "action": act.size + ACTION_DIM,
        }
