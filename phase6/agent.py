"""
Phase 6: Agent (same as phase5 but with coordination loss)
"""
import numpy as np
from typing import List, Tuple

HIDDEN_DIM = 32
ATT_DIM = 4
MAX_NEIGHBORS = 8
OBS_DIM = 20
ACTION_DIM = 7
GRID_SIZE = 32


class LSTMCell:
    def __init__(self, input_dim, hidden_dim, rng):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        n = 4 * hidden_dim
        self.W = rng.normal(0, np.sqrt(2.0/(input_dim+hidden_dim)), (n, input_dim+hidden_dim)).astype(np.float32)
        self.b = rng.normal(0, 0.01, (n,)).astype(np.float32)
    
    def set_params(self, flat):
        n = 4 * self.hidden_dim
        self.W = flat[:n*(self.input_dim+self.hidden_dim)].reshape(n, self.input_dim+self.hidden_dim)
        self.b = flat[n*(self.input_dim+self.hidden_dim):]
    
    def get_params(self):
        return np.concatenate([self.W.flatten(), self.b])
    
    def forward(self, x, h, c):
        combined = np.concatenate([x, h])
        g = self.W @ combined + self.b
        D = self.hidden_dim
        i, f, o, g_val = 1.0/(1+np.exp(-g[:D])), 1.0/(1+np.exp(-g[D:2*D])), 1.0/(1+np.exp(-g[2*D:3*D])), np.tanh(g[3*D:])
        c_new = f * c + i * g_val
        h_new = o * np.tanh(c_new)
        return h_new.astype(np.float32), c_new.astype(np.float32)


class AttentionQK:
    def __init__(self, hidden_dim, att_dim, rng):
        self.hidden_dim = hidden_dim
        self.att_dim = att_dim
        self.W_q = rng.normal(0, 0.01, (att_dim, hidden_dim)).astype(np.float32)
        self.W_k = rng.normal(0, 0.01, (att_dim, hidden_dim)).astype(np.float32)
    
    def set_params(self, flat):
        n = self.att_dim * self.hidden_dim
        self.W_q = flat[:n].reshape(self.att_dim, self.hidden_dim)
        self.W_k = flat[n:2*n].reshape(self.att_dim, self.hidden_dim)
    
    def get_params(self):
        return np.concatenate([self.W_q.flatten(), self.W_k.flatten()])
    
    def attend(self, h_self, h_neighbors):
        if isinstance(h_neighbors, list):
            if not h_neighbors: return np.zeros(self.att_dim, dtype=np.float32), np.array([], dtype=np.float32)
            h_neighbors = np.stack(h_neighbors)
        K = len(h_neighbors)
        if K == 0: return np.zeros(self.att_dim, dtype=np.float32), np.array([], dtype=np.float32)
        q = self.W_q @ h_self
        k = h_neighbors @ self.W_k.T
        scores = (k @ q) / np.sqrt(self.att_dim)
        x = scores - scores.max()
        weights = np.exp(x) / (np.exp(x).sum() + 1e-10)
        context = np.tanh(np.einsum('k,kd->d', weights, h_neighbors[:, :self.att_dim]))
        return context.astype(np.float32), weights.astype(np.float32)


class Agent:
    _next_id = 0
    
    def __init__(self, tribe_id, rng, tribe_templates=None):
        self.id = Agent._next_id
        Agent._next_id += 1
        self.tribe_id = tribe_id
        
        if tribe_templates is not None:
            self.lstm = tribe_templates[tribe_id]['lstm']
            self.attn = tribe_templates[tribe_id]['attn']
        else:
            self.lstm = LSTMCell(OBS_DIM, HIDDEN_DIM, rng)
            self.attn = AttentionQK(HIDDEN_DIM, ATT_DIM, rng)
        
        self.dec_W = rng.normal(0, 0.01, (HIDDEN_DIM, HIDDEN_DIM+ATT_DIM)).astype(np.float32)
        self.dec_b = rng.normal(0, 0.01, (HIDDEN_DIM,)).astype(np.float32)
        self.act_W = rng.normal(0, 0.01, (ACTION_DIM, HIDDEN_DIM)).astype(np.float32)
        self.act_b = rng.normal(0, 0.01, (ACTION_DIM,)).astype(np.float32)
        
        self.x = rng.uniform(0, GRID_SIZE)
        self.y = rng.uniform(0, GRID_SIZE)
        self.energy = 200.0
        self.alive = True
        self.food_collected = 0.0
        self.large_prey_captured = 0
        self.failed_attacks = 0
        self.signals_sent = 0
        self.signals_received = 0
        self.coord_attempts = 0  # New: coordination attempts
        self.age = 0
        self.h = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.c = np.zeros(HIDDEN_DIM, dtype=np.float32)
    
    def reset_hidden(self):
        self.h = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.c = np.zeros(HIDDEN_DIM, dtype=np.float32)
    
    def encode(self, obs):
        self.h, self.c = self.lstm.forward(obs.astype(np.float32), self.h, self.c)
    
    def decide(self, neighbor_hiddens):
        ctx, _ = self.attn.attend(self.h, neighbor_hiddens)
        dec_in = np.concatenate([self.h, ctx])
        decoded = np.tanh(self.dec_W @ dec_in + self.dec_b)
        action = self.act_W @ decoded + self.act_b
        raw = np.clip(action, -3, 3)
        return np.array([
            np.clip(raw[0], -1, 1),
            np.clip(raw[1], -1, 1),
            np.clip(raw[2], 0, 1),
            float(raw[3] > 0),
            float(raw[4] > 0),  # attack signal
            float(raw[5]),       # reserved
            float(np.clip(raw[6], 0, 4)),  # signal broadcast
        ], dtype=np.float32)
    
    def clone_weights_from(self, other):
        self.lstm.set_params(other.lstm.get_params())
        self.attn.set_params(other.attn.get_params())
        self.dec_W = other.dec_W.copy()
        self.dec_b = other.dec_b.copy()
        self.act_W = other.act_W.copy()
        self.act_b = other.act_b.copy()


def count_params():
    from phase5.agent import Agent as OldAgent
    return {"total": OldAgent.count_params()["total"] + 32} if hasattr(OldAgent, 'count_params') else {"total": 8455}
