"""
Phase 7: Recursive Attention (Theory of Mind)
==============================================
Key innovation: Agents attend not just to neighbors' hidden states,
but also to neighbors' attention patterns — "who is attending to whom."

This creates a recursive attention structure:
  Level 0: Self hidden state
  Level 1: Neighbor hidden states (standard attention)
  Level 2: Neighbor attention patterns (recursive — "what does my neighbor attend to?")

Hypothesis: Recursive attention enables Theory of Mind — agents can
predict what other agents will do by understanding what they perceive.

Architecture changes from Phase 6:
  - Attention output now includes neighbor attention weights
  - Decoder receives: [h_self | context | neighbor_attention_context]
  - OBS_DIM: 20 → 24 (+4 for recursive attention info)
  - Total params: ~10,000 (vs 8,455 in Phase 6)
"""

import numpy as np
from typing import List, Tuple

HIDDEN_DIM = 32
ATT_DIM = 4
MAX_NEIGHBORS = 8
OBS_DIM = 24     # 16 base + 4 coordination + 4 recursive attention
ACTION_DIM = 7   # 6 actions + 1 broadcast
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
        total = n * (self.input_dim + self.hidden_dim)
        self.W = flat[:total].reshape(n, self.input_dim + self.hidden_dim)
        self.b = flat[total:]
    
    def get_params(self):
        return np.concatenate([self.W.flatten(), self.b])
    
    def forward(self, x, h, c):
        combined = np.concatenate([x, h])
        g = self.W @ combined + self.b
        D = self.hidden_dim
        i = 1.0/(1+np.exp(-g[:D]))
        f = 1.0/(1+np.exp(-g[D:2*D]))
        o = 1.0/(1+np.exp(-g[2*D:3*D]))
        g_val = np.tanh(g[3*D:])
        c_new = f * c + i * g_val
        h_new = o * np.tanh(c_new)
        return h_new.astype(np.float32), c_new.astype(np.float32)


class RecursiveAttention:
    """
    Two-level attention:
    Level 1: Standard Q/K attention over neighbor hidden states
    Level 2: Aggregate neighbor attention patterns into recursive context
    
    The recursive context captures "what my neighbors are paying attention to"
    which is the foundation of Theory of Mind.
    """
    def __init__(self, hidden_dim, att_dim, rng):
        self.hidden_dim = hidden_dim
        self.att_dim = att_dim
        # Level 1: Standard attention
        self.W_q = rng.normal(0, 0.01, (att_dim, hidden_dim)).astype(np.float32)
        self.W_k = rng.normal(0, 0.01, (att_dim, hidden_dim)).astype(np.float32)
        # Level 2: Recursive projection (compress neighbor attention into 4D)
        self.W_rec = rng.normal(0, 0.01, (att_dim, MAX_NEIGHBORS)).astype(np.float32)
    
    def set_params(self, flat):
        n = self.att_dim * self.hidden_dim
        self.W_q = flat[:n].reshape(self.att_dim, self.hidden_dim)
        self.W_k = flat[n:2*n].reshape(self.att_dim, self.hidden_dim)
        self.W_rec = flat[2*n:2*n+self.att_dim*MAX_NEIGHBORS].reshape(self.att_dim, MAX_NEIGHBORS)
    
    def get_params(self):
        return np.concatenate([self.W_q.flatten(), self.W_k.flatten(), self.W_rec.flatten()])
    
    def attend(self, h_self, h_neighbors, neighbor_attn_weights=None):
        """
        Two-level recursive attention.
        
        Args:
            h_self: (hidden_dim,) self hidden state
            h_neighbors: list of (hidden_dim,) or (K, hidden_dim) array
            neighbor_attn_weights: list of K arrays, each is neighbor i's attention weights
                                   If None, Level 2 context is zero.
        
        Returns:
            context: (att_dim,) aggregated context
            recursive_context: (att_dim,) recursive attention context
            weights: (K,) attention weights
        """
        if isinstance(h_neighbors, list):
            if not h_neighbors:
                z = np.zeros(self.att_dim, dtype=np.float32)
                return z, z, np.array([], dtype=np.float32)
            h_neighbors = np.stack(h_neighbors)
        K = len(h_neighbors)
        if K == 0:
            z = np.zeros(self.att_dim, dtype=np.float32)
            return z, z, np.array([], dtype=np.float32)
        
        # Level 1: Standard attention
        q = self.W_q @ h_self
        k = h_neighbors @ self.W_k.T
        scores = (k @ q) / np.sqrt(self.att_dim)
        x = scores - scores.max()
        weights = np.exp(x) / (np.exp(x).sum() + 1e-10)
        context = np.tanh(np.einsum('k,kd->d', weights, h_neighbors[:, :self.att_dim]))
        
        # Level 2: Recursive attention
        # Aggregate neighbor attention patterns
        if neighbor_attn_weights is not None and len(neighbor_attn_weights) > 0:
            # Pad neighbor attention weights to MAX_NEIGHBORS
            attn_matrix = np.zeros((MAX_NEIGHBORS, MAX_NEIGHBORS), dtype=np.float32)
            for i, aw in enumerate(neighbor_attn_weights[:MAX_NEIGHBORS]):
                if len(aw) > 0:
                    attn_matrix[i, :len(aw)] = aw[:MAX_NEIGHBORS]
            
            # Sum over neighbors to get "what my neighbors attend to"
            neighbor_attn_summary = attn_matrix.sum(axis=0)[:MAX_NEIGHBORS]
            recursive_context = np.tanh(self.W_rec @ neighbor_attn_summary)
        else:
            recursive_context = np.zeros(self.att_dim, dtype=np.float32)
        
        return context.astype(np.float32), recursive_context.astype(np.float32), weights.astype(np.float32)


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
            self.attn = RecursiveAttention(HIDDEN_DIM, ATT_DIM, rng)
        
        # Decoder: concat(h_self, context, recursive_context) → hidden
        dec_input_dim = HIDDEN_DIM + ATT_DIM + ATT_DIM  # 32 + 4 + 4 = 40
        self.dec_W = rng.normal(0, 0.01, (HIDDEN_DIM, dec_input_dim)).astype(np.float32)
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
        self.age = 0
        self.h = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.c = np.zeros(HIDDEN_DIM, dtype=np.float32)
        
        # Cache last attention weights (for recursive attention by neighbors)
        self._last_attn_weights = np.array([], dtype=np.float32)
    
    @property
    def last_attn_weights(self):
        return self._last_attn_weights
    
    def reset_hidden(self):
        self.h = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.c = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self._last_attn_weights = np.array([], dtype=np.float32)
    
    def encode(self, obs):
        self.h, self.c = self.lstm.forward(obs.astype(np.float32), self.h, self.c)
    
    def decide(self, neighbor_hiddens, neighbor_attn_weights=None):
        ctx, rec_ctx, weights = self.attn.attend(self.h, neighbor_hiddens, neighbor_attn_weights)
        self._last_attn_weights = weights
        
        dec_in = np.concatenate([self.h, ctx, rec_ctx])
        decoded = np.tanh(self.dec_W @ dec_in + self.dec_b)
        action = self.act_W @ decoded + self.act_b
        raw = np.clip(action, -3, 3)
        return np.array([
            np.clip(raw[0], -1, 1),
            np.clip(raw[1], -1, 1),
            np.clip(raw[2], 0, 1),
            float(raw[3] > 0),
            float(raw[4] > 0),
            float(raw[5]),
            float(np.clip(raw[6], 0, 4)),
        ], dtype=np.float32)
    
    def clone_weights_from(self, other):
        self.lstm.set_params(other.lstm.get_params())
        self.attn.set_params(other.attn.get_params())
        self.dec_W = other.dec_W.copy()
        self.dec_b = other.dec_b.copy()
        self.act_W = other.act_W.copy()
        self.act_b = other.act_b.copy()
    
    @property
    def total_params(self):
        return (self.lstm.get_params().size + self.attn.get_params().size +
                self.dec_W.size + self.dec_b.size + self.act_W.size + self.act_b.size)
