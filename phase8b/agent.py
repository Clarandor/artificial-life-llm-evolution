"""
Phase 8B: Temporal Alignment
============================
Key innovation from Phase 7: Temporal Alignment + Weight Reduction

Phase 7 problem: Recursive attention grew 4x but coordination fell
(Phase 6: 1.08% → Phase 7: 0.70%). Hypothesis B: decision latency.

Root cause: Level 2 attention used "previous step" neighbor attention,
creating a 1-step temporal lag that compounds errors during fast prey movement.

Phase 8B solution:
  1. Reduce Level 2 recursive weight to 0.3 (was 1.0)
  2. Add temporal alignment: compare historical vs current attention,
     then align before aggregating (reduces lag error)
  3. Combined: rec_attn_output = rec_ctx * 0.3 + aligned_rec_ctx * 0.5

Architecture (same as Phase 7):
  Level 0: Self hidden state
  Level 1: Neighbor hidden states (standard Q/K attention)
  Level 2: Neighbor attention patterns (recursive + temporally aligned)
"""

import numpy as np
from typing import List, Tuple

HIDDEN_DIM = 32
ATT_DIM = 4
MAX_NEIGHBORS = 8
OBS_DIM = 24     # 16 base + 4 coordination + 4 recursive attention (unchanged)
ACTION_DIM = 7   # 6 actions + 1 broadcast
GRID_SIZE = 32

# Phase 8B key hyperparameters
REC_WEIGHT = 0.3          # Reduced from 1.0 (Phase 7)
ALIGN_WEIGHT = 0.5         # Temporal alignment weight
COST_WEIGHT = 0.05         # Attention loss weight (same as Phase 7)


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


class TemporalAlignedAttention:
    """
    Two-level attention with temporal alignment.
    
    Level 1: Standard Q/K attention over neighbor hidden states
    Level 2: Recursive + temporally-aligned attention over neighbor attention patterns
    
    Key improvement over Phase 7's RecursiveAttention:
    - Temporal alignment: cross-correlate historical (prev-step) attention weights
      with current attention weights to estimate and correct for temporal lag
    - Reduced weight: rec_ctx * 0.3 + aligned_rec_ctx * 0.5 (vs raw rec_ctx in Phase 7)
    """
    
    def __init__(self, hidden_dim, att_dim, rng):
        self.hidden_dim = hidden_dim
        self.att_dim = att_dim
        self.rec_weight = REC_WEIGHT
        self.align_weight = ALIGN_WEIGHT
        
        # Level 1: Standard attention
        self.W_q = rng.normal(0, 0.01, (att_dim, hidden_dim)).astype(np.float32)
        self.W_k = rng.normal(0, 0.01, (att_dim, hidden_dim)).astype(np.float32)
        
        # Level 2: Recursive projection (compress neighbor attention into 4D)
        self.W_rec = rng.normal(0, 0.01, (att_dim, MAX_NEIGHBORS)).astype(np.float32)
        
        # Level 2b: Temporal alignment — learns to align historical vs current attention
        self.W_align = rng.normal(0, 0.01, (att_dim, MAX_NEIGHBORS)).astype(np.float32)
        
        # Store last-level-1 attention weights for temporal alignment in next step
        self._prev_level1_weights = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
    
    def set_params(self, flat):
        n = self.att_dim * self.hidden_dim
        rec_n = self.att_dim * MAX_NEIGHBORS
        self.W_q = flat[:n].reshape(self.att_dim, self.hidden_dim)
        self.W_k = flat[n:2*n].reshape(self.att_dim, self.hidden_dim)
        self.W_rec = flat[2*n:2*n+rec_n].reshape(self.att_dim, MAX_NEIGHBORS)
        self.W_align = flat[2*n+rec_n:2*n+2*rec_n].reshape(self.att_dim, MAX_NEIGHBORS)
    
    def get_params(self):
        return np.concatenate([
            self.W_q.flatten(), self.W_k.flatten(),
            self.W_rec.flatten(), self.W_align.flatten()
        ])
    
    def attend(self, h_self, h_neighbors, neighbor_attn_weights=None):
        """
        Two-level recursive attention with temporal alignment.
        
        Args:
            h_self: (hidden_dim,) self hidden state
            h_neighbors: list of (hidden_dim,) or (K, hidden_dim) array
            neighbor_attn_weights: list of K arrays, each is neighbor i's attention weights
                                   (from previous step — used for Level 2)
        
        Returns:
            context: (att_dim,) Level 1 aggregated context
            recursive_output: (att_dim,) Level 2 output = rec_ctx*0.3 + aligned_rec_ctx*0.5
            weights: (K,) Level 1 attention weights
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
        
        # Save Level 1 weights for next step's temporal alignment
        prev_w = self._prev_level1_weights.copy()
        self._prev_level1_weights[:K] = weights
        
        # ── Level 2: Recursive attention with temporal alignment ──
        if neighbor_attn_weights is not None and len(neighbor_attn_weights) > 0:
            # Pad neighbor attention weights to MAX_NEIGHBORS
            attn_matrix = np.zeros((MAX_NEIGHBORS, MAX_NEIGHBORS), dtype=np.float32)
            for i, aw in enumerate(neighbor_attn_weights[:MAX_NEIGHBORS]):
                if len(aw) > 0:
                    attn_matrix[i, :len(aw)] = aw[:MAX_NEIGHBORS]
            
            # Sum over neighbors to get "what my neighbors attend to" (Phase 7 style)
            neighbor_attn_summary = attn_matrix.sum(axis=0)[:MAX_NEIGHBORS]
            raw_rec_ctx = self.W_rec @ neighbor_attn_summary
            
            # ── Temporal Alignment (NEW in Phase 8B) ──
            # Compare previous step's Level 1 weights with current Level 1 weights
            # to estimate how attention patterns have shifted over time
            if K > 0 and np.sum(prev_w[:K]) > 1e-6:
                # Correlation between previous and current attention weights
                # prev_w[:K] = attention from last step
                # weights[:K] = attention from current step
                
                # Cross-correlation as alignment signal
                cross_corr = prev_w[:K] * weights[:K]
                cross_corr_sum = cross_corr.sum()
                
                # Temporal shift: how different is current from previous?
                # High cross_corr = stable attention = low temporal lag error
                # Low cross_corr = attention shifted = high temporal lag error
                temporal_error = 1.0 - cross_corr_sum / (np.sum(prev_w[:K]**2)**0.5 * np.sum(weights[:K]**2)**0.5 + 1e-10)
                temporal_error = np.clip(temporal_error, 0, 1)
                
                # Alignment correction: weight the rec context more when temporal shift is high
                # (because the raw recursive context is more stale in that case)
                align_strength = 1.0 - 0.3 * temporal_error  # [0.7, 1.0]
                
                # Aligned attention = blend of previous and current weights
                # aligned_weights = align_strength * current + (1-align_strength) * cross_corr
                aligned_w = align_strength * weights[:K] + (1.0 - align_strength) * cross_corr
                aligned_w = np.abs(aligned_w)
                if aligned_w.sum() > 1e-10:
                    aligned_w = aligned_w / aligned_w.sum()
                
                # Pad aligned_w to MAX_NEIGHBORS=8 before matmul with W_align
                aligned_w_padded = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
                aligned_w_padded[:K] = aligned_w
                
                # Project aligned attention through alignment matrix
                aligned_rec_ctx = self.W_align @ aligned_w_padded
            else:
                # No history yet — use standard recursive context
                aligned_rec_ctx = raw_rec_ctx
                align_strength = 1.0
        else:
            # No neighbor attention weights — fallback to Phase 7 behavior
            raw_rec_ctx = np.zeros(self.att_dim, dtype=np.float32)
            aligned_rec_ctx = np.zeros(self.att_dim, dtype=np.float32)
            align_strength = 0.0
        
        # Phase 8B key formula: weighted combination
        # rec_attn_output = rec_ctx * 0.3 + aligned_rec_ctx * 0.5
        raw_rec_ctx_full = self.W_rec @ np.ones(MAX_NEIGHBORS) if neighbor_attn_weights is None else self.W_rec @ np.zeros(MAX_NEIGHBORS)
        
        if neighbor_attn_weights is not None and len(neighbor_attn_weights) > 0:
            attn_matrix = np.zeros((MAX_NEIGHBORS, MAX_NEIGHBORS), dtype=np.float32)
            for i, aw in enumerate(neighbor_attn_weights[:MAX_NEIGHBORS]):
                if len(aw) > 0:
                    attn_matrix[i, :len(aw)] = aw[:MAX_NEIGHBORS]
            neighbor_attn_summary = attn_matrix.sum(axis=0)[:MAX_NEIGHBORS]
            raw_rec_ctx_full = self.W_rec @ neighbor_attn_summary
        else:
            raw_rec_ctx_full = np.zeros(self.att_dim, dtype=np.float32)
        
        recursive_output = np.tanh(
            raw_rec_ctx_full * self.rec_weight + aligned_rec_ctx * self.align_weight
        ).astype(np.float32)
        
        return context.astype(np.float32), recursive_output, weights.astype(np.float32)


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
            self.attn = TemporalAlignedAttention(HIDDEN_DIM, ATT_DIM, rng)
        
        # Decoder: concat(h_self, context, recursive_output) → hidden
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
        # NEW: track temporal alignment strength per step
        self._last_align_strength = 0.0
    
    @property
    def last_attn_weights(self):
        return self._last_attn_weights
    
    def reset_hidden(self):
        self.h = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.c = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self._last_attn_weights = np.array([], dtype=np.float32)
        self._last_align_strength = 0.0
    
    def encode(self, obs):
        self.h, self.c = self.lstm.forward(obs.astype(np.float32), self.h, self.c)
    
    def decide(self, neighbor_hiddens, neighbor_attn_weights=None):
        ctx, rec_out, weights = self.attn.attend(
            self.h, neighbor_hiddens, neighbor_attn_weights
        )
        self._last_attn_weights = weights
        
        dec_in = np.concatenate([self.h, ctx, rec_out])
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
