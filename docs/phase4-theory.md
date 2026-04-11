# Phase 4 Theory: LSTM vs Feed-Forward Mirror Mode

**Date:** 2026-04-11  
**Author:** Subagent Analysis  
**Topic:** Explaining why LSTM reverses the "mirror mode" emergent in Phase 2.1 feedforward networks

---

## 1. Data Summary

| Metric | Feedforward (Phase 2.1) | LSTM (Phase 4) | Direction |
|--------|----------------------|----------------|-----------|
| **部落内相似度** (selfother_within) | +83% ↑ | -30% ↓ | **REVERSED** |
| **部落间相似度** (selfother_between) | -45% ↓ | +17% ↑ | **REVERSED** |
| **影响力方差** (gwt_influence_var) | +47% ↑ | -32% ↓ | **REVERSED** |
| GWT 熵 | -68% ↓ | -18% ↓ | Same (both decrease) |
| Φ (信息整合) | +3.1 ↑ | -1.0 ↓ | **REVERSED** |

Three metrics flip sign between architectures. This is not noise — it is a qualitative behavioral shift.

---

## 2. Architecture Comparison

### Phase 2.1 — Feedforward ( SupervisedAttentionMLP)

```
obs (16) → W_enc · ReLU → h_self (32)
                 ↓
         [h_self, neighbor_hiddens] → attention → context (4)
                 ↓
         concat(h_self, context) → W_dec · ReLU → action
```

- **No recurrence.** Each timestep starts from scratch — `encode()` computes `h = ReLU(W_enc @ obs)` fresh.
- **No temporal memory.** Agent state is purely reactive to the current observation.
- **Attention is stateless.** The attention target is computed every step, based on static geometric cues (prey direction, tribe membership).
- All weights optimized by **GA** (all parameters, not just attention).

### Phase 4 — LSTM (Agent with LSTMCell)

```
obs (16) ──→ ┌──────────────────────────┐
              │  LSTM Cell (HIDDEN_DIM=32) │
              │  h_t, c_t = LSTM(x, h_{t-1}, c_{t-1})  │
              └──────────┬─────────────────┘
                         ↓ h_t (32)
              [h_self, neighbors' h_t] → attention → context (4)
                         ↓
              concat → tanh decoder → action (6)
```

- **Temporal recurrence.** Hidden state `h_t` carries information from previous timesteps.
- **Gated memory.** Cell state `c_t` enables selective long-term information retention via input/forget/gate gates.
- **Attention over temporal states.** Attention is computed over neighbor LSTM hidden states, which encode *history*, not just the current observation.
- **Tribe templates shared within tribe.** LSTM cells are shared across all agents in a tribe — agents within the same tribe inherit the same recurrent dynamics.

---

## 3. Core Mechanism: Why the Mirror Effect?

### 3.1 Feedforward → Tribe Clustering (Mirror Mode)

In Phase 2.1, each agent is stateless. The GA must encode "which tribe I belong to" purely into static weights. Because:

1. **GA is blind to temporal structure** — it cannot propagate reward back through time, only through the current genotype.
2. **Stateless agents must be self-sufficient** — each agent must internally encode tribe identity, since it has no way to *communicate identity* through hidden state over time.
3. **Convergent evolution within tribe** — GA selection on tribe-level fitness pushes all agents in the same tribe toward similar weight configurations. Since each agent sees the world identically each step (only obs changes), the only way to signal tribe membership is through **weight convergence**.

This creates **intra-tribe weight convergence** — the "mirror" within each tribe. And since different tribes have different optimization trajectories, they diverge from each other, reducing between-tribe similarity.

**The feedforward network uses weights as a "tribal identity tag."**

### 3.2 LSTM → Tribe Blending (Anti-Mirror Mode)

With LSTM, the dynamics are fundamentally different:

1. **Identity is encoded in dynamics, not weights.** Because the LSTM is shared across the tribe (tribe templates), all tribe members share the *same* recurrent update rule. But they can differ in their **initial hidden state** and **action outputs**.
2. **Temporal behavior diverges even with shared weights.** Starting from different initial conditions, agents with the same LSTM dynamics will produce **different behavioral trajectories** — the "butterfly effect" of recurrent systems. Small differences in initial `h_0, c_0` compound over time into large behavioral divergence.
3. **Attention over temporal states amplifies divergence.** When agent A attends to agent B's hidden state, it sees not just B's current observation response, but B's entire temporal history. If B's hidden state is in a different dynamical regime than A's, the cross-tribe attention pulls A toward a different context vector, which feeds back into A's own LSTM update.
4. **Shared LSTM collapses tribal encoding.** Since all tribe members share the *same* LSTM parameters (tribe templates), the static weights cannot serve as a tribal identity marker. The identity must be carried in the **initial conditions** or **action outputs**, which GA can optimize separately. But initial conditions are reset each generation — the agent cannot rely on them for persistent identity.

**The LSTM encodes "what happened so far" rather than "who I am."**

---

## 4. Theoretical Hypotheses

### H1: Static vs. Dynamic Identity Encoding

**Claim:** Feedforward networks must encode tribal identity statically (in weights), while LSTMs encode identity dynamically (in hidden state trajectory).

- **Feedforward:** Tribe membership is a property of the genotype. GA selection on tribe fitness = selection on genotypes that encode tribe-specific behavior. Within-tribe convergence is a byproduct of GA selecting for the same tribe-specific behaviors.
- **LSTM:** Tribe membership is a property of the dynamics. The shared LSTM template means tribe identity is **not** in the weights. Agents within a tribe differentiate through behavioral dynamics, not weight convergence. This naturally *reduces* within-tribe similarity.

**Prediction:** In feedforward networks, within-tribe hidden state similarity should correlate with weight similarity. In LSTM networks, within-tribe hidden state similarity should be low regardless of weight similarity (because dynamics from different starting points diverge).

### H2: Attention Gradient Amplification Hypothesis

**Claim:** LSTM's attention mechanism operates over temporal representations, causing cross-tribe influence signals to propagate further than in feedforward networks.

In feedforward networks, attention is computed over static hidden vectors. The gradient of attention loss flows only into the current Q/K weights.

In LSTM networks, attention influences:
1. The context vector fed into the decoder → affects the LSTM's input for the *next* timestep
2. This creates a **temporal feedback loop**: attention at time `t` affects behavior at `t`, which affects hidden state at `t+1`, which affects attention at `t+1`

This feedback loop causes cross-tribe interactions to compound over time, reducing tribal coherence (within-tribe similarity drops) and increasing cross-tribe alignment (between-tribe similarity rises).

**Prediction:** The effect should be stronger in later generations (after more timesteps) than early generations. LSTM phase shows more behavioral change from gen 50→300 than gen 0→50.

### H3: Shared Templates → Competitive Dynamics

**Claim:** The tribe template sharing mechanism (Phase 4's `tribe_templates`) fundamentally changes the selection pressure on attention.

In Phase 2.1, each agent has independent weights. GA selection on attention means: agents whose attention patterns improve tribe fitness are selected, and their weights (including Q/K) are propagated.

In Phase 4, tribe templates mean:
- Mutations to the shared LSTM/attention affect all tribe members simultaneously
- A "good" mutation benefits the entire tribe; a "bad" mutation hurts the entire tribe
- This creates **stronger within-tribe coupling** but through dynamics, not through weight similarity
- The coupling is temporal: agents influence each other through attention dynamics, not through weight inheritance

**Prediction:** Attention entropy should be lower in LSTM (because attention converges to consistent patterns across the tribe through shared templates), but the *effects* of that attention (hidden state similarity) should be more diverse because dynamics are sensitive to initial conditions.

### H4: GWT Influence Variance Collapse

**Claim:** LSTM reduces influence variance because the recurrent mechanism **damps individual variability**.

In feedforward networks, each agent makes independent decisions based on its own observations. High-fitness agents have high influence because their actions consistently yield better outcomes. This creates high variance in influence scores across the population.

In LSTM networks, the hidden state acts as a **temporal smoothing filter**. Individual timestep decisions are influenced by the accumulated history, which itself encodes population-level information. The LSTM effectively **averages** over time, reducing the variance of individual influence.

**Evidence:** LSTM's gwt_influence_var trend is -32% vs feedforward's +47%.

---

## 5. Consolidated Explanation

The **"mirror mode"** in Phase 2.1 (feedforward) is a **static attractor** phenomenon:

> Because agents have no temporal memory, tribal identity must be permanently encoded in weights. GA selection drives within-tribe weight convergence → agents in the same tribe become "mirrors" of each other → high within-tribe similarity, low between-tribe similarity.

The **LSTM reversal** is a **dynamic bifurcation** phenomenon:

> Because agents share LSTM templates within a tribe, tribal identity cannot be encoded in weights. It must emerge from the interaction between shared dynamics and individual starting conditions. The recurrent dynamics cause behavioral trajectories to diverge from different starting points. Cross-tribe attention, operating over temporal representations, propagates influence across tribe boundaries. The net effect: tribal behavioral signatures blur, and cross-tribe alignment increases.

```
Feedforward Mirror Mode:
  Stateless → Identity in weights → Within-tribe weight convergence
  → Tribes are distinct clusters in weight-space
  → High within-tribe sim, low between-tribe sim

LSTM Anti-Mirror Mode:
  Recurrent + Shared Templates → Identity in dynamics, not weights
  → Same weights → different trajectories (sensitivity to initial conditions)
  → Cross-tribe attention propagates influence → tribal boundaries blur
  → Low within-tribe sim, higher between-tribe sim
```

---

## 6. Implications for Consciousness Metrics

The reversal in consciousness metrics has important implications:

1. **Φ (Information Integration):** Decreases in LSTM (-1.0) vs increases in feedforward (+3.1). This makes sense: stateless agents need strong internal integration (all identity must be in weights simultaneously), while LSTM agents can "offload" integration to temporal dynamics. The information is distributed across time rather than concentrated in a static representation.

2. **GWT Entropy:** Both decrease, but LSTM decreases less (-18% vs -68%). The shared templates may prevent full convergence to a single attention pattern.

3. **Self/Other Ratio:** Feedforward → more self/other distinction (ratio becomes more negative). LSTM → less distinction (ratio becomes less negative, approaching 0). This is a direct consequence of H2: cross-tribe attention amplifies inter-tribe influence.

---

## 7. Testable Predictions

| # | Prediction | Verification Method |
|---|-----------|--------------------|
| P1 | In Phase 4, within-tribe hidden state similarity should vary with timestep (growing divergence over time) | Track within-tribe similarity over timesteps within a generation |
| P2 | In Phase 4, removing attention should restore mirror mode (increase within-tribe similarity) | Ablation: run Phase 4 with attention disabled |
| P3 | In Phase 2.1, increasing attention loss weight should reduce mirror mode | Increase ATTN_LOSS_WEIGHT in Phase 2.1 |
| P4 | Individual agents with same LSTM weights but different initial hidden states should show diverging behaviors within 20 timesteps | Controlled experiment: fix LSTM weights, vary h_0 |
| P5 | Cross-tribe attention weight should be higher in LSTM than feedforward in later generations | Track attention weights to intra- vs inter-tribe neighbors |

---

## 8. Conclusions

The mirror mode reversal is not an anomaly — it is a **consequence of the fundamental architecture shift** from stateless to recurrent:

- **Feedforward** → Identity encoded in static weights → Tribes as discrete clusters → Mirror mode
- **LSTM** → Identity encoded in temporal dynamics → Tribes as overlapping dynamical attractors → Anti-mirror mode

This mirrors (pun intended) a broader principle in cognitive science: **explicit/declarative memory** (encoded in weights, like feedforward) and **procedural/sequential memory** (encoded in dynamics, like LSTM) produce fundamentally different representations of self and group identity.

The key architectural insight is the **tribe template sharing**: by sharing LSTM cells within a tribe, the system removes the substrate for tribal identity encoding in weights, forcing identity to emerge from dynamical interactions. This is a powerful design choice that reveals the representational consequences of architectural decisions.

---

*End of analysis. Generated by subagent for Phase 4 LSTM vs Phase 2.1 Feedforward comparison.*
