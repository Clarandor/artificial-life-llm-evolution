# Artificial Life × LLM Evolution

> **Can consciousness emerge from a swarm of evolving tiny AI minds?**

This project explores emergent behavior and consciousness-related phenomena in populations of small neural network agents evolving under survival pressure.

Inspired by Avida, Tierra, Lilith (arxiv:2507.04575), and multi-agent communication research.

---

## Research Questions

| Layer | Question |
|-------|----------|
| **Phase 0** | Can small MLPs evolve cooperative behavior via vector communication? |
| **Phase 1** | Can attention-based implicit coordination outperform explicit messaging? |
| **Phase 2** | Can gradient-based optimization (REINFORCE) solve what GA cannot? |
| **Phase 3** | Do consciousness-related metrics (Φ, GWT broadcast, self/other distinction) emerge? |

---

## Current: Phase 2 — Hybrid GA + REINFORCE

### Motivation

After 8 experiments across 2 architectures (Phase 0–1.1), GA was **exhaustively ruled out** as an optimizer for coordination. The core evidence:

- Communication collapse in all message-based variants (Phase 0–0.4a)
- Fixed encoding proves architecture can carry info; GA is the bottleneck (Phase 0.4c)
- Attention entropy stays maximal under GA — degenerates to uniform pooling (Phase 1, 1.1)
- Reducing parameter space 6× doesn't help (Phase 1.1)

Phase 2 introduces **gradient-based optimization for attention weights** while keeping GA for behavior.

### Architecture

```
Each agent (~2,182 parameters):
  1. Encoder:   obs(16) → W_enc(32×16) + b → h_self(32), ReLU
  2. Attention:  factored Q/K (4D), NO V projection, ≤8 neighbors
     Q = W_q(4×32) · h_self          → (4,)
     K = W_k(4×32) · h_neighbor      → (K, 4)
     context = softmax(Q·K^T / √4) · h_neighbor[:, :4]  → (4,)
  3. Decoder:   concat(h_self, context) = 36 → W_dec(32×36) → ReLU → action(6)

Optimization split:
  GA (behavior):     W_enc, b_enc, W_dec, b_dec, W_act, b_act  = 1,926 params
  REINFORCE (attn):  W_q, W_k                                   = 256 params
```

### Key Design (v2 — bug-fixed)

Phase 2 v1 had three critical bugs that killed REINFORCE gradient within 10 generations:
1. **Uniform reward** — `fitnesses[i] / len(buf)` gave identical reward per step → advantage ≈ 0 → ∇ ≈ 0
2. **Argmax as action** — used `argmax(attn)` instead of sampling → noise gradient direction
3. **Buffer-agent mismatch** — GA breed reshuffled agents but buffers indexed by position

v2 fixes:
- **Per-step delta reward**: `Δfood + 3×Δprey + 0.01(survival)` at each timestep
- **Sample from attention distribution**: `rng.choice(K, p=attn)` as REINFORCE action
- **Proper buffer reset** each generation, indexed by batch slot

Verified: gradient norm stays at 0.12–0.19 across generations (no longer dies).

### Status: Awaiting 300-gen run

---

## Experiment History

| Phase | Gens | Architecture | Key Finding | Status |
|-------|------|-------------|-------------|--------|
| **0** | 200 | MLP + 16D msg | Communication Collapse. Environment too simple. | ✅ |
| **0.1** | 300 | MLP + 4D msg | Heavy food + predators. Still collapses. | ✅ |
| **0.2** | 300 | MLP + 4D msg + tribes | Group selection + prey hunt. 1,709 captures, collision-based. | ✅ |
| **0.3** | 300 | MLP + 4D msg + shaping | Reward shaping. Captures ↑33% to 2,268. **Msg still collapse** (var 0.05). | ✅ |
| **0.4a** | 300 | MLP + 4D msg + recv shaping | Receiver shaping → **herding**, prey ↓46%, msg var 0.0008. | ✅ |
| **0.4c** | 300 | MLP + fixed encoding | Fixed msg = prey direction. Var preserved (0.22). GA weakly decodes (recv μ=0.58). | ✅ |
| **1** | 300 | AttentionMLP (QK/V 16D) | Attention over neighbor hiddens. **Entropy stays maximal** (1.89). Degenerates to avg pooling. ~3,846 params. | ✅ |
| **1.1** | 300 | FactoredAttention (QK 4D, no V) | Even with 6× fewer attn params (256 vs 1,536), entropy still maximal (1.91). **Parameter reduction doesn't help.** ~2,182 params. | ✅ |
| **2** | 5 (test) | **Hybrid GA+REINFORCE** | v2 bug-fixed. Gradient norm 0.12–0.19 (alive!). Entropy shows early drop (1.75). **Full 300-gen run pending.** | 🔄 |

### Cumulative Insights

1. **Communication collapse is GA-caused**: GA selects for silent/uniform signals (safe default). Fixed encoding (0.4c) proves the architecture can carry information.

2. **Neither bilateral (messaging) nor unilateral (attention) coordination evolves under GA**: Consistent across architectures — GA cannot navigate the fitness landscape to find coordination strategies.

3. **Reward shaping fixes behavior but not coordination**: Approach-prey strategy persists (0.3) but communication/attention remains unused.

4. **The optimizer is the bottleneck**: After 8 experiments across 2 architectures, the evidence consistently points to GA's inability to optimize coordination in >200-parameter attention spaces.

5. **Parameter space reduction doesn't help**: Phase 1.1 cut attention params 6× (1,536→256) — entropy still maximal. The problem is GA's search algorithm, not the search space size.

---

## Quick Start

```bash
pip install -r requirements.txt

# Phase 2 (hybrid GA + REINFORCE, current)
python run_phase2.py --generations 300 --save-log

# Phase 1 (attention-based, historical)
python run_phase1.py --generations 300 --save-log

# Phase 0 (message-based, historical)
python run_phase0.py --generations 300 --save-log

# Custom seed
python run_phase2.py --seed 0
```

Results saved to `results/`:
- `fitness_curve.png` — fitness over generations
- `prey_captures.png` — cooperative hunt metrics
- `attention_entropy.png` — attention selectivity tracking
- `gradient_norm.png` — REINFORCE gradient signal strength (Phase 2)
- `hidden_pca.png` — PCA of agent hidden states
- `tribe_competition.png` — inter-tribe fitness dynamics
- `generation_log.json` — full experiment data

---

## Project Structure

```
artificial-life-llm-evolution/
├── phase0/                  # Message-based agents (historical)
│   ├── environment.py       # GridWorld with messages + fixed encoding
│   ├── agent.py             # MLP with action + message heads
│   ├── evolution.py         # GA with reward shaping + curriculum
│   └── visualize.py         # Message PCA, variance plots
├── phase1/                  # Attention-based agents (historical)
│   ├── environment.py       # GridWorld with attention neighborhoods
│   ├── agent.py             # AttentionMLP / FactoredAttention
│   ├── evolution.py         # GA with attention forward pass
│   └── visualize.py         # Attention entropy, hidden PCA
├── phase2/                  # Hybrid GA + REINFORCE (current)
│   ├── agent.py             # HybridAttentionMLP (GA + RL weight split)
│   ├── environment.py       # GridWorld (same as Phase 1.1)
│   ├── evolution.py         # HybridEvolutionEngine (GA breed + REINFORCE update)
│   └── visualize.py         # Gradient norm plot added
├── docs/
│   └── research-plan.md
├── results/
├── run_phase0.py
├── run_phase1.py
├── run_phase2.py
└── requirements.txt
```

---

## Roadmap

- [x] Phase 0: MLP agents + GA + vector communication → communication collapse
- [x] Phase 0.1–0.4c: Environment enrichment + reward shaping + diagnostics → GA is the bottleneck
- [x] Phase 1: Attention-based coordination → attention degenerates to uniform pooling
- [x] Phase 1.1: Factored attention (reduced params) → still degenerates under GA
- [x] Phase 2: Hybrid GA + REINFORCE — code complete, v2 bug-fixed, gradient alive
- [ ] Phase 2 full experiment: 300-gen run + analysis
- [ ] Phase 3: Instrument Φ (IIT), global workspace broadcast, self/other distinction

---

## References

1. Lilith — arxiv:2507.04575
2. Words Evolution — arxiv:2505.05863
3. Self-Evolving Agent — arxiv:2601.11658
4. Tononi, G. (2004). IIT. *BMC Neuroscience.*
5. Baars, B.J. (1988). *A Cognitive Theory of Consciousness.*
