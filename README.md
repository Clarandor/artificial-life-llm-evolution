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
| **Phase 2** | Do consciousness-related metrics (Φ, GWT broadcast, self/other distinction) emerge? |

---

## Current: Phase 1 — Attention-based Coordination

### Experiment Purpose

Replace explicit message passing (which collapsed in Phase 0) with **single-head attention over neighbor hidden states**. This reduces the bilateral protocol problem (encode + decode) to a unilateral reading problem (learn what to attend to).

### Architecture

```
Each agent (~3,846 parameters):
  1. Encoder:   obs(16) → W_enc(32×16) + b → h_self(32), ReLU
  2. Attention:  single-head over ≤8 nearest neighbors (Manhattan radius 5)
     Q = W_q(16×32) · h_self          → (16,)
     K = W_k(16×32) · h_neighbor      → (K, 16)
     V = W_v(16×32) · h_neighbor      → (K, 16)
     context = softmax(Q·K^T / √16) · V   → (16,)
  3. Decoder:   concat(h_self, context) = 48 → W_dec(32×48) → ReLU → action(6)

No message channel. Communication is implicit via attention weights.
Observable: attention entropy reveals whether agents develop selective attention.
```

### Phase 1 Results (300 generations)

| Metric | Value |
|--------|-------|
| Total prey captures | **1,704** |
| Prey/gen (first 100) | 5.4 |
| Prey/gen (last 100) | 5.8 |
| Attn entropy (first 100) | 1.826 |
| Attn entropy (last 100) | 1.889 |
| Final mean fitness | 5.836 |

**Key finding**: Attention entropy stays near maximum (~1.9 vs ln(8)=2.08), meaning agents distribute attention uniformly across all neighbors. GA did NOT evolve selective attention — the attention mechanism degenerates into a simple average pooling. Prey captures (5.8/gen) are comparable to Phase 0.2 (5.7/gen without shaping) but below Phase 0.3 (7.8/gen with shaping).

**Hidden state PCA** shows progressive differentiation over generations (early states clustered at origin, late states spread outward), but this reflects general behavioral specialization, not attention-mediated coordination.

### Interpretation

The attention architecture **did not solve the coordination problem**. Even though attention reduces the problem from bilateral to unilateral, GA still cannot find useful attention patterns in the ~3,800 parameter space. The attention weights essentially learn "look at everyone equally" which is equivalent to having no attention at all.

This confirms that the bottleneck is **optimizer capability**, not protocol architecture. GA's random walk cannot navigate the landscape to find functionally meaningful attention weights.

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
| **1** | 300 | **AttentionMLP** | Attention over neighbor hiddens. **Entropy stays maximal** — degenerates to avg pooling. | ✅ |

### Cumulative Insights

1. **Communication collapse is GA-caused**: GA selects for silent/uniform signals (safe default). Fixed encoding (0.4c) proves the architecture can carry information.

2. **Neither bilateral (messaging) nor unilateral (attention) coordination evolves under GA**: The problem is consistent across architectures — GA cannot navigate the fitness landscape to find coordination strategies.

3. **Reward shaping fixes behavior but not coordination**: Approach-prey strategy persists (0.3) but communication/attention remains unused.

4. **The optimizer is the bottleneck**: After 7 experiments across 2 architectures, the evidence consistently points to GA's inability to optimize coordination in >2000-parameter spaces.

### Next Steps

- **Gradient-based optimization**: REINFORCE or policy gradient for communication/attention weights
- **Hybrid GA+gradient**: GA for high-level behavior + gradient descent for coordination
- **Reduce parameter space**: Factored attention (shared Q/K projections) or bottleneck architecture

---

## Quick Start

```bash
pip install -r requirements.txt

# Phase 1 (attention-based, current)
python run_phase1.py --generations 300 --save-log

# Phase 0 (message-based, historical)
python run_phase0.py --generations 300 --save-log

# Custom seed
python run_phase1.py --seed 0
```

Results saved to `results/`:
- `fitness_curve.png` — fitness over generations
- `prey_captures.png` — cooperative hunt metrics
- `attention_entropy.png` — attention selectivity tracking (Phase 1)
- `hidden_pca.png` — PCA of agent hidden states (Phase 1)
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
├── phase1/                  # Attention-based agents (current)
│   ├── environment.py       # GridWorld with attention neighborhoods
│   ├── agent.py             # AttentionMLP (encoder + Q/K/V + decoder)
│   ├── evolution.py         # GA with attention forward pass
│   └── visualize.py         # Attention entropy, hidden PCA
├── docs/
│   └── research-plan.md
├── results/
├── run_phase0.py
├── run_phase1.py
└── requirements.txt
```

---

## Roadmap

- [x] Phase 0: MLP agents + GA + vector communication → communication collapse
- [x] Phase 0.1–0.4c: Environment enrichment + reward shaping + diagnostics → GA is the bottleneck
- [x] Phase 1: Attention-based coordination → attention degenerates to uniform pooling
- [ ] Phase 1.1: Gradient-based optimization for attention/communication weights
- [ ] Phase 2: Open-ended communication protocol evolution
- [ ] Instrument: Φ (IIT), global workspace broadcast, self/other distinction

---

## References

1. Lilith — arxiv:2507.04575
2. Words Evolution — arxiv:2505.05863
3. Self-Evolving Agent — arxiv:2601.11658
4. Tononi, G. (2004). IIT. *BMC Neuroscience.*
5. Baars, B.J. (1988). *A Cognitive Theory of Consciousness.*
