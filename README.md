# Artificial Life × LLM Evolution

> **Can consciousness emerge from a swarm of evolving tiny AI minds?**

This project explores emergent behavior and consciousness-related phenomena in populations of small neural network agents evolving under survival pressure.

Inspired by Avida, Tierra, Lilith (arxiv:2507.04575), and multi-agent communication research.

---

## Research Questions

| Layer | Question |
|-------|----------|
| **Phase 0** | Can small MLPs evolve cooperative behavior via vector communication? |
| **Phase 1** | Can communication protocols self-organize into semantic structures? |
| **Phase 2** | Do consciousness-related metrics (Φ, GWT broadcast, self/other distinction) emerge? |

---

## Current: Phase 0.4c — Fixed Encoding Diagnostic

### Experiment Purpose

Diagnostic experiment to determine the **upper bound of GA's ability to evolve signal decoding**. Messages are hard-coded to the true direction of nearest prey (perfect encoding), so only the receiver/decoder network weights are subject to evolution.

### Key Mechanisms

| Mechanism | Purpose |
|-----------|---------|
| **Fixed Encoding** | `msg[:2]` = normalized direction to nearest prey (bypasses encoder evolution) |
| **Receiver Shaping** | Agent rewarded for moving in the direction indicated by neighbor messages |
| **Approach Reward** | Moving closer to prey → bonus |
| **Curriculum Decay** | Gen 0-100 full shaping, 100-200 linear decay, 200+ pure natural selection |
| **Tribes** (10 × 10) | Group selection for cooperation |
| **Prey Hunt** | 5 mobile prey requiring 2+ agents for cooperative capture |

### Architecture

```
Each agent:
  Input:   observation(16) + neighbour_message_aggregate(4) = 20 dims
  Hidden:  32 → ReLU → 32 → ReLU
  Output:
    action_head:  6 dims (up/down/left/right/collect/attack), softmax
    message_head: 4 dims, tanh  (msg[:2] overwritten by environment in fixed mode)

  ~2,100 parameters per agent
```

### Experiment History

| Phase | Gens | Key Finding | Status |
|-------|------|-------------|--------|
| **0** | 200 | Communication Collapse — 16-dim messages → 0. Environment too simple. | ✅ |
| **0.1** | 300 | Heavy food + predators + MSG_DIM=4. Cooperation near zero. Signal variance → 0. | ✅ |
| **0.2** | 300 | Group selection + prey hunt + kin clustering. 1,709 prey captures but no upward trend. Collision-based, not strategic. | ✅ |
| **0.3** | 300 | Reward shaping (sender alignment + approach + curriculum). Captures ↑33% to 2,268. Behavior fixed post-shaping. **But messages still collapse** (variance 0.05). | ✅ |
| **0.4a** | 300 | Added receiver shaping (follow neighbor signals). **Worse** — creates herding (all move same direction), prey captures ↓46% post-shaping. Msg variance 0.0008. | ✅ |
| **0.4c** | 300 | **Fixed encoding diagnostic**: msg[:2] = true prey direction. **Message variance preserved (0.22)**, PCA shows arc structure. Prey captures rebound to 5.6/gen post-shaping. GA evolves weak but non-zero signal utilization (recv μ=0.58). | ✅ |

### Key Insights

1. **Communication collapse is caused by GA selection pressure** on the message channel — not by architectural limitations. When messages are fixed (0.4c), variance stays at 0.22 vs 0.0008 when evolved (0.4a).

2. **Sender + receiver co-evolution creates degenerate equilibria** (Phase 0.4a): both learn to move in the same direction ("herding"), which hurts cooperative hunting that requires flanking from different angles.

3. **GA can weakly evolve decoders** but not strongly (0.4c): even with perfect signals, receiver utilization only reaches μ=0.58 and declines post-shaping. The 2000-parameter search space is too large for GA's random walk.

4. **Reward shaping successfully fixes behavior** (0.3): prey approach strategy persists after scaffolding removal. But it cannot bootstrap communication — the credit assignment chain for bidirectional communication is fundamentally too long for GA.

### Root Cause Analysis

The core barrier is that **communication is a bilateral protocol** requiring simultaneous co-evolution of compatible encoders and decoders. GA's mutation-based search cannot find these matching pairs in a 2000-parameter space. The problem is not environment design (tried: food density, predators, group selection, prey hunting, kin clustering, reward shaping) — it is **optimizer capability**.

### Next Steps

The diagnostic results point toward three possible directions:
- **Attention-based architecture**: Replace explicit messaging with learnable attention over neighbor states
- **Gradient-based optimization**: REINFORCE + differentiable communication channel
- **Hybrid**: GA for behavior policy + gradient descent for communication channel

---

## Quick Start

```bash
pip install -r requirements.txt

# Run current phase (0.4c fixed encoding diagnostic)
python run_phase0.py

# With log saving
python run_phase0.py --generations 300 --save-log

# Reproducible
python run_phase0.py --seed 0
```

Results saved to `results/`:
- `fitness_curve.png` — composite fitness over generations
- `prey_captures.png` — cooperative hunt metrics
- `shaping_curve.png` — reward shaping + receiver component + decay schedule
- `tribe_competition.png` — inter-tribe fitness dynamics
- `message_pca.png` — PCA of 4D message vectors
- `message_variance.png` — signal diversity tracking
- `generation_log.json` — full experiment data

---

## Project Structure

```
artificial-life-llm-evolution/
├── phase0/
│   ├── environment.py   # GridWorld: food, prey, predators, tribes, fixed encoding
│   ├── agent.py         # MLP with dual heads (action + message)
│   ├── evolution.py     # Two-level GA: group + individual selection
│   └── visualize.py     # All diagnostic plots
├── docs/
│   └── research-plan.md # Full research plan
├── results/             # Generated plots and logs
├── run_phase0.py        # Entry point
└── requirements.txt
```

---

## Roadmap

- [x] Phase 0: MLP agents + GA + vector communication → communication collapse
- [x] Phase 0.1: Heavy food + predators + compact signals → still collapses
- [x] Phase 0.2: Group selection + prey hunt + kin clustering → captures happen but not strategic
- [x] Phase 0.3: Reward shaping + curriculum → behavior fixed, but messages still collapse
- [x] Phase 0.4a: Receiver shaping → creates herding, worse than 0.3
- [x] Phase 0.4c: Fixed encoding diagnostic → GA weakly evolves decoders, confirms optimizer limit
- [ ] Phase 1: Architecture/optimizer pivot (attention or gradient-based communication)
- [ ] Phase 2: Open-ended communication protocol evolution
- [ ] Instrument: Φ (IIT), global workspace broadcast, self/other distinction

---

## References

1. Lilith — arxiv:2507.04575
2. Words Evolution — arxiv:2505.05863
3. Self-Evolving Agent — arxiv:2601.11658
4. Tononi, G. (2004). IIT. *BMC Neuroscience.*
5. Baars, B.J. (1988). *A Cognitive Theory of Consciousness.*
