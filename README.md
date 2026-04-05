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

## Current: Phase 0.2 — Group Selection + Prey Hunt

### Key Mechanisms

| Mechanism | Purpose |
|-----------|---------|
| **Tribes** (10 tribes × 10 agents) | Group selection — tribes with better cooperation get more breeding slots |
| **Prey Hunt** | 5 mobile prey requiring 2+ agents to capture cooperatively (80 energy reward) |
| **Predators** | 3 fast-moving lethal NPCs creating survival pressure for warning signals |
| **Kin Clustering** | Children spawn near tribe center, ensuring neighbors are relatives |
| **Compact Signals** | 4-dim message vectors (reduced from 16) for tractable GA search |

### Architecture (Phase 0.2)

```
Each agent:
  Input:   observation(16) + neighbour_message_aggregate(4) = 20 dims
  Hidden:  32 → ReLU → 32 → ReLU
  Output:
    action_head:  6 dims (up/down/left/right/collect/attack), softmax
    message_head: 4 dims, tanh

  ~2,100 parameters per agent
```

### Evolution

- **Two-level selection**: inter-tribe (proportional slot allocation by tribe avg fitness) + intra-tribe (tournament k=3)
- **Adaptive mutation**: σ decays from 0.05 → 0.01 over generations
- **Elitism**: top 1 per tribe preserved
- **No gradients** — pure evolutionary pressure

### Experiment History

| Phase | Generations | Key Finding |
|-------|-------------|-------------|
| **0** | 200 | Communication Collapse — 16-dim messages collapse to zero. No fitness improvement. Environment too simple for cooperation. |
| **0.1** | 300 | Added heavy food + predators + MSG_DIM=4. Cooperation still near zero (probability collision only). Signal variance → 0. |
| **0.2** | 300 | Group selection + prey hunt + kin clustering. **1,709 total cooperative prey captures** but no upward trend — captures are collision-based, not strategic. Message variance still collapses. |

### Root Cause Analysis

The **credit assignment problem** is the fundamental barrier: the causal chain from "sending a useful signal" to "receiving fitness reward" spans 4-5 indirect steps, which GA cannot trace through noise.

### Next: Phase 0.3 (Planned)

Reward shaping approach — directly reward signal-behavior alignment in fitness function, then decay the artificial reward over generations (curriculum learning). This shortens the credit assignment chain without prescribing specific signal encodings.

---

## Quick Start

```bash
pip install -r requirements.txt

# Run Phase 0.2 (default: 300 generations)
python run_phase0.py

# With log saving
python run_phase0.py --generations 300 --save-log

# Reproducible
python run_phase0.py --seed 0
```

Results saved to `results/`:
- `fitness_curve.png` — composite fitness over generations
- `prey_captures.png` — cooperative hunt metrics
- `tribe_competition.png` — inter-tribe fitness dynamics
- `message_pca.png` — PCA of 4D message vectors
- `message_variance.png` — signal diversity tracking
- `generation_log.json` — full experiment data

---

## Project Structure

```
artificial-life-llm-evolution/
├── phase0/
│   ├── environment.py   # GridWorld: food, prey, predators, tribes
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
- [ ] Phase 0.3: Reward shaping (signal-behavior alignment curriculum)
- [ ] Phase 1: GPT-2 small + chain-of-thought messaging
- [ ] Phase 2: Open-ended communication protocol evolution
- [ ] Instrument: Φ (IIT), global workspace broadcast, self/other distinction

---

## References

1. Lilith — arxiv:2507.04575
2. Words Evolution — arxiv:2505.05863
3. Self-Evolving Agent — arxiv:2601.11658
4. Tononi, G. (2004). IIT. *BMC Neuroscience.*
5. Baars, B.J. (1988). *A Cognitive Theory of Consciousness.*
