# Artificial Life × LLM Evolution

> **Can consciousness emerge from a swarm of evolving tiny AI minds?**

This project explores emergent behavior and consciousness-related phenomena in populations of small neural network agents evolving under survival pressure.

Inspired by Avida, Tierra, Lilith (arxiv:2507.04575), and multi-agent communication research.

---

## Research Questions

| Layer | Question |
|-------|----------|
| **Phase 0 (now)** | Can small MLPs evolve cooperative behavior via vector communication? |
| **Phase 1** | Can communication protocols self-organize into semantic structures? |
| **Phase 2** | Do consciousness-related metrics (Φ, GWT broadcast, self/other distinction) emerge? |

---

## Phase 0: Vector Communication + Genetic Algorithm

### Architecture

```
Each agent:
  Input:   observation(10) + neighbour_message_aggregate(16) = 26 dims
  Hidden:  64 → ReLU → 64 → ReLU
  Output:
    action_head:  5 dims (up/down/left/right/collect), softmax
    message_head: 16 dims, tanh

  ~6,400 parameters per agent
```

### Environment

- **Grid**: 32×32 discrete toroidal world
- **Food**: spawns randomly, replenishes at 2% per empty cell per step
- **Energy**: agents lose 1 energy/step, gain 20 on food collection
- **Reproduction**: threshold at 80 energy → child inherits weights + Gaussian noise (σ=0.01)

### Evolution

- **Selection**: Tournament selection (k=3)
- **Mutation**: Gaussian noise on all weights
- **Elitism**: top 10% kept unchanged each generation
- **No gradients** — pure evolutionary pressure

### Observation Metrics

1. Fitness curve (mean/max/min food collected per generation)
2. PCA of message vectors (tracking semantic drift)
3. Message vector variance (semantic differentiation signal)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Phase 0 (default: 200 generations, pop=100)
python run_phase0.py

# Longer run with plots and log
python run_phase0.py --generations 500 --population 200 --save-log

# Reproducible
python run_phase0.py --seed 0
```

Results are saved to `results/`:
- `fitness_curve.png`
- `message_pca.png`
- `message_variance.png`
- `generation_log.json` (if `--save-log`)

---

## Project Structure

```
artificial-life-llm-evolution/
├── phase0/
│   ├── environment.py   # GridWorld: food, energy, reproduction
│   ├── agent.py         # MLP with dual output heads (action + message)
│   ├── evolution.py     # Genetic algorithm engine
│   └── visualize.py     # Fitness curve, PCA, variance plots
├── phase1/              # (coming) GPT-2 small + CoT communication
├── phase2/              # (coming) Open-ended protocol evolution
├── docs/
│   └── research-plan.md # Full research plan
├── results/             # Generated plots and logs
├── run_phase0.py        # Entry point
└── requirements.txt
```

---

## Research Plan

See [docs/research-plan.md](docs/research-plan.md) for the full plan including the path from Phase 0 → consciousness-related metrics.

---

## Roadmap

- [x] Phase 0: MLP agents + GA + vector communication
- [ ] Phase 1: GPT-2 small + chain-of-thought messaging (4-group comparison)
- [ ] Phase 2: Communication protocol in genotype, open-ended evolution
- [ ] Instrument: Φ (IIT), global workspace broadcast, self/other distinction

---

## References

1. Lilith — arxiv:2507.04575
2. Words Evolution — arxiv:2505.05863
3. Self-Evolving Agent — arxiv:2601.11658
4. Tononi, G. (2004). IIT. *BMC Neuroscience.*
5. Baars, B.J. (1988). *A Cognitive Theory of Consciousness.*
