# Research Report: Artificial Life × LLM Evolution

## Overview

This project explored whether consciousness-related phenomena can emerge from populations of small neural network agents evolving under survival pressure.

**Repository**: https://github.com/Clarandor/artificial-life-llm-evolution

---

## Research Questions

| Level | Question | Status |
|-------|----------|--------|
| **Q1 (Engineering)** | Can small neural networks evolve cooperative behavior? | ✅ Solved - Yes, with reward shaping |
| **Q2 (Scientific)** | Can communication protocols spontaneously emerge? | ⚠️ Partial - Requires supervision |
| **Q3 (Philosophical)** | Can consciousness-related metrics emerge? | 🔄 In Progress - GWT shows emergence |

---

## Key Findings

### 1. REINFORCE Fails for Attention Learning

After 9 experiments across different hyperparameters, REINFORCE consistently failed to optimize attention weights:

| Experiment | Configuration | Gradient Vanishing Gen | Final Entropy |
|------------|---------------|------------------------|---------------|
| Baseline | entropy_bonus=0.005 | ~20 | 1.90 |
| B | entropy_bonus=0 | ~20 | 1.90 |
| C | PREY_BONUS=10x | ~20 | 1.90 |
| F | entropy=0 + LR=2x | ~30 | 1.89 |
| G | Attention shaping | ~30 | 1.91 |

**Root Cause**: Architecture separation problem - W_q/W_k (REINFORCE) and W_enc/W_dec (GA) evolved independently, forming "island effect".

---

### 2. Supervised Learning Successfully Trains Attention

Supervised attention learning worked:

| ATTN_LOSS_WEIGHT | Entropy Change | Fitness | GWT Emergence |
|------------------|----------------|---------|----------------|
| 0.5 | -73% | Crashed | ✅ YES |
| 0.1 | -73% | Crashed | ✅ YES |
| **0.01** | **-56%** | **Stable** | **✅ YES** |

**Key Finding**: ATTN_LOSS_WEIGHT=0.01 finds the balance point - attention focuses without crashing fitness.

---

### 3. GWT Entropy Shows Emergence Pattern

Global Workspace Theory entropy analysis:

| Metric | Early (0-99) | Late (200-299) | Trend | Emerged |
|--------|--------------|----------------|-------|----------|
| GWT Entropy | 1.50 | 0.46 | -1.04 | ✅ YES |
| Φ (IIT) | 9.29 | 9.21 | -0.08 | ❌ NO |

**Interpretation**: Attention becomes more focused (lower entropy = more selective), indicating global workspace-like properties emerging. However, integrated information (Φ) remains stable.

---

## Methodology

### Architecture

```
Agent (~2,182 parameters):
  1. Encoder: obs(16) → W_enc(32×16) + b → h_self(32), ReLU
  2. Attention: Q/K(4D), NO V projection, ≤8 neighbors
     Q = W_q(4×32) · h_self
     K = W_k(4×32) · h_neighbor
     context = softmax(Q·K^T / √4) · h_neighbor[:, :4]
  3. Decoder: concat(h_self, context)=36 → W_dec(32×36) → ReLU → action(6)
```

### Environment

- 32×32 grid world
- 10 tribes × 10 agents = 100 population
- 3 predators, 5 prey
- Food spawns at 2% rate
- 200 steps per generation

### Optimization

| Channel | Weights | Method |
|---------|---------|--------|
| Behavior | W_enc, W_dec, W_act (1,926 params) | GA + Tournament |
| Attention | W_q, W_k (256 params) | GA + Supervision |

---

## Limitations

1. **Observation space already contains prey direction** - Attention mechanism may be redundant
2. **Task may not require coordination** - Individual survival is achievable without cooperation
3. **Φ calculation simplified** - Full PyPhi computation needed for accurate IIT metrics

---

## Future Directions

### Phase 3 Continuation
- Implement full PyPhi Φ computation
- Analyze self/other distinction in hidden representations
- Test more complex environments requiring coordination

### Architecture Improvements
- Add memory mechanism (RNN/LSTM)
- Increase context dimension to match hidden dimension
- Add gating mechanism for neighbor information

### Alternative Approaches
- End-to-end PPO/A2C training
- Differentiable communication channels
- Population-level selection pressure for coordination

---

## Conclusion

This research demonstrates that:

1. **Gradient-based methods can train attention** when supervised signals are available
2. **Balance between supervision and survival** is critical - too much supervision crashes fitness
3. **GWT-like properties can emerge** during supervised attention training
4. **Spontaneous emergence of coordination** remains challenging without explicit supervision

The path to consciousness-related phenomena in artificial life systems appears to require:
- Careful balance of supervision and evolutionary pressure
- Tasks that genuinely require coordination
- Rich enough representations to support self-modeling

---

*Report generated: 2026-04-10*
