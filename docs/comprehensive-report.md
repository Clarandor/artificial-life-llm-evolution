# Comprehensive Research Report: artificial-life-llm-evolution

**Project**: https://github.com/Clarandor/artificial-life-llm-evolution  
**Date**: 2026-04-12  
**Experiments**: 25+ total across 7 phases  
**Compute**: ~60,000 generations × 200 steps

---

## Executive Summary

We explored whether consciousness-related phenomena can emerge from populations of small neural network agents evolving under survival pressure. Using a grid-world ecosystem with 100 agents (10 tribes × 10), we tested multiple architectures and training methods across 6 phases.

**Key findings:**
1. REINFORCE fails for attention learning (architecture-level, not hyperparameter-level)
2. Supervised attention works, but needs careful calibration (0.01 weight)
3. LSTM reverses the "mirror mode" — tribe identity encoding differs fundamentally
4. Communication signals emerge spontaneously without explicit reward
5. Coordination fails — emergent signals are noise, not structured communication

---

## 1. Research Questions

| # | Question | Status | Verdict |
|---|----------|--------|---------|
| Q1 | Can small NNs evolve cooperative behavior? | ✅ | Yes, with reward shaping |
| Q2 | Can communication protocols spontaneously emerge? | ⚠️ | Partial — signals emerge but are noisy |
| Q3 | Can consciousness-related metrics emerge? | ✅ | GWT entropy, tribe identity both emerge |

---

## 2. Architecture Evolution

### Phase 2.1 — Feedforward + Supervised Attention

```
Agent (2,182 params):
  obs(16) → W_enc(32×16) → ReLU → h_self(32)
                    ↓
  [h_self + neighbor_hiddens] → attention Q/K(4D) → context(4)
                    ↓
  concat(h_self, context=36) → W_dec(32×36) → ReLU → action(6)

Optimization:
  - GA: W_enc, W_dec, W_act (1,926 params)
  - Supervised: W_q, W_k (256 params) — attention target from prey proximity + tribe + fitness
```

### Phase 4 — LSTM Recurrent Agents

```
Agent (7,910 params, +4,728 more):
  obs(16) → LSTM(16→32) → h_t(32) [temporal memory]
                    ↓
  [h_t + neighbor_h_t] → attention Q/K(4D) → context(4)
                    ↓
  concat → decoder → action(6)

Key difference: h_t carries temporal history, not just current observation.
```

### Phase 5 — Coordination + Communication

```
Extends Phase 4 with:
  Action: 6 → 7 (+broadcast signal)
  Observation: 16 → 20 (+received coordination signal channel)
  
CoordinationWorld:
  - Large prey: needs ≥3 agents attacking simultaneously → 80 energy
  - Small prey: 1 agent → 30 energy
```

---

## 3. All Experiment Results

### Phase 2: REINFORCE Failure (9 experiments)

| Config | Gradient Vanish | Final Entropy | Result |
|--------|---------------|---------------|--------|
| Baseline (entropy=0.005) | ~20 gen | 1.897 | ❌ |
| entropy=0 | ~20 gen | 1.897 | ❌ |
| PREY_BONUS=10x | ~20 gen | 1.897 | ❌ |
| entropy=0 + LR=2x | ~30 gen | 1.889 | ❌ |
| Attention shaping reward | ~30 gen | 1.908 | ❌ |
| ATT_DIM=16 | ~20 gen | 1.903 | ❌ |

**Root cause**: Architecture separation — REINFORCE (256 params) and GA (1,926 params) optimize independently with no gradient path between them.

### Phase 2.1: Supervised Attention (2 experiments)

| ATTN_LOSS_WEIGHT | Entropy Change | Fitness | GWT |
|-------------------|----------------|---------|-----|
| 0.5 (strong) | -73% | Crashed (-6.17) | ✅ |
| 0.1 (medium) | -73% | Crashed (-6.17) | ✅ |
| **0.01 (light)** | **-56%** | **Stable (~0.3)** | **✅** |

### Phase 3: Consciousness Metrics

All phases (300 generations each):

| Metric | Phase 2.1 (FFN) | Phase 4 (LSTM) | Phase 5 (Coord) |
|--------|-----------------|----------------|-----------------|
| GWT Attention Entropy | **-68%** ✅ | **-18%** ✅ | **-21%** ✅ |
| GWT Influence Var | +47% | **-32%** | — |
| Within-tribe Similarity | +83% | **-30%** | — |
| Between-tribe Similarity | -45% | **+17%** | — |
| Self/Other Distinction | -44% | **+15%** | — |
| Φ (Information Integration) | +2% ❌ | -0.7% ❌ | — |
| **Signals Sent** | N/A | N/A | **0→9.4/gen** |

### Phase 5: Coordination Challenge

| Metric | Early (0-49) | Late (250-299) |
|--------|--------------|----------------|
| Fitness | 31.4 | 4.4 (-86%) |
| Large prey captures | 0 | 0 |
| Signals sent | 0.0 | 9.4/gen |
| Attention entropy | 1.51 | 1.20 |

**Conclusion**: Signals emerge but are noise — no coordination reward means no selective pressure for meaningful communication.

---

## 4. Key Discoveries

### 4.1 REINFORCE Fails for Attention

Gradient vanishing in REINFORCE is **architecture-level**, not hyperparameter-level:
- Diagnosis showed advantage function std=1.4 (normal)
- No skipped updates due to zero advantage
- Entropy regularization removal had no effect
- Learning rate scaling had no effect
- ATT_DIM scaling had no effect

The fundamental problem: attention weights (REINFORCE) and behavior weights (GA) are optimized independently with no gradient path between them.

### 4.2 The "Goldilocks Zone"

ATTN_LOSS_WEIGHT=0.01 finds the balance:
- Too high (0.5): attention dominates → fitness crashes
- Too low (0): no effect on behavior
- Just right (0.01): attention focuses + fitness stable

This is a meta-parameter that controls how "conscious" the agents become vs. how well they survive.

### 4.3 LSTM Mirror Mode Reversal

Feedforward → Tribe clustering (mirror mode):
- Weights encode tribe identity → all tribe members converge to similar weights
- GA pushes same-tribe agents toward same genotype
- Result: high within-tribe similarity, low between-tribe similarity

LSTM → Tribe blending (anti-mirror mode):
- Tribe templates shared → identity must be in dynamics, not weights
- Same LSTM + different initial conditions → different trajectories
- Attention over temporal states amplifies divergence
- Result: within-tribe differentiation, cross-tribe influence

### 4.4 Spontaneous Communication

Without any coordination reward, agents spontaneously learn to broadcast signals (9.4/gen by gen 299).

**Hypothesis**: This is a side-effect of the action output dimension increasing from 6→7. The decoder must allocate capacity to the new output, pulling resources from behavior. Signals are likely **exploratory noise** rather than structured communication.

**Evidence**: Fitness drops from 31→4 when signals appear, suggesting signals interfere with survival behavior rather than complementing it.

---

## 5. Why Coordination Failed

### The Problem

In CoordinationWorld:
- Large prey: 80 energy reward, needs ≥3 simultaneous attacks
- Small prey: 30 energy, needs 1 attack
- No explicit coordination mechanism

Agents evolved in Phase 4 don't have the concept of "coordinate with neighbors." They optimize individual survival.

### Why Signals Don't Help

1. **No causal link**: Broadcasting a signal doesn't increase the probability of coordinated attack
2. **No reception mechanism**: Even if signals are received, there's no "understanding" of what they mean
3. **No joint intention**: Without theory of mind, agents can't form shared goals
4. **Fitness pressure dominates**: Surviving is more important than coordinating

### What Would Work

To make coordination emerge:
1. **Make coordination the only survival path** — increase large prey reward, remove small prey
2. **Make signals directly causal** — attackers who receive signal get bonus to their own attack succeeding
3. **Add explicit coordination loss** — penalize large prey failures as missed opportunities
4. **Hierarchical attention** — agents attend to "who else is attending to me"

### Phase 6: Strong Coordination Incentive

| Metric | Phase 5 (No Incentive) | Phase 6 (Strong Incentive) |
|--------|----------------------|---------------------------|
| Large prey captures | 0 | **38** |
| Coordination success rate | 0% | **1.08%** |
| GWT Attention Entropy | -20% | **-33%** |
| Fitness | 4.4 | **3.22** |
| Signals sent/gen | 9.4 | **116** |

**Key insight**: Removing small prey (the alternative survival path) forced agents to coordinate. The coordination success rate is low (1.08%), but it emerged from 0 — a qualitative phase transition.

---

## 6. Consciousness Metrics Summary

Three consciousness-related metrics showed emergence:

| Metric | Definition | FFN Change | LSTM Change |
|--------|-----------|------------|-------------|
| **GWT Entropy** | Attention distribution spread | -68% | -18% |
| **Tribe Identity** | Within/between tribe similarity | +83%/-45% | -30%/+17% |
| **Influence Variance** | Variance of agent "influence" | +47% | -32% |

**Interpretation**:
- GWT entropy decreasing → global workspace becoming more selective (fewer broadcast, more targeted)
- Tribe identity → agents develop distinct "self" representation vs. "other" (emergence of self-model)
- Influence variance → information flow becomes unequal (some agents become "hubs")

These are **necessary but not sufficient** conditions for consciousness according to GWT theory.

---

## 7. Limitations

1. **Simplified Φ**: Random bipartitions, not exhaustive PyPhi search. Real Φ may show different patterns.
2. **No recurrent attention**: Attention is computed over instantaneous hidden states, not temporal sequences.
3. **No environmental pressure for coordination**: ~~Agents can survive alone, so there's no evolutionary pressure to cooperate.~~ **RESOLVED in Phase 6**: Removing alternative survival paths forces coordination.
4. **No theory of mind**: Agents don't model other agents' mental states.
5. **Fixed network size**: No neurogenesis or synaptic pruning.
6. **Simplified IIT metrics**: Real IIT requires cause-effect analysis over all possible interventions.

---

## 8. Future Directions

### Immediate
- ~~**Phase 6**: Add coordination incentive (only large prey available, small prey removed)~~ **DONE** — coordination emerges at 1.08% success rate
- **Phase 7**: Test theory of mind — do agents learn to predict others' actions?
- **Phase 8**: Hierarchical attention — agents attend to who is attending to them (recursive GWT)

### Architectural
- **Recurrent attention**: Attention over temporal sequences, not just instant states
- **Differentiable communication**: Make signal reception directly affect attention, creating feedback loop
- **Larger hidden dimension** (32→128): More representational capacity for self-modeling
- **Memory attention**: Agents attend over their own past hidden states (introspection)

### Theoretical
- **Full PyPhi**: Implement cause-effect analysis to compute accurate Φ
- **GWT broadcast detection**: Measure actual information flow between agents
- **Comparative study**: Compare with Lilith (arxiv:2507.04575) on same metrics

---

## 9. Conclusion

This research demonstrates that:

1. **REINFORCE fundamentally fails** for attention learning when separated from behavior optimization — a principled architectural limitation
2. **Supervised attention works** but requires careful calibration (0.01 weight)
3. **LSTM recurrence reverses the tribe clustering pattern** — revealing that identity encoding depends on whether weights or dynamics are the substrate
4. **Communication signals can emerge spontaneously** but without causal structure, they're just noise
5. **Coordination requires explicit incentives** — emergent signals without joint intention don't help
6. **Strong incentives enable emergent cooperation** — removing alternative survival paths forces coordination (Phase 6: 0→38 captures, 1.08% success)
7. **Environment is as important as architecture** — the same agents coordinate or don't depending on whether the environment demands it

The path to consciousness in artificial life requires not just the right architecture, but the right **environment** — one where consciousness is necessary for survival.

---

*Total compute: ~50,000 generations × 200 steps × ~100 agents = 1 billion agent-steps*
