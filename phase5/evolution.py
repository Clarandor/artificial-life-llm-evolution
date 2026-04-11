"""
Phase 5: Evolution Engine for CoordinationWorld
================================================
Key design:
- Large prey require ≥COORDINATION_REQUIRED agents attacking simultaneously.
- Agents can broadcast coordination signals to tribe members (action[6]).
- Small prey remain solo targets for baseline food economy.
- Fitness: rewards coordination (large_prey captured × 5 + small_prey × 2 + food).

Coordination signals enable emergent cooperation:
  - "Attack" signal → tribe members converge on a target
  - "Follow" signal → agents coordinate movement
  - Agents learn to recognize signals and respond appropriately.
"""

import numpy as np
import time
import json
import os
from typing import List, Dict

from .agent import HIDDEN_DIM, ATT_DIM, GRID_SIZE, Agent
from .environment import (
    World,
    COORDINATION_REQUIRED,
    LARGE_PREY_ENERGY,
    SMALL_PREY_ENERGY,
    NUM_SIGNALS,
    SIGNAL_RADIUS,
    SIGNAL_BROADCAST_COST,
)


# ── Attention supervision target ─────────────────────────────────────────

def compute_attention_target(
    agent_pos: tuple,
    neighbor_positions: List[tuple],
    large_prey_positions: List[tuple],
    small_prey_positions: List[tuple],
    neighbor_tribes: List[int],
    agent_tribe: int,
    neighbor_fitnesses: List[float],
    grid_size: int,
) -> np.ndarray:
    """
    Supervision target for attention weights.
    Reward: same-tribe neighbors + neighbors looking at prey + high-fitness neighbors.
    Large prey positions get extra weight (coordination incentive).
    """
    if not neighbor_positions:
        return np.array([], dtype=np.float32)

    def dist(p):
        dx = abs(p[0] - agent_pos[0]); dy = abs(p[1] - agent_pos[1])
        if dx > grid_size / 2: dx = grid_size - dx
        if dy > grid_size / 2: dy = grid_size - dy
        return max(dx + dy, 1.0)

    # Direction to nearest large prey (the coordination target)
    large_prey_dir = np.zeros(2)
    if large_prey_positions:
        nearest_lp = min(large_prey_positions, key=dist)
        ddx = nearest_lp[0] - agent_pos[0]; ddy = nearest_lp[1] - agent_pos[1]
        if abs(ddx) > grid_size/2: ddx -= np.sign(ddx) * grid_size
        if abs(ddy) > grid_size/2: ddy -= np.sign(ddy) * grid_size
        n = max(abs(ddx) + abs(ddy), 1.0)
        large_prey_dir = np.array([ddx/n, ddy/n])

    scores = []
    for i, (nx, ny) in enumerate(neighbor_positions):
        nd = dist((nx, ny))
        ndir = np.array([nx - agent_pos[0], ny - agent_pos[1]])
        if abs(ndir[0]) > grid_size/2: ndir[0] -= np.sign(ndir[0]) * grid_size
        if abs(ndir[1]) > grid_size/2: ndir[1] -= np.sign(ndir[1]) * grid_size
        ndir_norm = max(np.linalg.norm(ndir), 1e-8)
        ndir = ndir / ndir_norm

        # Large prey alignment bonus (key to coordination!)
        lp_alignment = max(0.0, np.dot(ndir, large_prey_dir)) if large_prey_positions else 0.0

        # Tribe bonus (same tribe = easier to coordinate)
        tribe_bonus = 1.0 if neighbor_tribes[i] == agent_tribe else 0.0

        # Fitness bonus
        total_fit = sum(neighbor_fitnesses) + 1e-8
        fit_bonus = neighbor_fitnesses[i] / total_fit if neighbor_fitnesses else 0.0

        # Proximity bonus (closer = more useful for coordination)
        prox_bonus = 1.0 / nd

        score = (
            lp_alignment * 2.0
            + tribe_bonus * 1.0
            + fit_bonus * 0.5
            + prox_bonus * 0.3
        )
        scores.append(score)

    scores = np.array(scores, dtype=np.float32)
    scores = scores - scores.max()
    weights = np.exp(scores)
    return (weights / weights.sum()).astype(np.float32)


# ── Evolution Engine ──────────────────────────────────────────────────────

class Evolution:
    """
    Phase 5: Coordination-aware evolutionary system.

    Population: N_TRIBES × TRIBE_SIZE = 10 × 10 = 100 agents.
    Agents share LSTM/Attention templates within tribe (Phase 4 design).

    Key differences from Phase 4:
    - Large prey require ≥3 simultaneous attacks
    - Agents can broadcast/receive coordination signals
    - Attention supervision favors large-prey-aware neighbors
    - Fitness rewards coordination (large prey worth 5pts vs small prey 2pts)
    """

    ATTN_LOSS_WEIGHT = 0.02          # Slightly higher weight to encourage signal learning
    MUTATION_SIGMA = 0.05
    TOURNAMENT_K = 3
    STEPS_PER_GEN = 200
    N_TRIBES = 10
    TRIBE_SIZE = 10

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.world = World(seed=seed)

        # Create population with tribe-shared templates
        self.agents: List[Agent] = []
        self.tribe_templates: dict = {}

        from .agent import LSTMCell, AttentionQK
        for t in range(self.N_TRIBES):
            rng_t = np.random.RandomState(seed + t)
            self.tribe_templates[t] = {
                'lstm': LSTMCell(20, HIDDEN_DIM, rng_t),   # OBS_DIM=20 for phase5
                'attn': AttentionQK(HIDDEN_DIM, ATT_DIM, rng_t),
            }

        for t in range(self.N_TRIBES):
            for _ in range(self.TRIBE_SIZE):
                agent = Agent(t, self.rng, tribe_templates=self.tribe_templates)
                self.agents.append(agent)
                self.world.add_agent(agent)

        counts = Agent.count_params()
        total = sum(counts.values())
        self.N_PARAMS = total

        Agent._next_id = len(self.agents)
        self.generation_log: List[dict] = []

        print(f"  Architecture: LSTM({HIDDEN_DIM}) + Attention({ATT_DIM})")
        print(f"  Params: lstm={counts['lstm']} attn={counts['attn']} "
              f"dec={counts['decoder']} act={counts['action']} total={total}")
        print(f"  ATTN_LOSS_WEIGHT: {self.ATTN_LOSS_WEIGHT}")
        print(f"  Coordination required: {COORDINATION_REQUIRED} agents for large prey "
              f"(reward: {LARGE_PREY_ENERGY} energy)")

    # ── Breeding ────────────────────────────────────────────────────────

    def _breed(self, agents: List[Agent], fitnesses: np.ndarray, gen: int, total_gens: int) -> List[Agent]:
        """Intra-tribe tournament selection + Gaussian mutation."""
        fitnesses = np.array(fitnesses, dtype=np.float32)
        fitnesses = fitnesses - fitnesses.min() + 1e-8

        by_tribe = {}
        for a, f in zip(agents, fitnesses):
            by_tribe.setdefault(a.tribe_id, []).append((a, f))

        new_agents: List[Agent] = []

        for tribe_id, members in by_tribe.items():
            tribe_a = [m[0] for m in members]
            tribe_f = np.array([m[1] for m in members], dtype=np.float32)

            for _ in range(len(tribe_a)):
                k = min(self.TOURNAMENT_K, len(tribe_a))
                idxs = self.rng.choice(len(tribe_a), k, replace=False)
                best = max(idxs, key=lambda i: tribe_f[i])
                parent = tribe_a[best]

                child = Agent(tribe_id, self.rng, tribe_templates=self.tribe_templates)
                child.clone_weights_from(parent)

                # Adaptive sigma (cools down over generations)
                sigma = self.MUTATION_SIGMA * (1 - 0.8 * gen / max(total_gens, 1))

                for attr, frac in [
                    (child.lstm.W, 0.1), (child.lstm.b, 0.1),
                    (child.attn.W_q, 0.1), (child.attn.W_k, 0.1),
                    (child.dec_W, 0.1), (child.dec_b, 0.1),
                    (child.act_W, 0.1), (child.act_b, 0.1),
                ]:
                    mask = self.rng.uniform(0, 1, attr.size) < frac
                    noise = self.rng.normal(0, sigma, mask.sum())
                    flat = attr.flatten()
                    flat[mask] += noise
                    attr[:] = flat.reshape(attr.shape)

                new_agents.append(child)

        return new_agents

    # ── Main run loop ────────────────────────────────────────────────────

    def run(self, generations: int = 300, save_log: bool = True) -> List[dict]:
        t0 = time.time()
        print(f"\nStarting Phase 5: CoordinationWorld")
        print(f"  {generations} gen × {self.STEPS_PER_GEN} steps")
        print(f"  Population: {self.N_TRIBES}×{self.TRIBE_SIZE}={len(self.agents)}")

        self.world.reset()

        for gen in range(generations):
            # ── Simulation ──────────────────────────────────────────────
            attn_losses = []
            attn_weights_list = []

            # Map alive agent id → observation index
            agent_obs_map: Dict[int, int] = {}
            agent_actions: Dict[int, dict] = {}  # collected before resolution

            for step in range(self.STEPS_PER_GEN):
                agents = self.world.agents
                if not any(a.alive for a in agents):
                    break

                # Build observations
                obs_list = self.world._build_observations()
                if not obs_list:
                    break

                alive_agents = [a for a in agents if a.alive]
                # Rebuild map: agent id → obs index
                agent_obs_map = {a.id: i for i, a in enumerate(alive_agents)
                                 if i < len(obs_list)}

                # Encode observations through LSTM
                for i, agent in enumerate(alive_agents[:len(obs_list)]):
                    agent.encode(obs_list[i])

                # Each agent decides + collects action
                for agent in alive_agents:
                    nh_h, nh_ids = self.world._find_neighbors(agent)
                    obs_idx = agent_obs_map.get(agent.id, -1)

                    # ── Attention supervision ────────────────────────────
                    if nh_h and obs_idx >= 0 and obs_idx < len(obs_list):
                        lp_pos = [(p.x, p.y) for p in self.world.large_prey if p.is_alive]
                        sp_pos = [(p.x, p.y) for p in self.world.small_prey if p.is_alive]

                        agent_fits = {
                            a.id: a.food_collected
                            + a.small_prey_captured * 2.0
                            + a.large_prey_captured * 5.0
                            for a in alive_agents
                        }

                        n_pos, n_tribe, n_fit = [], [], []
                        for nid in nh_ids:
                            for a in agents:
                                if a.id == nid:
                                    n_pos.append((a.x, a.y))
                                    n_tribe.append(a.tribe_id)
                                    n_fit.append(agent_fits.get(a.id, 0))
                                    break

                        if len(n_pos) == len(nh_ids) and len(n_pos) > 0:
                            target = compute_attention_target(
                                (agent.x, agent.y), n_pos, lp_pos, sp_pos, n_tribe,
                                agent.tribe_id, n_fit, GRID_SIZE
                            )
                            _, aw = agent.attn.attend(agent.h, nh_h)
                            if len(target) > 0 and len(aw) > 0:
                                min_len = min(len(target), len(aw))
                                kl = np.sum(
                                    target[:min_len] * np.log(
                                        target[:min_len] / (aw[:min_len] + 1e-10) + 1e-10
                                    )
                                )
                                attn_losses.append(kl)
                                attn_weights_list.append(aw[:min_len])

                    # ── Decision ───────────────────────────────────────
                    action = agent.decide(nh_h)
                    agent_actions[agent.id] = {
                        'dx': float(action[0]),
                        'dy': float(action[1]),
                        'speed': float(action[2]),
                        'eat_food': float(action[3]),
                        'attack_small': float(action[4]),
                        'attack_large': float(action[5]),
                        'broadcast_signal': int(round(action[6])),  # 0..4
                        'signal_target_x': float(agent.x),   # default: current position
                        'signal_target_y': float(agent.y),
                    }

                # ── Resolve all actions together ────────────────────────
                self.world.resolve_actions(agent_actions)
                agent_actions.clear()

                # ── World step: spawn food/prey, clear signals ───────────
                self.world.step()

                # Age all alive agents
                for a in self.world.agents:
                    if a.alive:
                        a.age += 1

            # ── Evaluation ──────────────────────────────────────────────
            agents = self.world.agents
            alive = [a for a in agents if a.alive]

            raw_food = np.array([a.food_collected for a in alive], dtype=np.float32)
            small_caps = np.array([a.small_prey_captured for a in alive], dtype=np.float32)
            large_caps = np.array([a.large_prey_captured for a in alive], dtype=np.float32)
            attacks = np.array([a.attacks_made for a in alive], dtype=np.float32)
            sig_sent = np.array([a.signals_sent for a in alive], dtype=np.float32)
            sig_recv = np.array([a.signals_received for a in alive], dtype=np.float32)

            # Fitness: large prey coordination is the primary objective
            raw_fit = raw_food + small_caps * 2.0 + large_caps * 5.0

            total_large = int(large_caps.sum())
            total_small = int(small_caps.sum())
            total_food = float(raw_food.sum())

            attn_ents = []
            for aw in attn_weights_list:
                if len(aw) > 1:
                    attn_ents.append(-np.sum(aw * np.log(aw + 1e-10)))
            mean_attn_entropy = float(np.mean(attn_ents)) if attn_ents else 0.0
            mean_attn_loss = float(np.mean(attn_losses)) if attn_losses else 0.0

            fitnesses = raw_fit + self.ATTN_LOSS_WEIGHT * mean_attn_loss
            if len(fitnesses) > 0:
                fitnesses = fitnesses - fitnesses.min() + 1e-8

            # Coordination quality: ratio of successful attacks to total attacks
            coord_quality = 0.0
            if attacks.sum() > 0:
                coord_quality = large_caps.sum() / (attacks.sum() + 1e-10)

            sample_h = np.stack([a.h for a in alive[:20]]).tolist() if (alive and gen % 10 == 0) else None

            log = {
                "generation": gen,
                "population": len(alive),
                "mean_fitness": float(fitnesses.mean()) if len(fitnesses) else 0.0,
                "max_fitness": float(fitnesses.max()) if len(fitnesses) else 0.0,
                "mean_raw_fitness": float(raw_fit.mean()) if len(raw_fit) else 0.0,
                "total_food_collected": total_food,
                "mean_food_per_agent": float(raw_food.mean()) if len(raw_food) else 0.0,
                "total_small_prey": total_small,
                "total_large_prey": total_large,
                "mean_large_prey_per_agent": float(large_caps.mean()) if len(large_caps) else 0.0,
                "coordination_quality": coord_quality,
                "total_attacks": int(attacks.sum()),
                "mean_signals_sent": float(sig_sent.mean()) if len(sig_sent) else 0.0,
                "mean_signals_recv": float(sig_recv.mean()) if len(sig_recv) else 0.0,
                "mean_attn_entropy": mean_attn_entropy,
                "mean_attn_loss": mean_attn_loss,
                "elapsed_sec": round(time.time() - t0, 2),
                "sample_hiddens": sample_h if sample_h is not None else None,
            }
            self.generation_log.append(log)

            if gen % 10 == 0:
                elapsed = round(time.time() - t0, 2)
                sigma = self.MUTATION_SIGMA * (1 - 0.8 * gen / max(generations, 1))
                print(
                    f"Gen {gen:4d} | pop={len(alive):3d} | "
                    f"fit={fitnesses.mean():.2f} raw={raw_fit.mean():.2f} | "
                    f"large={total_large:2d} small={total_small:2d} | "
                    f"H={mean_attn_entropy:.3f} CQ={coord_quality:.3f} | "
                    f"sig_sent={sig_sent.mean():.1f} | ⏱ {elapsed:.0f}s",
                    flush=True
                )

            # ── Breed next generation ────────────────────────────────────
            self.agents = self._breed(alive, fitnesses, gen, generations)
            self.world.agents = self.agents
            self.world._spawn_food(5)  # replenish food
            for a in self.agents:
                a.alive = True
                a.energy = 200.0
                a.food_collected = 0.0
                a.small_prey_captured = 0
                a.large_prey_captured = 0
                a.attacks_made = 0
                a.signals_sent = 0
                a.signals_received = 0
                a.age = 0
                a.reset_hidden()

        if save_log:
            os.makedirs("results/phase5", exist_ok=True)
            with open("results/phase5/generation_log.json", "w") as f:
                json.dump(
                    self.generation_log, f,
                    default=lambda x: float(x) if isinstance(x, np.floating) else x
                )

        print(f"\nPhase 5 complete in {time.time()-t0:.0f}s")
        return self.generation_log
