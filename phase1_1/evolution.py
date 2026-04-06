"""
Phase 1.1: Evolution with Factored Attention
===============================================
Key changes from Phase 1:
  - PopulationBatch has no W_v
  - attend_single uses raw neighbor hiddens[:, :ATT_DIM] as values
  - decide_batch input is HIDDEN_DIM + ATT_DIM = 36 (was 48)
  - Total params ~2,182 (was ~3,846)
"""

import numpy as np
from typing import List, Optional
import time

from .agent import (
    FactoredAttentionMLP, OBS_DIM, HIDDEN_DIM, ATT_DIM, DEC_DIM, ACTION_DIM, MAX_NEIGHBORS,
    relu, softmax,
)
from .environment import (
    GridWorld, Agent,
    GRID_SIZE, INITIAL_ENERGY,
    N_TRIBES, TRIBE_SIZE, NEIGHBOR_RADIUS,
    NUM_PREDATORS, NUM_PREY,
)


def softmax_rows(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / (e.sum(axis=1, keepdims=True) + 1e-8)


class PopulationBatch:
    """Vectorized weight storage — no W_v (factored attention)."""

    WEIGHT_ATTRS = [
        'W_enc', 'b_enc',
        'W_q', 'W_k',        # no W_v!
        'W_dec', 'b_dec',
        'W_act', 'b_act',
    ]

    WEIGHT_SHAPES = {
        'W_enc': (HIDDEN_DIM, OBS_DIM),
        'b_enc': (HIDDEN_DIM,),
        'W_q':   (ATT_DIM, HIDDEN_DIM),
        'W_k':   (ATT_DIM, HIDDEN_DIM),
        # no W_v
        'W_dec': (DEC_DIM, HIDDEN_DIM + ATT_DIM),   # 32 + 4 = 36
        'b_dec': (DEC_DIM,),
        'W_act': (ACTION_DIM, DEC_DIM),
        'b_act': (ACTION_DIM,),
    }

    def __init__(self, pop_size: int, rng: np.random.Generator):
        self.pop_size = pop_size
        for attr, shape in self.WEIGHT_SHAPES.items():
            if len(shape) == 1:
                setattr(self, attr, np.zeros((pop_size, *shape), dtype=np.float32))
            else:
                fan_in = shape[-1]
                setattr(self, attr,
                    rng.normal(0, np.sqrt(2/fan_in), (pop_size, *shape)).astype(np.float32))

    def encode_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        N = obs_batch.shape[0]
        return relu(np.einsum('nhi,ni->nh', self.W_enc[:N], obs_batch) + self.b_enc[:N])

    def attend_single(self, idx: int, h_self: np.ndarray,
                      neighbor_hiddens: np.ndarray) -> tuple:
        """Factored attention — no V projection."""
        K = neighbor_hiddens.shape[0]
        if K == 0:
            return np.zeros(ATT_DIM, dtype=np.float32), np.array([], dtype=np.float32)

        q = self.W_q[idx] @ h_self                           # (ATT_DIM,)
        keys = (self.W_k[idx] @ neighbor_hiddens.T).T        # (K, ATT_DIM)
        vals = neighbor_hiddens[:, :ATT_DIM]                  # (K, ATT_DIM) — raw hiddens

        scores = keys @ q / np.sqrt(ATT_DIM)                 # (K,)
        attn = softmax(scores)                                # (K,)
        context = attn @ vals                                 # (ATT_DIM,)
        return context, attn

    def decide_batch(self, h_self_batch: np.ndarray,
                     context_batch: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
        N = h_self_batch.shape[0]
        combined = np.concatenate([h_self_batch, context_batch], axis=1)  # (N, 36)
        h_dec = relu(np.einsum('ndi,ni->nd', self.W_dec[:N], combined) + self.b_dec[:N])
        logits = np.einsum('nai,ni->na', self.W_act[:N], h_dec) + self.b_act[:N]
        probs = softmax_rows(logits)
        cum = probs.cumsum(axis=1)
        u = rng.random((N, 1), dtype=np.float32)
        actions = (u > cum).sum(axis=1).clip(0, ACTION_DIM - 1)
        return actions.astype(np.int32)


class EvolutionEngine:
    """Two-level selection with factored attention agents."""

    STEPS_PER_GEN    = 200
    TOURNAMENT_K     = 3
    MUTATION_SIGMA_0 = 0.05
    MUTATION_SIGMA_F = 0.01
    ELITE_PER_TRIBE  = 1
    PREY_BONUS       = 3.0

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        seed: Optional[int] = 42,
        verbose: bool = True,
    ):
        self.population_size = N_TRIBES * TRIBE_SIZE
        self.grid_size = grid_size
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.generation_log: List[dict] = []

        self.world = GridWorld(grid_size=grid_size, seed=seed)
        self.world.seed_food(density=0.1)

        self.batch = PopulationBatch(self.population_size, self.rng)

        # Clustered tribe initialization
        tribe_centers = []
        for t in range(N_TRIBES):
            cx = int(self.rng.integers(4, grid_size - 4))
            cy = int(self.rng.integers(4, grid_size - 4))
            tribe_centers.append((cx, cy))

        for i in range(self.population_size):
            tribe_id = i // TRIBE_SIZE
            cx, cy = tribe_centers[tribe_id]
            agent = Agent(
                id=i,
                x=(cx + int(self.rng.integers(-3, 4))) % grid_size,
                y=(cy + int(self.rng.integers(-3, 4))) % grid_size,
                energy=INITIAL_ENERGY,
                weights=[],
                tribe_id=tribe_id,
            )
            self.world.add_agent(agent)
        self.world._next_agent_id = self.population_size

    def _get_sigma(self, gen: int, total_gens: int) -> float:
        frac = gen / max(total_gens - 1, 1)
        return self.MUTATION_SIGMA_0 * (1 - frac) + self.MUTATION_SIGMA_F * frac

    def run(self, generations: int = 300, callback=None):
        pop = self.population_size
        params = FactoredAttentionMLP().param_count
        print(f"Starting evolution: {generations} gen × {self.STEPS_PER_GEN} steps")
        print(f"  Population: {pop} ({N_TRIBES} tribes × {TRIBE_SIZE})")
        print(f"  Grid: {self.grid_size}² | Predators: {NUM_PREDATORS} | Prey: {NUM_PREY}")
        print(f"  Factored Attention: HIDDEN={HIDDEN_DIM} ATT={ATT_DIM} (no V proj) | params≈{params}")
        print(f"  Selection: 2-level (inter-tribe + intra-tribe tournament)")
        print(f"  Prey bonus: {self.PREY_BONUS}× | NO reward shaping")
        print(f"  Neighbor radius: {NEIGHBOR_RADIUS} (Manhattan)")
        print(f"  Mutation σ: {self.MUTATION_SIGMA_0} → {self.MUTATION_SIGMA_F}\n")

        for gen in range(generations):
            t0 = time.time()

            for _ in range(self.STEPS_PER_GEN):
                agents = self.world.agents
                if not agents:
                    break
                N = len(agents)
                if N > self.batch.pop_size:
                    self._grow_batch(N)

                obs_list = self.world._build_observations()
                if not obs_list:
                    break
                obs_batch = np.stack(obs_list).astype(np.float32)

                hiddens_batch = self.batch.encode_batch(obs_batch)

                context_batch = np.zeros((N, ATT_DIM), dtype=np.float32)
                attn_weights_list = []
                attn_neighbor_ids_list = []

                for i, agent in enumerate(agents):
                    neighbor_hiddens_list, neighbor_ids = self.world._find_neighbors(agent)
                    if neighbor_hiddens_list:
                        nh = np.stack(neighbor_hiddens_list)
                        ctx, aw = self.batch.attend_single(i, hiddens_batch[i], nh)
                        context_batch[i] = ctx
                        attn_weights_list.append(aw)
                        attn_neighbor_ids_list.append(neighbor_ids)
                    else:
                        attn_weights_list.append(np.array([], dtype=np.float32))
                        attn_neighbor_ids_list.append([])

                actions = self.batch.decide_batch(hiddens_batch, context_batch, self.rng)

                self.world.step(
                    list(actions),
                    list(hiddens_batch),
                    attn_weights_list,
                    attn_neighbor_ids_list,
                )

            # ── Evaluate ──
            agents = self.world.agents
            raw_food = np.array([a.food_collected for a in agents], dtype=np.float32)
            prey_caps = np.array([a.prey_captured for a in agents], dtype=np.float32)
            fitnesses = raw_food + self.PREY_BONUS * prey_caps

            tribe_fitness = {}
            for a, f in zip(agents, fitnesses):
                tribe_fitness.setdefault(a.tribe_id, []).append(f)
            tribe_avg = {t: np.mean(fs) for t, fs in tribe_fitness.items()}

            total_prey_caps = int(prey_caps.sum())

            attn_entropies = []
            for a in agents:
                if a.attn_step_count > 0:
                    attn_entropies.append(a.attn_entropy_sum / a.attn_step_count)
            mean_attn_entropy = float(np.mean(attn_entropies)) if attn_entropies else 0.0

            elapsed = round(time.time() - t0, 2)

            log_entry = {
                "generation":       gen,
                "population":       len(agents),
                "mean_fitness":     float(fitnesses.mean()) if len(fitnesses) else 0.0,
                "max_fitness":      float(fitnesses.max())  if len(fitnesses) else 0.0,
                "min_fitness":      float(fitnesses.min())  if len(fitnesses) else 0.0,
                "mean_raw_food":    float(raw_food.mean())  if len(raw_food)  else 0.0,
                "mean_prey_cap":    float(prey_caps.mean()) if len(prey_caps) else 0.0,
                "max_prey_cap":     float(prey_caps.max())  if len(prey_caps) else 0.0,
                "total_prey_caps":  total_prey_caps,
                "mean_attn_entropy": mean_attn_entropy,
                "tribe_avg":        tribe_avg,
                "elapsed_sec":      elapsed,
                "sample_hiddens":   np.stack([a.hidden for a in agents[:20]]) if agents else None,
            }
            self.generation_log.append(log_entry)

            if self.verbose and (gen % 10 == 0 or gen < 5):
                sigma = self._get_sigma(gen, generations)
                best_tribe = max(tribe_avg, key=tribe_avg.get) if tribe_avg else -1
                print(
                    f"Gen {gen:4d} | pop={log_entry['population']:4d} | "
                    f"fit μ={log_entry['mean_fitness']:.2f} max={log_entry['max_fitness']:.1f} | "
                    f"prey tot={total_prey_caps:3d} | "
                    f"attn H={mean_attn_entropy:.3f} | "
                    f"tribe★={best_tribe}({tribe_avg.get(best_tribe,0):.1f}) | "
                    f"σ={sigma:.4f} | ⏱ {elapsed}s"
                )

            if callback:
                callback(gen, log_entry)

            # ── Breed ──
            self._breed_group_selection(agents, fitnesses, gen, generations)

        print("\nEvolution complete.")
        return self.generation_log

    def _breed_group_selection(
        self, agents: List[Agent], fitnesses: np.ndarray, gen: int, total_gens: int
    ):
        sigma = self._get_sigma(gen, total_gens)

        tribe_agents = {}
        for i, (agent, fit) in enumerate(zip(agents, fitnesses)):
            tribe_agents.setdefault(agent.tribe_id, []).append((i, fit))

        tribe_ids = sorted(tribe_agents.keys())
        if not tribe_ids:
            return

        tribe_avg_fit = np.array([np.mean([f for _, f in tribe_agents[t]]) for t in tribe_ids])
        shifted = tribe_avg_fit - tribe_avg_fit.min() + 1.0
        tribe_probs = shifted / shifted.sum()

        raw_slots = tribe_probs * self.population_size
        tribe_slots = np.round(raw_slots).astype(int)
        diff = self.population_size - tribe_slots.sum()
        if diff > 0:
            for _ in range(diff):
                tribe_slots[np.argmax(tribe_probs)] += 1
        elif diff < 0:
            for _ in range(-diff):
                idx = np.argmin(tribe_slots)
                tribe_slots[idx] = max(1, tribe_slots[idx] - 1)

        old_weights = {}
        n = len(agents)
        for attr in PopulationBatch.WEIGHT_ATTRS:
            old_weights[attr] = getattr(self.batch, attr)[:n].copy()

        new_slot = 0
        new_agents = []

        tribe_centers = {}
        for t_idx, tid in enumerate(tribe_ids):
            members = tribe_agents[tid]
            xs = [agents[i].x for i, _ in members]
            ys = [agents[i].y for i, _ in members]
            tribe_centers[tid] = (int(np.mean(xs)), int(np.mean(ys)))

        for t_idx, tid in enumerate(tribe_ids):
            slots = int(tribe_slots[t_idx])
            members = tribe_agents[tid]
            member_indices = [i for i, _ in members]
            member_fits = np.array([f for _, f in members])

            if len(members) == 0:
                continue

            elite_count = min(self.ELITE_PER_TRIBE, slots, len(members))
            sorted_local = np.argsort(member_fits)[::-1]

            for e in range(elite_count):
                if new_slot >= self.batch.pop_size:
                    self._grow_batch(new_slot + 1)
                src = member_indices[sorted_local[e]]
                for attr in PopulationBatch.WEIGHT_ATTRS:
                    getattr(self.batch, attr)[new_slot] = old_weights[attr][src]
                cx, cy = tribe_centers.get(tid, (16, 16))
                new_agents.append(Agent(
                    id=self.world._next_agent_id,
                    x=(cx + int(self.rng.integers(-3, 4))) % self.grid_size,
                    y=(cy + int(self.rng.integers(-3, 4))) % self.grid_size,
                    energy=INITIAL_ENERGY,
                    weights=[],
                    tribe_id=tid,
                ))
                self.world._next_agent_id += 1
                new_slot += 1

            for _ in range(slots - elite_count):
                if new_slot >= self.batch.pop_size:
                    self._grow_batch(new_slot + 1)
                k = min(self.TOURNAMENT_K, len(members))
                comp = self.rng.choice(len(members), size=k, replace=False)
                winner_local = comp[np.argmax(member_fits[comp])]
                src = member_indices[winner_local]

                for attr in PopulationBatch.WEIGHT_ATTRS:
                    arr = getattr(self.batch, attr)
                    arr[new_slot] = old_weights[attr][src] + \
                        self.rng.normal(0, sigma, old_weights[attr][src].shape).astype(np.float32)

                cx, cy = tribe_centers.get(tid, (16, 16))
                new_agents.append(Agent(
                    id=self.world._next_agent_id,
                    x=(cx + int(self.rng.integers(-3, 4))) % self.grid_size,
                    y=(cy + int(self.rng.integers(-3, 4))) % self.grid_size,
                    energy=INITIAL_ENERGY,
                    weights=[],
                    tribe_id=tid,
                ))
                self.world._next_agent_id += 1
                new_slot += 1

        self.world.agents = new_agents
        self.world.grid[:] = 0.0
        self.world.seed_food(density=0.1)
        for prey in self.world.prey_list:
            prey.respawn(self.grid_size, self.rng)

    def _grow_batch(self, new_size: int):
        extra = new_size - self.batch.pop_size
        if extra <= 0:
            return
        for attr, shape in PopulationBatch.WEIGHT_SHAPES.items():
            old = getattr(self.batch, attr)
            pad = np.zeros((extra, *shape), dtype=np.float32)
            setattr(self.batch, attr, np.concatenate([old, pad], axis=0))
        self.batch.pop_size = new_size
