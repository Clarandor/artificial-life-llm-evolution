"""
Phase 0.4a: Evolution with Receiver Shaping
==============================================
Key change from 0.3: shaping_score now includes receiver_score in addition
to alignment_score + approach_score. This provides selection pressure on
BOTH the sender (encode signal matching movement) and receiver (decode
neighbor signal and follow it).

  fitness = raw_food + PREY_BONUS × prey_captures + decay(gen) × shaping_score

Curriculum schedule:
  gen 0-100:   decay = 1.0 (full shaping)
  gen 100-200: decay = linear 1.0 → 0.0
  gen 200+:    decay = 0.0 (pure natural selection)
"""

import numpy as np
from typing import List, Optional
import time

from .agent import MLP, INPUT_DIM, HIDDEN_DIM, ACTION_DIM, MSG_DIM
from .environment import (
    GridWorld, Agent,
    GRID_SIZE, INITIAL_ENERGY,
    N_TRIBES, TRIBE_SIZE,
    OBS_DIM, NUM_PREDATORS, NUM_PREY,
)


def relu(x):
    return np.maximum(0.0, x)

def softmax_rows(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def tanh(x):
    return np.tanh(x)


class PopulationBatch:
    """Vectorized weight storage for entire population."""

    def __init__(self, pop_size: int, rng: np.random.Generator):
        self.pop_size = pop_size
        scale = lambda fan_in: np.sqrt(2 / fan_in)
        self.W1 = rng.normal(0, scale(INPUT_DIM),  (pop_size, HIDDEN_DIM, INPUT_DIM )).astype(np.float32)
        self.b1 = np.zeros((pop_size, HIDDEN_DIM), dtype=np.float32)
        self.W2 = rng.normal(0, scale(HIDDEN_DIM), (pop_size, HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32)
        self.b2 = np.zeros((pop_size, HIDDEN_DIM), dtype=np.float32)
        self.Wa = rng.normal(0, scale(HIDDEN_DIM), (pop_size, ACTION_DIM, HIDDEN_DIM)).astype(np.float32)
        self.ba = np.zeros((pop_size, ACTION_DIM), dtype=np.float32)
        self.Wm = rng.normal(0, scale(HIDDEN_DIM), (pop_size, MSG_DIM,    HIDDEN_DIM)).astype(np.float32)
        self.bm = np.zeros((pop_size, MSG_DIM),    dtype=np.float32)

    def forward(self, obs_batch: np.ndarray, rng: np.random.Generator):
        N = obs_batch.shape[0]
        h1 = relu(np.einsum('nhi,ni->nh', self.W1[:N], obs_batch) + self.b1[:N])
        h2 = relu(np.einsum('nhi,ni->nh', self.W2[:N], h1)         + self.b2[:N])
        logits = np.einsum('nai,ni->na', self.Wa[:N], h2) + self.ba[:N]
        probs  = softmax_rows(logits)
        cum = probs.cumsum(axis=1)
        u   = rng.random((N, 1), dtype=np.float32)
        actions = (u > cum).sum(axis=1).clip(0, ACTION_DIM - 1)
        messages = tanh(np.einsum('nmi,ni->nm', self.Wm[:N], h2) + self.bm[:N])
        return actions.astype(np.int32), messages

    WEIGHT_ATTRS = ['W1', 'b1', 'W2', 'b2', 'Wa', 'ba', 'Wm', 'bm']


class EvolutionEngine:
    """
    Two-level selection + reward shaping with curriculum decay.
    """

    STEPS_PER_GEN    = 200
    TOURNAMENT_K     = 3
    MUTATION_SIGMA_0 = 0.05
    MUTATION_SIGMA_F = 0.01
    ELITE_PER_TRIBE  = 1
    PREY_BONUS       = 3.0

    # Curriculum schedule
    SHAPING_FULL_UNTIL = 100   # full reward shaping until this gen
    SHAPING_ZERO_AT    = 200   # shaping fully removed at this gen
    SHAPING_WEIGHT     = 0.5   # max weight of shaping in fitness

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

    def _get_shaping_decay(self, gen: int) -> float:
        """Curriculum decay: 1.0 → 0.0 over [SHAPING_FULL_UNTIL, SHAPING_ZERO_AT]."""
        if gen <= self.SHAPING_FULL_UNTIL:
            return 1.0
        elif gen >= self.SHAPING_ZERO_AT:
            return 0.0
        else:
            return 1.0 - (gen - self.SHAPING_FULL_UNTIL) / (self.SHAPING_ZERO_AT - self.SHAPING_FULL_UNTIL)

    def run(self, generations: int = 300, callback=None):
        pop = self.population_size
        print(f"Starting evolution: {generations} gen × {self.STEPS_PER_GEN} steps")
        print(f"  Population: {pop} ({N_TRIBES} tribes × {TRIBE_SIZE})")
        print(f"  Grid: {self.grid_size}² | Predators: {NUM_PREDATORS} | Prey: {NUM_PREY}")
        print(f"  MSG_DIM={MSG_DIM} | HIDDEN={HIDDEN_DIM} | params≈{MLP().param_count}")
        print(f"  Selection: 2-level (inter-tribe + intra-tribe tournament)")
        print(f"  Prey bonus: {self.PREY_BONUS}× | Shaping weight: {self.SHAPING_WEIGHT}")
        print(f"  Curriculum: full shaping gen 0-{self.SHAPING_FULL_UNTIL}, "
              f"decay {self.SHAPING_FULL_UNTIL}-{self.SHAPING_ZERO_AT}, "
              f"pure selection {self.SHAPING_ZERO_AT}+")
        print(f"  Mutation σ: {self.MUTATION_SIGMA_0} → {self.MUTATION_SIGMA_F}\n")

        for gen in range(generations):
            t0 = time.time()
            shaping_decay = self._get_shaping_decay(gen)

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

                actions, messages = self.batch.forward(obs_batch, self.rng)

                for i, agent in enumerate(agents):
                    agent.last_message = messages[i]

                self.world.step(list(actions), list(messages))

            # ── Evaluate ───────────────────────────────────────────────────────
            agents = self.world.agents
            raw_food = np.array([a.food_collected for a in agents], dtype=np.float32)
            prey_caps = np.array([a.prey_captured for a in agents], dtype=np.float32)
            shaping = np.array([a.shaping_score for a in agents], dtype=np.float32)

            # Composite fitness with curriculum-decayed shaping
            fitnesses = (raw_food
                        + self.PREY_BONUS * prey_caps
                        + self.SHAPING_WEIGHT * shaping_decay * shaping)

            # Per-tribe stats
            tribe_fitness = {}
            for a, f in zip(agents, fitnesses):
                tribe_fitness.setdefault(a.tribe_id, []).append(f)
            tribe_avg = {t: np.mean(fs) for t, fs in tribe_fitness.items()}

            total_prey_caps = int(prey_caps.sum())
            mean_shaping = float(shaping.mean()) if len(shaping) else 0.0
            receiver_scores = np.array([a.receiver_score for a in agents], dtype=np.float32)
            mean_receiver = float(receiver_scores.mean()) if len(receiver_scores) else 0.0
            elapsed = round(time.time() - t0, 2)

            log_entry = {
                "generation":      gen,
                "population":      len(agents),
                "mean_fitness":    float(fitnesses.mean()) if len(fitnesses) else 0.0,
                "max_fitness":     float(fitnesses.max())  if len(fitnesses) else 0.0,
                "min_fitness":     float(fitnesses.min())  if len(fitnesses) else 0.0,
                "mean_raw_food":   float(raw_food.mean())  if len(raw_food)  else 0.0,
                "mean_prey_cap":   float(prey_caps.mean()) if len(prey_caps) else 0.0,
                "max_prey_cap":    float(prey_caps.max())  if len(prey_caps) else 0.0,
                "total_prey_caps": total_prey_caps,
                "mean_shaping":    mean_shaping,
                "mean_receiver":   mean_receiver,
                "shaping_decay":   shaping_decay,
                "tribe_avg":       tribe_avg,
                "elapsed_sec":     elapsed,
                "sample_messages": np.stack([a.last_message for a in agents[:20]]) if agents else None,
            }
            self.generation_log.append(log_entry)

            if self.verbose and (gen % 10 == 0 or gen < 5):
                sigma = self._get_sigma(gen, generations)
                best_tribe = max(tribe_avg, key=tribe_avg.get) if tribe_avg else -1
                phase = ("SHAPING" if shaping_decay > 0.99
                         else f"DECAY({shaping_decay:.2f})" if shaping_decay > 0.01
                         else "NATURAL")
                print(
                    f"Gen {gen:4d} [{phase:12s}] | pop={log_entry['population']:4d} | "
                    f"fit μ={log_entry['mean_fitness']:.2f} | "
                    f"prey tot={total_prey_caps:3d} | "
                    f"shaping μ={mean_shaping:.1f} recv={mean_receiver:.1f} | "
                    f"tribe★={best_tribe}({tribe_avg.get(best_tribe,0):.1f}) | "
                    f"σ={sigma:.4f} | ⏱ {elapsed}s"
                )

            if callback:
                callback(gen, log_entry)

            # ── Breed ──────────────────────────────────────────────────────────
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
        for attr in PopulationBatch.WEIGHT_ATTRS:
            n = len(agents)
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
        for attr, shape_fn in [
            ('W1', lambda: (extra, HIDDEN_DIM, INPUT_DIM)),
            ('b1', lambda: (extra, HIDDEN_DIM)),
            ('W2', lambda: (extra, HIDDEN_DIM, HIDDEN_DIM)),
            ('b2', lambda: (extra, HIDDEN_DIM)),
            ('Wa', lambda: (extra, ACTION_DIM, HIDDEN_DIM)),
            ('ba', lambda: (extra, ACTION_DIM)),
            ('Wm', lambda: (extra, MSG_DIM, HIDDEN_DIM)),
            ('bm', lambda: (extra, MSG_DIM)),
        ]:
            old = getattr(self.batch, attr)
            pad = np.zeros(shape_fn(), dtype=np.float32)
            setattr(self.batch, attr, np.concatenate([old, pad], axis=0))
        self.batch.pop_size = new_size
