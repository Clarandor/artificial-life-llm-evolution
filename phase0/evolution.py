"""
Phase 0: Genetic Algorithm Evolution Loop (Vectorized)
=======================================================
Selection:  Tournament selection (k=3)
Mutation:   Gaussian noise σ=0.01
No gradient — pure evolutionary pressure.
Vectorized forward pass: all agents processed in a single batch per step.
"""

import numpy as np
from typing import List, Optional
import time

from .agent import MLP, INPUT_DIM, HIDDEN_DIM, ACTION_DIM, MSG_DIM
from .environment import (
    GridWorld, Agent,
    GRID_SIZE, INITIAL_ENERGY,
)


def relu(x):
    return np.maximum(0.0, x)

def softmax_rows(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def tanh(x):
    return np.tanh(x)


class PopulationBatch:
    """
    Holds weights for the entire population as stacked arrays.
    Forward pass is fully vectorized: O(pop) instead of O(pop) × Python loops.

    Weight shapes:
        W1: (pop, HIDDEN, INPUT)   b1: (pop, HIDDEN)
        W2: (pop, HIDDEN, HIDDEN)  b2: (pop, HIDDEN)
        Wa: (pop, ACTION, HIDDEN)  ba: (pop, ACTION)
        Wm: (pop, MSG,   HIDDEN)   bm: (pop, MSG)
    """

    def __init__(self, pop_size: int, rng: np.random.Generator):
        self.pop_size = pop_size
        scale = lambda fan_in: np.sqrt(2 / fan_in)
        self.W1 = rng.normal(0, scale(INPUT_DIM),  (pop_size, HIDDEN_DIM, INPUT_DIM )).astype(np.float32)
        self.b1 = np.zeros((pop_size, HIDDEN_DIM), dtype=np.float32)
        self.W2 = rng.normal(0, scale(HIDDEN_DIM), (pop_size, HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32)
        self.b2 = np.zeros((pop_size, HIDDEN_DIM), dtype=np.float32)
        self.Wa = rng.normal(0, scale(HIDDEN_DIM), (pop_size, ACTION_DIM, HIDDEN_DIM)).astype(np.float32)
        self.ba = np.zeros((pop_size, ACTION_DIM), dtype=np.float32)
        self.Wm = rng.normal(0, scale(HIDDEN_DIM), (pop_size, MSG_DIM,   HIDDEN_DIM)).astype(np.float32)
        self.bm = np.zeros((pop_size, MSG_DIM),    dtype=np.float32)

    def forward(self, obs_batch: np.ndarray, rng: np.random.Generator):
        """
        obs_batch: (N, INPUT_DIM)  where N == self.pop_size
        Returns:
            actions:  (N,) int
            messages: (N, MSG_DIM)
        """
        N = obs_batch.shape[0]
        # h1: (N, HIDDEN)
        h1 = relu(np.einsum('nhi,ni->nh', self.W1[:N], obs_batch) + self.b1[:N])
        h2 = relu(np.einsum('nhi,ni->nh', self.W2[:N], h1)         + self.b2[:N])
        # action
        logits = np.einsum('nai,ni->na', self.Wa[:N], h2) + self.ba[:N]
        probs  = softmax_rows(logits)                    # (N, ACTION)
        # sample actions per agent
        cum = probs.cumsum(axis=1)
        u   = rng.random((N, 1), dtype=np.float32)
        actions = (u > cum).sum(axis=1).clip(0, ACTION_DIM - 1)
        # message
        messages = tanh(np.einsum('nmi,ni->nm', self.Wm[:N], h2) + self.bm[:N])
        return actions.astype(np.int32), messages

    def get_agent_weights(self, idx: int) -> List[np.ndarray]:
        return [
            self.W1[idx], self.b1[idx],
            self.W2[idx], self.b2[idx],
            self.Wa[idx], self.ba[idx],
            self.Wm[idx], self.bm[idx],
        ]

    def set_agent_weights(self, idx: int, weights: List[np.ndarray]):
        self.W1[idx], self.b1[idx] = weights[0], weights[1]
        self.W2[idx], self.b2[idx] = weights[2], weights[3]
        self.Wa[idx], self.ba[idx] = weights[4], weights[5]
        self.Wm[idx], self.bm[idx] = weights[6], weights[7]

    def mutate_from(self, parent_idx: int, child_idx: int, sigma: float, rng: np.random.Generator):
        """Copy parent weights to child slot and add Gaussian noise."""
        for arr in [self.W1, self.b1, self.W2, self.b2, self.Wa, self.ba, self.Wm, self.bm]:
            arr[child_idx] = arr[parent_idx] + rng.normal(0, sigma, arr[parent_idx].shape).astype(np.float32)


class EvolutionEngine:
    """
    Population-level evolution with fully vectorized forward passes.
    Each 'generation' = STEPS_PER_GEN environment steps, then tournament selection.
    """

    STEPS_PER_GEN  = 200
    TOURNAMENT_K   = 3
    MUTATION_SIGMA = 0.01
    ELITE_FRACTION = 0.1

    def __init__(
        self,
        population_size: int = 100,
        grid_size: int = GRID_SIZE,
        seed: Optional[int] = 42,
        verbose: bool = True,
    ):
        self.population_size = population_size
        self.grid_size = grid_size
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        self.generation_log: List[dict] = []

        # Init world
        self.world = GridWorld(grid_size=grid_size, seed=seed)
        self.world.seed_food(density=0.1)

        # Init population batch
        self.batch = PopulationBatch(population_size, self.rng)

        # Seed agents into world (fitness tracking only; weights live in batch)
        for i in range(population_size):
            agent = Agent(
                id=i,
                x=int(self.rng.integers(0, grid_size)),
                y=int(self.rng.integers(0, grid_size)),
                energy=INITIAL_ENERGY,
                weights=[],   # unused — batch holds weights
            )
            self.world.add_agent(agent)
        self.world._next_agent_id = population_size

    # ── Main Loop ──────────────────────────────────────────────────────────────

    def run(self, generations: int = 200, callback=None):
        print(f"Starting evolution: {generations} gen × {self.STEPS_PER_GEN} steps | pop={self.population_size} | grid={self.grid_size}²\n")

        for gen in range(generations):
            t0 = time.time()

            for _ in range(self.STEPS_PER_GEN):
                agents = self.world.agents
                if not agents:
                    break
                N = len(agents)

                # Build obs batch
                obs_list = self.world._build_observations()   # list of (26,) arrays
                N = len(obs_list)  # may differ from initial N after reproduction
                if N == 0:
                    break
                # Grow batch if population exceeded initial size
                if N > self.batch.pop_size:
                    self._grow_batch(N)
                obs_batch = np.stack(obs_list).astype(np.float32)   # (N, 26)

                # Vectorized forward (only use slots 0..N-1 in batch)
                actions, messages = self.batch.forward(obs_batch, self.rng)

                # Update last_message on agents for environment messaging
                for i, agent in enumerate(agents):
                    agent.last_message = messages[i]

                self.world.step(list(actions), list(messages))

            # ── Evaluate & log ─────────────────────────────────────────────────
            agents = self.world.agents
            fitnesses = np.array([a.food_collected for a in agents], dtype=np.float32)

            log_entry = {
                "generation":     gen,
                "population":     len(agents),
                "mean_fitness":   float(fitnesses.mean()) if len(fitnesses) else 0.0,
                "max_fitness":    float(fitnesses.max())  if len(fitnesses) else 0.0,
                "min_fitness":    float(fitnesses.min())  if len(fitnesses) else 0.0,
                "mean_energy":    float(np.mean([a.energy for a in agents])) if agents else 0.0,
                "elapsed_sec":    round(time.time() - t0, 2),
                "sample_messages": np.stack([a.last_message for a in agents[:20]]) if agents else None,
            }
            self.generation_log.append(log_entry)

            if self.verbose and (gen % 5 == 0 or gen < 5):
                print(
                    f"Gen {gen:4d} | pop={log_entry['population']:4d} | "
                    f"fit μ={log_entry['mean_fitness']:.2f} max={log_entry['max_fitness']:.2f} | "
                    f"⏱ {log_entry['elapsed_sec']}s"
                )

            if callback:
                callback(gen, log_entry)

            # ── Breed next generation ──────────────────────────────────────────
            self._breed(agents, fitnesses)

        print("\nEvolution complete.")
        return self.generation_log

    def _grow_batch(self, new_size: int):
        """Expand batch arrays to accommodate a larger population."""
        extra = new_size - self.batch.pop_size
        rng = self.rng
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

    # ── Selection ──────────────────────────────────────────────────────────────

    def _breed(self, agents: List[Agent], fitnesses: np.ndarray):
        """Replace population via tournament selection + mutation (in-place on batch)."""
        n = len(agents)
        if n == 0:
            return

        n_elite = max(1, int(self.ELITE_FRACTION * self.population_size))
        sorted_idx = np.argsort(fitnesses)[::-1]

        # Build index map: agent list position → batch slot
        # (agents are stored at batch slots 0..n-1 currently)
        elite_slots = sorted_idx[:n_elite]

        # New batch: elites first, then tournament children
        new_W1 = self.batch.W1.copy()  # will overwrite non-elites

        # Temporary copy of current weights for selection
        old_W1 = self.batch.W1[:n].copy()
        old_b1 = self.batch.b1[:n].copy()
        old_W2 = self.batch.W2[:n].copy()
        old_b2 = self.batch.b2[:n].copy()
        old_Wa = self.batch.Wa[:n].copy()
        old_ba = self.batch.ba[:n].copy()
        old_Wm = self.batch.Wm[:n].copy()
        old_bm = self.batch.bm[:n].copy()

        def copy_weights_to_slot(src_slot, dst_slot):
            self.batch.W1[dst_slot] = old_W1[src_slot]
            self.batch.b1[dst_slot] = old_b1[src_slot]
            self.batch.W2[dst_slot] = old_W2[src_slot]
            self.batch.b2[dst_slot] = old_b2[src_slot]
            self.batch.Wa[dst_slot] = old_Wa[src_slot]
            self.batch.ba[dst_slot] = old_ba[src_slot]
            self.batch.Wm[dst_slot] = old_Wm[src_slot]
            self.batch.bm[dst_slot] = old_bm[src_slot]

        # Elites
        for dst, src in enumerate(elite_slots):
            copy_weights_to_slot(src, dst)

        # Children via tournament
        for dst in range(n_elite, self.population_size):
            k = min(self.TOURNAMENT_K, n)
            competitors = self.rng.choice(n, size=k, replace=False)
            winner = competitors[np.argmax(fitnesses[competitors])]
            copy_weights_to_slot(winner, dst)
            # Mutate in place
            sigma = self.MUTATION_SIGMA
            for arr in [self.batch.W1, self.batch.b1, self.batch.W2, self.batch.b2,
                        self.batch.Wa, self.batch.ba, self.batch.Wm, self.batch.bm]:
                arr[dst] += self.rng.normal(0, sigma, arr[dst].shape).astype(np.float32)

        # Reset world agents
        new_agents = []
        for i in range(self.population_size):
            agent = Agent(
                id=self.world._next_agent_id,
                x=int(self.rng.integers(0, self.grid_size)),
                y=int(self.rng.integers(0, self.grid_size)),
                energy=INITIAL_ENERGY,
                weights=[],
            )
            self.world._next_agent_id += 1
            new_agents.append(agent)

        self.world.agents = new_agents
        self.world.grid[:] = 0.0
        self.world.seed_food(density=0.1)
