"""
Phase 0: Genetic Algorithm Evolution Loop
==========================================
Selection:  Tournament selection (k=3)
Mutation:   Gaussian noise σ=0.01
No gradient-based learning — pure evolutionary pressure.

Usage:
    from phase0.evolution import EvolutionEngine
    engine = EvolutionEngine(population_size=100, seed=42)
    engine.run(generations=200)
"""

import numpy as np
from typing import List, Tuple, Optional
import time

from .agent import MLP
from .environment import (
    GridWorld, Agent,
    GRID_SIZE, INITIAL_ENERGY, MSG_DIM
)


class EvolutionEngine:
    """
    Manages the population-level evolution loop.

    Each 'generation' = STEPS_PER_GEN environment steps.
    After each generation, agents are evaluated by fitness,
    and a new population is seeded via tournament selection + mutation.
    """

    STEPS_PER_GEN   = 200    # environment steps per generation
    TOURNAMENT_K    = 3      # tournament size
    MUTATION_SIGMA  = 0.01   # weight perturbation std dev
    ELITE_FRACTION  = 0.1    # fraction of top agents kept unchanged

    def __init__(
        self,
        population_size: int = 100,
        grid_size: int = GRID_SIZE,
        seed: Optional[int] = 42,
        verbose: bool = True,
    ):
        self.population_size = population_size
        self.grid_size = grid_size
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        # Logging
        self.generation_log: List[dict] = []

        # Init world and population
        self.world = GridWorld(grid_size=grid_size, seed=seed)
        self.world.seed_food(density=0.1)
        self._init_population()

    # ── Population Init ────────────────────────────────────────────────────────

    def _init_population(self):
        """Seed the world with random agents."""
        for i in range(self.population_size):
            mlp = MLP(seed=int(self.rng.integers(0, 1_000_000)))
            agent = Agent(
                id=i,
                x=int(self.rng.integers(0, self.grid_size)),
                y=int(self.rng.integers(0, self.grid_size)),
                energy=INITIAL_ENERGY,
                weights=mlp.get_weights(),
            )
            self.world.add_agent(agent)
        self.world._next_agent_id = self.population_size

    # ── Main Loop ──────────────────────────────────────────────────────────────

    def run(self, generations: int = 200, callback=None):
        """
        Run the full evolution for N generations.

        Args:
            generations: number of generations to run
            callback:    optional callable(generation, log_entry) for custom logging
        """
        print(f"Starting evolution: {generations} generations × {self.STEPS_PER_GEN} steps/gen")
        print(f"Population: {self.population_size}  |  Grid: {self.grid_size}×{self.grid_size}\n")

        for gen in range(generations):
            t0 = time.time()

            # ── Run environment steps ──────────────────────────────────────────
            for _ in range(self.STEPS_PER_GEN):
                agents = self.world.agents
                if not agents:
                    break

                mlps = [MLP.from_weights(a.weights) for a in agents]
                obs_list = self.world._build_observations()

                actions, messages = [], []
                for mlp, obs in zip(mlps, obs_list):
                    action, msg = mlp.forward(obs)
                    actions.append(action)
                    messages.append(msg)

                self.world.step(actions, messages)

            # ── Evaluate fitness ───────────────────────────────────────────────
            agents = self.world.agents
            fitnesses = np.array([a.food_collected for a in agents], dtype=np.float32)

            log_entry = {
                "generation":     gen,
                "population":     len(agents),
                "mean_fitness":   float(fitnesses.mean()) if len(fitnesses) else 0.0,
                "max_fitness":    float(fitnesses.max())  if len(fitnesses) else 0.0,
                "min_fitness":    float(fitnesses.min())  if len(fitnesses) else 0.0,
                "mean_energy":    float(np.mean([a.energy for a in agents])) if agents else 0.0,
                "mean_age":       float(np.mean([a.age   for a in agents])) if agents else 0.0,
                "elapsed_sec":    round(time.time() - t0, 2),
                # Sample message vectors for later PCA
                "sample_messages": np.stack([a.last_message for a in agents[:20]]) if agents else None,
            }
            self.generation_log.append(log_entry)

            if self.verbose and gen % 10 == 0:
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

    # ── Selection & Reproduction ───────────────────────────────────────────────

    def _breed(self, agents: List[Agent], fitnesses: np.ndarray):
        """Replace current population with offspring via tournament selection."""
        if len(agents) == 0:
            self._init_population()
            return

        n_elite = max(1, int(self.ELITE_FRACTION * self.population_size))
        sorted_idx = np.argsort(fitnesses)[::-1]

        elite_agents = [agents[i] for i in sorted_idx[:n_elite]]

        new_population = []
        # Keep elites (reset stats, keep weights)
        for a in elite_agents:
            new_agent = Agent(
                id=self.world._next_agent_id,
                x=int(self.rng.integers(0, self.grid_size)),
                y=int(self.rng.integers(0, self.grid_size)),
                energy=INITIAL_ENERGY,
                weights=[w.copy() for w in a.weights],
            )
            self.world._next_agent_id += 1
            new_population.append(new_agent)

        # Fill rest via tournament selection + mutation
        while len(new_population) < self.population_size:
            parent = self._tournament_select(agents, fitnesses)
            parent_mlp = MLP.from_weights(parent.weights)
            child_mlp = parent_mlp.mutate(sigma=self.MUTATION_SIGMA, rng=self.rng)
            child = Agent(
                id=self.world._next_agent_id,
                x=int(self.rng.integers(0, self.grid_size)),
                y=int(self.rng.integers(0, self.grid_size)),
                energy=INITIAL_ENERGY,
                weights=child_mlp.get_weights(),
            )
            self.world._next_agent_id += 1
            new_population.append(child)

        # Replace world agents
        self.world.agents = new_population
        # Reset food
        self.world.grid[:] = 0.0
        self.world.seed_food(density=0.1)

    def _tournament_select(self, agents: List[Agent], fitnesses: np.ndarray) -> Agent:
        """Select one agent via tournament selection."""
        k = min(self.TOURNAMENT_K, len(agents))
        idx = self.rng.choice(len(agents), size=k, replace=False)
        winner_idx = idx[np.argmax(fitnesses[idx])]
        return agents[winner_idx]
