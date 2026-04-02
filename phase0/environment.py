"""
Phase 0: Grid World Environment
================================
A 2D discrete grid where agents survive, consume food, and reproduce.
Food spawns randomly; agents consume energy each step.
When energy exceeds REPRODUCTION_THRESHOLD, an agent reproduces (weight copy + mutation).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ── Constants ──────────────────────────────────────────────────────────────────
GRID_SIZE = 32
FOOD_ENERGY = 20.0
STEP_ENERGY_COST = 1.0
REPRODUCTION_THRESHOLD = 80.0
REPRODUCTION_COST = 40.0
INITIAL_ENERGY = 40.0
FOOD_SPAWN_RATE = 0.02       # probability per empty cell per step
MAX_AGENTS = 500
MIN_AGENTS = 10
MSG_DIM = 16                  # message vector dimensionality
OBS_DIM = 10                  # observation dimensionality


@dataclass
class Agent:
    id: int
    x: int
    y: int
    energy: float
    weights: List[np.ndarray]   # MLP weight matrices
    age: int = 0
    food_collected: int = 0
    alive: bool = True
    last_message: np.ndarray = field(default_factory=lambda: np.zeros(MSG_DIM))


class GridWorld:
    """
    2D grid environment for artificial life experiments.

    Grid cells:
        0 = empty
        1 = food
    Agents occupy cells (multiple agents can share a cell).
    """

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        food_spawn_rate: float = FOOD_SPAWN_RATE,
        max_agents: int = MAX_AGENTS,
        seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.food_spawn_rate = food_spawn_rate
        self.max_agents = max_agents
        self.rng = np.random.default_rng(seed)

        self.grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.agents: List[Agent] = []
        self.step_count = 0
        self._next_agent_id = 0

        # Observation metrics
        self.history_fitness: List[float] = []
        self.history_population: List[int] = []
        self.history_messages: List[np.ndarray] = []  # sampled message vectors

    # ── Initialization ─────────────────────────────────────────────────────────

    def seed_food(self, density: float = 0.1):
        """Randomly place food at the start."""
        mask = self.rng.random((self.grid_size, self.grid_size)) < density
        self.grid[mask] = 1.0

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    # ── Step ───────────────────────────────────────────────────────────────────

    def step(self, actions: List[int], messages: List[np.ndarray]) -> List[np.ndarray]:
        """
        Advance the world by one tick.

        Args:
            actions:  list of action indices (0=up,1=down,2=left,3=right,4=collect)
            messages: list of outgoing message vectors from each agent

        Returns:
            List of observation vectors for each (still-alive) agent.
        """
        assert len(actions) == len(self.agents)
        assert len(messages) == len(self.agents)

        # 1. Store outgoing messages
        for agent, msg in zip(self.agents, messages):
            agent.last_message = msg.copy()

        # 2. Move agents
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # up/down/left/right/stay
        for agent, action in zip(self.agents, actions):
            dx, dy = deltas[action]
            agent.x = (agent.x + dx) % self.grid_size
            agent.y = (agent.y + dy) % self.grid_size
            agent.energy -= STEP_ENERGY_COST
            agent.age += 1

        # 3. Collect food (action==4 and food present)
        for agent, action in zip(self.agents, actions):
            if action == 4 and self.grid[agent.x, agent.y] == 1.0:
                agent.energy += FOOD_ENERGY
                agent.food_collected += 1
                self.grid[agent.x, agent.y] = 0.0

        # 4. Kill agents with no energy
        for agent in self.agents:
            if agent.energy <= 0:
                agent.alive = False

        # 5. Reproduce
        new_agents = []
        for agent in self.agents:
            if agent.alive and agent.energy >= REPRODUCTION_THRESHOLD:
                if len(self.agents) + len(new_agents) < self.max_agents:
                    child = self._reproduce(agent)
                    new_agents.append(child)
                    agent.energy -= REPRODUCTION_COST

        self.agents = [a for a in self.agents if a.alive] + new_agents

        # Emergency respawn if population collapses
        if len(self.agents) < MIN_AGENTS:
            self._emergency_respawn()

        # 6. Spawn food
        self._spawn_food()

        # 7. Build observations
        observations = self._build_observations()

        # 8. Record metrics
        fitness = np.mean([a.food_collected for a in self.agents]) if self.agents else 0.0
        self.history_fitness.append(fitness)
        self.history_population.append(len(self.agents))
        if self.agents:
            sample = self.rng.choice(self.agents)
            self.history_messages.append(sample.last_message.copy())

        self.step_count += 1
        return observations

    # ── Observation ────────────────────────────────────────────────────────────

    def _build_observations(self) -> List[np.ndarray]:
        """
        Build observation vector for each agent.

        Observation (OBS_DIM = 10):
            [0]   food at current cell
            [1]   food north
            [2]   food south
            [3]   food west
            [4]   food east
            [5]   normalized energy  (energy / REPRODUCTION_THRESHOLD)
            [6]   neighbor count (normalized)
            [7-9] mean message from neighbors (first 3 dims, as proxy)

        Plus: neighbour message aggregate is passed separately (MSG_DIM dims).
        The full input to the MLP is OBS_DIM + MSG_DIM = 26 dims.
        """
        # Build spatial message map: average messages at each cell
        msg_map = np.zeros((self.grid_size, self.grid_size, MSG_DIM), dtype=np.float32)
        count_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for agent in self.agents:
            msg_map[agent.x, agent.y] += agent.last_message
            count_map[agent.x, agent.y] += 1

        # Avoid division by zero
        nonzero = count_map > 0
        msg_map[nonzero] /= count_map[nonzero, np.newaxis]

        observations = []
        for agent in self.agents:
            x, y = agent.x, agent.y
            g = self.grid_size

            food_here  = self.grid[x, y]
            food_north = self.grid[(x - 1) % g, y]
            food_south = self.grid[(x + 1) % g, y]
            food_west  = self.grid[x, (y - 1) % g]
            food_east  = self.grid[x, (y + 1) % g]
            energy_norm = min(agent.energy / REPRODUCTION_THRESHOLD, 2.0)

            # Neighbours: agents in adjacent cells
            neighbors = [
                a for a in self.agents
                if a is not agent and abs(a.x - x) <= 1 and abs(a.y - y) <= 1
            ]
            n_neighbors = min(len(neighbors), 8) / 8.0

            obs_core = np.array([
                food_here, food_north, food_south, food_west, food_east,
                energy_norm, n_neighbors,
                0.0, 0.0, 0.0,   # placeholder dims (for future use)
            ], dtype=np.float32)

            # Aggregate neighbour messages (mean over adjacent cells)
            neighbor_msg = np.zeros(MSG_DIM, dtype=np.float32)
            count = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = (x + dx) % g, (y + dy) % g
                    if count_map[nx, ny] > 0:
                        neighbor_msg += msg_map[nx, ny]
                        count += 1
            if count > 0:
                neighbor_msg /= count

            obs = np.concatenate([obs_core, neighbor_msg])  # shape: (26,)
            observations.append(obs)

        return observations

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _reproduce(self, parent: Agent) -> Agent:
        """Create a child agent with mutated weights."""
        child_weights = [
            w + self.rng.normal(0, 0.01, w.shape).astype(np.float32)
            for w in parent.weights
        ]
        child = Agent(
            id=self._next_agent_id,
            x=(parent.x + self.rng.integers(-2, 3)) % self.grid_size,
            y=(parent.y + self.rng.integers(-2, 3)) % self.grid_size,
            energy=REPRODUCTION_COST,
            weights=child_weights,
        )
        self._next_agent_id += 1
        return child

    def _emergency_respawn(self):
        """If population collapses, inject fresh random agents."""
        from .agent import MLP
        needed = MIN_AGENTS - len(self.agents)
        for _ in range(needed):
            mlp = MLP()
            agent = Agent(
                id=self._next_agent_id,
                x=int(self.rng.integers(0, self.grid_size)),
                y=int(self.rng.integers(0, self.grid_size)),
                energy=INITIAL_ENERGY,
                weights=mlp.get_weights(),
            )
            self._next_agent_id += 1
            self.agents.append(agent)

    def _spawn_food(self):
        """Randomly spawn food on empty cells."""
        empty = self.grid == 0.0
        spawn_mask = (self.rng.random((self.grid_size, self.grid_size)) < self.food_spawn_rate) & empty
        self.grid[spawn_mask] = 1.0

    # ── Utilities ──────────────────────────────────────────────────────────────

    def get_state_snapshot(self) -> dict:
        return {
            "step": self.step_count,
            "n_agents": len(self.agents),
            "food_count": int(self.grid.sum()),
            "mean_energy": float(np.mean([a.energy for a in self.agents])) if self.agents else 0.0,
            "mean_fitness": float(np.mean([a.food_collected for a in self.agents])) if self.agents else 0.0,
        }
