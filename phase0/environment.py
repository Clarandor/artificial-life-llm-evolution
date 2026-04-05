"""
Phase 0.2: Grid World — Kin Selection + Prey Hunt + Group Selection
====================================================================
Key changes:
  - Tribes: population split into N_TRIBES groups; group selection pressure
  - Prey: mobile high-value targets requiring 2+ agents to capture
  - Predators: faster (speed=2), maintain survival pressure
  - Agents get tribe_id for group-level selection
  - New action: 5 = attack (attempt prey capture)
  - Kin selection: children stay near parents (already ±2), tribe ensures kin proximity
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ── Constants ──────────────────────────────────────────────────────────────────
GRID_SIZE = 32
FOOD_ENERGY = 20.0
PREY_CAPTURE_ENERGY = 80.0        # huge reward for cooperative hunt
STEP_ENERGY_COST = 1.0
REPRODUCTION_THRESHOLD = 80.0
REPRODUCTION_COST = 40.0
INITIAL_ENERGY = 40.0
FOOD_SPAWN_RATE = 0.02
MAX_AGENTS = 500
MIN_AGENTS = 10
MSG_DIM = 4
OBS_DIM = 16                      # expanded for prey info + tribe signal
ACTION_DIM = 6                    # 0-3 move, 4 collect, 5 attack

NUM_PREDATORS = 3
PREDATOR_SPEED = 2                # faster predators
NUM_PREY = 5
PREY_SPEED = 1

N_TRIBES = 10
TRIBE_SIZE = 10                   # initial agents per tribe


@dataclass
class Agent:
    id: int
    x: int
    y: int
    energy: float
    weights: List[np.ndarray]
    tribe_id: int = 0
    age: int = 0
    food_collected: int = 0
    prey_captured: int = 0        # cooperative kills
    alive: bool = True
    last_message: np.ndarray = field(default_factory=lambda: np.zeros(MSG_DIM))


@dataclass
class Predator:
    """Lethal NPC that kills agents on contact."""
    x: int
    y: int

    def move(self, grid_size: int, rng: np.random.Generator, speed: int = PREDATOR_SPEED):
        for _ in range(speed):
            dx, dy = rng.choice([(-1,0),(1,0),(0,-1),(0,1),(0,0)], p=[0.25,0.25,0.25,0.25,0.0])
            self.x = (self.x + dx) % grid_size
            self.y = (self.y + dy) % grid_size


@dataclass
class Prey:
    """High-value mobile target. Requires 2+ agents attacking simultaneously to capture."""
    x: int
    y: int
    alive: bool = True

    def move(self, grid_size: int, rng: np.random.Generator):
        dx, dy = rng.choice([(-1,0),(1,0),(0,-1),(0,1),(0,0)], p=[0.2,0.2,0.2,0.2,0.2])
        self.x = (self.x + dx) % grid_size
        self.y = (self.y + dy) % grid_size

    def respawn(self, grid_size: int, rng: np.random.Generator):
        self.x = int(rng.integers(0, grid_size))
        self.y = int(rng.integers(0, grid_size))
        self.alive = True


class GridWorld:
    """
    2D grid with food, prey (cooperative hunt), and predators.
    Grid cells: 0=empty, 1=food
    """

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        food_spawn_rate: float = FOOD_SPAWN_RATE,
        num_predators: int = NUM_PREDATORS,
        num_prey: int = NUM_PREY,
        max_agents: int = MAX_AGENTS,
        seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.food_spawn_rate = food_spawn_rate
        self.max_agents = max_agents
        self.rng = np.random.default_rng(seed)

        self.grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.agents: List[Agent] = []
        self.predators: List[Predator] = []
        self.prey_list: List[Prey] = []
        self.step_count = 0
        self._next_agent_id = 0

        # Spawn predators
        for _ in range(num_predators):
            self.predators.append(Predator(
                x=int(self.rng.integers(0, grid_size)),
                y=int(self.rng.integers(0, grid_size)),
            ))

        # Spawn prey
        for _ in range(num_prey):
            self.prey_list.append(Prey(
                x=int(self.rng.integers(0, grid_size)),
                y=int(self.rng.integers(0, grid_size)),
            ))

        # Metrics
        self.history_fitness: List[float] = []
        self.history_population: List[int] = []
        self.history_messages: List[np.ndarray] = []
        self.history_prey_captures: List[int] = []
        self.history_predator_kills: List[int] = []

    def seed_food(self, density: float = 0.1):
        mask = self.rng.random((self.grid_size, self.grid_size)) < density
        self.grid[mask] = 1.0

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    # ── Step ───────────────────────────────────────────────────────────────────

    def step(self, actions: List[int], messages: List[np.ndarray]) -> List[np.ndarray]:
        assert len(actions) == len(self.agents)
        assert len(messages) == len(self.agents)

        prey_captures_this_step = 0
        kills_this_step = 0

        # 1. Store outgoing messages
        for agent, msg in zip(self.agents, messages):
            agent.last_message = msg.copy()

        # 2. Move agents (actions 0-3), collect (4), attack (5)
        deltas = [(-1,0),(1,0),(0,-1),(0,1),(0,0),(0,0)]  # 5=attack=stay
        for agent, action in zip(self.agents, actions):
            a = min(action, 5)
            dx, dy = deltas[a]
            agent.x = (agent.x + dx) % self.grid_size
            agent.y = (agent.y + dy) % self.grid_size
            agent.energy -= STEP_ENERGY_COST
            agent.age += 1

        # 3. Collect food (action==4)
        for agent, action in zip(self.agents, actions):
            if action == 4 and self.grid[agent.x, agent.y] == 1.0:
                agent.energy += FOOD_ENERGY
                agent.food_collected += 1
                self.grid[agent.x, agent.y] = 0.0

        # 4. Prey capture (action==5): need ≥2 attackers within 1 cell of prey
        for prey in self.prey_list:
            if not prey.alive:
                continue
            # Find agents attacking near this prey
            attackers = []
            for agent, action in zip(self.agents, actions):
                if action == 5 and agent.alive:
                    dist = abs(agent.x - prey.x) + abs(agent.y - prey.y)
                    # Handle toroidal wrapping
                    dx = abs(agent.x - prey.x)
                    dy = abs(agent.y - prey.y)
                    dx = min(dx, self.grid_size - dx)
                    dy = min(dy, self.grid_size - dy)
                    if dx + dy <= 1:  # adjacent or same cell
                        attackers.append(agent)
            if len(attackers) >= 2:
                # Successful cooperative hunt!
                for a in attackers:
                    a.energy += PREY_CAPTURE_ENERGY
                    a.prey_captured += 1
                    a.food_collected += 1  # also counts toward base fitness
                prey.alive = False
                prey_captures_this_step += 1

        # Respawn dead prey
        for prey in self.prey_list:
            if not prey.alive:
                prey.respawn(self.grid_size, self.rng)

        # 5. Move prey (alive ones)
        for prey in self.prey_list:
            prey.move(self.grid_size, self.rng)

        # 6. Move predators & kill agents
        for pred in self.predators:
            pred.move(self.grid_size, self.rng)

        for agent in self.agents:
            if not agent.alive:
                continue
            for pred in self.predators:
                if agent.x == pred.x and agent.y == pred.y:
                    agent.alive = False
                    agent.energy = 0
                    kills_this_step += 1
                    break

        # 7. Kill agents with no energy
        for agent in self.agents:
            if agent.energy <= 0:
                agent.alive = False

        # 8. Reproduce (children inherit tribe_id, stay near parent)
        new_agents = []
        for agent in self.agents:
            if agent.alive and agent.energy >= REPRODUCTION_THRESHOLD:
                if len(self.agents) + len(new_agents) < self.max_agents:
                    child = self._reproduce(agent)
                    new_agents.append(child)
                    agent.energy -= REPRODUCTION_COST

        self.agents = [a for a in self.agents if a.alive] + new_agents

        if len(self.agents) < MIN_AGENTS:
            self._emergency_respawn()

        # 9. Spawn food
        self._spawn_food()

        # 10. Build observations
        observations = self._build_observations()

        # 11. Record metrics
        if self.agents:
            fitness = np.mean([a.food_collected for a in self.agents])
        else:
            fitness = 0.0
        self.history_fitness.append(fitness)
        self.history_population.append(len(self.agents))
        self.history_prey_captures.append(prey_captures_this_step)
        self.history_predator_kills.append(kills_this_step)
        if self.agents:
            sample = self.rng.choice(self.agents)
            self.history_messages.append(sample.last_message.copy())

        self.step_count += 1
        return observations

    # ── Observation (vectorized) ───────────────────────────────────────────────

    def _build_observations(self) -> List[np.ndarray]:
        """
        OBS_DIM = 16:
            [0-4]   food here/N/S/W/E (normalized)
            [5]     energy (normalized)
            [6]     neighbor count (Moore)
            [7-8]   nearest predator dx, dy
            [9]     nearest predator proximity
            [10-11] nearest prey dx, dy
            [12]    nearest prey proximity
            [13]    tribe-mate density nearby (same-tribe neighbors / all neighbors)
            [14]    num agents at same cell
            [15]    placeholder

        Plus MSG_DIM=4 → total input = 20
        """
        g = self.grid_size
        half_g = g / 2.0
        N = len(self.agents)
        if N == 0:
            return []

        # Spatial maps
        msg_map = np.zeros((g, g, MSG_DIM), dtype=np.float32)
        count_map = np.zeros((g, g), dtype=np.float32)
        for agent in self.agents:
            msg_map[agent.x, agent.y] += agent.last_message
            count_map[agent.x, agent.y] += 1
        nonzero = count_map > 0
        msg_map[nonzero] /= count_map[nonzero, np.newaxis]

        # Neighbor count (Moore neighborhood via rolled count_map)
        neighbor_count_map = np.zeros((g, g), dtype=np.float32)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor_count_map += np.roll(np.roll(count_map, -dx, axis=0), -dy, axis=1)

        # Neighbor message map
        neighbor_msg_sum = np.zeros((g, g, MSG_DIM), dtype=np.float32)
        neighbor_msg_cnt = np.zeros((g, g), dtype=np.float32)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                rolled_msg = np.roll(np.roll(msg_map, -dx, axis=0), -dy, axis=1)
                rolled_cnt = np.roll(np.roll(count_map, -dx, axis=0), -dy, axis=1)
                has = rolled_cnt > 0
                neighbor_msg_sum[has] += rolled_msg[has]
                neighbor_msg_cnt += has.astype(np.float32)
        nz2 = neighbor_msg_cnt > 0
        neighbor_msg_map = np.zeros((g, g, MSG_DIM), dtype=np.float32)
        neighbor_msg_map[nz2] = neighbor_msg_sum[nz2] / neighbor_msg_cnt[nz2, np.newaxis]

        # Agent arrays
        ax = np.array([a.x for a in self.agents], dtype=np.int32)
        ay = np.array([a.y for a in self.agents], dtype=np.int32)
        energies = np.array([a.energy for a in self.agents], dtype=np.float32)
        tribe_ids = np.array([a.tribe_id for a in self.agents], dtype=np.int32)

        # Food sensing
        food_here  = self.grid[ax, ay]
        food_north = self.grid[(ax-1)%g, ay]
        food_south = self.grid[(ax+1)%g, ay]
        food_west  = self.grid[ax, (ay-1)%g]
        food_east  = self.grid[ax, (ay+1)%g]
        energy_norm = np.clip(energies / REPRODUCTION_THRESHOLD, 0.0, 2.0)
        n_neighbors_norm = np.clip(neighbor_count_map[ax, ay] / 8.0, 0.0, 1.0)

        # Predator sensing (vectorized)
        if self.predators:
            pred_pos = np.array([[p.x, p.y] for p in self.predators], dtype=np.float32)
            dxs = pred_pos[np.newaxis, :, 0] - ax[:, np.newaxis].astype(np.float32)
            dys = pred_pos[np.newaxis, :, 1] - ay[:, np.newaxis].astype(np.float32)
            dxs = np.where(np.abs(dxs) > half_g, dxs - np.sign(dxs)*g, dxs)
            dys = np.where(np.abs(dys) > half_g, dys - np.sign(dys)*g, dys)
            dists = np.abs(dxs) + np.abs(dys)
            nearest = np.argmin(dists, axis=1)
            pred_dx = dxs[np.arange(N), nearest] / half_g
            pred_dy = dys[np.arange(N), nearest] / half_g
            pred_prox = 1.0 - np.clip(dists[np.arange(N), nearest] / half_g, 0.0, 1.0)
        else:
            pred_dx = np.zeros(N, dtype=np.float32)
            pred_dy = np.zeros(N, dtype=np.float32)
            pred_prox = np.zeros(N, dtype=np.float32)

        # Prey sensing (vectorized)
        if self.prey_list:
            prey_pos = np.array([[p.x, p.y] for p in self.prey_list], dtype=np.float32)
            dxs_pr = prey_pos[np.newaxis, :, 0] - ax[:, np.newaxis].astype(np.float32)
            dys_pr = prey_pos[np.newaxis, :, 1] - ay[:, np.newaxis].astype(np.float32)
            dxs_pr = np.where(np.abs(dxs_pr) > half_g, dxs_pr - np.sign(dxs_pr)*g, dxs_pr)
            dys_pr = np.where(np.abs(dys_pr) > half_g, dys_pr - np.sign(dys_pr)*g, dys_pr)
            dists_pr = np.abs(dxs_pr) + np.abs(dys_pr)
            nearest_pr = np.argmin(dists_pr, axis=1)
            prey_dx = dxs_pr[np.arange(N), nearest_pr] / half_g
            prey_dy = dys_pr[np.arange(N), nearest_pr] / half_g
            prey_prox = 1.0 - np.clip(dists_pr[np.arange(N), nearest_pr] / half_g, 0.0, 1.0)
        else:
            prey_dx = np.zeros(N, dtype=np.float32)
            prey_dy = np.zeros(N, dtype=np.float32)
            prey_prox = np.zeros(N, dtype=np.float32)

        # Tribe-mate density: fraction of neighbors that share tribe_id
        # Build per-tribe count maps
        tribe_count_maps = np.zeros((N_TRIBES, g, g), dtype=np.float32)
        for agent in self.agents:
            tribe_count_maps[agent.tribe_id, agent.x, agent.y] += 1

        # For each agent: same-tribe neighbors / total neighbors
        tribe_mate_density = np.zeros(N, dtype=np.float32)
        for i, agent in enumerate(self.agents):
            x, y = agent.x, agent.y
            total_n = 0
            tribe_n = 0
            for ddx in [-1, 0, 1]:
                for ddy in [-1, 0, 1]:
                    if ddx == 0 and ddy == 0:
                        continue
                    nx, ny = (x+ddx)%g, (y+ddy)%g
                    total_n += count_map[nx, ny]
                    tribe_n += tribe_count_maps[agent.tribe_id, nx, ny]
            if total_n > 0:
                tribe_mate_density[i] = tribe_n / total_n

        same_cell = count_map[ax, ay] / 4.0

        # Stack
        obs_core = np.stack([
            food_here, food_north, food_south, food_west, food_east,
            energy_norm, n_neighbors_norm,
            pred_dx, pred_dy, pred_prox,
            prey_dx, prey_dy, prey_prox,
            tribe_mate_density,
            same_cell,
            np.zeros(N, dtype=np.float32),
        ], axis=1)  # (N, 16)

        n_msg = neighbor_msg_map[ax, ay]  # (N, MSG_DIM)
        obs_all = np.concatenate([obs_core, n_msg], axis=1)  # (N, 20)

        return [obs_all[i] for i in range(N)]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _reproduce(self, parent: Agent) -> Agent:
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
            tribe_id=parent.tribe_id,   # inherit tribe
        )
        self._next_agent_id += 1
        return child

    def _emergency_respawn(self):
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
                tribe_id=int(self.rng.integers(0, N_TRIBES)),
            )
            self._next_agent_id += 1
            self.agents.append(agent)

    def _spawn_food(self):
        empty = self.grid == 0.0
        spawn_mask = (self.rng.random((self.grid_size, self.grid_size)) < self.food_spawn_rate) & empty
        self.grid[spawn_mask] = 1.0

    def get_state_snapshot(self) -> dict:
        return {
            "step": self.step_count,
            "n_agents": len(self.agents),
            "food_count": int((self.grid == 1.0).sum()),
            "n_predators": len(self.predators),
            "n_prey": len([p for p in self.prey_list if p.alive]),
            "mean_energy": float(np.mean([a.energy for a in self.agents])) if self.agents else 0.0,
            "mean_fitness": float(np.mean([a.food_collected for a in self.agents])) if self.agents else 0.0,
            "mean_prey_cap": float(np.mean([a.prey_captured for a in self.agents])) if self.agents else 0.0,
        }
