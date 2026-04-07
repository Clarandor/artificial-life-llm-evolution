"""
Phase 2: Grid World — Hybrid GA + REINFORCE
=============================================
Same as Phase 1.1 environment. Agent dataclass unchanged.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from .agent import OBS_DIM, ACTION_DIM, HIDDEN_DIM, MAX_NEIGHBORS


# ── Constants ──
GRID_SIZE = 32
FOOD_ENERGY = 20.0
PREY_CAPTURE_ENERGY = 80.0
STEP_ENERGY_COST = 1.0
REPRODUCTION_THRESHOLD = 80.0
REPRODUCTION_COST = 40.0
INITIAL_ENERGY = 40.0
FOOD_SPAWN_RATE = 0.02
MAX_AGENTS = 500
MIN_AGENTS = 10

NUM_PREDATORS = 3
PREDATOR_SPEED = 2
NUM_PREY = 5
PREY_SPEED = 1

N_TRIBES = 10
TRIBE_SIZE = 10

NEIGHBOR_RADIUS = 5


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
    prey_captured: int = 0
    alive: bool = True
    hidden: np.ndarray = field(default_factory=lambda: np.zeros(HIDDEN_DIM, dtype=np.float32))
    last_attn_weights: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    last_attn_neighbor_ids: List[int] = field(default_factory=list)
    attn_entropy_sum: float = 0.0
    attn_step_count: int = 0


@dataclass
class Predator:
    x: int
    y: int

    def move(self, grid_size: int, rng: np.random.Generator, speed: int = PREDATOR_SPEED):
        for _ in range(speed):
            dx, dy = rng.choice([(-1,0),(1,0),(0,-1),(0,1),(0,0)], p=[0.25,0.25,0.25,0.25,0.0])
            self.x = (self.x + dx) % grid_size
            self.y = (self.y + dy) % grid_size


@dataclass
class Prey:
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
        self._next_agent_id = 0
        self.step_count = 0

        self.predators = [
            Predator(int(self.rng.integers(0, grid_size)),
                     int(self.rng.integers(0, grid_size)))
            for _ in range(NUM_PREDATORS)
        ]
        self.prey_list = [
            Prey(int(self.rng.integers(0, grid_size)),
                 int(self.rng.integers(0, grid_size)))
            for _ in range(NUM_PREY)
        ]

    def seed_food(self, density: float = 0.1):
        mask = self.rng.random((self.grid_size, self.grid_size)) < density
        self.grid[mask] = 1.0

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def _build_observations(self) -> List[np.ndarray]:
        g = self.grid_size
        half = g // 2
        obs_list = []
        for agent in self.agents:
            obs = np.zeros(OBS_DIM, dtype=np.float32)

            for d, (dx, dy) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
                nx, ny = (agent.x+dx)%g, (agent.y+dy)%g
                obs[d] = self.grid[nx, ny]

            obs[4] = self.grid[agent.x, agent.y]
            obs[5] = agent.energy / 100.0

            if self.predators:
                dists = []
                for p in self.predators:
                    dx = p.x - agent.x
                    dy = p.y - agent.y
                    if abs(dx) > half: dx -= np.sign(dx) * g
                    if abs(dy) > half: dy -= np.sign(dy) * g
                    dists.append((abs(dx)+abs(dy), dx, dy))
                _, pdx, pdy = min(dists)
                norm = max(abs(pdx)+abs(pdy), 1)
                obs[6] = pdx / norm
                obs[7] = pdy / norm

            alive_prey = [p for p in self.prey_list if p.alive]
            if alive_prey:
                dists = []
                for p in alive_prey:
                    dx = p.x - agent.x
                    dy = p.y - agent.y
                    if abs(dx) > half: dx -= np.sign(dx) * g
                    if abs(dy) > half: dy -= np.sign(dy) * g
                    dists.append((abs(dx)+abs(dy), dx, dy))
                _, prdx, prdy = min(dists)
                norm = max(abs(prdx)+abs(prdy), 1)
                obs[8]  = prdx / norm
                obs[9]  = prdy / norm
                obs[10] = min(dists)[0] / g

            n_nearby = sum(1 for a in self.agents
                           if a.id != agent.id and
                           (abs(a.x - agent.x) + abs(a.y - agent.y)) <= 3)
            obs[11] = min(n_nearby / 5.0, 1.0)

            t_nearby = sum(1 for a in self.agents
                           if a.tribe_id == agent.tribe_id and a.id != agent.id and
                           (abs(a.x - agent.x) + abs(a.y - agent.y)) <= 3)
            obs[12] = min(t_nearby / 3.0, 1.0)

            obs[13] = min(agent.age / 200.0, 1.0)

            tribe_mates = [a for a in self.agents
                           if a.tribe_id == agent.tribe_id and a.id != agent.id]
            if tribe_mates:
                dists_t = []
                for a in tribe_mates:
                    dx = a.x - agent.x
                    dy = a.y - agent.y
                    if abs(dx) > half: dx -= np.sign(dx) * g
                    if abs(dy) > half: dy -= np.sign(dy) * g
                    dists_t.append((abs(dx)+abs(dy), dx, dy))
                _, tdx, tdy = min(dists_t)
                norm = max(abs(tdx)+abs(tdy), 1)
                obs[14] = tdx / norm
                obs[15] = tdy / norm

            obs_list.append(obs)
        return obs_list

    def _find_neighbors(self, agent: Agent) -> Tuple[List[np.ndarray], List[int]]:
        g = self.grid_size
        half = g // 2
        candidates = []
        for other in self.agents:
            if other.id == agent.id:
                continue
            dx = abs(other.x - agent.x)
            dy = abs(other.y - agent.y)
            if dx > half: dx = g - dx
            if dy > half: dy = g - dy
            dist = dx + dy
            if dist <= NEIGHBOR_RADIUS:
                candidates.append((dist, other))

        candidates.sort(key=lambda x: x[0])
        neighbors = candidates[:MAX_NEIGHBORS]

        if not neighbors:
            return [], []

        hiddens = [n.hidden for _, n in neighbors]
        ids = [n.id for _, n in neighbors]
        return hiddens, ids

    def step(self, actions: List[int], hiddens: List[np.ndarray],
             attn_weights_list: List[np.ndarray], attn_neighbor_ids_list: List[List[int]]):
        g = self.grid_size
        self.step_count += 1

        for i, agent in enumerate(self.agents):
            agent.hidden = hiddens[i]
            agent.last_attn_weights = attn_weights_list[i]
            agent.last_attn_neighbor_ids = attn_neighbor_ids_list[i]
            aw = attn_weights_list[i]
            if len(aw) > 1:
                entropy = -float(np.sum(aw * np.log(aw + 1e-10)))
                agent.attn_entropy_sum += entropy
                agent.attn_step_count += 1

        for agent, action in zip(self.agents, actions):
            if not agent.alive:
                continue
            if action == 0:   agent.x = (agent.x - 1) % g
            elif action == 1: agent.x = (agent.x + 1) % g
            elif action == 2: agent.y = (agent.y - 1) % g
            elif action == 3: agent.y = (agent.y + 1) % g

        for agent, action in zip(self.agents, actions):
            if action == 4 and agent.alive and self.grid[agent.x, agent.y] > 0:
                agent.energy += FOOD_ENERGY
                agent.food_collected += 1
                self.grid[agent.x, agent.y] = 0.0

        for prey in self.prey_list:
            if not prey.alive:
                continue
            attackers = [a for a, act in zip(self.agents, actions)
                         if act == 5 and a.alive and abs(a.x - prey.x) + abs(a.y - prey.y) <= 1]
            if len(attackers) >= 2:
                share = PREY_CAPTURE_ENERGY / len(attackers)
                for a in attackers:
                    a.energy += share
                    a.prey_captured += 1
                prey.alive = False

        for pred in self.predators:
            for agent in self.agents:
                if agent.alive and agent.x == pred.x and agent.y == pred.y:
                    agent.energy -= 30.0
                    if agent.energy <= 0:
                        agent.alive = False

        for agent in self.agents:
            if agent.alive:
                agent.energy -= STEP_ENERGY_COST
                agent.age += 1
                if agent.energy <= 0:
                    agent.alive = False

        offspring = []
        for agent in self.agents:
            if (agent.alive and agent.energy >= REPRODUCTION_THRESHOLD
                and len(self.agents) + len(offspring) < self.max_agents):
                agent.energy -= REPRODUCTION_COST
                child = self._reproduce(agent)
                offspring.append(child)
        self.agents.extend(offspring)

        self.agents = [a for a in self.agents if a.alive]

        for pred in self.predators:
            pred.move(g, self.rng)
        for prey in self.prey_list:
            if prey.alive:
                prey.move(g, self.rng)
            else:
                prey.respawn(g, self.rng)

        self._spawn_food()

        if len(self.agents) < MIN_AGENTS:
            self._emergency_respawn()

    def _reproduce(self, parent: Agent) -> Agent:
        child = Agent(
            id=self._next_agent_id,
            x=(parent.x + self.rng.integers(-2, 3)) % self.grid_size,
            y=(parent.y + self.rng.integers(-2, 3)) % self.grid_size,
            energy=REPRODUCTION_COST,
            weights=[],
            tribe_id=parent.tribe_id,
        )
        self._next_agent_id += 1
        return child

    def _emergency_respawn(self):
        from .agent import HybridAttentionMLP
        needed = MIN_AGENTS - len(self.agents)
        for _ in range(needed):
            mlp = HybridAttentionMLP()
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
