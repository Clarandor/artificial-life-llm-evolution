"""
Phase 1: Grid World — Attention-based Coordination
======================================================
Key change from Phase 0:
  - NO explicit message channel. Communication is implicit via attention
    over neighbor hidden states.
  - Each step:
    1. All agents encode observations → hidden states
    2. Each agent attends over neighbor hidden states → context
    3. Agent decides action from (hidden, context)
  - Observable: attention weights per agent per step
  - No reward shaping. Pure natural selection from day one.
    (Phase 0 proved shaping fixes behavior but not communication.
     Attention should work without shaping because it reduces the
     bilateral protocol problem to a unilateral reading problem.)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from .agent import OBS_DIM, ACTION_DIM, HIDDEN_DIM, MAX_NEIGHBORS


# ── Constants ──────────────────────────────────────────────────────────────────
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

NEIGHBOR_RADIUS = 5   # Manhattan distance for attention neighborhood


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
    # Attention diagnostics (per-step, overwritten each step)
    last_attn_weights: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    last_attn_neighbor_ids: List[int] = field(default_factory=list)
    # Accumulated attention stats (per-generation)
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

        # Predators and prey
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

    # ── Observations ──────────────────────────────────────────────────────────

    def _build_observations(self) -> List[np.ndarray]:
        """Build OBS_DIM=16 observation for each agent (same as Phase 0 but no msg)."""
        g = self.grid_size
        half = g // 2
        obs_list = []
        for agent in self.agents:
            obs = np.zeros(OBS_DIM, dtype=np.float32)

            # [0-3] local food (4 directions)
            for d, (dx, dy) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
                nx, ny = (agent.x+dx)%g, (agent.y+dy)%g
                obs[d] = self.grid[nx, ny]

            # [4] food at own cell
            obs[4] = self.grid[agent.x, agent.y]

            # [5] normalized energy
            obs[5] = agent.energy / 100.0

            # [6-7] nearest predator direction (toroidal)
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

            # [8-9] nearest prey direction
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
                obs[10] = min(dists)[0] / g   # distance normalized

            # [11] agent density in neighborhood (radius 3)
            n_nearby = sum(1 for a in self.agents
                           if a.id != agent.id and
                           (abs(a.x - agent.x) + abs(a.y - agent.y)) <= 3)
            obs[11] = min(n_nearby / 5.0, 1.0)

            # [12] tribe density
            t_nearby = sum(1 for a in self.agents
                           if a.tribe_id == agent.tribe_id and a.id != agent.id and
                           (abs(a.x - agent.x) + abs(a.y - agent.y)) <= 3)
            obs[12] = min(t_nearby / 3.0, 1.0)

            # [13] age normalized
            obs[13] = min(agent.age / 200.0, 1.0)

            # [14-15] nearest same-tribe direction
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

    # ── Neighbor lookup ───────────────────────────────────────────────────────

    def _find_neighbors(self, agent: Agent) -> Tuple[List[np.ndarray], List[int]]:
        """Find up to MAX_NEIGHBORS nearest agents and return their hidden states + ids."""
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

        # Sort by distance, take closest MAX_NEIGHBORS
        candidates.sort(key=lambda x: x[0])
        neighbors = candidates[:MAX_NEIGHBORS]

        if not neighbors:
            return [], []

        hiddens = [n.hidden for _, n in neighbors]
        ids = [n.id for _, n in neighbors]
        return hiddens, ids

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, actions: List[int], hiddens: List[np.ndarray],
             attn_weights_list: List[np.ndarray], attn_neighbor_ids_list: List[List[int]]):
        """
        Execute one step. Unlike Phase 0, no messages.
        Args:
            actions: action per agent
            hiddens: hidden state per agent (stored for next step's neighbor lookup)
            attn_weights_list: attention weights per agent (for diagnostics)
            attn_neighbor_ids_list: neighbor ids attended to (for diagnostics)
        """
        g = self.grid_size
        half_g = g // 2
        self.step_count += 1

        # Store hidden states and attention diagnostics
        for i, agent in enumerate(self.agents):
            agent.hidden = hiddens[i]
            agent.last_attn_weights = attn_weights_list[i]
            agent.last_attn_neighbor_ids = attn_neighbor_ids_list[i]
            # Accumulate attention entropy
            aw = attn_weights_list[i]
            if len(aw) > 1:
                entropy = -float(np.sum(aw * np.log(aw + 1e-10)))
                agent.attn_entropy_sum += entropy
                agent.attn_step_count += 1

        # 1. Move agents
        for agent, action in zip(self.agents, actions):
            if not agent.alive:
                continue
            if action == 0:   agent.x = (agent.x - 1) % g
            elif action == 1: agent.x = (agent.x + 1) % g
            elif action == 2: agent.y = (agent.y - 1) % g
            elif action == 3: agent.y = (agent.y + 1) % g

        # 2. Collect food (action == 4)
        for agent, action in zip(self.agents, actions):
            if action == 4 and agent.alive and self.grid[agent.x, agent.y] > 0:
                agent.energy += FOOD_ENERGY
                agent.food_collected += 1
                self.grid[agent.x, agent.y] = 0.0

        # 3. Attack / cooperative prey capture (action == 5)
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

        # 4. Predator attacks
        for pred in self.predators:
            for agent in self.agents:
                if agent.alive and agent.x == pred.x and agent.y == pred.y:
                    agent.energy -= 30.0
                    if agent.energy <= 0:
                        agent.alive = False

        # 5. Energy cost
        for agent in self.agents:
            if agent.alive:
                agent.energy -= STEP_ENERGY_COST
                agent.age += 1
                if agent.energy <= 0:
                    agent.alive = False

        # 6. Reproduction
        offspring = []
        for agent in self.agents:
            if (agent.alive and agent.energy >= REPRODUCTION_THRESHOLD
                and len(self.agents) + len(offspring) < self.max_agents):
                agent.energy -= REPRODUCTION_COST
                child = self._reproduce(agent)
                offspring.append(child)
        self.agents.extend(offspring)

        # 7. Remove dead
        self.agents = [a for a in self.agents if a.alive]

        # 8. Move predators and prey
        for pred in self.predators:
            pred.move(g, self.rng)
        for prey in self.prey_list:
            if prey.alive:
                prey.move(g, self.rng)
            else:
                prey.respawn(g, self.rng)

        # 9. Spawn food
        self._spawn_food()

        # 10. Emergency respawn
        if len(self.agents) < MIN_AGENTS:
            self._emergency_respawn()

    # ── Helpers ────────────────────────────────────────────────────────────────

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
        from .agent import AttentionMLP
        needed = MIN_AGENTS - len(self.agents)
        for _ in range(needed):
            mlp = AttentionMLP()
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
        agents = self.agents
        attn_entropies = []
        for a in agents:
            if a.attn_step_count > 0:
                attn_entropies.append(a.attn_entropy_sum / a.attn_step_count)
        return {
            "step": self.step_count,
            "n_agents": len(agents),
            "food_count": int((self.grid == 1.0).sum()),
            "n_predators": len(self.predators),
            "n_prey": len([p for p in self.prey_list if p.alive]),
            "mean_energy": float(np.mean([a.energy for a in agents])) if agents else 0.0,
            "mean_fitness": float(np.mean([a.food_collected for a in agents])) if agents else 0.0,
            "mean_prey_cap": float(np.mean([a.prey_captured for a in agents])) if agents else 0.0,
            "mean_attn_entropy": float(np.mean(attn_entropies)) if attn_entropies else 0.0,
        }
