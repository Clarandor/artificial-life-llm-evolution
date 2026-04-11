"""
Phase 4: Grid World Environment
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

HIDDEN_DIM = 32
ATT_DIM = 4
MAX_NEIGHBORS = 8
OBS_DIM = 16
ACTION_DIM = 6
GRID_SIZE = 32
GRID = GRID_SIZE

FOOD_ENERGY = 30.0
PREY_ENERGY = 80.0
STEP_COST = 1.0
INITIAL_ENERGY = 200.0
FOOD_RATE = 0.02
NUM_PREY = 5
NUM_PREDATORS = 3
PRED_SPEED = 2
PREY_SPEED = 1.0
PREY_FLEE_RADIUS = 5


@dataclass
class Prey:
    x: float; y: float
    energy: float = 200.0


@dataclass
class Food:
    x: float; y: float
    energy: float = FOOD_ENERGY


class World:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.agents: List = []
        self.foods: List[Food] = []
        self.prey: List[Prey] = []
        self.predators: List[dict] = []
        self._next_id = 0
        self._spawn_food(30)
        self._spawn_prey()
        self._spawn_predators()

    def _spawn_food(self, n: int):
        for _ in range(n):
            self.foods.append(Food(self.rng.uniform(0, GRID), self.rng.uniform(0, GRID)))

    def _spawn_prey(self):
        for _ in range(NUM_PREY):
            self.prey.append(Prey(self.rng.uniform(0, GRID), self.rng.uniform(0, GRID)))

    def _spawn_predators(self):
        for _ in range(NUM_PREDATORS):
            self.predators.append({
                'x': self.rng.uniform(0, GRID),
                'y': self.rng.uniform(0, GRID),
                'angle': self.rng.uniform(0, 2*np.pi),
            })

    def add_agent(self, agent):
        self.agents.append(agent)

    def reset(self):
        for a in self.agents:
            a.x = self.rng.uniform(0, GRID)
            a.y = self.rng.uniform(0, GRID)
            a.energy = INITIAL_ENERGY
            a.alive = True
            a.food_collected = 0.0
            a.prey_captured = 0
            a.age = 0
            a.reset_hidden()

    def _periodic_dist(self, ax: float, ay: float, bx: float, by: float) -> float:
        dx = abs(bx - ax); dy = abs(by - ay)
        if dx > GRID / 2: dx = GRID - dx
        if dy > GRID / 2: dy = GRID - dy
        return dx + dy

    def _build_observations(self) -> List[np.ndarray]:
        obs_list = []
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return []
        for agent in alive:
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            obs[0] = float(agent.x < 1.0)
            obs[1] = float(agent.x > GRID - 1.0)
            obs[2] = float(agent.y < 1.0)
            obs[3] = float(agent.y > GRID - 1.0)
            for f in self.foods:
                d = self._periodic_dist(agent.x, agent.y, f.x, f.y)
                if d < 3.0:
                    ddx = f.x - agent.x; ddy = f.y - agent.y
                    if abs(ddx) > GRID/2: ddx -= np.sign(ddx)*GRID
                    if abs(ddy) > GRID/2: ddy -= np.sign(ddy)*GRID
                    norm = max(abs(ddx)+abs(ddy), 1)
                    for idx, (dd, ii) in enumerate([(ddx, 0), (-ddx, 1), (ddy, 2), (-ddy, 3)]):
                        obs[4 + ii] = max(obs[4 + ii], dd / norm * (1 - d/3.0))
            alive_prey = [p for p in self.prey if p.energy > 0]
            if alive_prey:
                dists = []
                for p in alive_prey:
                    dx = p.x - agent.x; dy = p.y - agent.y
                    if abs(dx) > GRID/2: dx -= np.sign(dx)*GRID
                    if abs(dy) > GRID/2: dy -= np.sign(dy)*GRID
                    dists.append((abs(dx)+abs(dy), dx, dy))
                _, prdx, prdy = min(dists)
                norm = max(abs(prdx)+abs(prdy), 1)
                obs[8] = prdx / norm
                obs[9] = prdy / norm
                obs[10] = min(dists)[0] / GRID
            n_nearby = sum(1 for a in self.agents if a.alive and a.id != agent.id
                           and self._periodic_dist(agent.x, agent.y, a.x, a.y) < 5.0)
            obs[11] = min(n_nearby / 10.0, 1.0)
            n_tribe = sum(1 for a in self.agents if a.alive and a.id != agent.id
                          and a.tribe_id == agent.tribe_id
                          and self._periodic_dist(agent.x, agent.y, a.x, a.y) < 5.0)
            obs[12] = min(n_tribe / 10.0, 1.0)
            obs[13] = min(agent.age / 200.0, 1.0)
            tribe_members = [a for a in self.agents if a.alive and a.id != agent.id
                             and a.tribe_id == agent.tribe_id
                             and self._periodic_dist(agent.x, agent.y, a.x, a.y) < 5.0]
            if tribe_members:
                m = tribe_members[self.rng.randint(len(tribe_members))]
                ddx = m.x - agent.x; ddy = m.y - agent.y
                if abs(ddx) > GRID/2: ddx -= np.sign(ddx)*GRID
                if abs(ddy) > GRID/2: ddy -= np.sign(ddy)*GRID
                norm = max(abs(ddx)+abs(ddy), 1)
                obs[14] = ddx / norm
                obs[15] = ddy / norm
            obs_list.append(obs)
        return obs_list

    def _find_neighbors(self, agent) -> Tuple[List[np.ndarray], List[int]]:
        neighbors_h: List[np.ndarray] = []
        neighbor_ids: List[int] = []
        for a in self.agents:
            if not a.alive or a.id == agent.id:
                continue
            if self._periodic_dist(agent.x, agent.y, a.x, a.y) < 5.0:
                neighbors_h.append(a.h.copy())
                neighbor_ids.append(a.id)
                if len(neighbors_h) >= MAX_NEIGHBORS:
                    break
        return neighbors_h, neighbor_ids

    def step(self):
        """World-level updates: food spawn, prey respawn."""
        # Spawn food
        if self.rng.random() < FOOD_RATE * max(len(self.foods), 1):
            self.foods.append(Food(self.rng.uniform(0, GRID), self.rng.uniform(0, GRID)))
        # Respawn prey
        alive_prey = sum(1 for p in self.prey if p.energy > 0)
        if alive_prey < NUM_PREY:
            self.prey = [p for p in self.prey if p.energy > 0]
            self._spawn_prey()
