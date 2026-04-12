"""
Phase 8A: Strong Coordination Incentive Environment (from Phase 6)
=================================================================
Copied from Phase 6's strong coordination environment.
This is the environment where coordination is the ONLY way to get
significant food (no small prey, only large prey requiring 2 same-tribe agents).

Key difference from Phase 7 environment: OBS_DIM = 24 to match Phase 7
(maintains compatibility with the recursive attention observation structure).
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

HIDDEN_DIM = 32
ATT_DIM = 4
MAX_NEIGHBORS = 8
OBS_DIM = 24  # 16 base + 4 coordination + 4 recursive attention
ACTION_DIM = 7  # 6 actions + 1 broadcast
GRID_SIZE = 32
GRID = GRID_SIZE

FOOD_ENERGY = 30.0
PREY_ENERGY = 80.0
STEP_COST = 0.5
INITIAL_ENERGY = 200.0

# Coordination settings
NUM_LARGE_PREY = 3  # Fewer but larger
COORDINATION_REQUIRED = 2  # 2 same-tribe agents needed
PREY_RADIUS = 2.0  # Larger hitbox (was 1.5)
PREY_SPEED = 0.3  # Slower prey
PREY_FLEE_RADIUS = 3  # Flee earlier
BIG_PREY_ENERGY = 100.0  # More reward
BIG_PREY_FLEE_ENERGY = 20.0  # Small reward even if prey escapes

NUM_PREDATORS = 3
PRED_SPEED = 2
PRED_DAMAGE = 5.0

FOOD_RATE = 0.01  # Slower food spawn
FOOD_COUNT_MAX = 10  # Fewer ambient food

GRID = GRID_SIZE


@dataclass
class LargePrey:
    """Large prey that requires coordinated attack."""
    x: float; y: float
    energy: float = 200.0
    fleeing: bool = False


@dataclass
class Food:
    x: float; y: float
    energy: float = FOOD_ENERGY


class World:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.agents: List = []
        self.foods: List[Food] = []
        self.prey: List[LargePrey] = []
        self.predators: List[dict] = []
        self._next_id = 0
        self._spawn_food(10)
        self._spawn_prey()

    def _spawn_food(self, n: int):
        for _ in range(n):
            if len(self.foods) < FOOD_COUNT_MAX:
                self.foods.append(Food(self.rng.uniform(0, GRID), self.rng.uniform(0, GRID)))

    def _spawn_prey(self):
        for _ in range(NUM_LARGE_PREY):
            self.prey.append(LargePrey(
                self.rng.uniform(0, GRID), self.rng.uniform(0, GRID)))

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
            a.large_prey_captured = 0
            a.failed_attacks = 0
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
        
        prey_alive = [p for p in self.prey]
        
        for agent in alive:
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            
            # Walls
            obs[0] = float(agent.x < 1.0)
            obs[1] = float(agent.x > GRID - 1.0)
            obs[2] = float(agent.y < 1.0)
            obs[3] = float(agent.y > GRID - 1.0)
            
            # Food
            for f in self.foods:
                d = self._periodic_dist(agent.x, agent.y, f.x, f.y)
                if d < 3.0:
                    ddx = f.x - agent.x; ddy = f.y - agent.y
                    if abs(ddx) > GRID/2: ddx -= np.sign(ddx)*GRID
                    if abs(ddy) > GRID/2: ddy -= np.sign(ddy)*GRID
                    norm = max(abs(ddx)+abs(ddy), 1)
                    for idx, (dd, ii) in enumerate([(ddx, 0), (-ddx, 1), (ddy, 2), (-ddy, 3)]):
                        obs[4 + ii] = max(obs[4 + ii], dd / norm * (1 - d/3.0))
            
            # Large prey direction
            if prey_alive:
                dists = []
                for p in prey_alive:
                    dx = p.x - agent.x; dy = p.y - agent.y
                    if abs(dx) > GRID/2: dx -= np.sign(dx)*GRID
                    if abs(dy) > GRID/2: dy -= np.sign(dy)*GRID
                    dists.append((abs(dx)+abs(dy), dx, dy, p.fleeing))
                
                dists.sort(key=lambda x: x[0])
                _, prdx, prdy, fleeing = dists[0]
                norm = max(abs(prdx)+abs(prdy), 1)
                obs[8] = prdx / norm
                obs[9] = prdy / norm
                obs[10] = dists[0][0] / GRID
                obs[11] = 1.0 if fleeing else 0.0  # Prey fleeing signal
            
            # Neighbors
            n_nearby = sum(1 for a in self.agents if a.alive and a.id != agent.id
                           and self._periodic_dist(agent.x, agent.y, a.x, a.y) < 5.0)
            obs[12] = min(n_nearby / 10.0, 1.0)
            
            n_tribe = sum(1 for a in self.agents if a.alive and a.id != agent.id
                          and a.tribe_id == agent.tribe_id
                          and self._periodic_dist(agent.x, agent.y, a.x, a.y) < 5.0)
            obs[13] = min(n_tribe / 10.0, 1.0)
            
            # Age
            obs[14] = min(agent.age / 200.0, 1.0)
            
            # Tribe member direction
            tribe_members = [a for a in self.agents if a.alive and a.id != agent.id
                             and a.tribe_id == agent.tribe_id
                             and self._periodic_dist(agent.x, agent.y, a.x, a.y) < 5.0]
            if tribe_members:
                m = tribe_members[self.rng.randint(len(tribe_members))]
                ddx = m.x - agent.x; ddy = m.y - agent.y
                if abs(ddx) > GRID/2: ddx -= np.sign(ddx)*GRID
                if abs(ddy) > GRID/2: ddy -= np.sign(ddy)*GRID
                norm = max(abs(ddx)+abs(ddy), 1)
                obs[15] = ddx / norm
                obs[16] = ddy / norm
            
            # Received coordination signal (from evolution)
            # obs[17:20] filled by evolution.py
            
            obs_list.append(obs)
        return obs_list

    def _find_neighbors(self, agent) -> Tuple[List[np.ndarray], List[int], List[int]]:
        """Returns: (hidden_list, id_list, tribe_list)"""
        h_list, id_list, tribe_list = [], [], []
        for a in self.agents:
            if not a.alive or a.id == agent.id:
                continue
            if self._periodic_dist(agent.x, agent.y, a.x, a.y) < 5.0:
                h_list.append(a.h.copy())
                id_list.append(a.id)
                tribe_list.append(a.tribe_id)
                if len(h_list) >= MAX_NEIGHBORS:
                    break
        return h_list, id_list, tribe_list

    def get_prey_proximity(self, agent) -> float:
        """
        Compute prey proximity for an agent: 1.0 = very close, 0.0 = no prey.
        Used as input to the gate network.
        """
        min_dist = float('inf')
        for p in self.prey:
            if p.energy <= 0:
                continue
            d = self._periodic_dist(agent.x, agent.y, p.x, p.y)
            if d < min_dist:
                min_dist = d
        
        if min_dist == float('inf'):
            return 0.0
        
        # Inverse distance: closer = higher proximity
        # Normalize so that distance 0 → proximity 1.0, distance GRID → proximity 0.0
        # Use a soft decay: proximity = exp(-min_dist / 5.0)
        proximity = np.exp(-min_dist / 5.0)
        return float(np.clip(proximity, 0.0, 1.0))

    def step(self):
        """World-level updates only."""
        if self.rng.random() < FOOD_RATE:
            self._spawn_food(1)
        
        # Respawn prey
        alive_prey = sum(1 for p in self.prey if p.energy > 0)
        if alive_prey < NUM_LARGE_PREY:
            self.prey = [p for p in self.prey if p.energy > 0]
            self._spawn_prey()
