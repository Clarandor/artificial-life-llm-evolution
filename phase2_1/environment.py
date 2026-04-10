"""
Phase 2.1: Supervised Evolution Engine
========================================
GA optimizes all weights (behavior + attention).
Attention weights get additional supervised signal.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import time

from .agent import (
    SupervisedAttentionMLP, OBS_DIM, HIDDEN_DIM, ATT_DIM, DEC_DIM, ACTION_DIM, MAX_NEIGHBORS,
    relu, softmax, compute_attention_target,
)


# Environment constants (same as phase2)
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
    """Same as phase2 environment."""
    
    def __init__(self, grid_size: int = GRID_SIZE, seed: Optional[int] = 42):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        self.agents: List[Agent] = []
        self.predators: List[Predator] = []
        self.prey_list: List[Prey] = []
        self.food_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        self._next_agent_id = 0
        
        for _ in range(NUM_PREDATORS):
            self.predators.append(Predator(
                x=int(self.rng.integers(0, grid_size)),
                y=int(self.rng.integers(0, grid_size))
            ))
        for _ in range(NUM_PREY):
            self.prey_list.append(Prey(
                x=int(self.rng.integers(0, grid_size)),
                y=int(self.rng.integers(0, grid_size))
            ))
    
    def seed_food(self, density: float = 0.1):
        n_food = int(self.grid_size * self.grid_size * density)
        for _ in range(n_food):
            x = int(self.rng.integers(0, self.grid_size))
            y = int(self.rng.integers(0, self.grid_size))
            self.food_grid[x, y] = 1.0
    
    def add_agent(self, agent: Agent):
        self.agents.append(agent)
    
    def _build_observations(self) -> List[np.ndarray]:
        """Build observations for all agents."""
        obs_list = []
        g = self.grid_size
        half = g // 2
        
        for agent in self.agents:
            if not agent.alive:
                continue
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            
            # Wall sensors (4 directions)
            obs[0] = 1.0 if agent.x == 0 else 0.0
            obs[1] = 1.0 if agent.x == g - 1 else 0.0
            obs[2] = 1.0 if agent.y == 0 else 0.0
            obs[3] = 1.0 if agent.y == g - 1 else 0.0
            
            # Food sensors (4 directions)
            food_positions = np.argwhere(self.food_grid > 0)
            for fx, fy in food_positions:
                dx = fx - agent.x
                dy = fy - agent.y
                if abs(dx) > half: dx -= int(np.sign(dx)) * g
                if abs(dy) > half: dy -= int(np.sign(dy)) * g
                if dx > 0 and abs(dx) > abs(dy): obs[4] = 1.0
                elif dx < 0 and abs(dx) > abs(dy): obs[5] = 1.0
                elif dy > 0 and abs(dy) >= abs(dx): obs[6] = 1.0
                elif dy < 0 and abs(dy) >= abs(dx): obs[7] = 1.0
            
            # Prey direction
            alive_prey = [p for p in self.prey_list if p.alive]
            if alive_prey:
                dists = []
                for p in alive_prey:
                    dx = p.x - agent.x
                    dy = p.y - agent.y
                    if abs(dx) > half: dx -= int(np.sign(dx)) * g
                    if abs(dy) > half: dy -= int(np.sign(dy)) * g
                    dists.append((abs(dx)+abs(dy), dx, dy))
                _, prdx, prdy = min(dists)
                norm = max(abs(prdx)+abs(prdy), 1)
                obs[8] = prdx / norm
                obs[9] = prdy / norm
                obs[10] = min(dists)[0] / g
            
            # Nearby agents
            n_nearby = sum(1 for a in self.agents
                          if a.id != agent.id and a.alive and
                          (abs(a.x - agent.x) + abs(a.y - agent.y)) <= 3)
            obs[11] = min(n_nearby / 5.0, 1.0)
            
            # Same tribe nearby
            t_nearby = sum(1 for a in self.agents
                          if a.tribe_id == agent.tribe_id and a.id != agent.id and a.alive and
                          (abs(a.x - agent.x) + abs(a.y - agent.y)) <= 3)
            obs[12] = min(t_nearby / 3.0, 1.0)
            
            # Age
            obs[13] = min(agent.age / 200.0, 1.0)
            
            # Nearest tribe mate
            tribe_mates = [a for a in self.agents
                          if a.tribe_id == agent.tribe_id and a.id != agent.id and a.alive]
            if tribe_mates:
                dists_t = []
                for a in tribe_mates:
                    dx = a.x - agent.x
                    dy = a.y - agent.y
                    if abs(dx) > half: dx -= int(np.sign(dx)) * g
                    if abs(dy) > half: dy -= int(np.sign(dy)) * g
                    dists_t.append((abs(dx)+abs(dy), dx, dy))
                _, tdx, tdy = min(dists_t)
                norm = max(abs(tdx)+abs(tdy), 1)
                obs[14] = tdx / norm
                obs[15] = tdy / norm
            
            obs_list.append(obs)
        return obs_list
    
    def _find_neighbors(self, agent: Agent) -> Tuple[List[np.ndarray], List[int]]:
        """Find neighbor agents within radius."""
        g = self.grid_size
        half = g // 2
        candidates = []
        
        for other in self.agents:
            if other.id == agent.id or not other.alive:
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
        
        neighbor_hiddens = [n.hidden for _, n in neighbors]
        neighbor_ids = [n.id for _, n in neighbors]
        return neighbor_hiddens, neighbor_ids
    
    def step(self, actions: List[int], hiddens: List[np.ndarray],
             attn_weights: List[np.ndarray], attn_neighbor_ids: List[List[int]]):
        """Execute one simulation step."""
        # Update agents
        for i, (agent, action, h, aw, nids) in enumerate(zip(self.agents, actions, hiddens, attn_weights, attn_neighbor_ids)):
            if not agent.alive:
                continue
            
            agent.hidden = h
            agent.last_attn_weights = aw
            agent.last_attn_neighbor_ids = nids
            
            # Track attention entropy
            if len(aw) > 0:
                entropy = -np.sum(aw * np.log(aw + 1e-10))
                agent.attn_entropy_sum += entropy
                agent.attn_step_count += 1
            
            # Execute action
            if action == 0:  # up
                agent.y = (agent.y - 1) % self.grid_size
            elif action == 1:  # down
                agent.y = (agent.y + 1) % self.grid_size
            elif action == 2:  # left
                agent.x = (agent.x - 1) % self.grid_size
            elif action == 3:  # right
                agent.x = (agent.x + 1) % self.grid_size
            elif action == 4:  # collect
                if self.food_grid[agent.x, agent.y] > 0:
                    self.food_grid[agent.x, agent.y] = 0
                    agent.energy += FOOD_ENERGY
                    agent.food_collected += 1
            elif action == 5:  # attack (prey)
                for prey in self.prey_list:
                    if prey.alive and prey.x == agent.x and prey.y == agent.y:
                        prey.alive = False
                        agent.energy += PREY_CAPTURE_ENERGY
                        agent.prey_captured += 1
                        break
            
            # Energy cost
            agent.energy -= STEP_ENERGY_COST
            agent.age += 1
            
            if agent.energy <= 0:
                agent.alive = False
        
        # Move predators
        for pred in self.predators:
            pred.move(self.grid_size, self.rng)
            # Kill agents at same position
            for agent in self.agents:
                if agent.alive and agent.x == pred.x and agent.y == pred.y:
                    agent.alive = False
        
        # Move prey
        for prey in self.prey_list:
            if prey.alive:
                prey.move(self.grid_size, self.rng)
        
        # Spawn food
        if self.rng.random() < FOOD_SPAWN_RATE:
            x = int(self.rng.integers(0, self.grid_size))
            y = int(self.rng.integers(0, self.grid_size))
            self.food_grid[x, y] = 1.0
