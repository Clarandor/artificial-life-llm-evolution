"""
Phase 4: Grid World Environment with Communication
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from .agent import HIDDEN_DIM, ATT_DIM, MAX_NEIGHBORS, OBS_DIM, ACTION_DIM, GRID_SIZE, CommunicationChannel

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
        
        # Communication channel
        self.comm_channel = CommunicationChannel()

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
            a.last_message = None
            a.messages_sent = 0
            a.coordination_events = 0

    def _periodic_dist(self, ax: float, ay: float, bx: float, by: float) -> float:
        dx = abs(bx - ax); dy = abs(by - ay)
        if dx > GRID / 2: dx = GRID - dx
        if dy > GRID / 2: dy = GRID - dy
        return dx + dy

    def _build_observation(self, agent) -> np.ndarray:
        """
        Build observation for a single agent.
        Shape: (48,) = 16 original + 32 for messages
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        
        # --- Original 16 dims ---
        # Border sensors (0-3)
        obs[0] = float(agent.x < 1.0)
        obs[1] = float(agent.x > GRID - 1.0)
        obs[2] = float(agent.y < 1.0)
        obs[3] = float(agent.y > GRID - 1.0)
        
        # Food direction sensors (4-7)
        for f in self.foods:
            d = self._periodic_dist(agent.x, agent.y, f.x, f.y)
            if d < 3.0:
                ddx = f.x - agent.x; ddy = f.y - agent.y
                if abs(ddx) > GRID/2: ddx -= np.sign(ddx)*GRID
                if abs(ddy) > GRID/2: ddy -= np.sign(ddy)*GRID
                norm = max(abs(ddx)+abs(ddy), 1)
                for idx, (dd, ii) in enumerate([(ddx, 0), (-ddx, 1), (ddy, 2), (-ddy, 3)]):
                    obs[4 + ii] = max(obs[4 + ii], dd / norm * (1 - d/3.0))
        
        # Prey direction (8-10)
        alive_prey = [p for p in self.prey if p.energy > 0]
        if alive_prey:
            dists = []
            for p in alive_prey:
                dx = p.x - agent.x; dy = p.y - agent.y
                if abs(dx) > GRID/2: dx -= np.sign(dx)*GRID
                if abs(dy) > GRID/2: dy -= np.sign(dy)*GRID
                dists.append((abs(dx)+abs(dy), dx, dy, p))
            _, prdx, prdy, nearest_prey = min(dists, key=lambda x: x[0])
            norm = max(abs(prdx)+abs(prdy), 1)
            obs[8] = prdx / norm
            obs[9] = prdy / norm
            obs[10] = min(dists)[0] / GRID
        
        # Nearby agents (11-12)
        n_nearby = sum(1 for a in self.agents if a.alive and a.id != agent.id
                       and self._periodic_dist(agent.x, agent.y, a.x, a.y) < 5.0)
        obs[11] = min(n_nearby / 10.0, 1.0)
        n_tribe = sum(1 for a in self.agents if a.alive and a.id != agent.id
                      and a.tribe_id == agent.tribe_id
                      and self._periodic_dist(agent.x, agent.y, a.x, a.y) < 5.0)
        obs[12] = min(n_tribe / 10.0, 1.0)
        
        # Age (13)
        obs[13] = min(agent.age / 200.0, 1.0)
        
        # Random tribe member direction (14-15)
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
        
        # --- Communication messages (16-47) ---
        # Receive messages from tribe members
        msg_summary = self.comm_channel.receive(agent.tribe_id, agent.id, max_messages=8)
        
        # Fill in relative positions for message senders
        if agent.tribe_id in self.comm_channel.messages:
            msgs = [(sid, msg, pos) for sid, msg, pos in self.comm_channel.messages[agent.tribe_id] 
                    if sid != agent.id]
            msgs = msgs[-8:]  # Same messages as in receive()
            
            for i, (sender_id, _, (sx, sy)) in enumerate(msgs):
                if i >= 8:
                    break
                # Compute relative position
                ddx = sx - agent.x; ddy = sy - agent.y
                if abs(ddx) > GRID/2: ddx -= np.sign(ddx)*GRID
                if abs(ddy) > GRID/2: ddy -= np.sign(ddy)*GRID
                norm = max(abs(ddx)+abs(ddy), 1)
                obs[16 + i*4 + 2] = ddx / norm  # dx to sender
                obs[16 + i*4 + 3] = ddy / norm  # dy to sender
        
        # Copy message stats from msg_summary
        obs[16:48] = msg_summary
        
        return obs

    def _build_observations(self) -> List[np.ndarray]:
        """Build observations for all alive agents."""
        obs_list = []
        alive = [a for a in self.agents if a.alive]
        for agent in alive:
            obs_list.append(self._build_observation(agent))
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
