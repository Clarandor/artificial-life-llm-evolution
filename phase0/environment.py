"""
Phase 0.4c: Grid World — Fixed Encoding Diagnostic
======================================================
Diagnostic experiment: messages are FIXED (hard-coded to nearest prey
direction) rather than evolved. Only the decoder (receiver weights) is
subject to evolution. This measures the upper bound of GA's ability to
evolve signal utilization.

When FIXED_ENCODING=True:
  - msg[:2] is overwritten with the normalized direction to nearest prey
  - Sender alignment shaping is skipped (encoding is perfect by design)
  - Receiver shaping + approach reward remain active
  - Curriculum decay still applies

When FIXED_ENCODING=False: behaves like Phase 0.4a.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


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
MSG_DIM = 4
OBS_DIM = 16
ACTION_DIM = 6                    # 0-3 move, 4 collect, 5 attack

NUM_PREDATORS = 3
PREDATOR_SPEED = 2
NUM_PREY = 5
PREY_SPEED = 1

N_TRIBES = 10
TRIBE_SIZE = 10

# Reward shaping constants
ALIGN_REWARD = 0.5        # max bonus per step for signal-action alignment
APPROACH_REWARD = 0.3     # bonus per step for moving toward prey
RECEIVER_REWARD = 0.4     # bonus per step for following neighbor signal direction

# Diagnostic mode: fixed encoding
FIXED_ENCODING = True     # True = msg[:2] hardcoded to prey direction (Phase 0.4c)


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
    last_message: np.ndarray = field(default_factory=lambda: np.zeros(MSG_DIM))
    # Reward shaping accumulators
    shaping_score: float = 0.0    # accumulated shaping bonus this generation
    alignment_score: float = 0.0  # signal-action alignment accumulator
    approach_score: float = 0.0   # prey-approach accumulator
    receiver_score: float = 0.0   # neighbor-signal following accumulator


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

        for _ in range(num_predators):
            self.predators.append(Predator(
                x=int(self.rng.integers(0, grid_size)),
                y=int(self.rng.integers(0, grid_size)),
            ))
        for _ in range(num_prey):
            self.prey_list.append(Prey(
                x=int(self.rng.integers(0, grid_size)),
                y=int(self.rng.integers(0, grid_size)),
            ))

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
        g = self.grid_size
        half_g = g / 2.0

        # Record pre-move positions for reward shaping
        pre_x = [a.x for a in self.agents]
        pre_y = [a.y for a in self.agents]

        # 0. Build neighbor message map BEFORE this step's messages overwrite
        #    This captures what each agent "heard" from neighbors last step
        neighbor_avg_msg = self._compute_neighbor_avg_msg()

        # 1. Store outgoing messages (with optional fixed encoding)
        if FIXED_ENCODING and self.prey_list:
            prey_pos = np.array([[p.x, p.y] for p in self.prey_list if p.alive],
                                dtype=np.float32)
            if len(prey_pos) == 0:
                prey_pos = np.array([[p.x, p.y] for p in self.prey_list],
                                    dtype=np.float32)
            for agent, msg in zip(self.agents, messages):
                # Compute direction to nearest prey (toroidal)
                dx = prey_pos[:, 0] - agent.x
                dy = prey_pos[:, 1] - agent.y
                dx = np.where(np.abs(dx) > half_g, dx - np.sign(dx) * g, dx)
                dy = np.where(np.abs(dy) > half_g, dy - np.sign(dy) * g, dy)
                dists = np.abs(dx) + np.abs(dy)
                nearest = np.argmin(dists)
                direction = np.array([dx[nearest], dy[nearest]], dtype=np.float32)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                # Override msg[:2] with true prey direction; keep msg[2:] from MLP
                fixed_msg = msg.copy()
                fixed_msg[:2] = direction
                agent.last_message = fixed_msg
        else:
            for agent, msg in zip(self.agents, messages):
                agent.last_message = msg.copy()

        # 2. Move agents
        deltas = [(-1,0),(1,0),(0,-1),(0,1),(0,0),(0,0)]
        for agent, action in zip(self.agents, actions):
            a = min(action, 5)
            dx, dy = deltas[a]
            agent.x = (agent.x + dx) % g
            agent.y = (agent.y + dy) % g
            agent.energy -= STEP_ENERGY_COST
            agent.age += 1

        # 3. Reward Shaping: Signal-Action Alignment
        # Skip when FIXED_ENCODING — sender is already perfect by design
        if not FIXED_ENCODING:
            action_dirs = np.array([[-1,0],[1,0],[0,-1],[0,1]], dtype=np.float32)
            for i, (agent, action, msg) in enumerate(zip(self.agents, actions, messages)):
                if action < 4:  # only for move actions
                    sig = msg[:2]
                    sig_norm = np.linalg.norm(sig)
                    if sig_norm > 0.1:
                        sig_unit = sig / sig_norm
                        act_dir = action_dirs[action]
                        alignment = float(np.dot(sig_unit, act_dir))
                        if alignment > 0:
                            agent.alignment_score += alignment * ALIGN_REWARD

        # 4. Reward Shaping: Approach Prey
        # Check if agent moved closer to nearest prey
        if self.prey_list:
            prey_pos = np.array([[p.x, p.y] for p in self.prey_list], dtype=np.float32)
            for i, agent in enumerate(self.agents):
                if not agent.alive:
                    continue
                # Distance before move
                ox, oy = pre_x[i], pre_y[i]
                dx_pre = prey_pos[:, 0] - ox
                dy_pre = prey_pos[:, 1] - oy
                dx_pre = np.where(np.abs(dx_pre) > half_g, dx_pre - np.sign(dx_pre)*g, dx_pre)
                dy_pre = np.where(np.abs(dy_pre) > half_g, dy_pre - np.sign(dy_pre)*g, dy_pre)
                dist_pre = np.min(np.abs(dx_pre) + np.abs(dy_pre))

                # Distance after move
                dx_post = prey_pos[:, 0] - agent.x
                dy_post = prey_pos[:, 1] - agent.y
                dx_post = np.where(np.abs(dx_post) > half_g, dx_post - np.sign(dx_post)*g, dx_post)
                dy_post = np.where(np.abs(dy_post) > half_g, dy_post - np.sign(dy_post)*g, dy_post)
                dist_post = np.min(np.abs(dx_post) + np.abs(dy_post))

                if dist_post < dist_pre:
                    # Bonus scales with proximity (closer = more reward)
                    prox_factor = max(0, 1.0 - dist_post / half_g)
                    agent.approach_score += APPROACH_REWARD * prox_factor

        # 4b. Reward Shaping: Receiver — follow neighbor signal direction
        # If agent receives neighbor msg[:2] pointing in a direction and agent moves
        # that way, it gets a receiver bonus.
        action_dirs_recv = np.array([[-1,0],[1,0],[0,-1],[0,1]], dtype=np.float32)
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            if action < 4 and agent.alive:  # only for move actions
                recv_msg = neighbor_avg_msg[i]  # average msg from neighbors
                sig = recv_msg[:2]
                sig_norm = np.linalg.norm(sig)
                if sig_norm > 0.1:  # neighbors sent meaningful signal
                    sig_unit = sig / sig_norm
                    act_dir = action_dirs_recv[action]
                    alignment = float(np.dot(sig_unit, act_dir))
                    if alignment > 0:
                        agent.receiver_score += alignment * RECEIVER_REWARD

        # 5. Collect food (action==4)
        for agent, action in zip(self.agents, actions):
            if action == 4 and self.grid[agent.x, agent.y] == 1.0:
                agent.energy += FOOD_ENERGY
                agent.food_collected += 1
                self.grid[agent.x, agent.y] = 0.0

        # 6. Prey capture (action==5): need ≥2 attackers within 1 cell
        for prey in self.prey_list:
            if not prey.alive:
                continue
            attackers = []
            for agent, action in zip(self.agents, actions):
                if action == 5 and agent.alive:
                    dx = abs(agent.x - prey.x)
                    dy = abs(agent.y - prey.y)
                    dx = min(dx, g - dx)
                    dy = min(dy, g - dy)
                    if dx + dy <= 1:
                        attackers.append(agent)
            if len(attackers) >= 2:
                for a in attackers:
                    a.energy += PREY_CAPTURE_ENERGY
                    a.prey_captured += 1
                    a.food_collected += 1
                prey.alive = False
                prey_captures_this_step += 1

        # Respawn dead prey
        for prey in self.prey_list:
            if not prey.alive:
                prey.respawn(g, self.rng)

        # 7. Move prey
        for prey in self.prey_list:
            prey.move(g, self.rng)

        # 8. Move predators & kill
        for pred in self.predators:
            pred.move(g, self.rng)

        for agent in self.agents:
            if not agent.alive:
                continue
            for pred in self.predators:
                if agent.x == pred.x and agent.y == pred.y:
                    agent.alive = False
                    agent.energy = 0
                    kills_this_step += 1
                    break

        # 9. Kill agents with no energy
        for agent in self.agents:
            if agent.energy <= 0:
                agent.alive = False

        # 10. Reproduce
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

        # 11. Spawn food
        self._spawn_food()

        # 12. Update shaping scores
        for agent in self.agents:
            agent.shaping_score = agent.alignment_score + agent.approach_score + agent.receiver_score

        # 13. Build observations
        observations = self._build_observations()

        # 14. Record metrics
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
        """OBS_DIM=16 + MSG_DIM=4 → 20 dims total."""
        g = self.grid_size
        half_g = g / 2.0
        N = len(self.agents)
        if N == 0:
            return []

        msg_map = np.zeros((g, g, MSG_DIM), dtype=np.float32)
        count_map = np.zeros((g, g), dtype=np.float32)
        for agent in self.agents:
            msg_map[agent.x, agent.y] += agent.last_message
            count_map[agent.x, agent.y] += 1
        nonzero = count_map > 0
        msg_map[nonzero] /= count_map[nonzero, np.newaxis]

        neighbor_count_map = np.zeros((g, g), dtype=np.float32)
        neighbor_msg_sum = np.zeros((g, g, MSG_DIM), dtype=np.float32)
        neighbor_msg_cnt = np.zeros((g, g), dtype=np.float32)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                rolled_cnt = np.roll(np.roll(count_map, -dx, axis=0), -dy, axis=1)
                neighbor_count_map += rolled_cnt
                rolled_msg = np.roll(np.roll(msg_map, -dx, axis=0), -dy, axis=1)
                has = rolled_cnt > 0
                neighbor_msg_sum[has] += rolled_msg[has]
                neighbor_msg_cnt += has.astype(np.float32)
        nz2 = neighbor_msg_cnt > 0
        neighbor_msg_map = np.zeros((g, g, MSG_DIM), dtype=np.float32)
        neighbor_msg_map[nz2] = neighbor_msg_sum[nz2] / neighbor_msg_cnt[nz2, np.newaxis]

        ax = np.array([a.x for a in self.agents], dtype=np.int32)
        ay = np.array([a.y for a in self.agents], dtype=np.int32)
        energies = np.array([a.energy for a in self.agents], dtype=np.float32)

        food_here  = self.grid[ax, ay]
        food_north = self.grid[(ax-1)%g, ay]
        food_south = self.grid[(ax+1)%g, ay]
        food_west  = self.grid[ax, (ay-1)%g]
        food_east  = self.grid[ax, (ay+1)%g]
        energy_norm = np.clip(energies / REPRODUCTION_THRESHOLD, 0.0, 2.0)
        n_neighbors_norm = np.clip(neighbor_count_map[ax, ay] / 8.0, 0.0, 1.0)

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

        # Tribe-mate density
        tribe_count_maps = np.zeros((N_TRIBES, g, g), dtype=np.float32)
        for agent in self.agents:
            tribe_count_maps[agent.tribe_id, agent.x, agent.y] += 1

        tribe_mate_density = np.zeros(N, dtype=np.float32)
        for i, agent in enumerate(self.agents):
            x, y = agent.x, agent.y
            total_n = 0.0
            tribe_n = 0.0
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

        obs_core = np.stack([
            food_here, food_north, food_south, food_west, food_east,
            energy_norm, n_neighbors_norm,
            pred_dx, pred_dy, pred_prox,
            prey_dx, prey_dy, prey_prox,
            tribe_mate_density,
            same_cell,
            np.zeros(N, dtype=np.float32),
        ], axis=1)

        n_msg = neighbor_msg_map[ax, ay]
        obs_all = np.concatenate([obs_core, n_msg], axis=1)
        return [obs_all[i] for i in range(N)]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _compute_neighbor_avg_msg(self) -> np.ndarray:
        """Compute average neighbor message for each agent (from last_message).
        Returns array of shape (N, MSG_DIM)."""
        g = self.grid_size
        N = len(self.agents)
        if N == 0:
            return np.zeros((0, MSG_DIM), dtype=np.float32)

        # Build message map from current last_message
        msg_map = np.zeros((g, g, MSG_DIM), dtype=np.float32)
        count_map = np.zeros((g, g), dtype=np.float32)
        for agent in self.agents:
            msg_map[agent.x, agent.y] += agent.last_message
            count_map[agent.x, agent.y] += 1
        nonzero = count_map > 0
        msg_map[nonzero] /= count_map[nonzero, np.newaxis]

        # Sum neighbor messages (8-connected)
        neighbor_msg_sum = np.zeros((g, g, MSG_DIM), dtype=np.float32)
        neighbor_msg_cnt = np.zeros((g, g), dtype=np.float32)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                rolled_cnt = np.roll(np.roll(count_map, -dx, axis=0), -dy, axis=1)
                rolled_msg = np.roll(np.roll(msg_map, -dx, axis=0), -dy, axis=1)
                has = rolled_cnt > 0
                neighbor_msg_sum[has] += rolled_msg[has]
                neighbor_msg_cnt += has.astype(np.float32)

        nz = neighbor_msg_cnt > 0
        neighbor_avg = np.zeros((g, g, MSG_DIM), dtype=np.float32)
        neighbor_avg[nz] = neighbor_msg_sum[nz] / neighbor_msg_cnt[nz, np.newaxis]

        # Gather per-agent
        ax = np.array([a.x for a in self.agents], dtype=np.int32)
        ay = np.array([a.y for a in self.agents], dtype=np.int32)
        return neighbor_avg[ax, ay]

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
            tribe_id=parent.tribe_id,
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
            "mean_shaping": float(np.mean([a.shaping_score for a in self.agents])) if self.agents else 0.0,
            "mean_receiver": float(np.mean([a.receiver_score for a in self.agents])) if self.agents else 0.0,
        }
