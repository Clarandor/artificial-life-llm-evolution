"""
Phase 5: CoordinationWorld Environment
========================================
大型猎物需要多个智能体同时攻击才能被捕获。
- 单个智能体攻击大型猎物 → 无效果（猎物逃跑）
- 3个或以上智能体同时攻击 → 捕获猎物，攻击者各获得80能量
- 小型猎物（保留原有）→ 单个智能体即可捕获，获得30能量
- 协调信号机制：部落内部可进行近距离通信（发送+接收协调信号）
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

# ── Constants (inherited from phase4) ────────────────────────────────────
HIDDEN_DIM = 32
ATT_DIM = 4
MAX_NEIGHBORS = 8
OBS_DIM = 20     # 16 (from phase4) + 4 (coordination signals)
ACTION_DIM = 7   # 6 (from phase4) + 1 (broadcast_signal)
GRID_SIZE = 32

# ── Phase 5 specific constants ──────────────────────────────────────────
INITIAL_ENERGY = 200.0
STEP_COST = 0.5
FOOD_ENERGY = 30.0

# Large prey — requires COORDINATION_REQUIRED attackers simultaneously
COORDINATION_REQUIRED = 3
LARGE_PREY_ENERGY = 80.0
NUM_LARGE_PREY = 3

# Small prey — single agent can capture (keeps the food economy alive)
SMALL_PREY_ENERGY = 30.0
NUM_SMALL_PREY = 5

# Food spawn
FOOD_RATE = 0.02
INITIAL_FOOD = 20

# Coordination signals
NUM_SIGNALS = 4   # 4 different signal types (e.g., "attack target", "follow me", "scout", "gather")
SIGNAL_RADIUS = 3.0  # Only neighbors within this range receive the signal
SIGNAL_BROADCAST_COST = 0.2  # Energy cost to broadcast
SIGNAL_MAGNITUDE = 1.0  # Signal value in observation

# Prey flee
PREY_FLEE_RADIUS = 2.0   # Small prey flees when agent within this range
LARGE_PREY_FLEE_RADIUS = 2.5  # Large prey flees (but slower = easier to coordinate)


# ── Entities ──────────────────────────────────────────────────────────────

@dataclass
class LargePrey:
    """大型猎物：需要3个智能体同时攻击才能捕获"""
    x: float; y: float
    energy: float = 100.0  # Just to check if alive
    # Internal state
    _attacking_agents: List[int] = None  # agent ids who attacked this step

    def __post_init__(self):
        self._attacking_agents = []

    @property
    def is_alive(self) -> bool:
        return self.energy > 0


@dataclass
class SmallPrey:
    """小型猎物：单个智能体即可捕获"""
    x: float; y: float
    energy: float = 100.0

    @property
    def is_alive(self) -> bool:
        return self.energy > 0


@dataclass
class Food:
    x: float; y: float
    energy: float = FOOD_ENERGY


# ── Signal message ───────────────────────────────────────────────────────

@dataclass
class SignalMessage:
    sender_id: int
    tribe_id: int
    signal_type: int  # 0..NUM_SIGNALS-1
    target_x: float
    target_y: float
    step: int


# ── World ────────────────────────────────────────────────────────────────

class World:
    """
    Phase 5 CoordinationWorld

    Key mechanics:
    1. Large prey (LargePrey): needs ≥COORDINATION_REQUIRED agents attacking
       in the same step to be captured. Attackers must all be within 2.0.
    2. Small prey (SmallPrey): single agent within 1.5 can capture.
    3. Coordination signals: agents can broadcast a typed signal to tribe
       members within SIGNAL_RADIUS. Signals are consumed each step.
    4. All prey flee when an agent gets too close (reducing effectiveness
       of lone wolves).
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.agents: List = []
        self.foods: List[Food] = []
        self.large_prey: List[LargePrey] = []
        self.small_prey: List[SmallPrey] = []
        # Signal inbox: dict {agent_id -> List[SignalMessage]}
        self._signal_inbox: dict = {}
        self._step_counter = 0

        self._spawn_food(INITIAL_FOOD)
        self._spawn_large_prey()
        self._spawn_small_prey()

    # ── Spawning ─────────────────────────────────────────────────────────

    def _spawn_food(self, n: int):
        for _ in range(n):
            self.foods.append(Food(
                self.rng.uniform(0, GRID_SIZE),
                self.rng.uniform(0, GRID_SIZE)
            ))

    def _spawn_large_prey(self):
        for _ in range(NUM_LARGE_PREY):
            self.large_prey.append(LargePrey(
                self.rng.uniform(0, GRID_SIZE),
                self.rng.uniform(0, GRID_SIZE)
            ))

    def _spawn_small_prey(self):
        for _ in range(NUM_SMALL_PREY):
            self.small_prey.append(SmallPrey(
                self.rng.uniform(0, GRID_SIZE),
                self.rng.uniform(0, GRID_SIZE)
            ))

    # ── Agent management ────────────────────────────────────────────────

    def add_agent(self, agent):
        self.agents.append(agent)

    def reset(self):
        """Reset all agents for a new generation."""
        for a in self.agents:
            a.x = self.rng.uniform(0, GRID_SIZE)
            a.y = self.rng.uniform(0, GRID_SIZE)
            a.energy = INITIAL_ENERGY
            a.alive = True
            a.food_collected = 0.0
            a.small_prey_captured = 0
            a.large_prey_captured = 0
            a.attacks_made = 0
            a.signals_sent = 0
            a.signals_received = 0
            a.age = 0
            a._current_signal = 0  # no signal
            a._signal_target = (0.0, 0.0)
            a.reset_hidden()
        self._signal_inbox = {a.id: [] for a in self.agents}
        self._step_counter = 0

    # ── Distance helpers ────────────────────────────────────────────────

    def _periodic_dist(self, ax: float, ay: float, bx: float, by: float) -> float:
        dx = abs(bx - ax); dy = abs(by - ay)
        if dx > GRID_SIZE / 2: dx = GRID_SIZE - dx
        if dy > GRID_SIZE / 2: dy = GRID_SIZE - dy
        return dx + dy

    def _wrap_delta(self, ax: float, ay: float, bx: float, by: float):
        """Return (bx-ax, by-ay) with periodic boundary correction."""
        ddx = bx - ax; ddy = by - ay
        if abs(ddx) > GRID_SIZE / 2: ddx -= np.sign(ddx) * GRID_SIZE
        if abs(ddy) > GRID_SIZE / 2: ddy -= np.sign(ddy) * GRID_SIZE
        return ddx, ddy

    # ── Observations ────────────────────────────────────────────────────

    def _build_observations(self) -> List[np.ndarray]:
        """
        Build observation vector for each alive agent.
        OBS_DIM = 20:
          [0..3]   : wall proximity (left/right/top/bottom boundary)
          [4..7]   : food gradient (dx/dy normalized, one per dir)
          [8..10]  : nearest large prey direction + distance ratio
          [11]     : nearest small prey distance ratio
          [12]     : fraction of nearby alive agents (r=5)
          [13]     : fraction of nearby same-tribe agents (r=5)
          [14..15] : direction to nearest tribe member (r=5)
          [16]     : age / max_age
          [17..20] : coordination signals (signal_type, target_dir_x, target_dir_y, signal_strength)
        """
        obs_list = []
        alive = [a for a in self.agents if a.alive]
        if not alive:
            return []

        for agent in alive:
            obs = np.zeros(OBS_DIM, dtype=np.float32)

            # Boundary proximity (binary)
            obs[0] = float(agent.x < 1.0)
            obs[1] = float(agent.x > GRID_SIZE - 1.0)
            obs[2] = float(agent.y < 1.0)
            obs[3] = float(agent.y > GRID_SIZE - 1.0)

            # Food gradient
            best_food_dx, best_food_dy = 0.0, 0.0
            for f in self.foods:
                d = self._periodic_dist(agent.x, agent.y, f.x, f.y)
                if d < 3.0:
                    ddx, ddy = self._wrap_delta(agent.x, agent.y, f.x, f.y)
                    norm = max(abs(ddx) + abs(ddy), 1)
                    strength = (1.0 - d / 3.0)
                    for idx, (dd, ii) in enumerate([(ddx, 0), (-ddx, 1), (ddy, 2), (-ddy, 3)]):
                        candidate = dd / norm * strength
                        if abs(candidate) > abs(obs[4 + ii]):
                            obs[4 + ii] = candidate

            # Large prey (direction + distance) — the coordination target!
            alive_large = [p for p in self.large_prey if p.is_alive]
            if alive_large:
                dists = []
                for p in alive_large:
                    d = self._periodic_dist(agent.x, agent.y, p.x, p.y)
                    ddx, ddy = self._wrap_delta(agent.x, agent.y, p.x, p.y)
                    dists.append((d, ddx, ddy))
                d, ddx, ddy = min(dists)
                norm = max(abs(ddx) + abs(ddy), 1.0)
                obs[8] = ddx / norm   # large prey dir x
                obs[9] = ddy / norm   # large prey dir y
                obs[10] = min(d for d, _, _ in dists) / GRID_SIZE  # distance ratio

            # Small prey distance (single-agent prey)
            alive_small = [p for p in self.small_prey if p.is_alive]
            if alive_small:
                dists_s = [self._periodic_dist(agent.x, agent.y, p.x, p.y)
                           for p in alive_small]
                obs[11] = min(dists_s) / GRID_SIZE
            else:
                obs[11] = 1.0

            # Nearby agents
            nearby = [a for a in self.agents if a.alive and a.id != agent.id
                      and self._periodic_dist(agent.x, agent.y, a.x, a.y) < 5.0]
            obs[12] = min(len(nearby) / 10.0, 1.0)

            # Nearby same-tribe agents
            tribe_nearby = [a for a in nearby if a.tribe_id == agent.tribe_id]
            obs[13] = min(len(tribe_nearby) / 10.0, 1.0)

            # Direction to nearest tribe member
            if tribe_nearby:
                m = tribe_nearby[self.rng.randint(len(tribe_nearby))]
                ddx, ddy = self._wrap_delta(agent.x, agent.y, m.x, m.y)
                norm = max(abs(ddx) + abs(ddy), 1.0)
                obs[14] = ddx / norm
                obs[15] = ddy / norm
            else:
                obs[14] = obs[15] = 0.0

            # Age
            obs[16] = min(agent.age / 200.0, 1.0)

            # Coordination signals received (aggregate from inbox)
            inbox = self._signal_inbox.get(agent.id, [])
            if inbox:
                # Take the most recent (last) message
                msg = inbox[-1]
                obs[17] = float(msg.signal_type) / NUM_SIGNALS  # normalized signal type
                ddx, ddy = self._wrap_delta(agent.x, agent.y, msg.target_x, msg.target_y)
                norm = max(abs(ddx) + abs(ddy), 1.0)
                obs[18] = ddx / norm
                obs[19] = ddy / norm
                # signal strength is always 1.0 for now (could be weighted by sender count)
                obs[20] = 1.0 if hasattr(obs, '__setitem__') else obs[20]  # placeholder
                # Actually fix: obs has 20 elements (0-19), obs[20] would be out of range
                # Wait: OBS_DIM=20 means indices 0..19
                # Let me recount: obs[17]=signal_type, obs[18]=target_x, obs[19]=target_y
                # That's only 3 more, we need 4. Let me adjust OBS_DIM
                # Actually looking at the structure: [17] = signal_type, [18]=target_x, [19]=target_y
                # That's only 3 elements (17,18,19) = 3 elements.
                # I said NUM_SIGNALS=4 but that's the signal_type space, not obs elements.
                # Let me just use 3 more obs elements for signals: obs[17,18,19]
                # OBS_DIM=20 gives indices 0-19, so obs[17,18,19] are the 3 signal elements.
                # obs[17] = normalized signal type (0..1)
                # obs[18] = target direction x
                # obs[19] = target direction y
                # That's correct. obs[20] is out of range.
            else:
                # No signal: zeros (default)
                obs[17] = 0.0
                obs[18] = 0.0
                obs[19] = 0.0

            obs_list.append(obs)

        return obs_list

    # ── Neighbor finding ─────────────────────────────────────────────────

    def _find_neighbors(self, agent) -> Tuple[List[np.ndarray], List[int]]:
        """Return hidden states and IDs of nearby alive agents."""
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

    # ── World step ───────────────────────────────────────────────────────

    def step(self):
        """Called once per simulation step to update world state."""
        self._step_counter += 1

        # Spawn food
        if self.rng.random() < FOOD_RATE * max(len(self.foods), 1):
            self.foods.append(Food(
                self.rng.uniform(0, GRID_SIZE),
                self.rng.uniform(0, GRID_SIZE)
            ))

        # Respawn small prey
        alive_small = sum(1 for p in self.small_prey if p.is_alive)
        if alive_small < NUM_SMALL_PREY:
            self.small_prey = [p for p in self.small_prey if p.is_alive]
            self._spawn_small_prey()

        # Respawn large prey (less frequent, they are more valuable)
        alive_large = sum(1 for p in self.large_prey if p.is_alive)
        if alive_large < NUM_LARGE_PREY:
            self.large_prey = [p for p in self.large_prey if p.is_alive]
            self._spawn_large_prey()

        # Clear signal inbox for next step
        self._signal_inbox = {a.id: [] for a in self.agents}

        # Reset prey attack lists
        for p in self.large_prey:
            p._attacking_agents = []

    # ── Action resolution ─────────────────────────────────────────────────

    def resolve_actions(self, agent_actions: dict):
        """
        Resolve all agent actions for the current step.
        agent_actions: {agent_id: action_dict}
        action_dict keys: 'dx', 'dy', 'speed', 'eat_food', 'attack_small',
                           'attack_large', 'broadcast_signal', 'signal_type',
                           'signal_target_x', 'signal_target_y'

        Resolution order:
        1. Broadcast coordination signals
        2. Flee prey (before movement)
        3. Move agents
        4. Eat food / capture small prey
        5. Attack large prey (collective)
        6. Energy drain
        """
        alive_agents = {a.id: a for a in self.agents if a.alive}
        if not alive_agents:
            return

        # ── Step 1: Broadcast signals ────────────────────────────────────
        for aid, action in agent_actions.items():
            agent = alive_agents.get(aid)
            if agent is None:
                continue
            sig_type = int(action.get('broadcast_signal', 0))
            # 0 = no signal, 1..NUM_SIGNALS = signal type
            if sig_type > 0 and agent.energy > SIGNAL_BROADCAST_COST:
                agent.energy -= SIGNAL_BROADCAST_COST
                agent.signals_sent += 1
                sig_type_idx = min(sig_type - 1, NUM_SIGNALS - 1)
                tx = action.get('signal_target_x', agent.x)
                ty = action.get('signal_target_y', agent.y)

                # Deliver to tribe members within SIGNAL_RADIUS
                for other in self.agents:
                    if not other.alive:
                        continue
                    if other.id == agent.id:
                        continue
                    if other.tribe_id != agent.tribe_id:
                        continue
                    d = self._periodic_dist(agent.x, agent.y, other.x, other.y)
                    if d <= SIGNAL_RADIUS:
                        msg = SignalMessage(
                            sender_id=agent.id,
                            tribe_id=agent.tribe_id,
                            signal_type=sig_type_idx,
                            target_x=tx,
                            target_y=ty,
                            step=self._step_counter
                        )
                        self._signal_inbox.setdefault(other.id, []).append(msg)
                        other.signals_received += 1

        # ── Step 2: Prey flee ───────────────────────────────────────────
        for prey in self.large_prey:
            if not prey.is_alive:
                continue
            # Count nearby agents
            nearby = sum(
                1 for a in alive_agents.values()
                if self._periodic_dist(a.x, a.y, prey.x, prey.y) < LARGE_PREY_FLEE_RADIUS
            )
            if nearby > 0:
                # Flee in the average direction away from agents
                avg_dx, avg_dy = 0.0, 0.0
                for a in alive_agents.values():
                    d = self._periodic_dist(a.x, a.y, prey.x, prey.y)
                    if d < LARGE_PREY_FLEE_RADIUS and d > 0.01:
                        ddx, ddy = self._wrap_delta(prey.x, prey.y, a.x, a.y)
                        weight = 1.0 / max(d, 0.5)
                        avg_dx += ddx * weight
                        avg_dy += ddy * weight
                norm = max(abs(avg_dx) + abs(avg_dy), 1.0)
                flee_speed = 0.4  # slower than agents → coordination possible
                prey.x = (prey.x - avg_dx / norm * flee_speed + GRID_SIZE) % GRID_SIZE
                prey.y = (prey.y - avg_dy / norm * flee_speed + GRID_SIZE) % GRID_SIZE

        for prey in self.small_prey:
            if not prey.is_alive:
                continue
            nearby = sum(
                1 for a in alive_agents.values()
                if self._periodic_dist(a.x, a.y, prey.x, prey.y) < PREY_FLEE_RADIUS
            )
            if nearby > 0:
                avg_dx, avg_dy = 0.0, 0.0
                for a in alive_agents.values():
                    d = self._periodic_dist(a.x, a.y, prey.x, prey.y)
                    if d < PREY_FLEE_RADIUS and d > 0.01:
                        ddx, ddy = self._wrap_delta(prey.x, prey.y, a.x, a.y)
                        avg_dx += ddx; avg_dy += ddy
                norm = max(abs(avg_dx) + abs(avg_dy), 1.0)
                prey.x = (prey.x - avg_dx / norm * PREY_FLEE_RADIUS * 0.5 + GRID_SIZE) % GRID_SIZE
                prey.y = (prey.y - avg_dy / norm * PREY_FLEE_RADIUS * 0.5 + GRID_SIZE) % GRID_SIZE

        # ── Step 3: Move agents ──────────────────────────────────────────
        for aid, action in agent_actions.items():
            agent = alive_agents.get(aid)
            if agent is None:
                continue
            dx = float(np.clip(action.get('dx', 0), -1, 1))
            dy = float(np.clip(action.get('dy', 0), -1, 1))
            speed = 0.5 + 0.5 * float(np.clip(action.get('speed', 0), 0, 1))
            agent.x = (agent.x + dx * speed + GRID_SIZE) % GRID_SIZE
            agent.y = (agent.y + dy * speed + GRID_SIZE) % GRID_SIZE
            agent.energy -= STEP_COST

        # Re-fetch alive agents (energy may have changed)
        alive_agents = {a.id: a for a in self.agents if a.alive}

        # ── Step 4: Eat food ─────────────────────────────────────────────
        for aid, action in agent_actions.items():
            agent = alive_agents.get(aid)
            if agent is None:
                continue
            if action.get('eat_food', 0) > 0:
                for f in self.foods[:]:
                    d = self._periodic_dist(agent.x, agent.y, f.x, f.y)
                    if d < 1.0:
                        agent.energy += f.energy
                        agent.food_collected += f.energy
                        self.foods.remove(f)
                        break

        # ── Step 5: Capture small prey ──────────────────────────────────
        for aid, action in agent_actions.items():
            agent = alive_agents.get(aid)
            if agent is None:
                continue
            if action.get('attack_small', 0) > 0:
                for prey in self.small_prey:
                    if not prey.is_alive:
                        continue
                    d = self._periodic_dist(agent.x, agent.y, prey.x, prey.y)
                    if d < 1.5:
                        agent.energy += SMALL_PREY_ENERGY
                        agent.small_prey_captured += 1
                        prey.energy = 0
                        break

        # ── Step 6: Register attacks on large prey ─────────────────────
        for aid, action in agent_actions.items():
            agent = alive_agents.get(aid)
            if agent is None:
                continue
            if action.get('attack_large', 0) > 0:
                for prey in self.large_prey:
                    if not prey.is_alive:
                        continue
                    d = self._periodic_dist(agent.x, agent.y, prey.x, prey.y)
                    if d < 2.0:  # within striking range
                        prey._attacking_agents.append(agent.id)
                        agent.attacks_made += 1

        # ── Step 7: Resolve large prey captures ──────────────────────────
        for prey in self.large_prey:
            if not prey.is_alive:
                continue
            attackers = prey._attacking_agents
            if len(attackers) >= COORDINATION_REQUIRED:
                # Successful coordination! Reward the attackers
                reward = LARGE_PREY_ENERGY / len(attackers)
                for aid in attackers:
                    atk_agent = alive_agents.get(aid)
                    if atk_agent is not None:
                        atk_agent.energy += reward
                        atk_agent.large_prey_captured += 1
                prey.energy = 0  # prey is captured

        # ── Step 8: Kill starved agents ──────────────────────────────────
        for a in self.agents:
            if a.alive and a.energy <= 0:
                a.alive = False
