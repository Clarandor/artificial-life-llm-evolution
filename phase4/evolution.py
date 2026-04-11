"""
Phase 4: Evolution Engine for LSTM Agents with Communication
=============================================================
Enhanced with coordination rewards for cooperative hunting.
"""

import numpy as np
import time
import json
import os
from typing import List

from .agent import HIDDEN_DIM, ATT_DIM, GRID_SIZE, COMM_DIM, Agent
from .environment import World


def compute_attention_target(
    agent_pos: tuple, neighbor_positions: List[tuple],
    prey_positions: List[tuple], neighbor_tribes: List[int],
    agent_tribe: int, neighbor_fitnesses: List[float], grid_size: int
) -> np.ndarray:
    if not neighbor_positions:
        return np.array([], dtype=np.float32)
    
    def dist(p):
        dx = abs(p[0] - agent_pos[0]); dy = abs(p[1] - agent_pos[1])
        if dx > grid_size/2: dx = grid_size - dx
        if dy > grid_size/2: dy = grid_size - dy
        return max(dx + dy, 1)
    
    prey_dir = np.zeros(2)
    if prey_positions:
        nearest = min(prey_positions, key=dist)
        d = dist(nearest)
        prey_dir = np.array([nearest[0] - agent_pos[0], nearest[1] - agent_pos[1]])
        prey_dir /= (np.linalg.norm(prey_dir) + 1e-8)
    
    scores = []
    for i, (nx, ny) in enumerate(neighbor_positions):
        nd = dist((nx, ny))
        ndir = np.array([nx - agent_pos[0], ny - agent_pos[1]])
        ndir /= (np.linalg.norm(ndir) + 1e-8)
        prey_sim = max(0, np.dot(ndir, prey_dir)) if prey_positions else 0
        tribe_bonus = 1.0 if neighbor_tribes[i] == agent_tribe else 0.0
        fit_bonus = neighbor_fitnesses[i] / max(sum(neighbor_fitnesses) + 1e-8, 1)
        score = prey_sim * 2.0 + tribe_bonus * 1.0 + fit_bonus * 0.5 + (1.0/nd) * 0.2
        scores.append(score)
    
    scores = np.array(scores, dtype=np.float32)
    scores = scores - scores.max()
    weights = np.exp(scores)
    return (weights / weights.sum()).astype(np.float32)


class Evolution:
    ATTN_LOSS_WEIGHT = 0.01
    MUTATION_SIGMA = 0.05
    TOURNAMENT_K = 3
    STEPS_PER_GEN = 200
    N_TRIBES = 10
    TRIBE_SIZE = 10
    
    # Coordination reward parameters
    COORDINATION_RADIUS = 6.0  # Distance for cooperative hunting
    COORDINATION_REWARD = 2.0  # Base reward for coordinated behavior
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.world = World(seed=seed)
        
        # Create population — share LSTM/Attention templates within tribes
        self.agents: List[Agent] = []
        self.tribe_templates: dict = {}
        
        from .agent import LSTMCell, AttentionQK
        for t in range(self.N_TRIBES):
            rng_t = np.random.RandomState(seed + t)
            self.tribe_templates[t] = {
                'lstm': LSTMCell(48, HIDDEN_DIM, rng_t),  # OBS_DIM = 48
                'attn': AttentionQK(HIDDEN_DIM, ATT_DIM, rng_t),
            }
        
        for t in range(self.N_TRIBES):
            for _ in range(self.TRIBE_SIZE):
                agent = Agent(t, self.rng, tribe_templates=self.tribe_templates)
                self.agents.append(agent)
                self.world.add_agent(agent)
        
        counts = Agent.count_params()
        total = sum(counts.values())
        self.N_PARAMS = total
        
        Agent._next_id = len(self.agents)
        self.generation_log: List[dict] = []
        
        # Track coordination events
        self.coordination_events_total = 0
        
        print(f"  Architecture: LSTM({HIDDEN_DIM}) + Attention({ATT_DIM}) + Comm({COMM_DIM})")
        print(f"  Obs dim: 48 (16 base + 32 messages)")
        print(f"  Action dim: 7 (6 base + 1 comm trigger)")
        print(f"  Params: lstm={counts['lstm']} attn={counts['attn']} dec={counts['decoder']} "
              f"act={counts['action']} comm={counts['communication']} total={total}")
        print(f"  ATTN_LOSS_WEIGHT: {self.ATTN_LOSS_WEIGHT}")
        print(f"  COORDINATION_REWARD: {self.COORDINATION_REWARD}")

    def _compute_coordination_reward(self, agents: List[Agent]) -> dict:
        """
        Compute coordination rewards for tribe members.
        
        Coordination is measured by:
        1. Tribe members near same prey (potential cooperative hunting)
        2. Messages sent during hunting opportunities
        3. Proximity coordination (moving together)
        """
        rewards = {a.id: 0.0 for a in agents if a.alive}
        coordination_events = 0
        
        alive_agents = [a for a in agents if a.alive]
        alive_prey = [p for p in self.world.prey if p.energy > 0]
        
        for prey in alive_prey:
            # Find agents near this prey
            nearby_tribe_agents = {}
            for a in alive_agents:
                dist = self.world._periodic_dist(a.x, a.y, prey.x, prey.y)
                if dist < self.COORDINATION_RADIUS:
                    if a.tribe_id not in nearby_tribe_agents:
                        nearby_tribe_agents[a.tribe_id] = []
                    nearby_tribe_agents[a.tribe_id].append(a)
            
            # Reward coordinated hunting (2+ tribe members near same prey)
            for tribe_id, tribe_members in nearby_tribe_agents.items():
                if len(tribe_members) >= 2:
                    coordination_events += 1
                    for a in tribe_members:
                        # Higher reward for more coordinated agents
                        rewards[a.id] += self.COORDINATION_REWARD * len(tribe_members) / 2.0
                        a.coordination_events += 1
        
        # Bonus for agents that sent messages while near prey
        for a in alive_agents:
            if a.messages_sent > 0:
                for prey in alive_prey:
                    dist = self.world._periodic_dist(a.x, a.y, prey.x, prey.y)
                    if dist < self.COORDINATION_RADIUS:
                        rewards[a.id] += 0.5  # Small bonus for communicating near prey
        
        return {"rewards": rewards, "events": coordination_events}

    def _breed(self, agents: List[Agent], fitnesses: np.ndarray, gen: int, total_gens: int) -> List[Agent]:
        """Intra-tribe tournament selection + mutation."""
        fitnesses = np.array(fitnesses, dtype=np.float32)
        fitnesses = fitnesses - fitnesses.min() + 1e-8
        
        by_tribe = {}
        for a, f in zip(agents, fitnesses):
            by_tribe.setdefault(a.tribe_id, []).append((a, f))
        
        new_agents: List[Agent] = []
        
        for tribe_id, members in by_tribe.items():
            tribe_a = [m[0] for m in members]
            tribe_f = np.array([m[1] for m in members], dtype=np.float32)
            
            for _ in range(len(tribe_a)):
                idxs = self.rng.choice(len(tribe_a), self.TOURNAMENT_K, replace=False)
                best = max(idxs, key=lambda i: tribe_f[i])
                parent = tribe_a[best]
                
                child = Agent(tribe_id, self.rng, tribe_templates=self.tribe_templates)
                child.clone_weights_from(parent)
                
                # Mutate all weight matrices
                sigma = self.MUTATION_SIGMA * (1 - 0.8 * gen / total_gens)
                
                for attr, frac in [
                    (child.lstm.W, 0.1), (child.lstm.b, 0.1),
                    (child.attn.W_q, 0.1), (child.attn.W_k, 0.1),
                    (child.dec_W, 0.1), (child.dec_b, 0.1),
                    (child.act_W, 0.1), (child.act_b, 0.1),
                    (child.comm_W, 0.1), (child.comm_b, 0.1),  # Also mutate communication
                ]:
                    mask = self.rng.uniform(0, 1, attr.size) < frac
                    noise = self.rng.normal(0, sigma, mask.sum())
                    flat = attr.flatten()
                    flat[mask] += noise
                    attr[:] = flat.reshape(attr.shape)
                
                new_agents.append(child)
        
        return new_agents

    def run(self, generations: int = 100, save_log: bool = True) -> List[dict]:
        t0 = time.time()
        
        print(f"\nStarting: {generations} gen × {self.STEPS_PER_GEN} steps")
        print(f"  Population: {self.N_TRIBES}×{self.TRIBE_SIZE}={len(self.agents)}")
        
        self.world.reset()
        
        for gen in range(generations):
            alive_prev = len([a for a in self.agents if a.alive])
            
            # ── Simulate ──
            attn_losses = []
            attn_weights_list = []
            messages_sent_this_gen = 0
            
            for step in range(self.STEPS_PER_GEN):
                agents = self.world.agents
                if not any(a.alive for a in agents):
                    break
                
                # Clear communication channel at start of step
                self.world.comm_channel.clear()
                
                # Encode observations through LSTM
                obs_list = self.world._build_observations()
                if not obs_list:
                    break
                
                alive_agents = [a for a in agents if a.alive]
                
                # Encode
                for i, agent in enumerate(alive_agents[:len(obs_list)]):
                    agent.encode(obs_list[i])
                
                # First pass: generate and broadcast messages
                for agent in alive_agents:
                    # Generate potential message
                    message = agent.generate_message()
                    
                    # Decision (without messages for now, they're in obs)
                    nh_h, nh_ids = self.world._find_neighbors(agent)
                    action = agent.decide(nh_h)
                    
                    # Broadcast if communication action triggered
                    if action[5] > 0:  # Communication trigger
                        self.world.comm_channel.broadcast(
                            agent.tribe_id, agent.id, message, (agent.x, agent.y)
                        )
                        agent.last_message = message
                        agent.messages_sent += 1
                        messages_sent_this_gen += 1
                
                # Second pass: execute actions
                for agent in alive_agents:
                    nh_h, nh_ids = self.world._find_neighbors(agent)
                    action = agent.decide(nh_h)
                    
                    # Attention supervision
                    if nh_h:
                        prey_pos = [(p.x, p.y) for p in self.world.prey if p.energy > 0]
                        agent_fits = {a.id: a.food_collected + a.prey_captured * 3.0 
                                     for a in alive_agents}
                        
                        n_pos, n_tribe, n_fit = [], [], []
                        for nid in nh_ids:
                            for a in agents:
                                if a.id == nid:
                                    n_pos.append((a.x, a.y))
                                    n_tribe.append(a.tribe_id)
                                    n_fit.append(agent_fits.get(a.id, 0))
                                    break
                        
                        if len(n_pos) == len(nh_ids):
                            target = compute_attention_target(
                                (agent.x, agent.y), n_pos, prey_pos, n_tribe,
                                agent.tribe_id, n_fit, GRID_SIZE)
                            _, aw = agent.attn.attend(agent.h, nh_h)
                            if len(target) > 0 and len(aw) > 0:
                                kl = np.sum(target * np.log(target / (aw + 1e-10) + 1e-10))
                                attn_losses.append(kl)
                                attn_weights_list.append(aw)
                    
                    # Move
                    speed = 0.5 + 0.5 * action[2]
                    agent.x = (agent.x + action[0] * speed + GRID_SIZE) % GRID_SIZE
                    agent.y = (agent.y + action[1] * speed + GRID_SIZE) % GRID_SIZE
                    agent.energy -= 0.5
                    agent.age += 1
                    
                    # Eat food
                    if action[3] > 0:
                        for f in self.world.foods[:]:
                            dx = abs(f.x - agent.x); dy = abs(f.y - agent.y)
                            if dx > GRID_SIZE/2: dx = GRID_SIZE - dx
                            if dy > GRID_SIZE/2: dy = GRID_SIZE - dy
                            if dx + dy < 1.0:
                                agent.energy += f.energy
                                agent.food_collected += f.energy
                                self.world.foods.remove(f)
                                break
                    
                    # Catch prey
                    if action[4] > 0:
                        for p in self.world.prey:
                            if p.energy > 0:
                                dx = abs(p.x - agent.x); dy = abs(p.y - agent.y)
                                if dx > GRID_SIZE/2: dx = GRID_SIZE - dx
                                if dy > GRID_SIZE/2: dy = GRID_SIZE - dy
                                if dx + dy < 1.5:
                                    agent.energy += 80.0
                                    agent.prey_captured += 1
                                    p.energy = 0
                    
                    if agent.energy <= 0:
                        agent.alive = False
                
                # World
                self.world.step()
            
            # ── Evaluate ──
            agents = self.world.agents
            alive = [a for a in agents if a.alive]
            
            # Compute coordination rewards
            coord_result = self._compute_coordination_reward(alive)
            coord_rewards = coord_result["rewards"]
            coord_events = coord_result["events"]
            self.coordination_events_total += coord_events
            
            raw_food = np.array([a.food_collected for a in alive], dtype=np.float32)
            prey_caps = np.array([a.prey_captured for a in alive], dtype=np.float32)
            raw_fit = raw_food + prey_caps * 3.0
            
            # Add coordination rewards
            coord_fit = np.array([coord_rewards.get(a.id, 0.0) for a in alive], dtype=np.float32)
            
            total_prey = int(prey_caps.sum())
            
            attn_ents = []
            for aw in attn_weights_list:
                if len(aw) > 1:
                    attn_ents.append(-np.sum(aw * np.log(aw + 1e-10)))
            mean_attn_entropy = np.mean(attn_ents) if attn_ents else 0.0
            mean_attn_loss = np.mean(attn_losses) if attn_losses else 0.0
            
            # Combined fitness
            fitnesses = raw_fit + coord_fit + self.ATTN_LOSS_WEIGHT * mean_attn_loss
            if len(fitnesses) > 0:
                fitnesses = fitnesses - fitnesses.min() + 1e-8
            
            # Sample hiddens
            sample_h = np.stack([a.h for a in alive[:50]]) if alive else None
            
            # Communication stats
            total_messages = sum(a.messages_sent for a in alive)
            avg_coord_events = np.mean([a.coordination_events for a in alive]) if alive else 0.0
            
            log = {
                "generation": gen,
                "population": len(alive),
                "mean_fitness": float(fitnesses.mean()) if len(fitnesses) else 0.0,
                "max_fitness": float(fitnesses.max()) if len(fitnesses) else 0.0,
                "mean_raw_fitness": float(raw_fit.mean()) if len(raw_fit) else 0.0,
                "mean_raw_food": float(raw_food.mean()) if len(raw_food) else 0.0,
                "mean_prey_cap": float(prey_caps.mean()) if len(prey_caps) else 0.0,
                "total_prey_caps": total_prey,
                "mean_coord_reward": float(coord_fit.mean()) if len(coord_fit) else 0.0,
                "coord_events": coord_events,
                "total_coord_events": self.coordination_events_total,
                "messages_sent": messages_sent_this_gen,
                "total_messages": total_messages,
                "mean_attn_entropy": float(mean_attn_entropy),
                "mean_attn_loss": float(mean_attn_loss),
                "elapsed_sec": round(time.time() - t0, 2),
                "sample_hiddens": sample_h.tolist() if sample_h is not None else None,
            }
            self.generation_log.append(log)
            
            if gen % 10 == 0:
                elapsed = round(time.time() - t0, 2)
                sigma = self.MUTATION_SIGMA * (1 - 0.8 * gen / generations)
                print(f"Gen {gen:4d} | pop={len(alive):3d} | "
                      f"fit μ={fitnesses.mean():.2f} raw={raw_fit.mean():.2f} | "
                      f"prey={total_prey:3d} | coord={coord_events:3d} | "
                      f"msg={messages_sent_this_gen:4d} | H={mean_attn_entropy:.3f} | "
                      f"σ={sigma:.4f} | ⏱ {elapsed:.0f}s")
            
            # ── Breed ──
            self.agents = self._breed(alive, fitnesses, gen, generations)
            self.world.agents = self.agents
            self.world._spawn_food(5)
            for a in self.agents:
                a.alive = True
                a.energy = 200.0
                a.food_collected = 0.0
                a.prey_captured = 0
                a.age = 0
                a.reset_hidden()
                a.last_message = None
                a.messages_sent = 0
                a.coordination_events = 0
        
        if save_log:
            os.makedirs("results/phase4", exist_ok=True)
            with open("results/phase4/generation_log.json", "w") as f:
                json.dump(self.generation_log, f, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        print(f"\nDone in {time.time()-t0:.0f}s")
        return self.generation_log
