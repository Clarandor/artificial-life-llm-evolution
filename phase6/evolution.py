"""
Phase 6: Evolution with Strong Coordination Incentive
====================================================
Key changes:
- REMOVED small prey entirely
- Large prey requires 2 same-tribe agents attacking simultaneously
- Added coordination supervision loss (penalize failed coordinated attacks)
- Added coordination-aware attention target
- Sparse food (only 10 pieces max) - coordination is the survival path
"""

import numpy as np
import time
import json
import os
from typing import List, Tuple

from .agent import HIDDEN_DIM, ATT_DIM, GRID_SIZE, Agent
from .environment import World


def compute_coord_attention_target(
    agent_pos, agent_tribe, neighbor_positions, neighbor_tribes,
    neighbor_ids, agent_id, grid_size
) -> np.ndarray:
    """Attention target that rewards same-tribe neighbors."""
    if not neighbor_positions:
        return np.array([], dtype=np.float32)
    
    K = len(neighbor_positions)
    ax, ay = agent_pos
    
    scores = []
    for i in range(K):
        nx, ny = neighbor_positions[i]
        dx = abs(nx - ax); dy = abs(ny - ay)
        if dx > grid_size/2: dx = grid_size - dx
        if dy > grid_size/2: dy = grid_size - dy
        dist = max(dx + dy, 1)
        
        # Same tribe bonus (strong)
        tribe_bonus = 2.0 if neighbor_tribes[i] == agent_tribe else 0.0
        
        # Closer = better (for coordination)
        proximity_bonus = 3.0 / dist
        
        score = tribe_bonus + proximity_bonus
        scores.append(score)
    
    scores = np.array(scores, dtype=np.float32)
    scores = scores - scores.max()
    weights = np.exp(scores)
    return (weights / (weights.sum() + 1e-10)).astype(np.float32)


class Evolution:
    ATTN_LOSS_WEIGHT = 0.05  # Higher to encourage coordination
    MUTATION_SIGMA = 0.05
    TOURNAMENT_K = 3
    STEPS_PER_GEN = 200
    N_TRIBES = 10
    TRIBE_SIZE = 10
    COORDINATION_REQUIRED = 2
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.world = World(seed=seed)
        
        # Create population
        self.agents: List[Agent] = []
        self.tribe_templates = {}
        
        from .agent import LSTMCell, AttentionQK
        for t in range(self.N_TRIBES):
            rng_t = np.random.RandomState(seed + t)
            self.tribe_templates[t] = {
                'lstm': LSTMCell(20, HIDDEN_DIM, rng_t),  # OBS_DIM=20
                'attn': AttentionQK(HIDDEN_DIM, ATT_DIM, rng_t),
            }
        
        for t in range(self.N_TRIBES):
            for _ in range(self.TRIBE_SIZE):
                agent = Agent(t, self.rng, tribe_templates=self.tribe_templates)
                self.agents.append(agent)
                self.world.add_agent(agent)
        
        Agent._next_id = len(self.agents)
        self.generation_log: List[dict] = []
        
        print(f"  Coordination required: {self.COORDINATION_REQUIRED} same-tribe agents")
        print(f"  No small prey — coordination is survival")
        print(f"  ATTN_LOSS_WEIGHT: {self.ATTN_LOSS_WEIGHT}")

    def _breed(self, agents, fitnesses, gen, total_gens):
        fitnesses = np.array(fitnesses, dtype=np.float32)
        fitnesses = fitnesses - fitnesses.min() + 1e-8
        
        by_tribe = {}
        for a, f in zip(agents, fitnesses):
            by_tribe.setdefault(a.tribe_id, []).append((a, f))
        
        new_agents = []
        
        for tribe_id, members in by_tribe.items():
            tribe_a = [m[0] for m in members]
            tribe_f = np.array([m[1] for m in members], dtype=np.float32)
            
            for _ in range(len(tribe_a)):
                k = min(self.TOURNAMENT_K, len(tribe_a))
                idxs = self.rng.choice(len(tribe_a), k, replace=False)
                best = max(idxs, key=lambda i: tribe_f[i])
                parent = tribe_a[best]
                
                child = Agent(tribe_id, self.rng, tribe_templates=self.tribe_templates)
                child.clone_weights_from(parent)
                
                sigma = self.MUTATION_SIGMA * (1 - 0.8 * gen / total_gens)
                
                for attr, frac in [
                    (child.lstm.W, 0.1), (child.lstm.b, 0.1),
                    (child.attn.W_q, 0.1), (child.attn.W_k, 0.1),
                    (child.dec_W, 0.1), (child.dec_b, 0.1),
                    (child.act_W, 0.1), (child.act_b, 0.1),
                ]:
                    mask = self.rng.uniform(0, 1, attr.size) < frac
                    noise = self.rng.normal(0, sigma, mask.sum())
                    flat = attr.flatten()
                    flat[mask] += noise
                    attr[:] = flat.reshape(attr.shape)
                
                new_agents.append(child)
        
        return new_agents

    def run(self, generations=100, save_log=True):
        t0 = time.time()
        
        print(f"\nStarting Phase 6: Strong Coordination")
        print(f"  {generations} gen × {self.STEPS_PER_GEN} steps")
        print(f"  Population: {self.N_TRIBES}×{self.TRIBE_SIZE}={len(self.agents)}")
        
        self.world.reset()
        
        for gen in range(generations):
            # ── Simulate ──
            attn_losses = []
            attn_weights_list = []
            
            for step in range(self.STEPS_PER_GEN):
                agents = self.world.agents
                if not any(a.alive for a in agents):
                    break
                
                obs_list = self.world._build_observations()
                if not obs_list:
                    break
                
                alive = [a for a in agents if a.alive]
                
                # Encode
                for i, agent in enumerate(alive[:len(obs_list)]):
                    agent.encode(obs_list[i])
                
                # Track attack intentions
                attack_intents = {}  # agent_id -> prey_id
                
                for agent in alive:
                    nh_h, nh_ids, nh_tribes = self.world._find_neighbors(agent)
                    
                    # Coordination-aware attention target
                    if nh_h:
                        n_pos = []
                        n_tribe_list = []
                        for nid, nt in zip(nh_ids, nh_tribes):
                            for a in agents:
                                if a.id == nid:
                                    n_pos.append((a.x, a.y))
                                    n_tribe_list.append(nt)
                                    break
                        
                        if len(n_pos) == len(nh_ids):
                            target = compute_coord_attention_target(
                                (agent.x, agent.y), agent.tribe_id,
                                n_pos, n_tribe_list, nh_ids, agent.id, GRID_SIZE)
                            _, aw = agent.attn.attend(agent.h, nh_h)
                            if len(target) > 0 and len(aw) > 0:
                                kl = np.sum(target * np.log(target / (aw + 1e-10) + 1e-10))
                                attn_losses.append(kl)
                                attn_weights_list.append(aw)
                    
                    action = agent.decide(nh_h)
                    
                    # Track attack intention
                    if action[4] > 0:
                        # Find nearest prey
                        nearest_prey = None
                        min_d = float('inf')
                        for pi, p in enumerate(self.world.prey):
                            if p.energy <= 0: continue
                            d = self.world._periodic_dist(agent.x, agent.y, p.x, p.y)
                            if d < min_d:
                                min_d = d
                                nearest_prey = pi
                        
                        if nearest_prey is not None and min_d < 3.0:
                            attack_intents[agent.id] = (nearest_prey, agent)
                
                # ── Resolve attacks ──
                # Group attacks by prey
                prey_attacks = {}  # prey_id -> list of agents
                for agent_id, (prey_id, agent) in attack_intents.items():
                    prey_attacks.setdefault(prey_id, []).append(agent)
                
                for prey_id, attackers in prey_attacks.items():
                    prey = self.world.prey[prey_id]
                    if prey.energy <= 0:
                        continue
                    
                    # Count same-tribe attackers
                    tribe_attacks = {}
                    for a in attackers:
                        tribe_attacks.setdefault(a.tribe_id, []).append(a)
                    
                    # Check if any tribe has enough coordinated attackers
                    coord_successful = False
                    for tribe_id, tribe_attacker_list in tribe_attacks.items():
                        if len(tribe_attacker_list) >= self.COORDINATION_REQUIRED:
                            # Successful coordinated attack!
                            for a in tribe_attacker_list:
                                a.energy += 100.0  # BIG_ENERGY
                                a.large_prey_captured += 1
                            prey.energy = 0
                            coord_successful = True
                            break
                    
                    if not coord_successful:
                        # Failed attack - penalize
                        for a in attackers:
                            a.failed_attacks += 1
                            a.energy -= 5.0  # Penalty for wasted attack
                
                # ── Move agents ──
                for agent in alive:
                    action = agent.decide([a.h for a in alive if a.alive and a.id != agent.id][:8])
                    speed = 0.5 + 0.5 * action[2]
                    agent.x = (agent.x + action[0] * speed + GRID_SIZE) % GRID_SIZE
                    agent.y = (agent.y + action[1] * speed + GRID_SIZE) % GRID_SIZE
                    agent.energy -= 0.5
                    agent.age += 1
                    
                    # Ambient food (small reward)
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
                    
                    # Signal broadcasting
                    if action[6] > 0:
                        agent.signals_sent += 1
                    
                    if agent.energy <= 0:
                        agent.alive = False
                
                # ── Move prey ──
                for p in self.world.prey:
                    if p.energy <= 0: continue
                    
                    alive_agents = [a for a in agents if a.alive]
                    threats = [a for a in alive_agents
                               if self.world._periodic_dist(p.x, p.y, a.x, a.y) < 3.0]
                    if threats:
                        t = threats[self.rng.randint(len(threats))]
                        ddx = p.x - t.x; ddy = p.y - t.y
                        if abs(ddx) > GRID_SIZE/2: ddx -= np.sign(ddx)*GRID_SIZE
                        if abs(ddy) > GRID_SIZE/2: ddy -= np.sign(ddy)*GRID_SIZE
                        norm = max(abs(ddx)+abs(ddy), 1)
                        p.x = (p.x + ddx/norm * 0.3 + GRID_SIZE) % GRID_SIZE
                        p.y = (p.y + ddy/norm * 0.3 + GRID_SIZE) % GRID_SIZE
                    else:
                        p.x = (p.x + self.rng.uniform(-0.2, 0.2) + GRID_SIZE) % GRID_SIZE
                        p.y = (p.y + self.rng.uniform(-0.2, 0.2) + GRID_SIZE) % GRID_SIZE
                
                self.world.step()
            
            # ── Evaluate ──
            agents = self.world.agents
            alive = [a for a in agents if a.alive]
            
            raw_food = np.array([a.food_collected for a in alive], dtype=np.float32)
            large_caps = np.array([a.large_prey_captured for a in alive], dtype=np.float32)
            failed = np.array([a.failed_attacks for a in alive], dtype=np.float32)
            
            # Fitness: food + large_prey bonus - failed_attack penalty
            raw_fit = raw_food + large_caps * 50.0 - failed * 5.0
            
            total_large = int(large_caps.sum())
            total_failed = int(failed.sum())
            
            # Coordination quality
            coord_attempts = sum(1 for a in alive for _ in range(int(a.coord_attempts)) if a.coord_attempts > 0)
            
            attn_ents = []
            for aw in attn_weights_list:
                if len(aw) > 1:
                    attn_ents.append(-np.sum(aw * np.log(aw + 1e-10)))
            mean_attn_entropy = np.mean(attn_ents) if attn_ents else 0.0
            mean_attn_loss = np.mean(attn_losses) if attn_losses else 0.0
            
            fitnesses = raw_fit + self.ATTN_LOSS_WEIGHT * mean_attn_loss
            if len(fitnesses) > 0:
                fitnesses = fitnesses - fitnesses.min() + 1e-8
            
            sample_h = np.stack([a.h for a in alive[:20]]).tolist() if (alive and gen % 10 == 0) else None
            
            sig_sent = np.array([a.signals_sent for a in alive], dtype=np.float32)
            
            log = {
                "generation": gen,
                "population": len(alive),
                "mean_fitness": float(fitnesses.mean()) if len(fitnesses) else 0.0,
                "max_fitness": float(fitnesses.max()) if len(fitnesses) else 0.0,
                "mean_raw_fitness": float(raw_fit.mean()) if len(raw_fit) else 0.0,
                "mean_raw_food": float(raw_food.mean()) if len(raw_food) else 0.0,
                "large_prey_captured": total_large,
                "failed_attacks": total_failed,
                "mean_attn_entropy": float(mean_attn_entropy),
                "mean_attn_loss": float(mean_attn_loss),
                "mean_signals_sent": float(sig_sent.mean()) if len(sig_sent) else 0.0,
                "elapsed_sec": round(time.time() - t0, 2),
                "sample_hiddens": sample_h,
            }
            self.generation_log.append(log)
            
            if gen % 10 == 0:
                elapsed = round(time.time() - t0, 2)
                sigma = self.MUTATION_SIGMA * (1 - 0.8 * gen / max(generations, 1))
                coord_rate = total_large / max(total_failed + total_large, 1)
                print(f"Gen {gen:4d} | pop={len(alive):3d} | "
                      f"fit={fitnesses.mean():.2f} raw={raw_fit.mean():.2f} | "
                      f"large={total_large:2d} failed={total_failed:2d} | "
                      f"CQ={coord_rate:.2f} | H={mean_attn_entropy:.3f} | "
                      f"σ={sigma:.4f} | ⏱ {elapsed:.0f}s",
                      flush=True)
            
            # Breed
            self.agents = self._breed(alive, fitnesses, gen, generations)
            self.world.agents = self.agents
            self.world._spawn_food(2)
            for a in self.agents:
                a.alive = True
                a.energy = 200.0
                a.food_collected = 0.0
                a.large_prey_captured = 0
                a.failed_attacks = 0
                a.signals_sent = 0
                a.age = 0
                a.reset_hidden()
        
        if save_log:
            os.makedirs("results/phase6", exist_ok=True)
            with open("results/phase6/generation_log.json", "w") as f:
                json.dump(self.generation_log, f, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        print(f"\nDone in {time.time()-t0:.0f}s")
        return self.generation_log
