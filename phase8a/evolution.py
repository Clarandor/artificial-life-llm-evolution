"""
Phase 8A: Gated Recursive Attention Evolution
===============================================
Based on Phase 7 evolution, with key modifications:

1. Gate mechanism: Level-2 recursive attention only activates when
   gate_value > 0.5 (prey nearby)
2. Prey proximity is computed from world state and passed to agent.decide()
3. Track:
   - gate_values: distribution of gate activations
   - recursive_usage: fraction of steps where Level 2 was active
   - prey_proximity: average proximity when Level 2 active vs inactive
4. Fitness: same as Phase 7, plus a small bonus for using Level 2 only when useful
   (low prey_proximity when Level 2 is active = selective usage = good)
"""

import numpy as np
import time
import json
import os
from typing import List

from .agent import HIDDEN_DIM, ATT_DIM, GRID_SIZE, Agent, GatedRecursiveAttention, LSTMCell
from .environment import World


class Evolution:
    ATTN_LOSS_WEIGHT = 0.05
    MUTATION_SIGMA = 0.05
    TOURNAMENT_K = 3
    STEPS_PER_GEN = 200
    N_TRIBES = 10
    TRIBE_SIZE = 10
    COORDINATION_REQUIRED = 2
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.world = World(seed=seed)
        
        # Tribe templates
        self.tribe_templates = {}
        for t in range(self.N_TRIBES):
            rng_t = np.random.RandomState(seed + t)
            self.tribe_templates[t] = {
                'lstm': LSTMCell(24, HIDDEN_DIM, rng_t),  # OBS_DIM=24
                'attn': GatedRecursiveAttention(HIDDEN_DIM, ATT_DIM, rng_t),
            }
        
        self.agents: List[Agent] = []
        for t in range(self.N_TRIBES):
            for _ in range(self.TRIBE_SIZE):
                agent = Agent(t, self.rng, tribe_templates=self.tribe_templates)
                self.agents.append(agent)
                self.world.add_agent(agent)
        
        Agent._next_id = len(self.agents)
        self.generation_log: List[dict] = []
        
        total_params = self.agents[0].total_params if self.agents else 0
        lstm_p = HIDDEN_DIM * 4 * (24 + HIDDEN_DIM)
        attn_p = 2 * ATT_DIM * HIDDEN_DIM + ATT_DIM * 8 + 2  # +2 for gate
        print(f"  GatedRecursiveAttention: Level1 + Gate-controlled Level2")
        print(f"  Params: LSTM={lstm_p} attn={attn_p} dec={self.agents[0].dec_W.size+self.agents[0].dec_b.size} total={total_params}")
        print(f"  Gate: W_gate={float(self.agents[0].attn.W_gate[0]):.2f} b_gate={float(self.agents[0].attn.b_gate[0]):.2f} threshold=0.5")

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
                    (child.attn.W_rec, 0.1),
                    # Gate network also mutates
                    (child.attn.W_gate, 0.2), (child.attn.b_gate, 0.2),
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
        
        print(f"\nStarting Phase 8A: Gated Recursive Attention")
        print(f"  {generations} gen × {self.STEPS_PER_GEN} steps")
        print(f"  Population: {self.N_TRIBES}×{self.TRIBE_SIZE}={len(self.agents)}")
        print(f"  Gate activates Level 2 only when prey nearby (threshold=0.5)")
        
        self.world.reset()
        
        for gen in range(generations):
            attn_losses = []
            attn_weights_list = []
            attn_entropy_list = []
            gate_values = []          # All gate values
            recursive_active_count = 0  # Steps where Level 2 was active
            prey_proximity_when_active = []
            prey_proximity_when_inactive = []
            
            # Collect previous-step attention weights for each agent
            prev_attn = {a.id: np.array([], dtype=np.float32) for a in self.agents}
            
            for step in range(self.STEPS_PER_GEN):
                agents = self.world.agents
                if not any(a.alive for a in agents):
                    break
                
                obs_list = self.world._build_observations()
                if not obs_list:
                    break
                
                alive = [a for a in agents if a.alive]
                
                # ── Encode ──
                for i, agent in enumerate(alive[:len(obs_list)]):
                    agent.encode(obs_list[i])
                
                # ── Phase 1: Compute attention + store weights for next step ──
                attack_intents = {}
                
                for agent in alive:
                    nh_h, nh_ids, nh_tribes = self.world._find_neighbors(agent)
                    
                    # Get neighbor attention weights from previous step
                    nh_attn_weights = [prev_attn[nid] for nid in nh_ids]
                    
                    # Compute prey proximity for gate
                    prey_proximity = self.world.get_prey_proximity(agent)
                    
                    # decide() uses prev-step weights + prey proximity; sets _last_attn_weights + _last_gate_value
                    action = agent.decide(nh_h, neighbor_attn_weights=nh_attn_weights, prey_proximity=prey_proximity)
                    
                    # Track attention
                    if nh_h:
                        _, _, aw, gate_val = agent.attn.attend(agent.h, nh_h, nh_attn_weights, prey_proximity)
                        if len(aw) > 1:
                            H = -np.sum(aw * np.log(aw + 1e-10))
                            attn_entropy_list.append(H)
                            attn_losses.append(H)
                            attn_weights_list.append(aw)
                    
                    # Track gate statistics
                    gate_values.append(agent._last_gate_value)
                    if agent._last_gate_value > 0.5:
                        recursive_active_count += 1
                        prey_proximity_when_active.append(prey_proximity)
                    else:
                        prey_proximity_when_inactive.append(prey_proximity)
                    
                    # Store attention weights for next step's recursive attention
                    prev_attn[agent.id] = agent._last_attn_weights
                    
                    # Track attack intention
                    if action[4] > 0:
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
                prey_attacks = {}
                for agent_id, (prey_id, agent) in attack_intents.items():
                    prey_attacks.setdefault(prey_id, []).append(agent)
                
                for prey_id, attackers in prey_attacks.items():
                    prey = self.world.prey[prey_id]
                    if prey.energy <= 0:
                        continue
                    
                    tribe_attacks = {}
                    for a in attackers:
                        tribe_attacks.setdefault(a.tribe_id, []).append(a)
                    
                    coord_successful = False
                    for tribe_id, tribe_attacker_list in tribe_attacks.items():
                        if len(tribe_attacker_list) >= self.COORDINATION_REQUIRED:
                            for a in tribe_attacker_list:
                                a.energy += 100.0
                                a.large_prey_captured += 1
                            prey.energy = 0
                            coord_successful = True
                            break
                    
                    if not coord_successful:
                        for a in attackers:
                            a.failed_attacks += 1
                            a.energy -= 5.0
                
                # ── Move agents ──
                for agent in alive:
                    # Need to re-get action for move (lost after attack check)
                    nh_h2, nh_ids2, _ = self.world._find_neighbors(agent)
                    nh_attn2 = [prev_attn[nid] for nid in nh_ids2]
                    prey_proximity2 = self.world.get_prey_proximity(agent)
                    action = agent.decide(nh_h2, neighbor_attn_weights=nh_attn2, prey_proximity=prey_proximity2)
                    
                    speed = 0.5 + 0.5 * action[2]
                    agent.x = (agent.x + action[0] * speed + GRID_SIZE) % GRID_SIZE
                    agent.y = (agent.y + action[1] * speed + GRID_SIZE) % GRID_SIZE
                    agent.energy -= 0.5
                    agent.age += 1
                    
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
                        t_ = threats[self.rng.randint(len(threats))]
                        ddx = p.x - t_.x; ddy = p.y - t_.y
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
            
            raw_fit = raw_food + large_caps * 50.0 - failed * 5.0
            
            total_large = int(large_caps.sum())
            total_failed = int(failed.sum())
            
            mean_attn_entropy = np.mean(attn_entropy_list) if attn_entropy_list else 0.0
            mean_attn_loss = np.mean(attn_losses) if attn_losses else 0.0
            
            # Gating statistics
            mean_gate = np.mean(gate_values) if gate_values else 0.0
            total_steps = len(gate_values)
            recursive_usage = recursive_active_count / max(total_steps, 1)
            mean_prox_active = np.mean(prey_proximity_when_active) if prey_proximity_when_active else 0.0
            mean_prox_inactive = np.mean(prey_proximity_when_inactive) if prey_proximity_when_inactive else 0.0
            
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
                "large_prey_captured": total_large,
                "failed_attacks": total_failed,
                "mean_attn_entropy": float(mean_attn_entropy),
                "mean_attn_loss": float(mean_attn_loss),
                "mean_signals_sent": float(sig_sent.mean()) if len(sig_sent) else 0.0,
                "mean_gate_value": float(mean_gate),
                "recursive_usage_rate": float(recursive_usage),
                "prey_prox_when_active": float(mean_prox_active),
                "prey_prox_when_inactive": float(mean_prox_inactive),
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
                      f"Gate={mean_gate:.3f} RecUse={recursive_usage:.2f} | "
                      f"P(active)={mean_prox_active:.2f} P(inact)={mean_prox_inactive:.2f} | "
                      f"⏱ {elapsed:.0f}s",
                      flush=True)
            
            # Breed
            self.agents = self._breed(alive, fitnesses, gen, generations)
            self.world.agents = self.agents
            self.world._spawn_food(2)
            
            # Reset attention history for new population
            prev_attn = {a.id: np.array([], dtype=np.float32) for a in self.agents}
            
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
            os.makedirs("results/phase8a", exist_ok=True)
            with open("results/phase8a/generation_log.json", "w") as f:
                json.dump(self.generation_log, f, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        print(f"\nDone in {time.time()-t0:.0f}s")
        return self.generation_log
