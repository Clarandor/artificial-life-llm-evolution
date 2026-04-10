"""
Phase 2: Hybrid GA + REINFORCE Evolution Engine (v2 — fixed)
================================================================
Fixes from v1:
  1. Per-step reward tracking (delta food + delta prey), not uniform distribution
  2. Sample from attention distribution (not argmax) as REINFORCE action
  3. RL buffer keyed by batch slot index, properly reset each generation
  4. Survival reward bonus to give gradient signal even without food/prey events

GA channel: W_enc, b_enc, W_dec, b_dec, W_act, b_act — tournament + mutation
REINFORCE channel: W_q, W_k — policy gradient on attention distribution
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass, field
import time

from .agent import (
    HybridAttentionMLP, OBS_DIM, HIDDEN_DIM, ATT_DIM, DEC_DIM, ACTION_DIM, MAX_NEIGHBORS,
    relu, softmax,
)
from .environment import (
    GridWorld, Agent,
    GRID_SIZE, INITIAL_ENERGY,
    N_TRIBES, TRIBE_SIZE, NEIGHBOR_RADIUS,
    NUM_PREDATORS, NUM_PREY,
    FOOD_ENERGY, PREY_CAPTURE_ENERGY,
)


def softmax_rows(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / (e.sum(axis=1, keepdims=True) + 1e-8)


@dataclass
class RLTransition:
    """Single step of attention data for REINFORCE."""
    attn: np.ndarray         # attention weights (K,)
    scores: np.ndarray       # raw logits (K,)
    h_self: np.ndarray       # hidden state (HIDDEN_DIM,)
    neighbor_h: np.ndarray   # neighbor hiddens (K, HIDDEN_DIM)
    sampled_idx: int         # which neighbor was "sampled" from attn distribution
    reward: float            # immediate step reward


class PopulationBatch:
    """Vectorized weight storage — split into GA and RL parts."""

    GA_ATTRS = ['W_enc', 'b_enc', 'W_dec', 'b_dec', 'W_act', 'b_act']
    RL_ATTRS = ['W_q', 'W_k']
    ALL_ATTRS = GA_ATTRS + RL_ATTRS

    WEIGHT_SHAPES = {
        'W_enc': (HIDDEN_DIM, OBS_DIM),
        'b_enc': (HIDDEN_DIM,),
        'W_q':   (ATT_DIM, HIDDEN_DIM),
        'W_k':   (ATT_DIM, HIDDEN_DIM),
        'W_dec': (DEC_DIM, HIDDEN_DIM + ATT_DIM),
        'b_dec': (DEC_DIM,),
        'W_act': (ACTION_DIM, DEC_DIM),
        'b_act': (ACTION_DIM,),
    }

    def __init__(self, pop_size: int, rng: np.random.Generator):
        self.pop_size = pop_size
        for attr, shape in self.WEIGHT_SHAPES.items():
            if len(shape) == 1:
                setattr(self, attr, np.zeros((pop_size, *shape), dtype=np.float32))
            else:
                fan_in = shape[-1]
                setattr(self, attr,
                    rng.normal(0, np.sqrt(2/fan_in), (pop_size, *shape)).astype(np.float32))

    def encode_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        N = obs_batch.shape[0]
        return relu(np.einsum('nhi,ni->nh', self.W_enc[:N], obs_batch) + self.b_enc[:N])

    def attend_single(self, idx: int, h_self: np.ndarray,
                      neighbor_hiddens: np.ndarray,
                      rng: np.random.Generator) -> tuple:
        """Returns (context, attn_weights, raw_scores, sampled_neighbor_idx)."""
        K = neighbor_hiddens.shape[0]
        if K == 0:
            return (np.zeros(ATT_DIM, dtype=np.float32),
                    np.array([], dtype=np.float32),
                    np.array([], dtype=np.float32),
                    -1)

        q = self.W_q[idx] @ h_self
        keys = (self.W_k[idx] @ neighbor_hiddens.T).T
        vals = neighbor_hiddens[:, :ATT_DIM]

        scores = keys @ q / np.sqrt(ATT_DIM)
        attn = softmax(scores)
        context = attn @ vals

        # Sample from attention distribution (for REINFORCE)
        sampled_idx = int(rng.choice(K, p=attn))

        return context, attn, scores, sampled_idx

    def decide_batch(self, h_self_batch: np.ndarray,
                     context_batch: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
        N = h_self_batch.shape[0]
        combined = np.concatenate([h_self_batch, context_batch], axis=1)
        h_dec = relu(np.einsum('ndi,ni->nd', self.W_dec[:N], combined) + self.b_dec[:N])
        logits = np.einsum('nai,ni->na', self.W_act[:N], h_dec) + self.b_act[:N]
        probs = softmax_rows(logits)
        cum = probs.cumsum(axis=1)
        u = rng.random((N, 1), dtype=np.float32)
        actions = (u > cum).sum(axis=1).clip(0, ACTION_DIM - 1)
        return actions.astype(np.int32)


class HybridEvolutionEngine:
    """Two-level selection + REINFORCE for attention weights."""

    STEPS_PER_GEN    = 200
    TOURNAMENT_K     = 3
    MUTATION_SIGMA_0 = 0.05
    MUTATION_SIGMA_F = 0.01
    ELITE_PER_TRIBE  = 1
    PREY_BONUS       = 3.0

    # REINFORCE hyperparams
    RL_LR_0          = 0.005    # initial learning rate
    RL_LR_F          = 0.001    # final learning rate
    RL_GAMMA         = 0.99     # discount factor
    RL_GRAD_CLIP     = 0.5      # gradient clipping norm (tighter)
    RL_ENTROPY_BONUS = 0.005    # entropy regularization
    RL_LR_0          = 0.005    # initial learning rate
    SURVIVAL_REWARD  = 0.01     # small reward per step for being alive

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        seed: Optional[int] = 42,
        verbose: bool = True,
    ):
        self.population_size = N_TRIBES * TRIBE_SIZE
        self.grid_size = grid_size
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.generation_log: List[dict] = []

        self.world = GridWorld(grid_size=grid_size, seed=seed)
        self.world.seed_food(density=0.1)

        self.batch = PopulationBatch(self.population_size, self.rng)

        # RL buffers — per batch slot (list of RLTransition)
        self.rl_buffers: List[List[RLTransition]] = [[] for _ in range(self.population_size * 2)]

        # Clustered tribe initialization
        tribe_centers = []
        for t in range(N_TRIBES):
            cx = int(self.rng.integers(4, grid_size - 4))
            cy = int(self.rng.integers(4, grid_size - 4))
            tribe_centers.append((cx, cy))

        for i in range(self.population_size):
            tribe_id = i // TRIBE_SIZE
            cx, cy = tribe_centers[tribe_id]
            agent = Agent(
                id=i,
                x=(cx + int(self.rng.integers(-3, 4))) % grid_size,
                y=(cy + int(self.rng.integers(-3, 4))) % grid_size,
                energy=INITIAL_ENERGY,
                weights=[],
                tribe_id=tribe_id,
            )
            self.world.add_agent(agent)
        self.world._next_agent_id = self.population_size

    def _get_sigma(self, gen: int, total_gens: int) -> float:
        frac = gen / max(total_gens - 1, 1)
        return self.MUTATION_SIGMA_0 * (1 - frac) + self.MUTATION_SIGMA_F * frac

    def _get_rl_lr(self, gen: int, total_gens: int) -> float:
        frac = gen / max(total_gens - 1, 1)
        return self.RL_LR_0 * (1 - frac) + self.RL_LR_F * frac

    def run(self, generations: int = 300, callback=None):
        pop = self.population_size
        params = HybridAttentionMLP().param_count
        ga_p = HybridAttentionMLP().ga_param_count
        rl_p = HybridAttentionMLP().rl_param_count
        print(f"Starting hybrid evolution: {generations} gen × {self.STEPS_PER_GEN} steps")
        print(f"  Population: {pop} ({N_TRIBES} tribes × {TRIBE_SIZE})")
        print(f"  Grid: {self.grid_size}² | Predators: {NUM_PREDATORS} | Prey: {NUM_PREY}")
        print(f"  Total params: {params} | GA: {ga_p} | REINFORCE: {rl_p}")
        print(f"  GA: mutation σ {self.MUTATION_SIGMA_0}→{self.MUTATION_SIGMA_F} + tournament")
        print(f"  REINFORCE: lr {self.RL_LR_0}→{self.RL_LR_F} | γ={self.RL_GAMMA} | "
              f"entropy={self.RL_ENTROPY_BONUS} | survival_r={self.SURVIVAL_REWARD}")
        print(f"  Prey bonus: {self.PREY_BONUS}× | Neighbor radius: {NEIGHBOR_RADIUS}\n")

        for gen in range(generations):
            t0 = time.time()

            # Clear RL buffers for active slots
            N_cur = len(self.world.agents)
            for idx in range(min(N_cur * 2, len(self.rl_buffers))):
                self.rl_buffers[idx] = []

            # Track per-step deltas for reward computation
            prev_food: Dict[int, int] = {}
            prev_prey: Dict[int, int] = {}
            for a in self.world.agents:
                prev_food[a.id] = a.food_collected
                prev_prey[a.id] = a.prey_captured

            for step in range(self.STEPS_PER_GEN):
                agents = self.world.agents
                if not agents:
                    break
                N = len(agents)
                if N > self.batch.pop_size:
                    self._grow_batch(N)
                while len(self.rl_buffers) < N:
                    self.rl_buffers.append([])

                obs_list = self.world._build_observations()
                if not obs_list:
                    break
                obs_batch = np.stack(obs_list).astype(np.float32)

                hiddens_batch = self.batch.encode_batch(obs_batch)

                context_batch = np.zeros((N, ATT_DIM), dtype=np.float32)
                attn_weights_list = []
                attn_neighbor_ids_list = []
                step_attn_data = []

                for i, agent in enumerate(agents):
                    neighbor_hiddens_list, neighbor_ids = self.world._find_neighbors(agent)
                    if neighbor_hiddens_list:
                        nh = np.stack(neighbor_hiddens_list)
                        ctx, aw, sc, sampled = self.batch.attend_single(
                            i, hiddens_batch[i], nh, self.rng)
                        context_batch[i] = ctx
                        attn_weights_list.append(aw)
                        attn_neighbor_ids_list.append(neighbor_ids)
                        step_attn_data.append((i, aw, sc, hiddens_batch[i].copy(),
                                               nh.copy(), sampled))
                    else:
                        attn_weights_list.append(np.array([], dtype=np.float32))
                        attn_neighbor_ids_list.append([])

                actions = self.batch.decide_batch(hiddens_batch, context_batch, self.rng)

                self.world.step(
                    list(actions),
                    list(hiddens_batch),
                    attn_weights_list,
                    attn_neighbor_ids_list,
                )

                # Record transitions with ACTUAL per-step reward
                for (idx, aw, sc, h_s, nh, sampled) in step_attn_data:
                    if idx >= len(agents):
                        continue
                    agent = agents[idx]
                    if not agent.alive:
                        continue

                    # Compute delta reward for this step
                    food_delta = agent.food_collected - prev_food.get(agent.id, 0)
                    prey_delta = agent.prey_captured - prev_prey.get(agent.id, 0)
                    step_reward = (float(food_delta) +
                                   self.PREY_BONUS * float(prey_delta) +
                                   self.SURVIVAL_REWARD)

                    self.rl_buffers[idx].append(RLTransition(
                        attn=aw, scores=sc, h_self=h_s, neighbor_h=nh,
                        sampled_idx=sampled, reward=step_reward,
                    ))

                # Update prev trackers for next step
                for agent in agents:
                    if agent.alive:
                        prev_food[agent.id] = agent.food_collected
                        prev_prey[agent.id] = agent.prey_captured

            # ── Evaluate ──
            agents = self.world.agents
            raw_food = np.array([a.food_collected for a in agents], dtype=np.float32)
            prey_caps = np.array([a.prey_captured for a in agents], dtype=np.float32)
            fitnesses = raw_food + self.PREY_BONUS * prey_caps

            tribe_fitness = {}
            for a, f in zip(agents, fitnesses):
                tribe_fitness.setdefault(a.tribe_id, []).append(f)
            tribe_avg = {t: np.mean(fs) for t, fs in tribe_fitness.items()}

            total_prey_caps = int(prey_caps.sum())

            attn_entropies = []
            for a in agents:
                if a.attn_step_count > 0:
                    attn_entropies.append(a.attn_entropy_sum / a.attn_step_count)
            mean_attn_entropy = float(np.mean(attn_entropies)) if attn_entropies else 0.0

            # ── REINFORCE update ──
            rl_lr = self._get_rl_lr(gen, generations)
            mean_grad_norm, diag = self._reinforce_update(len(agents), rl_lr)

            elapsed = round(time.time() - t0, 2)

            log_entry = {
                "generation":       gen,
                "population":       len(agents),
                "mean_fitness":     float(fitnesses.mean()) if len(fitnesses) else 0.0,
                "max_fitness":      float(fitnesses.max())  if len(fitnesses) else 0.0,
                "min_fitness":      float(fitnesses.min())  if len(fitnesses) else 0.0,
                "mean_raw_food":    float(raw_food.mean())  if len(raw_food)  else 0.0,
                "mean_prey_cap":    float(prey_caps.mean()) if len(prey_caps) else 0.0,
                "max_prey_cap":     float(prey_caps.max())  if len(prey_caps) else 0.0,
                "total_prey_caps":  total_prey_caps,
                "mean_attn_entropy": mean_attn_entropy,
                "mean_grad_norm":   mean_grad_norm,
                "rl_lr":            rl_lr,
                "diag_adv_mean":    diag.get('adv_mean', 0),
                "diag_adv_std":     diag.get('adv_std', 0),
                "diag_buf_len":     diag.get('buf_len', 0),
                "diag_skip_zero":   diag.get('skip_zero_std', 0),
                "tribe_avg":        tribe_avg,
                "elapsed_sec":      elapsed,
                "sample_hiddens":   np.stack([a.hidden for a in agents[:20]]) if agents else None,
            }
            self.generation_log.append(log_entry)

            if self.verbose and (gen % 10 == 0 or gen < 5):
                sigma = self._get_sigma(gen, generations)
                best_tribe = max(tribe_avg, key=tribe_avg.get) if tribe_avg else -1
                print(
                    f"Gen {gen:4d} | pop={log_entry['population']:4d} | "
                    f"fit μ={log_entry['mean_fitness']:.2f} max={log_entry['max_fitness']:.1f} | "
                    f"prey={total_prey_caps:3d} | "
                    f"H={mean_attn_entropy:.3f} | "
                    f"∇={mean_grad_norm:.4f} lr={rl_lr:.4f} | "
                    f"σ={sigma:.4f} | ⏱ {elapsed}s"
                )

            if callback:
                callback(gen, log_entry)

            # ── GA Breed ──
            self._breed_group_selection(agents, fitnesses, gen, generations)

        print("\nEvolution complete.")
        return self.generation_log

    def _reinforce_update(self, N: int, lr: float) -> tuple:
        """
        Apply REINFORCE gradient to W_q and W_k for each agent slot.
        Uses proper sampled action from attention distribution.
        Returns: (mean_grad_norm, diagnostic_dict)
        """
        grad_norms = []
        adv_means = []
        adv_stds = []
        buf_lens = []
        skip_zero_count = 0

        for i in range(min(N, len(self.rl_buffers))):
            buf = self.rl_buffers[i]
            if len(buf) < 10:  # need minimum trajectory
                continue

            buf_lens.append(len(buf))
            T = len(buf)
            rewards = np.array([t.reward for t in buf], dtype=np.float32)

            # Compute discounted returns
            returns = np.zeros(T, dtype=np.float32)
            G = 0.0
            for t in range(T - 1, -1, -1):
                G = rewards[t] + self.RL_GAMMA * G
                returns[t] = G

            # Baseline = mean return
            baseline = returns.mean()
            advantages = returns - baseline

            # Normalize advantages
            std = advantages.std()
            adv_means.append(float(advantages.mean()))
            adv_stds.append(float(std))
            
            if std > 1e-6:
                advantages = advantages / std
            else:
                skip_zero_count += 1
                continue  # no signal if all returns are identical

            # Accumulate gradients
            grad_Wq = np.zeros_like(self.batch.W_q[i])
            grad_Wk = np.zeros_like(self.batch.W_k[i])

            for t in range(T):
                tr = buf[t]
                attn = tr.attn
                K = len(attn)
                if K == 0 or tr.sampled_idx < 0:
                    continue

                adv = advantages[t]
                a_star = tr.sampled_idx  # SAMPLED from attn distribution

                # Score function gradient: ∂log π / ∂scores_j = (δ_{j,a*} - attn_j)
                target = np.zeros(K, dtype=np.float32)
                target[a_star] = 1.0
                d_scores = adv * (target - attn)

                # Entropy bonus
                log_attn = np.log(attn + 1e-10)
                mean_log = np.sum(attn * log_attn)
                d_entropy = -attn * (log_attn - mean_log)
                d_scores += self.RL_ENTROPY_BONUS * d_entropy

                h_self = tr.h_self
                nh = tr.neighbor_h

                q = self.batch.W_q[i] @ h_self
                keys = (self.batch.W_k[i] @ nh.T).T

                # ∂L/∂W_q
                dL_dq = (d_scores[:, None] * keys).sum(axis=0) / np.sqrt(ATT_DIM)
                grad_Wq += np.outer(dL_dq, h_self)

                # ∂L/∂W_k — only for the sampled neighbor (sparse update, much faster)
                grad_Wk += d_scores[a_star] * np.outer(q, nh[a_star]) / np.sqrt(ATT_DIM)

            grad_Wq /= T
            grad_Wk /= T

            # Clip
            total_norm = np.sqrt(np.linalg.norm(grad_Wq)**2 + np.linalg.norm(grad_Wk)**2)
            if total_norm > self.RL_GRAD_CLIP:
                scale = self.RL_GRAD_CLIP / total_norm
                grad_Wq *= scale
                grad_Wk *= scale
                total_norm = self.RL_GRAD_CLIP

            grad_norms.append(total_norm)

            # Apply gradient ascent
            self.batch.W_q[i] += lr * grad_Wq
            self.batch.W_k[i] += lr * grad_Wk

        diag = {
            'adv_mean': float(np.mean(adv_means)) if adv_means else 0.0,
            'adv_std': float(np.mean(adv_stds)) if adv_stds else 0.0,
            'buf_len': float(np.mean(buf_lens)) if buf_lens else 0.0,
            'skip_zero_std': skip_zero_count,
        }
        return (float(np.mean(grad_norms)) if grad_norms else 0.0, diag)

    def _breed_group_selection(
        self, agents: List[Agent], fitnesses: np.ndarray, gen: int, total_gens: int
    ):
        """GA breeds behavior weights; COPIES attention weights from parent."""
        sigma = self._get_sigma(gen, total_gens)

        tribe_agents = {}
        for i, (agent, fit) in enumerate(zip(agents, fitnesses)):
            tribe_agents.setdefault(agent.tribe_id, []).append((i, fit))

        tribe_ids = sorted(tribe_agents.keys())
        if not tribe_ids:
            return

        tribe_avg_fit = np.array([np.mean([f for _, f in tribe_agents[t]]) for t in tribe_ids])
        shifted = tribe_avg_fit - tribe_avg_fit.min() + 1.0
        tribe_probs = shifted / shifted.sum()

        raw_slots = tribe_probs * self.population_size
        tribe_slots = np.round(raw_slots).astype(int)
        diff = self.population_size - tribe_slots.sum()
        if diff > 0:
            for _ in range(diff):
                tribe_slots[np.argmax(tribe_probs)] += 1
        elif diff < 0:
            for _ in range(-diff):
                idx = np.argmin(tribe_slots)
                tribe_slots[idx] = max(1, tribe_slots[idx] - 1)

        old_weights = {}
        n = len(agents)
        for attr in PopulationBatch.ALL_ATTRS:
            old_weights[attr] = getattr(self.batch, attr)[:n].copy()

        new_slot = 0
        new_agents = []

        tribe_centers = {}
        for t_idx, tid in enumerate(tribe_ids):
            members = tribe_agents[tid]
            xs = [agents[i].x for i, _ in members]
            ys = [agents[i].y for i, _ in members]
            tribe_centers[tid] = (int(np.mean(xs)), int(np.mean(ys)))

        for t_idx, tid in enumerate(tribe_ids):
            slots = int(tribe_slots[t_idx])
            members = tribe_agents[tid]
            member_indices = [i for i, _ in members]
            member_fits = np.array([f for _, f in members])

            if len(members) == 0:
                continue

            elite_count = min(self.ELITE_PER_TRIBE, slots, len(members))
            sorted_local = np.argsort(member_fits)[::-1]

            for e in range(elite_count):
                if new_slot >= self.batch.pop_size:
                    self._grow_batch(new_slot + 1)
                src = member_indices[sorted_local[e]]
                for attr in PopulationBatch.ALL_ATTRS:
                    getattr(self.batch, attr)[new_slot] = old_weights[attr][src]
                cx, cy = tribe_centers.get(tid, (16, 16))
                new_agents.append(Agent(
                    id=self.world._next_agent_id,
                    x=(cx + int(self.rng.integers(-3, 4))) % self.grid_size,
                    y=(cy + int(self.rng.integers(-3, 4))) % self.grid_size,
                    energy=INITIAL_ENERGY,
                    weights=[],
                    tribe_id=tid,
                ))
                self.world._next_agent_id += 1
                new_slot += 1

            for _ in range(slots - elite_count):
                if new_slot >= self.batch.pop_size:
                    self._grow_batch(new_slot + 1)
                k = min(self.TOURNAMENT_K, len(members))
                comp = self.rng.choice(len(members), size=k, replace=False)
                winner_local = comp[np.argmax(member_fits[comp])]
                src = member_indices[winner_local]

                # GA mutation on behavior weights ONLY
                for attr in PopulationBatch.GA_ATTRS:
                    arr = getattr(self.batch, attr)
                    arr[new_slot] = old_weights[attr][src] + \
                        self.rng.normal(0, sigma, old_weights[attr][src].shape).astype(np.float32)

                # COPY attention weights (RL-optimized) without mutation
                for attr in PopulationBatch.RL_ATTRS:
                    getattr(self.batch, attr)[new_slot] = old_weights[attr][src].copy()

                cx, cy = tribe_centers.get(tid, (16, 16))
                new_agents.append(Agent(
                    id=self.world._next_agent_id,
                    x=(cx + int(self.rng.integers(-3, 4))) % self.grid_size,
                    y=(cy + int(self.rng.integers(-3, 4))) % self.grid_size,
                    energy=INITIAL_ENERGY,
                    weights=[],
                    tribe_id=tid,
                ))
                self.world._next_agent_id += 1
                new_slot += 1

        self.world.agents = new_agents
        self.world.grid[:] = 0.0
        self.world.seed_food(density=0.1)
        for prey in self.world.prey_list:
            prey.respawn(self.grid_size, self.rng)

    def _grow_batch(self, new_size: int):
        extra = new_size - self.batch.pop_size
        if extra <= 0:
            return
        for attr, shape in PopulationBatch.WEIGHT_SHAPES.items():
            old = getattr(self.batch, attr)
            pad = np.zeros((extra, *shape), dtype=np.float32)
            setattr(self.batch, attr, np.concatenate([old, pad], axis=0))
        self.batch.pop_size = new_size
