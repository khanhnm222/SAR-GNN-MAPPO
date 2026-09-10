"""MADDPG baseline (Lowe et al., 2017, NeurIPS) — Bang 5, nhom "MARL (khong
do thi)". No GNN encoder: the actor sees only its own local observation, and
the centralized critic sees the *concatenation* of all agents' observations
and actions (no message passing / attention), which is exactly the structural
gap GNN-MAPPO is designed to close (RQ1).

Discrete actions are handled via the Gumbel-Softmax reparameterization, as
recommended in the original MADDPG paper for discrete action spaces. Because
the swarm is homogeneous and shares a single global reward (muc 1.5.4), a
single shared actor and a single shared centralized critic are used across
agents (parameter sharing) instead of N independent actor-critic pairs — this
keeps the parameter count tractable while retaining MADDPG's core mechanism:
off-policy actor-critic learning with a critic conditioned on the joint
observation-action of the whole swarm (unlike MAPPO's on-policy PPO-clip).
"""
from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MADDPGActor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)  # logits


class MADDPGCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, n_agents: int, hidden_dim: int = 128):
        super().__init__()
        in_dim = n_agents * (obs_dim + n_actions)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        b = joint_obs.shape[0]
        x = torch.cat([joint_obs.reshape(b, -1), joint_actions.reshape(b, -1)], dim=-1)
        return self.net(x).squeeze(-1)


class MADDPGConfig:
    def __init__(self, **kwargs):
        self.hidden_dim = kwargs.get("hidden_dim", 64)
        self.lr = kwargs.get("lr", 3e-4)
        self.gamma = kwargs.get("gamma", 0.99)
        self.tau = kwargs.get("tau", 0.01)
        self.buffer_size = kwargs.get("buffer_size", 20000)
        self.batch_size = kwargs.get("batch_size", 128)
        self.warmup_steps = kwargs.get("warmup_steps", 500)
        self.rollout_len = kwargs.get("rollout_len", 512)
        self.gumbel_tau = kwargs.get("gumbel_tau", 1.0)
        self.updates_per_iteration = kwargs.get("updates_per_iteration", 8)


class MADDPGTrainer:
    def __init__(self, env_factory, obs_dim: int, n_actions: int, n_agents: int,
                 config: MADDPGConfig | None = None, device: str = "cpu", seed: int = 0):
        self.cfg = config or MADDPGConfig()
        self.device = torch.device(device)
        self.obs_dim, self.n_actions, self.n_agents = obs_dim, n_actions, n_agents

        self.actor = MADDPGActor(obs_dim, n_actions, self.cfg.hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = MADDPGCritic(obs_dim, n_actions, n_agents, self.cfg.hidden_dim * 2).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

        self.buffer = deque(maxlen=self.cfg.buffer_size)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.env_factory = env_factory
        self.env = env_factory(seed=seed)
        self.obs, _ = self.env.reset(seed=seed)
        self.global_step = 0

    def _joint_obs(self) -> np.ndarray:
        arr = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for i, a in enumerate(self.env.agents):
            if a in self.obs:
                arr[i] = self.obs[a]
        return arr

    @torch.no_grad()
    def _select_actions(self, joint_obs: np.ndarray, explore: bool = True):
        obs_t = torch.tensor(joint_obs, dtype=torch.float32, device=self.device)
        logits = self.actor(obs_t)
        if explore:
            probs = F.gumbel_softmax(logits, tau=self.cfg.gumbel_tau, hard=False)
        else:
            probs = F.softmax(logits, dim=-1)
        onehot = F.one_hot(probs.argmax(dim=-1), self.n_actions).float()
        actions = probs.argmax(dim=-1).cpu().numpy()
        return actions, onehot.cpu().numpy()

    def collect_and_train(self, n_steps: int) -> dict:
        rewards_collected = []
        for _ in range(n_steps):
            joint_obs = self._joint_obs()
            alive_mask = np.array([1.0 if (a in self.env.agents and self.env.uavs[i].alive) else 0.0
                                    for i, a in enumerate(self.env.agents)], dtype=np.float32)
            actions, onehot = self._select_actions(joint_obs, explore=True)

            actions_dict = {a: int(actions[i]) for i, a in enumerate(self.env.agents)
                             if self.env.uavs[i].alive}
            next_obs, rewards, terms, truncs, infos = self.env.step(actions_dict)
            # MADDPG's critic is a single joint critic taking one scalar reward,
            # so it must see the TEAM reward. Taking `next(iter(rewards))` used
            # to be equivalent because every agent got an identical global
            # reward; with ScenarioConfig.per_agent_reward that would silently
            # train the whole swarm on UAV 0's individual reward.
            reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
            done = (len(self.env.agents) == 0)
            rewards_collected.append(reward)

            next_joint_obs = np.zeros_like(joint_obs)
            if not done:
                for i, a in enumerate(self.env.agents):
                    if a in next_obs:
                        next_joint_obs[i] = next_obs[a]

            self.buffer.append((joint_obs, onehot * alive_mask[:, None], reward, next_joint_obs, float(done)))
            self.global_step += 1

            if done:
                self.seed += 1
                self.obs, _ = self.env.reset(seed=self.seed)
            else:
                self.obs = next_obs

        stats = {"critic_loss": 0.0, "actor_loss": 0.0, "n_updates": 0}
        if len(self.buffer) >= max(self.cfg.warmup_steps, self.cfg.batch_size):
            for _ in range(self.cfg.updates_per_iteration):
                u = self._update()
                stats["critic_loss"] += u["critic_loss"]
                stats["actor_loss"] += u["actor_loss"]
                stats["n_updates"] += 1
            for k in ("critic_loss", "actor_loss"):
                stats[k] /= max(stats["n_updates"], 1)

        stats["global_step"] = self.global_step
        stats["mean_reward"] = float(np.mean(rewards_collected))
        return stats

    def _update(self) -> dict:
        idx = self.rng.integers(0, len(self.buffer), size=self.cfg.batch_size)
        batch = [self.buffer[i] for i in idx]
        joint_obs = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        joint_actions = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_joint_obs = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            b, n, _ = next_joint_obs.shape
            next_logits = self.actor_target(next_joint_obs.reshape(b * n, -1))
            next_probs = F.softmax(next_logits, dim=-1).reshape(b, n, -1)
            target_q = self.critic_target(next_joint_obs, next_probs)
            y = rewards + self.cfg.gamma * (1 - dones) * target_q

        q = self.critic(joint_obs, joint_actions)
        critic_loss = F.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        b, n, _ = joint_obs.shape
        logits = self.actor(joint_obs.reshape(b * n, -1))
        probs = F.gumbel_softmax(logits, tau=self.cfg.gumbel_tau, hard=False).reshape(b, n, -1)
        actor_loss = -self.critic(joint_obs, probs).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1 - self.cfg.tau) + sp.data * self.cfg.tau)

    def train_iteration(self) -> dict:
        return self.collect_and_train(self.cfg.rollout_len)

    # ------------------------------------------------------------- inference
    def act_greedy(self, obs_dict: dict) -> dict:
        joint_obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        agent_list = list(obs_dict.keys())
        for i, a in enumerate(agent_list):
            joint_obs[i] = obs_dict[a]
        actions, _ = self._select_actions(joint_obs, explore=False)
        return {a: int(actions[i]) for i, a in enumerate(agent_list)}

    def act_sampled(self, obs_dict: dict, rng: np.random.Generator | None = None) -> dict:
        """Sample a ~ softmax(actor(o)) instead of argmax.

        Same rationale as evaluation/controllers.MAPPOController: an
        undertrained actor's argmax collapses onto one action for every agent
        at every step, which in this environment means "fly in a straight line
        until the map edge, then freeze". Sampling evaluates the stochastic
        policy that training actually optimised.
        """
        rng = rng or self.rng
        agent_list = list(obs_dict.keys())
        joint_obs = np.zeros((len(agent_list), self.obs_dim), dtype=np.float32)
        for i, a in enumerate(agent_list):
            joint_obs[i] = obs_dict[a]
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(joint_obs, device=self.device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        actions = [int(rng.choice(self.n_actions, p=p / p.sum())) for p in probs]
        return {a: actions[i] for i, a in enumerate(agent_list)}
 