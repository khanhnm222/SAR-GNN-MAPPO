"""Unified controller interface so evaluation/plots code can drive any of the
7 methods (2 heuristics + MADDPG + 4 MAPPO encoder variants) identically:
`controller.reset()` at episode start, `controller.act(env, obs_dict)` per step.

IMPORTANT (fixed 2026-09-03) — evaluation must SAMPLE from the policy, not take
argmax. The trained policies are high-entropy (H ~ 1.4-2.3 nats out of
ln(13)=2.565 after 500k steps), so argmax collapses them onto a single action:
measured on results_stage7_argmax_eval/easy/gnn_mappo/seed_0, argmax picked
action 11 (FAST_FORWARD) on 480/480 agent-steps. FAST_FORWARD advances along
the UAV's *stored heading* and never changes it, so every UAV flew north in
formation until it hit the map edge and then froze for the rest of the episode.
That made every learned method score far below Random Walk and produced replay
videos in which nothing moves.

Sampling matches the distribution PPO actually optimized (the training-time
reward is measured under sampling) and lifts Easy VDR from ~0.16 to ~0.59.
Set `deterministic=True` only to reproduce the old (broken) numbers.
"""
from __future__ import annotations

import numpy as np
import torch

from algorithms.heuristics import GreedyPolicy, RandomWalkPolicy


class HeuristicController:
    def __init__(self, kind: str, seed: int = 0):
        self.policy = RandomWalkPolicy(seed=seed) if kind == "random_walk" else GreedyPolicy(seed=seed)

    def reset(self):
        pass

    def act(self, env, obs_dict):
        return {a: self.policy.act(o) for a, o in obs_dict.items()}


class MAPPOController:
    """Rollout of a trained GNNActorCritic policy.

    `deterministic=False` (default) samples a ~ pi(.|o), i.e. the same policy
    PPO optimised. `deterministic=True` takes the argmax — kept only for
    ablation/back-compat, see the module docstring for why it is not the
    default.
    """

    def __init__(self, policy, hidden_dim: int, device: str = "cpu",
                 deterministic: bool = False, seed: int | None = None):
        self.policy = policy
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.hidden = {}
        if seed is not None:
            self._gen = torch.Generator(device="cpu").manual_seed(int(seed))
        else:
            self._gen = None

    def reset(self):
        self.hidden = {}

    @torch.no_grad()
    def act(self, env, obs_dict):
        graph = env.build_communication_graph()
        alive_idx = graph["alive_idx"]
        if not alive_idx:
            return {}
        uids = [env.uavs[i].uid for i in alive_idx]
        agent_names = [env.agents[i] for i in alive_idx]
        x = torch.tensor(np.stack([obs_dict[a] for a in agent_names]), dtype=torch.float32, device=self.device)
        h_prev = torch.stack([
            self.hidden.get(uid, torch.zeros(self.hidden_dim, device=self.device)) for uid in uids
        ])
        h = self.policy.encode(x, graph["edge_index"].to(self.device), graph["edge_attr"].to(self.device), h_prev)
        for uid, hi in zip(uids, h):
            self.hidden[uid] = hi.detach()
        logits = self.policy.actor.net(h)
        if self.deterministic:
            actions = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1, generator=self._gen).squeeze(-1)
        return {a: int(actions[i].item()) for i, a in enumerate(agent_names)}


class MADDPGController:
    """Rollout of a trained MADDPG actor. Same argmax-vs-sampling caveat as
    MAPPOController: `deterministic=False` samples from the actor's softmax."""

    def __init__(self, trainer, deterministic: bool = False, seed: int | None = None):
        self.trainer = trainer
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, env, obs_dict):
        if self.deterministic:
            return self.trainer.act_greedy(obs_dict)
        return self.trainer.act_sampled(obs_dict, self.rng)
