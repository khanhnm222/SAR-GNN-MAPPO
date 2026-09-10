"""Actor (decentralized) and Centralized Critic for the CTDE framework
(muc 1.5.4). The critic pools node embeddings with a permutation-invariant
operator (mean or attention pooling) so that it can process a varying number
of UAVs N between episodes without changing its architecture — the condition
required for zero-shot scalability (RQ3, muc 1.5.3).

Two corrections were made after the Giai doan 1-7 review (2026-09-03):

1. The critic now produces a value PER AGENT, V_i = f([h_i || Pool(h)]),
   instead of one scalar for the whole team. With a single team value every
   UAV received the identical advantage at every timestep, so PPO's gradient
   could not distinguish a UAV that was doing useful work from one that was
   idling. Division of labour -- exactly the behaviour the graph is supposed
   to enable -- was therefore unlearnable at any budget. The pooled term is
   still there, so the critic remains centralized and permutation-invariant.

2. `separate_critic_encoder=True` gives the critic its own encoder. Sharing
   one encoder let the value loss dominate: measured over the Giai doan 6
   runs, value_coef x value_loss exceeded |actor_loss| by 22-26x for all three
   architectures, so the representation the actor reads was driven almost
   entirely by the critic's regression target.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def _scatter_sum(h: torch.Tensor, batch_index: torch.Tensor, num_graphs: int) -> torch.Tensor:
    out = torch.zeros(num_graphs, h.shape[-1], device=h.device, dtype=h.dtype)
    return out.index_add_(0, batch_index, h)


def _scatter_mean(h: torch.Tensor, batch_index: torch.Tensor, num_graphs: int) -> torch.Tensor:
    sums = _scatter_sum(h, batch_index, num_graphs)
    counts = torch.zeros(num_graphs, device=h.device, dtype=h.dtype).index_add_(
        0, batch_index, torch.ones(h.shape[0], device=h.device, dtype=h.dtype))
    return sums / counts.clamp(min=1).unsqueeze(-1)


class Actor(nn.Module):
    """Decentralized policy pi_theta(a_i | o_i, h_i) — uses only the local node
    embedding h_i produced by the encoder, no global information at execution."""

    def __init__(self, hidden_dim: int, n_actions: int = 13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        # Small final-layer gain: a near-uniform initial policy is what PPO
        # expects, and it keeps early logits from locking onto one action.
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> Categorical:
        logits = self.net(h)
        return Categorical(logits=logits)


class CentralizedCritic(nn.Module):
    """V_phi(s, G)_i = f_phi([h_i || Pool({h_1..h_N})]) — one value per agent,
    conditioned on the whole graph. Centralized training only; never used at
    decentralized execution time (muc 1.5.4, buoc 5-6)."""

    def __init__(self, hidden_dim: int, pool: str = "mean", per_agent: bool = True):
        super().__init__()
        self.pool = pool
        self.per_agent = per_agent
        if pool == "attention":
            self.att = nn.Linear(hidden_dim, 1)
        in_dim = hidden_dim * 2 if per_agent else hidden_dim
        self.value_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _pool(self, h: torch.Tensor, batch_index: torch.Tensor, num_graphs: int) -> torch.Tensor:
        if self.pool == "mean":
            return _scatter_mean(h, batch_index, num_graphs)
        if self.pool == "attention":
            scores = self.att(h).squeeze(-1)
            score_max = torch.full((num_graphs,), -1e9, device=h.device, dtype=h.dtype)
            score_max = score_max.scatter_reduce(0, batch_index, scores, reduce="amax", include_self=True)
            exp_scores = torch.exp(scores - score_max[batch_index])
            denom = _scatter_sum(exp_scores.unsqueeze(-1), batch_index, num_graphs).clamp(min=1e-8)
            weights = exp_scores.unsqueeze(-1) / denom[batch_index]
            return _scatter_sum(h * weights, batch_index, num_graphs)
        raise ValueError(f"Unknown pool type {self.pool}")

    def forward(self, h: torch.Tensor, batch_index: torch.Tensor, num_graphs: int) -> torch.Tensor:
        pooled = self._pool(h, batch_index, num_graphs)
        if not self.per_agent:
            return self.value_head(pooled).squeeze(-1)          # [num_graphs]
        joint = torch.cat([h, pooled[batch_index]], dim=-1)
        return self.value_head(joint).squeeze(-1)                # [n_nodes]


class GNNActorCritic(nn.Module):
    """Wraps an encoder (MLP / GCN / GAT / DGAT) with the decentralized actor
    and centralized critic. `needs_graph` / `use_gru` are read off the encoder
    so the same trainer code (algorithms/mappo.py) works for every method in
    Bang 5 by only swapping `encoder_name`.
    """

    def __init__(self, encoder: nn.Module, hidden_dim: int, n_actions: int = 13,
                 critic_pool: str = "mean", per_agent_critic: bool = True,
                 separate_critic_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        self.separate_critic_encoder = separate_critic_encoder
        self.critic_encoder = copy.deepcopy(encoder) if separate_critic_encoder else None
        self.actor = Actor(hidden_dim, n_actions)
        self.critic = CentralizedCritic(hidden_dim, pool=critic_pool, per_agent=per_agent_critic)
        self.needs_graph = getattr(encoder, "needs_graph", False)
        self.use_gru = getattr(encoder, "use_gru", False)
        self.per_agent_critic = per_agent_critic

    # ------------------------------------------------------------- encoding
    def encode(self, x, edge_index, edge_attr, h_prev=None):
        return self.encoder(x, edge_index, edge_attr, h_prev)

    def encode_critic(self, x, edge_index, edge_attr, h_prev=None):
        enc = self.critic_encoder if self.critic_encoder is not None else self.encoder
        return enc(x, edge_index, edge_attr, h_prev)

    def actor_parameters(self):
        return list(self.encoder.parameters()) + list(self.actor.parameters())

    def critic_parameters(self):
        params = list(self.critic.parameters())
        if self.critic_encoder is not None:
            params += list(self.critic_encoder.parameters())
        return params

    # -------------------------------------------------------------- acting
    def act(self, h: torch.Tensor):
        dist = self.actor(h)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(self, h: torch.Tensor, actions: torch.Tensor):
        dist = self.actor(h)
        return dist.log_prob(actions), dist.entropy()

    def value(self, h: torch.Tensor, batch_index: torch.Tensor, num_graphs: int):
        return self.critic(h, batch_index, num_graphs)
