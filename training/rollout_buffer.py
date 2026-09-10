"""Rollout storage + GAE for the MAPPO trainer.

Each `Transition` corresponds to one environment timestep. Because UAVs can be
masked out of the graph when their energy is depleted (muc 1.5.3), the number
of nodes `n_i` varies between transitions, so transitions are stored as a
plain Python list rather than a fixed-size tensor and manually collated into
minibatches by `collate_transitions` (avoids relying on PyTorch Geometric's
`Batch` heuristics for the mix of node-level and graph-level fields we need).

GAE is computed PER AGENT (`compute_gae_per_agent`), following each UAV's own
uid through the rollout. The previous scalar version ran one 1-D recursion over
timesteps and broadcast the resulting advantage to every UAV alive at that
step, so all N agents received an identical learning signal and nothing in the
gradient distinguished a UAV that found a victim from one that hovered in an
already-scanned corner. A UAV that runs out of energy simply stops appearing in
the node list, which ends its trajectory with a zero bootstrap — the correct
terminal treatment for that agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Transition:
    x: torch.Tensor            # [n_i, obs_dim]
    edge_index: torch.Tensor   # [2, E_i]
    edge_attr: torch.Tensor    # [E_i, edge_dim]
    h_prev: torch.Tensor       # [n_i, hidden_dim] actor encoder hidden state
    h_prev_critic: torch.Tensor  # [n_i, hidden_dim] critic encoder hidden state
    uids: list                 # len n_i, uav id per node
    actions: torch.Tensor      # [n_i]
    log_probs_old: torch.Tensor  # [n_i]
    values_old: torch.Tensor   # [n_i] per-agent value estimate
    rewards: torch.Tensor      # [n_i] per-agent reward
    done: bool
    advantages: torch.Tensor | None = None  # [n_i]
    returns: torch.Tensor | None = None     # [n_i]


class RolloutBuffer:
    def __init__(self):
        self.transitions: list[Transition] = []

    def add(self, transition: Transition):
        self.transitions.append(transition)

    def __len__(self):
        return len(self.transitions)

    def clear(self):
        self.transitions = []

    # ---------------------------------------------------------------- GAE
    def compute_gae_per_agent(self, gamma: float = 0.99, lam: float = 0.95,
                               last_values: dict[int, float] | None = None):
        """Per-UAV GAE(lambda).

        `last_values[uid]` bootstraps a UAV whose trajectory is still running at
        the end of the rollout. A uid missing from a later timestep (dead, or
        the episode reset) terminates that agent's trajectory with value 0.
        """
        last_values = last_values or {}
        n = len(self.transitions)
        for tr in self.transitions:
            tr.advantages = torch.zeros_like(tr.values_old)
            tr.returns = torch.zeros_like(tr.values_old)

        # uid -> ordered list of (transition index, node position)
        occurrences: dict[int, list[tuple[int, int]]] = {}
        episode_break = [False] * n
        for t, tr in enumerate(self.transitions):
            for pos, uid in enumerate(tr.uids):
                occurrences.setdefault(uid, []).append((t, pos))
            episode_break[t] = tr.done

        for uid, occ in occurrences.items():
            # split the uid's occurrences into contiguous within-episode runs
            runs: list[list[tuple[int, int]]] = []
            current: list[tuple[int, int]] = []
            for k, (t, pos) in enumerate(occ):
                current.append((t, pos))
                is_last = k == len(occ) - 1
                # a run ends at an episode boundary, at a gap in the timeline,
                # or at the end of the rollout
                if episode_break[t] or is_last or occ[k + 1][0] != t + 1:
                    runs.append(current)
                    current = []
            for run in runs:
                t_last, _ = run[-1]
                if episode_break[t_last] or t_last != n - 1:
                    # episode ended, or the UAV vanished (died) mid-rollout
                    next_value = 0.0
                else:
                    next_value = float(last_values.get(uid, 0.0))
                adv = 0.0
                for t, pos in reversed(run):
                    tr = self.transitions[t]
                    v = float(tr.values_old[pos])
                    r = float(tr.rewards[pos])
                    delta = r + gamma * next_value - v
                    adv = delta + gamma * lam * adv
                    tr.advantages[pos] = adv
                    tr.returns[pos] = adv + v
                    next_value = v
                    # once we step past the first element of a run the
                    # bootstrap is the previous value, and the mask is 1 by
                    # construction (a run never crosses an episode boundary)

    def normalize_advantages(self, eps: float = 1e-8):
        advs = torch.cat([t.advantages for t in self.transitions if t.advantages is not None])
        if advs.numel() < 2:
            return
        mean, std = advs.mean(), advs.std()
        for t in self.transitions:
            t.advantages = (t.advantages - mean) / (std + eps)

    def all_returns(self) -> torch.Tensor:
        return torch.cat([t.returns for t in self.transitions if t.returns is not None])


def collate_transitions(subset: list[Transition]) -> dict:
    n_list = [t.x.shape[0] for t in subset]
    x = torch.cat([t.x for t in subset], dim=0)
    h_prev = torch.cat([t.h_prev for t in subset], dim=0)
    h_prev_critic = torch.cat([t.h_prev_critic for t in subset], dim=0)
    actions = torch.cat([t.actions for t in subset], dim=0)
    log_probs_old = torch.cat([t.log_probs_old for t in subset], dim=0)
    values_old = torch.cat([t.values_old for t in subset], dim=0)
    advantages = torch.cat([t.advantages for t in subset], dim=0)
    returns = torch.cat([t.returns for t in subset], dim=0)

    node_to_graph = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(n_list)])

    offset = 0
    edge_index_parts, edge_attr_parts = [], []
    for t, n in zip(subset, n_list):
        if t.edge_index.numel() > 0:
            edge_index_parts.append(t.edge_index + offset)
            edge_attr_parts.append(t.edge_attr)
        offset += n
    edge_index = torch.cat(edge_index_parts, dim=1) if edge_index_parts else torch.zeros((2, 0), dtype=torch.long)
    edge_dim = subset[0].edge_attr.shape[-1] if subset[0].edge_attr.numel() else 6
    edge_attr = torch.cat(edge_attr_parts, dim=0) if edge_attr_parts else torch.zeros((0, edge_dim))

    return {
        "x": x, "edge_index": edge_index, "edge_attr": edge_attr, "h_prev": h_prev,
        "h_prev_critic": h_prev_critic,
        "actions": actions, "log_probs_old": log_probs_old, "values_old": values_old,
        "node_to_graph": node_to_graph,
        "advantages": advantages, "returns": returns, "num_graphs": len(subset),
    }


class RunningNorm:
    """Welford running mean/std, used to normalise the critic's regression
    target. Un-normalised returns in this environment span roughly [0, 40]
    (alpha=20 coverage plus beta=5 per victim), so an un-normalised MSE is
    orders of magnitude larger than the PPO surrogate and, under a shared
    gradient-norm clip, crowds the policy gradient out entirely."""

    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: torch.Tensor):
        x = x.detach().cpu().numpy().astype(np.float64)
        if x.size == 0:
            return
        batch_mean, batch_var, batch_count = x.mean(), x.var(), x.size
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / tot) / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(max(self.var, 1e-6) ** 0.5)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]
