"""MAPPO (Multi-Agent PPO) trainer under the CTDE paradigm (muc 1.5.4), used
for all four encoder variants in Bang 5 (MAPPO-MLP, MAPPO-GCN, MAPPO-GAT,
GNN-MAPPO) — only `models.build_encoder(name, ...)` changes between methods,
the training loop, PPO-clip objective and GAE estimator (Bang 6 hyperparams)
are shared, ensuring the "so sanh cong bang" (fair comparison) requirement.

Corrections applied 2026-09-03 after reviewing Giai doan 1-7 (see
EXPERIMENT_LOG.md). Every one of them is toggleable so the earlier runs stay
reproducible via `MAPPOConfig(legacy=True)`:

  * per-agent advantage — the old code ran a scalar GAE over timesteps and
    handed the identical advantage to all N UAVs, so no gradient could
    distinguish their behaviour;
  * separate actor/critic encoders + return normalisation — the shared encoder
    made value_coef*value_loss exceed |actor_loss| by 22-26x;
  * linear learning-rate and entropy-coefficient annealing — the 2M-step run
    oscillated instead of converging, which is what an un-annealed entropy
    bonus plus a constant LR looks like;
  * the critic keeps its own recurrent state (`h_prev_critic`). It previously
    shared the actor's, so the value the update regressed was not the value
    that produced the stored advantage.

Truncated BPTT (added 2026-09-05). A recurrent encoder used to be updated from
the stored h_prev of each timestep with the timesteps shuffled, so
backpropagation through time had length 1: the GRU carried state but never
received a gradient for carrying it. Measured cost of that on Easy: GNN-MAPPO
(DGAT with GRU) scored 0.890 against 0.910 for the same architecture without
the GRU, p = 0.0116 -- the "temporal memory" the thesis proposes was actively
harmful purely because of how it was trained.

`bptt_len` (default 8) now chunks each rollout into contiguous within-episode
sequences, shuffles the CHUNKS, and rolls the encoder forward through each
chunk carrying the hidden state WITH gradient. Only recurrent encoders take
this path; MLP/GCN/GAT keep the original flat, faster update, so the fair
comparison of Bang 5 is preserved. `bptt_len=1` restores the old behaviour.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.actor_critic import GNNActorCritic
from training.rollout_buffer import (RolloutBuffer, RunningNorm, Transition,
                                      collate_transitions)


class MAPPOConfig:
    def __init__(self, **kwargs):
        legacy = kwargs.get("legacy", False)
        self.hidden_dim = kwargs.get("hidden_dim", 64)
        self.lr = kwargs.get("lr", 3e-4)
        self.gamma = kwargs.get("gamma", 0.99)
        self.gae_lambda = kwargs.get("gae_lambda", 0.95)
        self.clip_eps = kwargs.get("clip_eps", 0.2)
        self.ppo_epochs = kwargs.get("ppo_epochs", 4)          # reduced from Bang 6's 10 for CPU POC
        self.minibatch_size = kwargs.get("minibatch_size", 256)  # in timesteps, not 4096 (POC-scale)
        self.entropy_coef = kwargs.get("entropy_coef", 0.01)
        self.value_coef = kwargs.get("value_coef", 0.5)
        self.max_grad_norm = kwargs.get("max_grad_norm", 0.5)
        self.rollout_len = kwargs.get("rollout_len", 512)
        self.critic_pool = kwargs.get("critic_pool", "mean")

        # ---- corrections (all default ON; legacy=True restores Giai doan 6/7)
        self.per_agent_critic = kwargs.get("per_agent_critic", not legacy)
        self.separate_critic_encoder = kwargs.get("separate_critic_encoder", not legacy)
        self.normalize_returns = kwargs.get("normalize_returns", not legacy)
        self.anneal_lr = kwargs.get("anneal_lr", not legacy)
        self.entropy_coef_final = kwargs.get("entropy_coef_final",
                                              0.001 if not legacy else self.entropy_coef)
        self.total_env_steps = kwargs.get("total_env_steps", 500_000)
        # Truncated BPTT length for recurrent encoders (1 = legacy, no BPTT).
        self.bptt_len = kwargs.get("bptt_len", 1 if legacy else 8)


class MAPPOTrainer:
    def __init__(self, env_factory, encoder, obs_dim: int, n_actions: int = 13,
                 config: MAPPOConfig | None = None, device: str = "cpu", seed: int = 0):
        self.cfg = config or MAPPOConfig()
        self.device = torch.device(device)
        self.policy = GNNActorCritic(
            encoder, self.cfg.hidden_dim, n_actions,
            critic_pool=self.cfg.critic_pool,
            per_agent_critic=self.cfg.per_agent_critic,
            separate_critic_encoder=self.cfg.separate_critic_encoder,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.ret_norm = RunningNorm() if self.cfg.normalize_returns else None
        self.env_factory = env_factory
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.env = env_factory(seed=seed)
        self.obs, _ = self.env.reset(seed=seed)
        self.hidden = {}       # uid -> actor hidden vector
        self.hidden_c = {}     # uid -> critic hidden vector
        self.global_step = 0

    # ---------------------------------------------------------------- helpers
    def _progress(self) -> float:
        return min(1.0, self.global_step / max(self.cfg.total_env_steps, 1))

    def _current_entropy_coef(self) -> float:
        if not self.cfg.anneal_lr:
            return self.cfg.entropy_coef
        p = self._progress()
        return self.cfg.entropy_coef + p * (self.cfg.entropy_coef_final - self.cfg.entropy_coef)

    def _apply_lr(self):
        if not self.cfg.anneal_lr:
            return
        lr = self.cfg.lr * (1.0 - self._progress())
        for g in self.optimizer.param_groups:
            g["lr"] = max(lr, self.cfg.lr * 0.05)

    def _denorm(self, v: torch.Tensor) -> torch.Tensor:
        return self.ret_norm.denormalize(v) if self.ret_norm else v

    # ------------------------------------------------------------------ acting
    def _current_graph_inputs(self):
        graph = self.env.build_communication_graph()
        alive_idx = graph["alive_idx"]
        uids = [self.env.uavs[i].uid for i in alive_idx]
        agent_names = [self.env.agents[i] for i in alive_idx]
        if not uids:
            empty = torch.zeros((0, self.cfg.hidden_dim), device=self.device)
            return graph, uids, agent_names, torch.zeros((0, 1), device=self.device), empty, empty
        x = torch.tensor(np.stack([self.obs[a] for a in agent_names]), dtype=torch.float32,
                          device=self.device)
        zero = torch.zeros(self.cfg.hidden_dim, device=self.device)
        h_prev = torch.stack([self.hidden.get(uid, zero) for uid in uids])
        h_prev_c = torch.stack([self.hidden_c.get(uid, zero) for uid in uids])
        return graph, uids, agent_names, x, h_prev, h_prev_c

    @torch.no_grad()
    def collect_rollout(self, n_steps: int) -> RolloutBuffer:
        buffer = RolloutBuffer()
        for _ in range(n_steps):
            graph, uids, agent_names, x, h_prev, h_prev_c = self._current_graph_inputs()
            if len(uids) == 0:
                self._reset_env()
                continue

            ei = graph["edge_index"].to(self.device)
            ea = graph["edge_attr"].to(self.device)
            h = self.policy.encode(x, ei, ea, h_prev)
            hc = self.policy.encode_critic(x, ei, ea, h_prev_c)
            action, log_prob, _ = self.policy.act(h)
            batch_index = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
            values = self.policy.value(hc, batch_index, 1)
            if not self.cfg.per_agent_critic:
                values = values.expand(x.shape[0])
            values = self._denorm(values)

            for uid, hi, hci in zip(uids, h, hc):
                self.hidden[uid] = hi.detach()
                self.hidden_c[uid] = hci.detach()

            actions_dict = {a: int(action[i].item()) for i, a in enumerate(agent_names)}
            next_obs, rewards, terms, truncs, infos = self.env.step(actions_dict)
            reward_vec = torch.tensor([float(rewards.get(a, 0.0)) for a in agent_names],
                                       dtype=torch.float32)
            done = (len(self.env.agents) == 0)

            buffer.add(Transition(
                x=x.cpu(), edge_index=graph["edge_index"], edge_attr=graph["edge_attr"],
                h_prev=h_prev.cpu(), h_prev_critic=h_prev_c.cpu(), uids=uids, actions=action.cpu(),
                log_probs_old=log_prob.detach().cpu(), values_old=values.detach().cpu(),
                rewards=reward_vec, done=done,
            ))
            self.global_step += 1

            if done:
                self._reset_env()
            else:
                self.obs = next_obs

        # bootstrap values for UAVs whose trajectory is still running
        last_values: dict[int, float] = {}
        if len(self.env.agents) > 0:
            graph, uids, agent_names, x, h_prev, h_prev_c = self._current_graph_inputs()
            if uids:
                hc = self.policy.encode_critic(x, graph["edge_index"].to(self.device),
                                                graph["edge_attr"].to(self.device), h_prev_c)
                batch_index = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
                v = self.policy.value(hc, batch_index, 1)
                if not self.cfg.per_agent_critic:
                    v = v.expand(x.shape[0])
                v = self._denorm(v)
                last_values = {uid: float(v[i]) for i, uid in enumerate(uids)}

        buffer.compute_gae_per_agent(self.cfg.gamma, self.cfg.gae_lambda, last_values)
        if self.ret_norm is not None:
            self.ret_norm.update(buffer.all_returns())
        buffer.normalize_advantages()
        return buffer

    def _reset_env(self):
        self.seed += 1
        self.obs, _ = self.env.reset(seed=self.seed)
        self.hidden = {}
        self.hidden_c = {}

    # ------------------------------------------------------------------- PPO
    def update(self, buffer: RolloutBuffer) -> dict:
        if self.policy.use_gru and self.cfg.bptt_len > 1:
            return self._update_recurrent(buffer)
        return self._update_flat(buffer)

    # ---------------------------------------------------------- truncated BPTT
    def _chunks(self, buffer: RolloutBuffer) -> list[list[int]]:
        """Split the rollout into contiguous within-episode index runs of at
        most `bptt_len` timesteps. A chunk never crosses an episode boundary,
        so the hidden state carried through it is always valid."""
        out, cur = [], []
        for t, tr in enumerate(buffer.transitions):
            cur.append(t)
            if tr.done or len(cur) == self.cfg.bptt_len:
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
        return out

    def _update_recurrent(self, buffer: RolloutBuffer) -> dict:
        chunks = self._chunks(buffer)
        ent_coef = self._current_entropy_coef()
        self._apply_lr()
        stats = {"actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0, "n_updates": 0}
        mb_chunks = max(1, self.cfg.minibatch_size // self.cfg.bptt_len)
        order = np.arange(len(chunks))

        for _ in range(self.cfg.ppo_epochs):
            self.rng.shuffle(order)
            for start in range(0, len(order), mb_chunks):
                sel = [chunks[i] for i in order[start:start + mb_chunks]]
                loss, s_ = self._recurrent_minibatch_loss(buffer, sel, ent_coef)
                if loss is None:
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                for k, v in s_.items():
                    stats[k] += v
                stats["n_updates"] += 1

        for k in ("actor_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
            stats[k] /= max(stats["n_updates"], 1)
        stats["entropy_coef"] = ent_coef
        stats["lr"] = self.optimizer.param_groups[0]["lr"]
        stats["bptt_len"] = self.cfg.bptt_len
        return stats

    def _recurrent_minibatch_loss(self, buffer: RolloutBuffer, sel: list[list[int]],
                                   ent_coef: float):
        """Roll every chunk in `sel` forward together, one timestep at a time.

        At step k the k-th transition of each chunk is collated into a single
        graph batch, so the cost is `bptt_len` forward passes per minibatch
        regardless of how many chunks it holds. Hidden state is carried per
        (chunk, uid) WITH gradient; a UAV appearing for the first time inside
        a chunk starts from its stored (detached) rollout hidden state.
        """
        dev = self.device
        hid_a: dict = {}
        hid_c: dict = {}
        lp_new, lp_old, ent, vals, rets, advs = [], [], [], [], [], []
        max_len = max(len(c) for c in sel)

        for k in range(max_len):
            members = [(ci, c[k]) for ci, c in enumerate(sel) if k < len(c)]
            if not members:
                continue
            subset = [buffer.transitions[t] for _, t in members]
            batch = collate_transitions(subset)

            x = batch["x"].to(dev)
            edge_index = batch["edge_index"].to(dev)
            edge_attr = batch["edge_attr"].to(dev)
            node_to_graph = batch["node_to_graph"].to(dev)
            num_graphs = batch["num_graphs"]

            ha, hc, keys = [], [], []
            for ci, t in members:
                tr = buffer.transitions[t]
                for pos, uid in enumerate(tr.uids):
                    key = (ci, uid)
                    keys.append(key)
                    ha.append(hid_a.get(key, tr.h_prev[pos].to(dev)))
                    hc.append(hid_c.get(key, tr.h_prev_critic[pos].to(dev)))
            h_prev = torch.stack(ha)
            h_prev_c = torch.stack(hc)

            h = self.policy.encode(x, edge_index, edge_attr, h_prev)
            hcx = self.policy.encode_critic(x, edge_index, edge_attr, h_prev_c)
            for i, key in enumerate(keys):
                hid_a[key] = h[i]
                hid_c[key] = hcx[i]

            actions = batch["actions"].to(dev)
            l_new, e = self.policy.evaluate_actions(h, actions)
            v = self.policy.value(hcx, node_to_graph, num_graphs)
            if not self.cfg.per_agent_critic:
                v = v[node_to_graph]

            lp_new.append(l_new)
            lp_old.append(batch["log_probs_old"].to(dev))
            ent.append(e)
            vals.append(v)
            rets.append(batch["returns"].to(dev))
            advs.append(batch["advantages"].to(dev))

        if not lp_new:
            return None, {}

        log_probs_new = torch.cat(lp_new)
        log_probs_old = torch.cat(lp_old)
        entropy = torch.cat(ent)
        values = torch.cat(vals)
        returns = torch.cat(rets)
        advantages = torch.cat(advs)

        target = self.ret_norm.normalize(returns) if self.ret_norm else returns
        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.functional.mse_loss(values, target)
        entropy_mean = entropy.mean()
        loss = actor_loss + self.cfg.value_coef * value_loss - ent_coef * entropy_mean

        with torch.no_grad():
            logr = log_probs_new - log_probs_old
            s_ = {"actor_loss": actor_loss.item(), "value_loss": value_loss.item(),
                  "entropy": entropy_mean.item(),
                  "approx_kl": float(((torch.exp(logr) - 1) - logr).mean()),
                  "clip_frac": float(((ratio - 1).abs() > self.cfg.clip_eps).float().mean())}
        return loss, s_

    # -------------------------------------------------------------- flat PPO
    def _update_flat(self, buffer: RolloutBuffer) -> dict:
        n = len(buffer)
        idx = np.arange(n)
        ent_coef = self._current_entropy_coef()
        self._apply_lr()
        stats = {"actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0, "n_updates": 0}

        for _ in range(self.cfg.ppo_epochs):
            self.rng.shuffle(idx)
            for start in range(0, n, self.cfg.minibatch_size):
                mb_idx = idx[start:start + self.cfg.minibatch_size]
                subset = [buffer.transitions[i] for i in mb_idx]
                batch = collate_transitions(subset)

                x = batch["x"].to(self.device)
                edge_index = batch["edge_index"].to(self.device)
                edge_attr = batch["edge_attr"].to(self.device)
                h_prev = batch["h_prev"].to(self.device)
                h_prev_critic = batch["h_prev_critic"].to(self.device)
                actions = batch["actions"].to(self.device)
                log_probs_old = batch["log_probs_old"].to(self.device)
                node_to_graph = batch["node_to_graph"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)
                num_graphs = batch["num_graphs"]

                h = self.policy.encode(x, edge_index, edge_attr, h_prev)
                log_probs_new, entropy = self.policy.evaluate_actions(h, actions)

                # the critic's recurrent state is its own, not the actor's --
                # feeding h_prev here would make the value the update regresses
                # differ from the value that produced the stored advantage
                hc = self.policy.encode_critic(x, edge_index, edge_attr, h_prev_critic)
                values = self.policy.value(hc, node_to_graph, num_graphs)
                if not self.cfg.per_agent_critic:
                    values = values[node_to_graph]

                target = self.ret_norm.normalize(returns) if self.ret_norm else returns

                ratio = torch.exp(log_probs_new - log_probs_old)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, target)
                entropy_mean = entropy.mean()

                loss = actor_loss + self.cfg.value_coef * value_loss - ent_coef * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    logr = log_probs_new - log_probs_old
                    stats["approx_kl"] += float(((torch.exp(logr) - 1) - logr).mean())
                    stats["clip_frac"] += float(((ratio - 1).abs() > self.cfg.clip_eps).float().mean())
                stats["actor_loss"] += actor_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy_mean.item()
                stats["n_updates"] += 1

        for k in ("actor_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
            stats[k] /= max(stats["n_updates"], 1)
        stats["entropy_coef"] = ent_coef
        stats["lr"] = self.optimizer.param_groups[0]["lr"]
        stats["bptt_len"] = 1
        return stats

    def train_iteration(self) -> dict:
        buffer = self.collect_rollout(self.cfg.rollout_len)
        stats = self.update(buffer)
        stats["global_step"] = self.global_step
        stats["mean_reward"] = float(
            torch.cat([t.rewards for t in buffer.transitions]).mean()) if buffer.transitions else 0.0
        return stats
 