"""SARSwarmEnv with test-time communication degradation.

In the base environment a radio link between two alive UAVs exists iff their
distance is at most r_comm, and that link is used by two separate channels:

  * belief channel  - `_merge_beliefs_over_comm_graph` merges the per-UAV belief
    maps over every connected component of the link graph. Every method that
    reads its observation (MLP, GCN, GAT, DGAT and the Greedy heuristic) uses it.
  * message channel - `build_communication_graph` hands the same links to the
    policy as `edge_index`/`edge_attr`. Only the graph encoders (GCN, GAT, DGAT)
    and the coordinated heuristics of baselines.py read it; MLPEncoder ignores
    the edges.

`DegradedSARSwarmEnv` drops links independently per step with probability
`p_belief` on the first channel and `p_message` on the second. Both channels
read the same per-step uniform draw U[t, i, j], so with p_belief == p_message a
link that is down is down for both, and a link that is down at loss p is also
down at every p' > p (the realisations are nested). Setting one probability to
0 and the other to 1 cuts a single channel.

The draw comes from its own generator, seeded from the episode seed and
`link_salt`, never from the environment RNG. With p_belief = p_message = 0 the
environment therefore reproduces SARSwarmEnv step for step, which run_eval.py
checks against the stored eval_metrics.json of every checkpoint.

A second, spatially correlated model shrinks the radio range instead:
`range_scale` = rho keeps only links with distance <= rho * r_comm, on both
channels. Long links fail first, which is what interference or terrain shadowing
does, whereas the independent loss above removes short and long links alike.

Edge features of the links that survive are left unchanged (they are still
normalised by the nominal r_comm), so a surviving message looks exactly like it
did in training; only the set of links shrinks.

Extensions added for the second round (14/10/2026); every one is off by default,
so the conditions of the first round are reproduced unchanged:

  * `burst` = (loss, mean_burst): Gilbert-Elliott two-state channel per
    unordered pair. A link is down while its pair is in the bad state; the chain
    starts from its stationary distribution, the long-run fraction of bad steps
    is `loss` and a bad run lasts `mean_burst` steps on average. Applies to both
    channels, like the independent loss.
  * `delay` = D steps on the belief channel: the map merged within a connected
    component at step t reaches each member at step t + D, with its content as
    of step t. A UAV's own sensing is immediate.
  * `bandwidth` = B cells on the belief channel: per step each UAV receives at
    most B cells of map information that its component knows and it does not,
    nearest cells first. The rest stays pending and is offered again later.
  * `no_death`: a UAV whose energy reaches zero keeps flying. Its observation
    still reports zero energy, which is within the range seen in training,
    because depleted UAVs were removed only after their ratio had reached zero.

Delay and bandwidth are applied to the belief channel only. A GNN message is a
64-dimensional vector per neighbour, so a cell budget does not bind it, and
delaying it would require changing the encoder's forward pass.
"""
from __future__ import annotations

import numpy as np
import torch

from sar_env.entities import UAV
from sar_env.sar_parallel_env import SARSwarmEnv

LINK_SALT = 2_000_003
BURST_SALT = 3_000_017


class _ImmortalUAV(UAV):
    """UAV that is never removed for running out of energy (energy stays at 0)."""

    def apply_action(self, action, width, height, terrain, c0=1.0, c1=2.0):
        cost = super().apply_action(action, width, height, terrain, c0, c1)
        if not self.alive:
            self.alive = True
        return cost


class DegradedSARSwarmEnv(SARSwarmEnv):
    def __init__(self, scenario, seed=None, reward_weights=None,
                 p_belief: float = 0.0, p_message: float = 0.0, range_scale: float = 1.0,
                 burst: tuple[float, float] | None = None, delay: int = 0,
                 bandwidth: int | None = None, no_death: bool = False,
                 link_salt: int = LINK_SALT):
        super().__init__(scenario, seed=seed, reward_weights=reward_weights)
        if not (0.0 <= p_belief <= 1.0 and 0.0 <= p_message <= 1.0):
            raise ValueError("loss probabilities must lie in [0, 1]")
        if not 0.0 < range_scale <= 1.0:
            raise ValueError("range_scale must lie in (0, 1]")
        if burst is not None and not (0.0 < burst[0] < 1.0 and burst[1] >= 1.0):
            raise ValueError("burst = (loss in (0,1), mean burst length >= 1)")
        self.range_scale = float(range_scale)
        self.p_belief = float(p_belief)
        self.p_message = float(p_message)
        self.burst = burst
        self.delay = int(delay)
        self.bandwidth = None if bandwidth is None else int(bandwidth)
        self.no_death = bool(no_death)
        self.link_salt = int(link_salt)
        self._u = None
        self._bad = None
        self._queue = []
        # Link statistics for the current episode, read by run_eval.py.
        self.link_stats = {"belief_links_nominal": 0, "belief_links_kept": 0,
                           "message_links_nominal": 0, "message_links_kept": 0,
                           "message_steps": 0, "message_isolated_node_steps": 0,
                           "message_node_steps": 0, "bw_cells_sent": 0, "bw_cells_pending": 0,
                           "alive_uav_steps": 0}

    # ------------------------------------------------------------------ links
    def _spawn_uavs(self):
        uavs = super()._spawn_uavs()
        if self.no_death:
            uavs = [_ImmortalUAV(**u.__dict__) for u in uavs]
        return uavs

    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        n = self.cfg.n_uavs
        rng = np.random.default_rng([self.link_salt, 0 if seed is None else int(seed)])
        u = rng.random((self.cfg.max_steps + 2, n, n))
        # symmetric: one draw per unordered pair and step
        self._u = np.triu(u, 1) + np.transpose(np.triu(u, 1), (0, 2, 1))
        self._bad = self._burst_states(seed) if self.burst is not None else None
        self._queue = []
        for k in self.link_stats:
            self.link_stats[k] = 0
        return out

    def _burst_states(self, seed):
        """bad[t, i, j]: Gilbert-Elliott state of pair (i, j) at step t, symmetric."""
        loss, mean_burst = self.burst
        b = 1.0 / mean_burst                 # bad -> good per step
        a = loss * b / (1.0 - loss)          # good -> bad per step, so that a / (a + b) = loss
        n, T = self.cfg.n_uavs, self.cfg.max_steps + 2
        rng = np.random.default_rng([BURST_SALT, 0 if seed is None else int(seed)])
        iu = np.triu_indices(n, 1)
        state = rng.random(len(iu[0])) < loss          # stationary start
        bad = np.zeros((T, n, n), dtype=bool)
        for t in range(T):
            bad[t][iu] = state
            r = rng.random(len(state))
            state = np.where(state, r >= b, r < a)
        return bad | np.transpose(bad, (0, 2, 1))

    def _dropped(self, uid_i: int, uid_j: int, p: float) -> bool:
        t = min(self.t, self._u.shape[0] - 1)
        if self._bad is not None and self._bad[t, uid_i, uid_j]:
            return True
        if p <= 0.0:
            return False
        if p >= 1.0:
            return True
        return bool(self._u[t, uid_i, uid_j] < p)

    # --------------------------------------------------------- belief channel
    def _merge_beliefs_over_comm_graph(self):
        """Component merge over the surviving links, optionally delayed or rate-limited."""
        self.link_stats["alive_uav_steps"] += sum(1 for u in self.uavs if u.alive)
        if not self.cfg.local_belief:
            return
        alive = [i for i, u in enumerate(self.uavs) if u.alive]
        components: dict[int, list[int]] = {}
        if len(alive) >= 2:
            r2 = self.cfg.r_comm ** 2
            r2_kept = (self.range_scale * self.cfg.r_comm) ** 2
            parent = {i: i for i in alive}

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            for pos, i in enumerate(alive):
                ui = self.uavs[i]
                for j in alive[pos + 1:]:
                    uj = self.uavs[j]
                    d2 = (ui.x - uj.x) ** 2 + (ui.y - uj.y) ** 2
                    if d2 <= r2:
                        self.link_stats["belief_links_nominal"] += 1
                        if d2 > r2_kept or self._dropped(ui.uid, uj.uid, self.p_belief):
                            continue
                        self.link_stats["belief_links_kept"] += 1
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[rj] = ri
            for i in alive:
                components.setdefault(find(i), []).append(i)

        groups = [m for m in components.values() if len(m) >= 2]
        merged = []
        for members in groups:
            maps = [self._beliefs[i] for i in members]
            prob = maps[0].prob.copy()
            visited = maps[0].visited.copy()
            for bm in maps[1:]:
                np.minimum(prob, bm.prob, out=prob)
                np.maximum(visited, bm.visited, out=visited)
            merged.append((members, prob, visited))

        if self.delay > 0:
            # what is exchanged now arrives D steps later, with its content as of now
            self._queue.append(merged)
            if len(self._queue) > self.delay:
                for members, prob, visited in self._queue.pop(0):
                    for i in members:
                        bm = self._beliefs[i]
                        np.minimum(bm.prob, prob, out=bm.prob)
                        np.maximum(bm.visited, visited, out=bm.visited)
            return

        if self.bandwidth is not None:
            for members, prob, visited in merged:
                for i in members:
                    self._receive_limited(i, prob, visited)
            return

        for members, prob, visited in merged:
            for i in members:
                bm = self._beliefs[i]
                bm.prob[:] = prob
                bm.visited[:] = visited

    def _receive_limited(self, i, prob, visited):
        """Copy at most `bandwidth` cells of the component map into UAV i's map, nearest first."""
        bm = self._beliefs[i]
        new = np.flatnonzero(((prob < bm.prob - 1e-7) | (visited > bm.visited)).ravel())
        if new.size == 0:
            return
        if new.size > self.bandwidth:
            u = self.uavs[i]
            h = bm.prob.shape[1]
            d2 = (new // h - u.x) ** 2 + (new % h - u.y) ** 2
            keep = np.argpartition(d2, self.bandwidth - 1)[:self.bandwidth]
            self.link_stats["bw_cells_pending"] += int(new.size - self.bandwidth)
            new = new[keep]
        self.link_stats["bw_cells_sent"] += int(new.size)
        bp, bv = bm.prob.reshape(-1), bm.visited.reshape(-1)
        bp[new] = np.minimum(bp[new], prob.reshape(-1)[new])
        bv[new] = np.maximum(bv[new], visited.reshape(-1)[new])

    # -------------------------------------------------------- message channel
    def build_communication_graph(self):
        graph = super().build_communication_graph()
        ei = graph["edge_index"]
        n_nodes = graph["n_nodes"]
        n_edges = ei.shape[1]
        self.link_stats["message_steps"] += 1
        self.link_stats["message_node_steps"] += n_nodes
        self.link_stats["message_links_nominal"] += n_edges // 2
        if n_edges and (self.p_message > 0.0 or self.range_scale < 1.0 or self._bad is not None):
            uid = [self.uavs[k].uid for k in graph["alive_idx"]]
            src, dst = ei[0].tolist(), ei[1].tolist()
            # column 4 is the distance; normalised by r_comm when normalize_edge_attr
            dist = graph["edge_attr"][:, 4].tolist()
            r_max = self.range_scale * (1.0 if self.cfg.normalize_edge_attr else self.cfg.r_comm)
            keep = [d <= r_max + 1e-9 and not self._dropped(uid[a], uid[b], self.p_message)
                    for a, b, d in zip(src, dst, dist)]
            mask = torch.tensor(keep, dtype=torch.bool)
            graph["edge_index"] = ei[:, mask]
            graph["edge_attr"] = graph["edge_attr"][mask]
            adj = graph["adjacency"]
            for a, b, k in zip(src, dst, keep):
                if not k:
                    adj[a, b] = 0.0
        kept = graph["edge_index"]
        self.link_stats["message_links_kept"] += kept.shape[1] // 2
        if n_nodes:
            deg = np.bincount(kept[0].numpy(), minlength=n_nodes) if kept.shape[1] else np.zeros(n_nodes)
            self.link_stats["message_isolated_node_steps"] += int((deg == 0).sum())
        return graph
