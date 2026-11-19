"""Two coordinated, non-learned search baselines that read the same belief maps.

  * FrontierController: frontier-based exploration (Yamauchi, CIRA 1997). Each UAV
    flies to the nearest frontier cell of its own belief map, i.e. an unvisited
    cell next to a visited one.
  * BayesController: myopic Bayesian search in the spirit of Bourgault et al.
    (IROS 2003). Each UAV flies to the cell that maximises the victim probability
    its sensor footprint would cover there, divided by the travel distance.

Both are decentralised. A UAV reads only its own belief map (which the belief
channel keeps merged) and coordinates through the message channel: UAVs are
processed in uid order, and a UAV avoids targets within 2 * r_sense of the targets
already claimed this step by lower-uid UAVs it is linked to in the message graph,
and avoids moving next to their positions. They therefore use both channels, so a
message cut affects them, unlike Greedy.

Movement uses the eight one-cell moves and HOVER only: FAST_FORWARD doubles the
energy cost and RETURN_BASE is never needed in these episodes. A move is rejected
if the target cell is blocked by terrain or within SAFETY_DIST of a linked
neighbour; the best remaining move by distance to the target is taken.
"""
from __future__ import annotations

import numpy as np

from sar_env.sar_parallel_env import SAFETY_DIST

_MOVES = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0), 4: (1, 1), 5: (-1, 1), 6: (1, -1), 7: (-1, -1)}
HOVER = 10


def _frontier_mask(visited: np.ndarray) -> np.ndarray:
    v = visited > 0.5
    grown = v.copy()
    grown[1:, :] |= v[:-1, :]
    grown[:-1, :] |= v[1:, :]
    grown[:, 1:] |= v[:, :-1]
    grown[:, :-1] |= v[:, 1:]
    return grown & ~v


def _box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Sum of `a` over the (2r+1)x(2r+1) window centred on every cell (zero padding)."""
    p = np.pad(a, r + 1).cumsum(0).cumsum(1)
    k = 2 * r + 1
    return p[k:, k:] - p[:-k, k:] - p[k:, :-k] + p[:-k, :-k]


class _CoordinatedSearch:
    replan_every = 1

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.targets = {}
        self.age = {}

    def reset(self):
        self.targets = {}
        self.age = {}

    # ---------------------------------------------------------------- planning
    def _candidates(self, env, i):
        raise NotImplementedError

    def _pick(self, env, i, claimed):
        cand, score = self._candidates(env, i)
        if cand.size == 0:
            return None
        r_claim = 2.0 * env.cfg.r_sense
        h = env._beliefs[i].prob.shape[1]
        cx, cy = cand // h, cand % h
        ok = np.ones(cand.size, dtype=bool)
        for tx, ty in claimed:
            ok &= (cx - tx) ** 2 + (cy - ty) ** 2 > r_claim ** 2
        if ok.any():
            cand, score, cx, cy = cand[ok], score[ok], cx[ok], cy[ok]
        best = int(np.argmax(score + self.rng.uniform(0, 1e-9, size=score.size)))
        return int(cx[best]), int(cy[best])

    def _valid(self, env, i, target, claimed):
        if target is None:
            return False
        tx, ty = target
        if env._beliefs[i].visited[tx, ty] > 0.5 or env.terrain.is_blocked(tx, ty):
            return False
        r_claim = 2.0 * env.cfg.r_sense
        return all((tx - cx) ** 2 + (ty - cy) ** 2 > r_claim ** 2 for cx, cy in claimed)

    # ---------------------------------------------------------------- acting
    def act(self, env, obs_dict):
        graph = env.build_communication_graph()
        alive_idx = graph["alive_idx"]
        if not alive_idx:
            return {}
        ei = graph["edge_index"]
        nbrs = {k: set() for k in alive_idx}
        if ei.shape[1]:
            for a, b in zip(ei[0].tolist(), ei[1].tolist()):
                nbrs[alive_idx[a]].add(alive_idx[b])
        planned_pos, claimed_by = {}, {}
        actions = {}
        for i in sorted(alive_idx, key=lambda k: env.uavs[k].uid):
            linked_lower = [j for j in nbrs[i] if j in claimed_by]
            claimed = [claimed_by[j] for j in linked_lower]
            uid = env.uavs[i].uid
            tgt = self.targets.get(uid)
            self.age[uid] = self.age.get(uid, 0) + 1
            if not self._valid(env, i, tgt, claimed) or self.age[uid] >= self.replan_every:
                tgt = self._pick(env, i, claimed)
                self.targets[uid] = tgt
                self.age[uid] = 0
            if tgt is not None:
                claimed_by[i] = tgt
            avoid = [planned_pos[j] for j in linked_lower if j in planned_pos]
            avoid += [(env.uavs[j].x, env.uavs[j].y) for j in nbrs[i] if j not in planned_pos]
            a, nxt = self._move(env, i, tgt, avoid)
            planned_pos[i] = nxt
            actions[env.agents[i]] = a
        return actions

    def _move(self, env, i, tgt, avoid):
        u = env.uavs[i]
        here = (u.x, u.y)
        if tgt is None:
            return HOVER, here
        best, best_d = HOVER, (tgt[0] - u.x) ** 2 + (tgt[1] - u.y) ** 2
        best_pos = here
        for a, (dx, dy) in _MOVES.items():
            nx, ny = u.x + dx, u.y + dy
            if not (0 <= nx <= env.cfg.width - 1 and 0 <= ny <= env.cfg.height - 1):
                continue
            if env.terrain.is_blocked(int(round(nx)), int(round(ny))):
                continue
            if any((nx - px) ** 2 + (ny - py) ** 2 < SAFETY_DIST ** 2 for px, py in avoid):
                continue
            d = (tgt[0] - nx) ** 2 + (tgt[1] - ny) ** 2
            if d < best_d - 1e-9:
                best, best_d, best_pos = a, d, (nx, ny)
        return best, best_pos


class FrontierController(_CoordinatedSearch):
    """Nearest frontier cell of the UAV's own belief map (Yamauchi 1997)."""

    def _candidates(self, env, i):
        bm = env._beliefs[i]
        mask = _frontier_mask(bm.visited) & (env.terrain.occupancy < 0.5)
        if not mask.any():
            mask = (bm.visited < 0.5) & (env.terrain.occupancy < 0.5)
        cand = np.flatnonzero(mask.ravel())
        if cand.size == 0:
            return cand, np.zeros(0)
        u = env.uavs[i]
        h = bm.prob.shape[1]
        d2 = (cand // h - u.x) ** 2 + (cand % h - u.y) ** 2
        return cand, -d2.astype(np.float64)


class BayesController(_CoordinatedSearch):
    """Expected detections of the sensor footprint at a cell per unit travel distance."""

    replan_every = 5

    def _candidates(self, env, i):
        bm = env._beliefs[i]
        r = int(round(env.cfg.r_sense))
        gain = _box_sum(bm.prob.astype(np.float64), r)
        free = (env.terrain.occupancy < 0.5) & (bm.visited < 0.5)
        cand = np.flatnonzero(free.ravel())
        if cand.size == 0:
            return cand, np.zeros(0)
        u = env.uavs[i]
        h = bm.prob.shape[1]
        d = np.sqrt((cand // h - u.x) ** 2 + (cand % h - u.y) ** 2)
        return cand, gain.ravel()[cand] / (1.0 + d / (2.0 * r))
