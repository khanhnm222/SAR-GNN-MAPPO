"""Round 3 (15/11/2026): two checks asked for by a pre-submission review of 14/11.

1. Map gap (review issue 10). The paper explains the VDR loss by how late map
   information reaches a UAV, but never measured it. `GapSARSwarmEnv` measures,
   after the belief merge of every step and for every alive UAV i,

       gap_i^t = |S^t \\ S_i^t| / |S^t|

   where S^t is the set of cells scanned by any UAV up to step t (the team union
   map, kept for metrics only) and S_i^t the cells marked visited in UAV i's own
   belief map. gap = 0 means UAV i knows everything the swarm has scanned. The
   same ratio restricted to the 25 x 25 window the policy observes is `wgap`.
   Both are averaged over alive UAVs and steps of an episode. The measurement only
   reads arrays, so with no degradation the episode is identical to the original
   evaluation (checked: p0 VDR equals the stored round-1 values).

2. One hop per step (review issue 2). The base environment merges belief maps over
   the whole connected component in the same step, i.e. relaying is instantaneous.
   With `hop1=True` every UAV merges only with its direct neighbours (distance
   <= r_comm) per step, synchronously from the maps of the previous step, so
   information travels one hop per step and a k-hop path needs k steps.

    python -m comm_robustness.round3 --workers 14
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import torch

from .run_eval import (BASELINES, CKPT_DIR, HIDDEN_DIM, HERE, env_kwargs,
                       n_eval_episodes)
from .degraded_env import DegradedSARSwarmEnv

RAW3 = os.path.join(HERE, "results", "raw3")
WIN = 12  # half-width of the 25 x 25 observation window (LOCAL_MAP_RADIUS 2, block 5)

GAP_CONDS = ["p0", "p50", "p75", "p100", "belief_cut", "message_cut", "r75", "r50", "r25",
             "ge75L10", "ge75L50", "delay5", "delay25", "bw100", "bw25", "hop1"]
HOP_METHODS = ["mappo_mlp", "mappo_gat", "gnn_mappo", "greedy"]


class GapSARSwarmEnv(DegradedSARSwarmEnv):
    def __init__(self, *a, hop1: bool = False, **kw):
        super().__init__(*a, **kw)
        self.hop1 = bool(hop1)
        self.gap_stats = {"gap_sum": 0.0, "wgap_sum": 0.0, "n": 0}

    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self.gap_stats = {"gap_sum": 0.0, "wgap_sum": 0.0, "n": 0}
        return out

    def _merge_beliefs_over_comm_graph(self):
        if self.hop1:
            self._merge_one_hop()
        else:
            super()._merge_beliefs_over_comm_graph()
        self._measure_gap()

    def _merge_one_hop(self):
        self.link_stats["alive_uav_steps"] += sum(1 for u in self.uavs if u.alive)
        if not self.cfg.local_belief:
            return
        alive = [i for i, u in enumerate(self.uavs) if u.alive]
        if len(alive) < 2:
            return
        r2 = self.cfg.r_comm ** 2
        old = {i: (self._beliefs[i].prob.copy(), self._beliefs[i].visited.copy()) for i in alive}
        for i in alive:
            ui = self.uavs[i]
            bm = self._beliefs[i]
            for j in alive:
                if j == i:
                    continue
                uj = self.uavs[j]
                if (ui.x - uj.x) ** 2 + (ui.y - uj.y) ** 2 <= r2:
                    np.minimum(bm.prob, old[j][0], out=bm.prob)
                    np.maximum(bm.visited, old[j][1], out=bm.visited)

    def _measure_gap(self):
        team = self.belief.visited > 0.5
        n_team = int(team.sum())
        if n_team == 0:
            return
        W, H = team.shape
        for i, u in enumerate(self.uavs):
            if not u.alive:
                continue
            own = self._belief_of(i).visited > 0.5
            missing = team & ~own
            self.gap_stats["gap_sum"] += float(missing.sum()) / n_team
            x, y = int(round(u.x)), int(round(u.y))
            x0, x1 = max(0, x - WIN), min(W, x + WIN + 1)
            y0, y1 = max(0, y - WIN), min(H, y + WIN + 1)
            tw = int(team[x0:x1, y0:y1].sum())
            self.gap_stats["wgap_sum"] += (float(missing[x0:x1, y0:y1].sum()) / tw) if tw else 0.0
            self.gap_stats["n"] += 1


def raw3_path(scenario, method, seed, cond):
    return os.path.join(RAW3, scenario, method, f"seed_{seed}", f"{cond}.json")


def run_cell(task):
    scenario, method, seed, cond = task
    out = raw3_path(scenario, method, seed, cond)
    if os.path.exists(out):
        return out, 0.0
    torch.set_num_threads(1)

    from evaluation.controllers import HeuristicController, MAPPOController
    from evaluation.evaluate import run_episode
    from evaluation.zero_shot import load_policy
    from sar_env import get_scenario, obs_dim_for

    kw = {"hop1": True} if cond == "hop1" else env_kwargs(cond)
    cfg = get_scenario(scenario)
    if method == "greedy":
        controller = HeuristicController("greedy", seed=seed)
    elif method in BASELINES:
        from .baselines import BayesController, FrontierController
        controller = (FrontierController if method == "frontier" else BayesController)(seed=seed)
    else:
        ckpt = os.path.join(CKPT_DIR, scenario, method, f"seed_{seed}", "policy.pt")
        policy = load_policy(ckpt, method, obs_dim_for(cfg), HIDDEN_DIM)
        controller = MAPPOController(policy, hidden_dim=HIDDEN_DIM, seed=seed)

    t0 = time.time()
    episodes, gaps, links = [], [], []
    seed_start = 90_000 + seed * 1000
    for k in range(n_eval_episodes(scenario)):
        ep_seed = seed_start + k
        env = GapSARSwarmEnv(cfg, seed=ep_seed, **kw)
        ep = run_episode(env, controller, seed=ep_seed)
        episodes.append(ep)
        g = env.gap_stats
        gaps.append({"gap": g["gap_sum"] / max(g["n"], 1), "wgap": g["wgap_sum"] / max(g["n"], 1)})
        links.append(dict(env.link_stats))

    metrics = {}
    for key in ["coverage_rate", "victim_detection_rate", "ttfd", "collision_rate", "energy_efficiency"]:
        values = np.array([e[key] for e in episodes], dtype=np.float64)
        metrics[key] = {"mean": float(values.mean()), "values": values.tolist()}
    for key in ["gap", "wgap"]:
        values = np.array([g[key] for g in gaps], dtype=np.float64)
        metrics[key] = {"mean": float(values.mean()), "values": values.tolist()}
    record = {"scenario": scenario, "method": method, "seed": seed, "condition": cond,
              "env_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()},
              "n_episodes": len(episodes), "seed_start": seed_start, "metrics": metrics,
              "links": {k: int(sum(l[k] for l in links)) for k in links[0]},
              "elapsed_s": time.time() - t0}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f)
    os.replace(tmp, out)
    return out, record["elapsed_s"]


def tasks():
    out = list(itertools.product(["easy", "medium", "hard"], ["mappo_gcn"], range(10), GAP_CONDS))
    out += list(itertools.product(["easy", "medium", "hard"], HOP_METHODS, range(10), ["p0", "hop1"]))
    # the MLP and Greedy never read messages: message_cut == p0 there, nothing else to skip
    order = {"hard": 0, "medium": 1, "easy": 2}
    out = [t for t in dict.fromkeys(out) if not os.path.exists(raw3_path(*t))]
    out.sort(key=lambda t: (order[t[0]], t[1] != "gnn_mappo"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    todo = tasks()
    print(f"{len(todo)} cells to run on {a.workers} workers", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (path, dt) in enumerate(pool.imap_unordered(run_cell, todo), 1):
            print(f"[{i}/{len(todo)}] {os.path.relpath(path, HERE)} {dt:.0f}s "
                  f"(wall {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
