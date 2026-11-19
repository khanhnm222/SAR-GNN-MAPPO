"""Diagnostic for Sec. 4.3: does Greedy spread out when belief maps are not shared?

    python -m comm_robustness.diag_greedy_spread

Runs Greedy on the Easy and Medium test episodes (10 seeds) with intact links and with
every link cut, and records the mean nominal degree (pairs within r_comm per alive UAV)
and the mean pairwise distance per step. Writes data/greedy_spread.json.
Result used in the paper: on Medium the pairwise distance falls from 52.9 to 42.5 cells
without map sharing, so the Greedy gain does not come from the UAVs spreading out.
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def cell(task):
    sc, seed, p = task
    from evaluation.controllers import HeuristicController
    from sar_env import get_scenario

    from .degraded_env import DegradedSARSwarmEnv
    from .run_eval import n_eval_episodes

    cfg = get_scenario(sc)
    ctrl = HeuristicController("greedy", seed=seed)
    deg, dist = [], []
    for k in range(n_eval_episodes(sc)):
        s = 90_000 + seed * 1000 + k
        env = DegradedSARSwarmEnv(cfg, seed=s, p_belief=p, p_message=p)
        ctrl.reset()
        obs, _ = env.reset(seed=s)
        while env.agents:
            obs, *_ = env.step(ctrl.act(env, obs))
            alive = [u for u in env.uavs if u.alive]
            if len(alive) > 1:
                P = np.array([[u.x, u.y] for u in alive])
                D = np.linalg.norm(P[:, None] - P[None], axis=-1)
                n = len(alive)
                deg.append(((D <= cfg.r_comm).sum() - n) / n)
                dist.append(D[np.triu_indices(n, 1)].mean())
    return sc, seed, p, float(np.mean(deg)), float(np.mean(dist))


def main():
    tasks = [(sc, s, p) for sc in ("easy", "medium") for s in range(10) for p in (0.0, 1.0)]
    with Pool(14) as pool:
        res = pool.map(cell, tasks)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "greedy_spread.json"), "w", encoding="utf-8") as f:
        json.dump(res, f)
    for sc in ("easy", "medium"):
        for p in (0.0, 1.0):
            r = [x for x in res if x[0] == sc and x[2] == p]
            print(sc, "p", p, "degree %.2f" % np.mean([x[3] for x in r]),
                  "pairwise distance %.1f" % np.mean([x[4] for x in r]))


if __name__ == "__main__":
    main()
