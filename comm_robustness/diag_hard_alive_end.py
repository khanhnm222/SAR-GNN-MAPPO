# -*- coding: utf-8 -*-
"""Hard, intact links: fraction of UAVs still active when each test episode ends, per method and seed.

Re-runs the same 20 test episodes per seed as run_eval (p0), so VDR must equal Table 1.
Run from the repo root: python comm_robustness/diag_hard_alive_end.py (about 25 min on 14 workers)
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "comm_robustness", "data", "hard_alive_end.json")
METHODS = ["mappo_gcn", "mappo_gat", "gnn_mappo", "mappo_mlp", "greedy"]


def cell(task):
    method, seed = task
    import torch
    torch.set_num_threads(1)
    from evaluation.controllers import HeuristicController, MAPPOController
    from evaluation.evaluate import run_episode
    from evaluation.zero_shot import load_policy
    from sar_env import get_scenario, obs_dim_for
    from comm_robustness.degraded_env import DegradedSARSwarmEnv
    from comm_robustness.run_eval import CKPT_DIR, HIDDEN_DIM, n_eval_episodes

    cfg = get_scenario("hard")
    if method == "greedy":
        ctrl = HeuristicController("greedy", seed=seed)
    else:
        pol = load_policy(os.path.join(CKPT_DIR, "hard", method, f"seed_{seed}", "policy.pt"),
                          method, obs_dim_for(cfg), HIDDEN_DIM)
        ctrl = MAPPOController(pol, hidden_dim=HIDDEN_DIM, seed=seed)
    rows = []
    for k in range(n_eval_episodes("hard")):
        s = 90_000 + 1000 * seed + k
        env = DegradedSARSwarmEnv(cfg, seed=s)
        ep = run_episode(env, ctrl, seed=s)
        alive = sum(1 for u in env.uavs if u.alive)
        rows.append(dict(vdr=ep["victim_detection_rate"], alive_end=alive / len(env.uavs),
                         steps=int(env.t), all_dead=alive == 0,
                         energy_left=float(np.mean([u.energy for u in env.uavs]))
                         if hasattr(env.uavs[0], "energy") else None))
    return method, seed, rows


if __name__ == "__main__":
    os.chdir(ROOT)
    tasks = [(m, s) for m in METHODS for s in range(10)]
    res = {}
    with Pool(14) as pool:
        for m, s, rows in pool.imap_unordered(cell, tasks):
            res.setdefault(m, {})[str(s)] = rows
            print(m, s, "alive_end %.3f" % np.mean([r["alive_end"] for r in rows]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    for m in METHODS:
        seeds = [np.mean([r["alive_end"] for r in res[m][str(s)]]) for s in range(10)]
        vdr = np.mean([np.mean([r["vdr"] for r in res[m][str(s)]]) for s in range(10)])
        dead = np.mean([r["all_dead"] for s in range(10) for r in res[m][str(s)]])
        steps = np.mean([r["steps"] for s in range(10) for r in res[m][str(s)]])
        print(f"{m:10s} alive_end mean {100*np.mean(seeds):.1f} % (seed range {100*min(seeds):.1f}-{100*max(seeds):.1f}),"
              f" all-dead episodes {100*dead:.0f} %, steps {steps:.0f}, VDR {vdr:.3f}")
