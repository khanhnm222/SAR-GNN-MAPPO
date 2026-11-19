"""Evaluate the stored checkpoints (results/ at the repo root) under communication degradation.

    python -m comm_robustness.run_eval --check          # p = 0 reproduction only
    python -m comm_robustness.run_eval --workers 14     # round-1 grid, resumable
    python -m comm_robustness.run_eval --round 2        # round-2 grid (burst, delay,
                                                               # bandwidth, no-death, baselines)

Every (scenario, method, seed, condition) cell is one JSON file under
comm_robustness/results/raw/. Existing files are skipped, so the run can
be stopped and restarted. Test maps, number of episodes and controller seed are
those of the final evaluation in training/train.py (seed_start = 90_000 +
seed*1000), so condition "p0" must reproduce eval_metrics.json episode by
episode; `--check` verifies this before any degraded condition is trusted.
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
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT_DIR = os.path.join(ROOT, "results")
RAW_DIR = os.path.join(HERE, "results", "raw")

SCENARIOS = ["easy", "medium", "hard"]
LEARNED = ["mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo"]
METHODS = LEARNED + ["greedy"]
SEEDS = list(range(10))
HIDDEN_DIM = 64

# name -> (p_belief, p_message, range_scale)
CONDITIONS = {
    "p0": (0.0, 0.0, 1.0),
    "p25": (0.25, 0.25, 1.0),
    "p50": (0.5, 0.5, 1.0),
    "p75": (0.75, 0.75, 1.0),
    "p100": (1.0, 1.0, 1.0),
    "belief_cut": (1.0, 0.0, 1.0),    # belief maps never merged, GNN messages intact
    "message_cut": (0.0, 1.0, 1.0),   # belief maps merged as usual, GNN sees no edges
    "r75": (0.0, 0.0, 0.75),          # radio range shrunk to 75 % of r_comm
    "r50": (0.0, 0.0, 0.5),
    "r25": (0.0, 0.0, 0.25),
}


# Round 2 (14/10/2026). Each entry is the keyword arguments of DegradedSARSwarmEnv.
CONDITIONS2 = {
    "ge75L10": dict(burst=(0.75, 10)),     # Gilbert-Elliott, 75 % of steps bad, bursts of 10 steps
    "ge75L50": dict(burst=(0.75, 50)),     # same loss rate, bursts of 50 steps
    "delay5": dict(delay=5),               # belief channel: maps arrive 5 steps late
    "delay25": dict(delay=25),
    "bw100": dict(bandwidth=100),          # belief channel: at most 100 cells per UAV and step
    "bw25": dict(bandwidth=25),
    "p0_nd": dict(no_death=True),          # Hard energy control: UAVs never removed
    "p75_nd": dict(p_belief=0.75, p_message=0.75, no_death=True),
    "p100_nd": dict(p_belief=1.0, p_message=1.0, no_death=True),
    "belief_cut_nd": dict(p_belief=1.0, no_death=True),
    "r25_nd": dict(range_scale=0.25, no_death=True),
}
COMM2 = ["ge75L10", "ge75L50", "delay5", "delay25", "bw100", "bw25"]
ENERGY2 = ["p0_nd", "p75_nd", "p100_nd", "belief_cut_nd", "r25_nd"]
BASELINES = ["frontier", "bayes"]
BASELINE_CONDS = ["p0", "p50", "p75", "p100", "belief_cut", "message_cut", "r25"]


def env_kwargs(cond: str) -> dict:
    if cond in CONDITIONS:
        p_belief, p_message, range_scale = CONDITIONS[cond]
        return dict(p_belief=p_belief, p_message=p_message, range_scale=range_scale)
    return dict(CONDITIONS2[cond])


def round2_tasks():
    """(priority, task): energy control first, then the new channel models, then the baselines."""
    out = []
    for m, s, c in itertools.product(METHODS, SEEDS, ENERGY2):
        out.append((0, ("hard", m, s, c)))
    for sc, m, s, c in itertools.product(SCENARIOS, METHODS + BASELINES, SEEDS, COMM2):
        out.append((1, (sc, m, s, c)))
    for sc, m, s, c in itertools.product(SCENARIOS, BASELINES, SEEDS, BASELINE_CONDS):
        out.append((2, (sc, m, s, c)))
    return out


def needed(method: str, cond: str) -> bool:
    """Cells that are not identical by construction to another cell.

    MLP and Greedy never read the message channel, so for them message_cut is
    p0 and belief_cut is p100; those cells are filled by copy in analyze.py.
    """
    if method in ("mappo_mlp", "greedy") and cond in ("belief_cut", "message_cut", "belief_cut_nd"):
        return False
    return True


def n_eval_episodes(scenario: str) -> int:
    with open(os.path.join(ROOT, "training", "configs", f"{scenario}.yaml"), encoding="utf-8") as f:
        return int(yaml.safe_load(f)["n_eval_episodes"])


def raw_path(scenario, method, seed, cond):
    return os.path.join(RAW_DIR, scenario, method, f"seed_{seed}", f"{cond}.json")


def run_cell(task):
    scenario, method, seed, cond = task
    out = raw_path(scenario, method, seed, cond)
    if os.path.exists(out):
        return out, 0.0
    torch.set_num_threads(1)

    from evaluation.controllers import HeuristicController, MAPPOController
    from evaluation.evaluate import run_episode
    from evaluation.zero_shot import load_policy
    from sar_env import get_scenario, obs_dim_for

    from .degraded_env import DegradedSARSwarmEnv

    kw = env_kwargs(cond)
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
    episodes, links = [], []
    seed_start = 90_000 + seed * 1000
    for k in range(n_eval_episodes(scenario)):
        ep_seed = seed_start + k
        env = DegradedSARSwarmEnv(cfg, seed=ep_seed, **kw)
        ep = run_episode(env, controller, seed=ep_seed)
        episodes.append(ep)
        links.append(dict(env.link_stats))

    metrics = {}
    for key in ["coverage_rate", "victim_detection_rate", "ttfd", "collision_rate", "energy_efficiency"]:
        values = np.array([e[key] for e in episodes], dtype=np.float64)
        metrics[key] = {"mean": float(values.mean()), "std": float(values.std()),
                        "values": values.tolist()}
    tot = {k: int(sum(l[k] for l in links)) for k in links[0]}
    record = {"scenario": scenario, "method": method, "seed": seed, "condition": cond,
              "env_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()},
              "n_episodes": len(episodes), "seed_start": seed_start,
              "metrics": metrics, "links": tot, "per_episode_links": links,
              "elapsed_s": time.time() - t0}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f)
    os.replace(tmp, out)
    return out, record["elapsed_s"]


def check_reproduction(scenarios, methods, seeds) -> bool:
    ok = True
    for sc, m, s in itertools.product(scenarios, methods, seeds):
        run_cell((sc, m, s, "p0"))
        with open(raw_path(sc, m, s, "p0"), encoding="utf-8") as f:
            mine = json.load(f)["metrics"]["victim_detection_rate"]["values"]
        with open(os.path.join(CKPT_DIR, sc, m, f"seed_{s}", "eval_metrics.json"), encoding="utf-8") as f:
            ref = json.load(f)["metrics"]["victim_detection_rate"]["values"]
        same = len(mine) == len(ref) and np.allclose(mine, ref, atol=1e-12)
        ok &= same
        print(f"{sc:6s} {m:10s} seed{s}: p0 VDR {np.mean(mine):.4f} vs stored {np.mean(ref):.4f} "
              f"{'IDENTICAL' if same else 'DIFFERENT'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--check", action="store_true", help="p0 reproduction check only")
    ap.add_argument("--scenarios", nargs="*", default=SCENARIOS)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--round", type=int, default=1)
    a = ap.parse_args()

    if a.check:
        raise SystemExit(0 if check_reproduction(a.scenarios, a.methods, a.seeds) else 1)

    order = {"hard": 0, "medium": 1, "easy": 2}
    if a.round == 2:
        pri = [(p, t) for p, t in round2_tasks()
               if needed(t[1], t[3]) and not os.path.exists(raw_path(*t))]
        # energy control first, then hardest cells first inside each group
        pri.sort(key=lambda pt: (pt[0], order[pt[1][0]], pt[1][1] == "greedy"))
        tasks = [t for _, t in pri]
    else:
        tasks = [t for t in itertools.product(a.scenarios, a.methods, a.seeds, a.conditions)
                 if needed(t[1], t[3]) and not os.path.exists(raw_path(*t))]
        # longest cells first so the pool does not end on one straggler
        tasks.sort(key=lambda t: (order[t[0]], t[1] == "greedy"))
    print(f"{len(tasks)} cells to run on {a.workers} workers", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (path, dt) in enumerate(pool.imap_unordered(run_cell, tasks), 1):
            print(f"[{i}/{len(tasks)}] {os.path.relpath(path, HERE)} {dt:.0f}s "
                  f"(wall {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
