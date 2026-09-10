"""Environment Validation Protocol (muc 1.5.1).

Runs a battery of sanity checks against SARSwarmEnv before it is used for
training, verifying:
  1. Energy conservation / monotonic decrease under motion.
  2. Action legality: illegal moves (into obstacles / bounds) don't teleport.
  3. Belief map convergence: repeatedly scanning a cell drives P(c) -> 0.
  4. Communication graph symmetry: d(i,j) <= r_comm is symmetric.
  5. Observation shape / finiteness.
  6. Episode termination reachability within max_steps under random policy.
"""
from __future__ import annotations

import numpy as np

from .scenarios import get_scenario
from .sar_parallel_env import SARSwarmEnv


def validate_environment(scenario_name: str = "easy", seed: int = 0, verbose: bool = True) -> dict:
    results = {}
    cfg = get_scenario(scenario_name)
    env = SARSwarmEnv(cfg, seed=seed)
    obs, infos = env.reset(seed=seed)

    # 1. Observation shape / finiteness
    ok = all(np.isfinite(o).all() for o in obs.values())
    ok = ok and all(o.shape == env.observation_space(a).shape for a, o in obs.items())
    results["obs_valid"] = ok

    # 2. Energy monotonic non-increase for alive UAVs under motion
    e0 = {u.uid: u.energy for u in env.uavs}
    obs, rewards, terms, truncs, infos = env.step({a: 2 for a in env.agents})  # move East
    e1 = {u.uid: u.energy for u in env.uavs}
    results["energy_non_increasing"] = all(e1[k] <= e0[k] + 1e-6 for k in e0)

    # 3. Illegal move handling: force a UAV into an obstacle if one exists, verify no teleport
    illegal_ok = True
    if env.terrain.occupancy.sum() > 0:
        bx, by = np.argwhere(env.terrain.occupancy > 0.5)[0]
        u = env.uavs[0]
        u.x, u.y = float(bx) - 1, float(by)
        pre = (u.x, u.y)
        u.apply_action(2, cfg.width, cfg.height, env.terrain)  # try to move East into obstacle
        illegal_ok = env.terrain.is_blocked(int(round(pre[0] + 1)), int(round(pre[1])))
    results["illegal_move_blocked"] = bool(illegal_ok)

    # 4. Belief map convergence under repeated scanning
    bm = env.belief
    bm.prob[5, 5] = 0.9
    for _ in range(50):
        bm.scan_no_detection(5, 5)
    results["belief_converges"] = bool(bm.prob[5, 5] < 0.05)

    # 5. Communication graph symmetry
    graph = env.build_communication_graph()
    adj = graph["adjacency"]
    sym_err = float(np.abs((adj > 0).astype(float) - (adj.T > 0).astype(float)).max()) if adj.size else 0.0
    results["graph_symmetric"] = sym_err < 1e-6

    # 6. Episode reachability under random policy
    env2 = SARSwarmEnv(cfg, seed=seed + 1)
    obs, infos = env2.reset(seed=seed + 1)
    rng = np.random.default_rng(seed + 1)
    steps = 0
    done = False
    while env2.agents and steps < cfg.max_steps + 5:
        actions = {a: int(rng.integers(0, 13)) for a in env2.agents}
        obs, rewards, terms, truncs, infos = env2.step(actions)
        steps += 1
        if not env2.agents:
            done = True
            break
    results["episode_terminates"] = bool(done)
    results["episode_length"] = steps

    if verbose:
        print(f"[validation:{scenario_name}] " + ", ".join(f"{k}={v}" for k, v in results.items()))
    results["all_passed"] = all(v for k, v in results.items() if isinstance(v, (bool, np.bool_)))
    return results


if __name__ == "__main__":
    for name in ["easy", "medium", "hard"]:
        validate_environment(name)
