"""Runs test episodes and computes the five metrics from muc 1.5.5:
Coverage Rate, Victim Detection Rate, TTFD, Collision Rate, Energy Efficiency.
"""
from __future__ import annotations

import numpy as np


def run_episode(env, controller, seed: int) -> dict:
    controller.reset()
    obs, _ = env.reset(seed=seed)
    collision_steps = 0
    energy_spent_total = 0.0
    steps = 0
    while env.agents:
        actions = controller.act(env, obs)
        obs, rewards, terms, truncs, infos = env.step(actions)
        steps += 1
        if infos:
            any_info = next(iter(infos.values()))
            # "Collision Rate: ti le BUOC co va cham tren tong so buoc" (thesis
            # muc 1.5.5) -> dem SO BUOC co it nhat 1 va cham, khong cong don
            # tong so cap vi pham (truoc day lam vay khien collision_rate co
            # the >1 va lan at hoan toan tin hieu reward, xem sar_env/rewards.py).
            if any_info.get("collisions", 0) > 0:
                collision_steps += 1
            energy_spent_total += any_info.get("reward_components", {}).get("energy", 0.0) * 1000.0
        if steps > env.cfg.max_steps + 5:
            break

    coverage_rate = env.belief.coverage_rate()
    n_victims = max(len(env.victims), 1)
    victim_detection_rate = env._n_victims_detected / n_victims
    detected_steps = [v.detected_at_step for v in env.victims if v.detected]
    ttfd = min(detected_steps) if detected_steps else env.cfg.max_steps
    collision_rate = collision_steps / max(steps, 1)
    energy_efficiency = env._n_victims_detected / max(energy_spent_total, 1e-6)

    return {
        "coverage_rate": coverage_rate,
        "victim_detection_rate": victim_detection_rate,
        "ttfd": ttfd,
        "collision_rate": collision_rate,
        "energy_efficiency": energy_efficiency,
        "steps": steps,
    }


def evaluate_policy(env_factory, controller, n_episodes: int, seed_start: int = 10_000) -> dict:
    episodes = [run_episode(env_factory(seed=seed_start + i), controller, seed=seed_start + i)
                for i in range(n_episodes)]
    metrics = {}
    for key in ["coverage_rate", "victim_detection_rate", "ttfd", "collision_rate", "energy_efficiency"]:
        values = np.array([e[key] for e in episodes], dtype=np.float64)
        metrics[key] = {"mean": float(values.mean()), "std": float(values.std()), "values": values.tolist()}
    metrics["n_episodes"] = n_episodes
    return metrics
