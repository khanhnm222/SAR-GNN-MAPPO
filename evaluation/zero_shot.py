"""Zero-shot scalability evaluation (RQ3, muc 1.5.5): load a policy checkpoint
trained at the scenario's native N, and evaluate it — with NO retraining — on
swarms of different size N, in fixed-map mode (map size held constant, only N
changes). All four MAPPO encoders (MLP/GCN/GAT/DGAT) here are permutation-
invariant by construction (their centralized critic pools over a variable
number of node embeddings, muc 1.5.3), so they can architecturally run at any
N; MADDPG cannot (its critic concatenates a fixed number of agents) and is
therefore excluded from this test, matching Bang 1's "Zero-shot N" column.

Two corrections (2026-09-03):
  * the controller now samples instead of taking argmax, so this table is no
    longer measuring the degenerate one-action policy described in
    REVIEW_2026-09-03.md section 1;
  * obs_dim is read from the environment instead of being hard-coded to 116,
    which was wrong for any scenario with `include_neighbor_obs=False`.
"""
from __future__ import annotations

import json
import os

import torch

from evaluation.controllers import MAPPOController
from evaluation.evaluate import evaluate_policy
from models import GNNActorCritic, build_encoder
from sar_env import get_scenario
from sar_env.sar_parallel_env import SARSwarmEnv
from sar_env.scenarios import LEGACY_OVERRIDES, obs_dim_for
from training.train import METHOD_TO_ENCODER

DEFAULT_TEST_N = {"easy": [6, 8], "medium": [6, 10, 12], "hard": [10, 20]}


def load_policy(checkpoint_path: str, method: str, obs_dim: int, hidden_dim: int = 64,
                 legacy_env: bool = False) -> GNNActorCritic:
    encoder = build_encoder(METHOD_TO_ENCODER[method], in_dim=obs_dim, hidden_dim=hidden_dim)
    policy = GNNActorCritic(encoder, hidden_dim,
                             per_agent_critic=not legacy_env,
                             separate_critic_encoder=not legacy_env)
    policy.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    policy.eval()
    return policy


def zero_shot_eval(out_dir: str, scenario: str, method: str, seed: int,
                    test_n_values: list[int] | None = None, n_episodes: int = 6,
                    hidden_dim: int = 64, legacy_env: bool = False) -> dict:
    ckpt_path = os.path.join(out_dir, scenario, method, f"seed_{seed}", "policy.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    overrides = dict(LEGACY_OVERRIDES) if legacy_env else {}
    base_cfg = get_scenario(scenario, **overrides)
    obs_dim = obs_dim_for(base_cfg)
    policy = load_policy(ckpt_path, method, obs_dim, hidden_dim, legacy_env=legacy_env)

    test_n_values = test_n_values or DEFAULT_TEST_N.get(scenario, [base_cfg.n_uavs + 2])
    results = {"scenario": scenario, "method": method, "seed": seed,
               "train_n": base_cfg.n_uavs, "fixed_map": True,
               "eval_protocol": "sampled", "legacy_env": legacy_env, "by_n": {}}

    for n in [base_cfg.n_uavs] + test_n_values:
        def env_factory(seed, n=n):
            cfg = get_scenario(scenario, n_uavs=n, **overrides)
            return SARSwarmEnv(cfg, seed=seed)

        controller = MAPPOController(policy, hidden_dim=hidden_dim, seed=seed)
        metrics = evaluate_policy(env_factory, controller, n_episodes=n_episodes, seed_start=70_000 + n)
        results["by_n"][str(n)] = metrics

    return results


def save_zero_shot_report(out_dir: str, scenario: str, method: str = "gnn_mappo", seed: int = 0,
                           legacy_env: bool = False):
    report = zero_shot_eval(out_dir, scenario, method, seed, legacy_env=legacy_env)
    path = os.path.join(out_dir, scenario, f"zero_shot_{method}_seed{seed}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Zero-shot report written to {path}")
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="easy")
    ap.add_argument("method", nargs="?", default="gnn_mappo")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--legacy-env", action="store_true",
                    help="checkpoint was trained with the Giai doan 6/7 environment")
    a = ap.parse_args()
    save_zero_shot_report(a.out_dir, a.scenario, a.method, a.seed, legacy_env=a.legacy_env)
