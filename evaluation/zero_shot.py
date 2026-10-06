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
import math
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


def density_overrides(base_cfg, n: int) -> dict:
    """Density-controlled scaling (de cuong muc 1.5.5, che do thu hai cua RQ3).

    Giu mat do UAV va mat do nan nhan tren ban do khong doi khi N thay doi:
    dien tich ti le voi N (canh ti le sqrt(N / N_train)), so nan nhan ti le voi N.
    r_comm giu nguyen vi bac trung binh ky vong deg = (N-1) pi r^2 / (W H) chi
    phu thuoc mat do; r_sense va max_steps giu nguyen. Nho do phep do tach anh
    huong cua *so luong* UAV khoi anh huong cua *mat do* khong gian, khac voi
    fixed-map noi them UAV dong nghia voi ban do dong hon.
    """
    ratio = n / base_cfg.n_uavs
    side = math.sqrt(ratio)
    lo, hi = base_cfg.n_victims_range
    return dict(
        width=max(10, int(round(base_cfg.width * side))),
        height=max(10, int(round(base_cfg.height * side))),
        n_victims_range=(max(1, int(round(lo * ratio))), max(1, int(round(hi * ratio)))),
    )


def zero_shot_eval(out_dir: str, scenario: str, method: str, seed: int,
                    test_n_values: list[int] | None = None, n_episodes: int = 6,
                    hidden_dim: int = 64, legacy_env: bool = False,
                    mode: str = "fixed_map") -> dict:
    """mode = "fixed_map" (ban do giu nguyen, chi N doi) hoac "density"
    (dien tich va so nan nhan scale theo N, xem `density_overrides`)."""
    if mode not in ("fixed_map", "density"):
        raise ValueError(f"mode phai la fixed_map hoac density, nhan {mode!r}")
    ckpt_path = os.path.join(out_dir, scenario, method, f"seed_{seed}", "policy.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    overrides = dict(LEGACY_OVERRIDES) if legacy_env else {}
    base_cfg = get_scenario(scenario, **overrides)
    obs_dim = obs_dim_for(base_cfg)
    policy = load_policy(ckpt_path, method, obs_dim, hidden_dim, legacy_env=legacy_env)

    test_n_values = test_n_values or DEFAULT_TEST_N.get(scenario, [base_cfg.n_uavs + 2])
    results = {"scenario": scenario, "method": method, "seed": seed,
               "train_n": base_cfg.n_uavs, "mode": mode, "fixed_map": mode == "fixed_map",
               "eval_protocol": "sampled", "legacy_env": legacy_env, "by_n": {},
               "env_by_n": {}}

    # Khoi seed rieng cho tung che do de hai che do khong dung chung ban do test.
    seed_base = 70_000 if mode == "fixed_map" else 80_000

    for n in [base_cfg.n_uavs] + test_n_values:
        extra = density_overrides(base_cfg, n) if mode == "density" else {}

        def env_factory(seed, n=n, extra=extra):
            cfg = get_scenario(scenario, n_uavs=n, **overrides, **extra)
            return SARSwarmEnv(cfg, seed=seed)

        cfg_n = get_scenario(scenario, n_uavs=n, **overrides, **extra)
        results["env_by_n"][str(n)] = {"width": cfg_n.width, "height": cfg_n.height,
                                        "n_victims_range": list(cfg_n.n_victims_range),
                                        "r_comm": cfg_n.r_comm}
        controller = MAPPOController(policy, hidden_dim=hidden_dim, seed=seed)
        metrics = evaluate_policy(env_factory, controller, n_episodes=n_episodes,
                                  seed_start=seed_base + n)
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
