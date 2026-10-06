"""CLI entrypoint: python -m training.train --method gnn_mappo --scenario easy --seed 0

Trains (or, for heuristics, just evaluates) one (method, scenario, seed)
configuration, periodically evaluating and finally writing learning_curve.json,
eval_metrics.json, trajectory_sample.json and a policy checkpoint under
results/<scenario>/<method>/seed_<seed>/.

`--legacy-env` reproduces the Giai doan 6/7 environment and trainer (global
belief map, neighbour block in the observation, global reward, shared
actor/critic encoder, no normalisation) — see REVIEW_2026-09-03.md.
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
import yaml

from algorithms.heuristics import GreedyPolicy, RandomWalkPolicy
from algorithms.maddpg import MADDPGConfig, MADDPGTrainer
from algorithms.mappo import MAPPOConfig, MAPPOTrainer
from evaluation.controllers import HeuristicController, MADDPGController, MAPPOController
from evaluation.evaluate import evaluate_policy
from models import build_encoder
from sar_env import make_env
from sar_env.rewards import RewardWeights
from sar_env.scenarios import LEGACY_OVERRIDES
from training.logger import RunLogger

METHOD_TO_ENCODER = {"mappo_mlp": "mlp", "mappo_gcn": "gcn", "mappo_gat": "gat", "gnn_mappo": "dgat"}
HEURISTIC_METHODS = {"random_walk", "greedy"}
ALL_METHODS = list(HEURISTIC_METHODS) + ["maddpg"] + list(METHOD_TO_ENCODER.keys())


def seed_everything(seed: int) -> None:
    """Seed every RNG the run touches.

    `torch.manual_seed` was never called anywhere in this project, so network
    weight initialisation and every `dist.sample()` during rollout came from an
    unseeded global RNG: the `--seed` argument only controlled the environment
    and the numpy minibatch shuffle. Runs were therefore NOT reproducible,
    contrary to the seed protocol of muc 1.5.5. Measured effect of this: with
    the code byte-identical between results_v3_rcomm and results_v4, MAPPO-MLP
    on Easy moved 0.877 -> 0.860 and MAPPO-GCN 0.942 -> 0.925 purely from
    unseeded initialisation -- a shift comparable to the between-seed std.

    Must be called BEFORE `build_encoder`, since that is where weights are
    created.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_eval_controller(method: str, trainer, cfg: dict, seed: int = 0):
    if method == "maddpg":
        return MADDPGController(trainer, seed=seed)
    return MAPPOController(trainer.policy, hidden_dim=cfg["hidden_dim"], seed=seed)


def save_trajectory_sample(env_factory, controller, logger: RunLogger, seed: int):
    env = env_factory(seed=seed + 777)
    controller.reset()
    obs, _ = env.reset(seed=seed + 777)
    # static obstacle layout (Hinh 2 minh hoa: "vat can tinh" = o vuong xam) —
    # saved once; cells added later by flood growth arrive per-frame as
    # `terrain_new` inside get_render_state().
    terrain_occupancy = np.argwhere(env.terrain.occupancy > 0.5).tolist()
    frames = [env.get_render_state()]
    steps = 0
    while env.agents and steps < env.cfg.max_steps:
        actions = controller.act(env, obs)
        obs, rewards, terms, truncs, infos = env.step(actions)
        frames.append(env.get_render_state())
        steps += 1
    logger.save_trajectory(frames, terrain_occupancy=terrain_occupancy)


REWARD_COMPONENTS = ("cover", "victim", "collision", "energy")


def run(method: str, scenario: str, seed: int, out_dir: str = "results",
        config_path: str | None = None, total_env_steps_override: int | None = None,
        legacy_env: bool = False, reward_zero: str | None = None):
    cfg_path = config_path or f"training/configs/{scenario}.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if total_env_steps_override:
        cfg["total_env_steps"] = total_env_steps_override

    seed_everything(seed)
    env_overrides = dict(LEGACY_OVERRIDES) if legacy_env else {}
    if reward_zero:
        # Ablation ham thuong (de cuong, muc 1.5.4): bo dung MOT thanh phan bang
        # cach dat trong so cua no = 0, moi thu khac giu nguyen (ke ca doi tuong
        # danh gia, vi cac chi so danh gia khong doc reward).
        if reward_zero not in REWARD_COMPONENTS:
            raise ValueError(f"--reward-zero must be one of {REWARD_COMPONENTS}")
        env_overrides["reward_weights"] = RewardWeights(**{reward_zero: 0.0})

    def env_factory(seed):
        return make_env(scenario, seed=seed, **env_overrides)

    env0 = env_factory(seed)
    obs_dim = env0.obs_dim
    n_agents = env0.cfg.n_uavs

    logger = RunLogger(out_dir, method, scenario, seed)
    logger.meta["legacy_env"] = legacy_env
    logger.meta["obs_dim"] = obs_dim
    logger.meta["seeded"] = True
    logger.meta["reward_zero"] = reward_zero
    tag = f"[{method}/{scenario}/seed{seed}]"

    if method in HEURISTIC_METHODS:
        controller = HeuristicController(method, seed=seed)
        t0 = time.time()
        # Same test maps as the learned methods (90_000 + seed*1000): the
        # heuristics used to be scored on a DIFFERENT seed range (10_000+),
        # so every heuristic-vs-learned comparison was across different maps.
        metrics = evaluate_policy(env_factory, controller, n_episodes=cfg["n_eval_episodes"],
                                   seed_start=90_000 + seed * 1000)
        logger.log_iteration({"global_step": 0, "elapsed": time.time() - t0,
                               "eval_vdr": metrics["victim_detection_rate"]["mean"],
                               "eval_coverage": metrics["coverage_rate"]["mean"]})
        logger.save_eval(metrics)
        save_trajectory_sample(env_factory, controller, logger, seed)
        print(f"{tag} done. VDR={metrics['victim_detection_rate']['mean']:.3f} "
              f"cov={metrics['coverage_rate']['mean']:.3f}")
        return

    if method == "maddpg":
        mcfg = MADDPGConfig(hidden_dim=cfg["hidden_dim"], lr=cfg["lr"], gamma=cfg["gamma"],
                             rollout_len=cfg["rollout_len"])
        trainer = MADDPGTrainer(env_factory, obs_dim, 13, n_agents, config=mcfg, seed=seed)
    elif method in METHOD_TO_ENCODER:
        encoder = build_encoder(METHOD_TO_ENCODER[method], in_dim=obs_dim, hidden_dim=cfg["hidden_dim"])
        pcfg = MAPPOConfig(hidden_dim=cfg["hidden_dim"], lr=cfg["lr"], gamma=cfg["gamma"],
                            gae_lambda=cfg["gae_lambda"], clip_eps=cfg["clip_eps"], ppo_epochs=cfg["ppo_epochs"],
                            minibatch_size=cfg["minibatch_size"], entropy_coef=cfg["entropy_coef"],
                            value_coef=cfg["value_coef"], max_grad_norm=cfg["max_grad_norm"],
                            rollout_len=cfg["rollout_len"], critic_pool=cfg["critic_pool"],
                            total_env_steps=cfg["total_env_steps"], legacy=legacy_env)
        trainer = MAPPOTrainer(env_factory, encoder, obs_dim, config=pcfg, seed=seed)
    else:
        raise ValueError(f"Unknown method {method}. Choose from {ALL_METHODS}")

    total_steps = cfg["total_env_steps"]
    eval_interval = cfg["eval_interval_steps"]
    next_eval = eval_interval
    eval_idx = 0
    t_start = time.time()

    while trainer.global_step < total_steps:
        stats = trainer.train_iteration()
        stats["elapsed"] = time.time() - t_start

        if trainer.global_step >= next_eval:
            controller = build_eval_controller(method, trainer, cfg, seed)
            # Bounded, disjoint seed block for intermediate evals. The old
            # `50_000 + seed*1000 + global_step` grew with training and swept
            # straight through the final test block (90_000..94_029) once
            # global_step passed ~40k. No gradient ever came from these
            # episodes, so results are unaffected, but the test maps must not
            # be touched during development.
            eval_metrics = evaluate_policy(env_factory, controller,
                                            n_episodes=cfg.get("n_eval_episodes_intermediate", 4),
                                            seed_start=50_000 + seed * 1000 + eval_idx * 10)
            eval_idx += 1
            stats["eval_vdr"] = eval_metrics["victim_detection_rate"]["mean"]
            stats["eval_coverage"] = eval_metrics["coverage_rate"]["mean"]
            print(f"{tag} step={trainer.global_step}/{total_steps} "
                  f"VDR={stats['eval_vdr']:.3f} cov={stats['eval_coverage']:.3f} "
                  f"({stats['elapsed']:.0f}s)")
            # Checkpoint tam thoi sau moi lan danh gia trung gian (23/09/2026): mot lan
            # khoi dong lai Windows da giet lượt DGAT/Hard 2M ngay truoc danh gia cuoi va
            # mat 22 gio vi policy.pt chi duoc luu o cuoi. Ghi file khong dung RNG nen
            # khong doi ket qua; file cuoi cung van la policy.pt.
            if method != "maddpg":
                logger.save_checkpoint(trainer.policy.state_dict(), name="policy_latest.pt")
            next_eval += eval_interval

        logger.log_iteration(stats)

    controller = build_eval_controller(method, trainer, cfg, seed)
    final_metrics = evaluate_policy(env_factory, controller, n_episodes=cfg["n_eval_episodes"],
                                     seed_start=90_000 + seed * 1000)
    final_metrics["eval_protocol"] = "sampled"
    logger.save_eval(final_metrics)
    save_trajectory_sample(env_factory, controller, logger, seed)
    # MADDPG used to be skipped here, which left its runs impossible to
    # re-evaluate without retraining (found during the 2026-09-03 review).
    if method == "maddpg":
        logger.save_checkpoint({"actor": trainer.actor.state_dict(),
                                 "critic": trainer.critic.state_dict()})
    else:
        logger.save_checkpoint(trainer.policy.state_dict())
    print(f"{tag} DONE. Final VDR={final_metrics['victim_detection_rate']['mean']:.3f} "
          f"cov={final_metrics['coverage_rate']['mean']:.3f} "
          f"total_time={time.time() - t_start:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=ALL_METHODS)
    parser.add_argument("--scenario", required=True, choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--config", default=None)
    parser.add_argument("--total_env_steps", type=int, default=None)
    parser.add_argument("--legacy-env", action="store_true",
                        help="reproduce the Giai doan 6/7 environment + trainer")
    parser.add_argument("--reward-zero", default=None, choices=list(REWARD_COMPONENTS),
                        help="ablation ham thuong: dat trong so cua thanh phan nay = 0")
    args = parser.parse_args()
    run(args.method, args.scenario, args.seed, args.out_dir, args.config,
        args.total_env_steps, args.legacy_env, args.reward_zero)


if __name__ == "__main__":
    main()
