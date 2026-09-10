"""Re-evaluate existing policy checkpoints with the corrected (stochastic)
evaluation protocol — no retraining required.

Background: evaluation/controllers.MAPPOController used to roll out the policy
with argmax. The trained policies are high-entropy (H ~ 1.4-2.3 nats out of
ln(13) = 2.565), so argmax collapsed every UAV onto one action for the whole
episode — measured on easy/gnn_mappo/seed_0 it was action 11 (FAST_FORWARD)
480/480 agent-steps, which flies along a frozen heading until the map edge and
then stops. Every learned method therefore scored below Random Walk, and the
replay showed a formation drifting into a wall. See REVIEW_2026-09-03.md.

This script reloads each `policy.pt`, re-runs `evaluate_policy` with sampling
(the policy PPO actually optimised), and rewrites `eval_metrics.json` and
`trajectory_sample.json` in place. Heuristics are re-run too so their
trajectory files pick up the new replay fields (comm_edges / action / blocked).

`--legacy-env` (default ON) builds the environment the checkpoints were trained
with. Use `--no-legacy-env` for checkpoints produced after the review, i.e. by
`training.train` without `--legacy-env`.

Usage:
    python -m scripts.reevaluate --out-dir results
    python -m scripts.reevaluate --out-dir results_v2 --no-legacy-env
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import yaml

sys.path.insert(0, ".")

from evaluation.controllers import HeuristicController, MAPPOController  # noqa: E402
from evaluation.evaluate import evaluate_policy  # noqa: E402
from models import build_encoder  # noqa: E402
from models.actor_critic import GNNActorCritic  # noqa: E402
from sar_env import make_env  # noqa: E402
from sar_env.scenarios import LEGACY_OVERRIDES, get_scenario, obs_dim_for  # noqa: E402
from training.logger import RunLogger  # noqa: E402
from training.train import HEURISTIC_METHODS, METHOD_TO_ENCODER, save_trajectory_sample  # noqa: E402

SCENARIOS = ["easy", "medium", "hard"]
METHODS = list(HEURISTIC_METHODS) + ["maddpg"] + list(METHOD_TO_ENCODER)


def _load_cfg(scenario: str) -> dict:
    with open(f"training/configs/{scenario}.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def reevaluate_one(out_dir: str, scenario: str, method: str, seed: int,
                    deterministic: bool = False, legacy_env: bool = True) -> dict:
    seed_dir = os.path.join(out_dir, scenario, method, f"seed_{seed}")
    cfg = _load_cfg(scenario)
    overrides = dict(LEGACY_OVERRIDES) if legacy_env else {}

    def env_factory(seed):
        return make_env(scenario, seed=seed, **overrides)

    if method in HEURISTIC_METHODS:
        controller = HeuristicController(method, seed=seed)
    elif method in METHOD_TO_ENCODER:
        ckpt = os.path.join(seed_dir, "policy.pt")
        if not os.path.exists(ckpt):
            return {"status": "skipped_no_checkpoint", "dir": seed_dir}
        obs_dim = obs_dim_for(get_scenario(scenario, **overrides))
        encoder = build_encoder(METHOD_TO_ENCODER[method], in_dim=obs_dim,
                                hidden_dim=cfg["hidden_dim"])
        policy = GNNActorCritic(encoder, cfg["hidden_dim"], 13,
                                 per_agent_critic=not legacy_env,
                                 separate_critic_encoder=not legacy_env)
        policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
        policy.eval()
        controller = MAPPOController(policy, hidden_dim=cfg["hidden_dim"],
                                      deterministic=deterministic, seed=seed)
    else:  # maddpg — checkpointing was only enabled after the review
        return {"status": "skipped_needs_retrain", "dir": seed_dir}

    t0 = time.time()
    metrics = evaluate_policy(env_factory, controller, n_episodes=cfg["n_eval_episodes"],
                               seed_start=90_000 + seed * 1000)
    logger = RunLogger(out_dir, method, scenario, seed)
    logger.meta["legacy_env"] = legacy_env
    # preserve the learning curve that train.py wrote; only replace the eval
    curve_path = os.path.join(seed_dir, "learning_curve.json")
    if os.path.exists(curve_path):
        with open(curve_path, "r", encoding="utf-8") as f:
            logger.curve = json.load(f).get("curve", [])
    metrics["eval_protocol"] = "argmax" if deterministic else "sampled"
    logger.save_eval(metrics)
    save_trajectory_sample(env_factory, controller, logger, seed)
    return {"status": "ok", "dir": seed_dir,
            "vdr": metrics["victim_detection_rate"]["mean"],
            "coverage": metrics["coverage_rate"]["mean"],
            "secs": round(time.time() - t0, 1)}


def _job(args):
    return reevaluate_one(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--scenarios", nargs="*", default=SCENARIOS)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--max-jobs", type=int, default=4)
    ap.add_argument("--deterministic", action="store_true",
                    help="reproduce the old argmax protocol instead")
    ap.add_argument("--legacy-env", dest="legacy_env", action="store_true", default=True)
    ap.add_argument("--no-legacy-env", dest="legacy_env", action="store_false")
    args = ap.parse_args()

    tasks = []
    for scenario in args.scenarios:
        for method in args.methods:
            mdir = os.path.join(args.out_dir, scenario, method)
            if not os.path.isdir(mdir):
                continue
            for d in sorted(os.listdir(mdir)):
                if d.startswith("seed_"):
                    tasks.append((args.out_dir, scenario, method,
                                   int(d.split("_")[1]), args.deterministic, args.legacy_env))

    print(f"re-evaluating {len(tasks)} runs, "
          f"{'argmax' if args.deterministic else 'sampled'} protocol, "
          f"{'legacy' if args.legacy_env else 'reviewed'} env, {args.max_jobs} workers",
          flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=args.max_jobs) as ex:
        futures = {ex.submit(_job, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            done += 1
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{done}/{len(tasks)}] FAILED {t[1]}/{t[2]}/seed_{t[3]}: {e}", flush=True)
                continue
            if r["status"] == "ok":
                print(f"[{done}/{len(tasks)}] {t[1]}/{t[2]}/seed_{t[3]} "
                      f"VDR={r['vdr']:.3f} cov={r['coverage']:.3f} ({r['secs']}s)", flush=True)
            else:
                print(f"[{done}/{len(tasks)}] {t[1]}/{t[2]}/seed_{t[3]} {r['status']}", flush=True)


if __name__ == "__main__":
    main()
