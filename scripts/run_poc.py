"""Runs the full POC experiment matrix: 7 methods x scenario x num_seeds
(counts per training/configs/*.yaml), sequentially, logging progress to stdout
and to results/run_poc.log. Designed to run unattended in the background.

Usage: python -m scripts.run_poc [--scenarios easy medium hard] [--out_dir results]
"""
from __future__ import annotations

import argparse
import sys
import time

import yaml

sys.path.insert(0, ".")
from training.train import ALL_METHODS, run  # noqa: E402

METHOD_ORDER = ["random_walk", "greedy", "maddpg", "mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--methods", nargs="+", default=METHOD_ORDER)
    args = parser.parse_args()

    t0 = time.time()
    total_runs = 0
    plan = []
    for scenario in args.scenarios:
        with open(f"training/configs/{scenario}.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        n_seeds = cfg["num_seeds"]
        for method in args.methods:
            n = 1 if method in ("random_walk", "greedy") else n_seeds
            for seed in range(n):
                plan.append((method, scenario, seed))
    total_runs = len(plan)
    print(f"=== run_poc: {total_runs} runs planned across scenarios={args.scenarios} ===", flush=True)

    for i, (method, scenario, seed) in enumerate(plan, 1):
        print(f"\n--- [{i}/{total_runs}] method={method} scenario={scenario} seed={seed} "
              f"(elapsed {time.time() - t0:.0f}s) ---", flush=True)
        try:
            run(method, scenario, seed, out_dir=args.out_dir)
        except Exception as e:  # keep going even if one run fails
            print(f"!!! run failed: method={method} scenario={scenario} seed={seed}: {e}", flush=True)

    print(f"\n=== run_poc DONE: {total_runs} runs in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
