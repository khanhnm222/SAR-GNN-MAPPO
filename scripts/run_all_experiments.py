"""Chay TOAN BO ma tran thuc nghiem (heuristic + baseline + GNN-MAPPO de xuat)
bang MOT lenh duy nhat, cross-platform (Windows/macOS/Linux) — thay the cho
`scripts/run_research.sh` (dung bash 4+ associative array, khong chay duoc voi
bash 3.2 mac dinh tren macOS).

Vi du dung tren MacBook (macOS/Apple Silicon):
    python -m scripts.run_all_experiments --max-jobs 8

Chay day du dung nhu Giai doan 6 (EXPERIMENT_LOG.md) + dot longrun GNN-MAPPO/Easy:
    python -m scripts.run_all_experiments --longrun --max-jobs 8

Chi chay thu nhanh de kiem tra pipeline (1 seed, 2000 buoc moi run):
    python -m scripts.run_all_experiments --seeds 1 --steps-override 2000 \
        --out-dir results_smoketest --no-postprocess

Script tu dong BO QUA cac cau hinh da co eval_metrics.json (idempotent) —
chay lai an toan sau khi bi ngat giua chung, dung --force de chay lai tat ca.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import yaml

HEURISTIC_METHODS = {"random_walk", "greedy"}
METHOD_ORDER = ["random_walk", "greedy", "maddpg", "mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo"]
LONGRUN_SEEDS = 3
LONGRUN_STEPS = 2_000_000


@dataclass
class Task:
    method: str
    scenario: str
    seed: int
    out_dir: str
    total_env_steps: int | None = None
    extra_tag: str = ""
    legacy_env: bool = False

    @property
    def name(self) -> str:
        tag = f"_{self.extra_tag}" if self.extra_tag else ""
        return f"{self.scenario}_{self.method}_seed{self.seed}{tag}"

    @property
    def eval_metrics_path(self) -> str:
        return os.path.join(self.out_dir, self.scenario, self.method, f"seed_{self.seed}", "eval_metrics.json")

    def is_done(self) -> bool:
        return os.path.exists(self.eval_metrics_path)

    def command(self, python_exe: str) -> list[str]:
        cmd = [python_exe, "-m", "training.train",
               "--method", self.method, "--scenario", self.scenario, "--seed", str(self.seed),
               "--out_dir", self.out_dir]
        if self.total_env_steps:
            cmd += ["--total_env_steps", str(self.total_env_steps)]
        if self.legacy_env:
            cmd += ["--legacy-env"]
        return cmd


def build_plan(scenarios: list[str], methods: list[str], out_dir: str,
                seeds_override: int | None, longrun: bool, longrun_out_dir: str,
                legacy_env: bool = False, seed_offset: int = 0) -> list[Task]:
    plan: list[Task] = []
    for scenario in scenarios:
        with open(f"training/configs/{scenario}.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        n_seeds = seeds_override or cfg["num_seeds"]
        for method in methods:
            # Heuristics used to get a single seed, which left their std at 0
            # and made every statistical test against them impossible. Random
            # Walk is stochastic and Greedy has random tie-breaking, so they
            # need the same n as everything else (and they cost seconds).
            n = n_seeds
            # seed_offset: an independent replicate needs DIFFERENT seeds now
            # that every RNG is seeded -- re-running seeds 0..4 would reproduce
            # the exact same numbers, not a second sample.
            for seed in range(seed_offset, seed_offset + n):
                plan.append(Task(method=method, scenario=scenario, seed=seed, out_dir=out_dir,
                                  legacy_env=legacy_env))
    if longrun:
        for seed in range(LONGRUN_SEEDS):
            plan.append(Task(method="gnn_mappo", scenario="easy", seed=seed, out_dir=longrun_out_dir,
                              total_env_steps=LONGRUN_STEPS, extra_tag="longrun",
                              legacy_env=legacy_env))
    return plan


def run_task(task: Task, python_exe: str, log_dir: str) -> tuple[Task, int, float]:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{task.name}.log")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(task.command(python_exe), stdout=logf, stderr=subprocess.STDOUT)
    return task, proc.returncode, time.time() - t0


def run_postprocess(python_exe: str, legacy_env: bool = False) -> None:
    print("\n=== Post-processing: ablation + zero-shot + plots (out_dir=results) ===", flush=True)
    zs = ["--legacy-env"] if legacy_env else []
    steps = [
        [python_exe, "-m", "evaluation.ablation"],
        [python_exe, "-m", "evaluation.zero_shot", "easy", "gnn_mappo", *zs],
        [python_exe, "-m", "evaluation.zero_shot", "medium", "gnn_mappo", *zs],
        [python_exe, "-m", "evaluation.zero_shot", "hard", "gnn_mappo", *zs],
        [python_exe, "-m", "evaluation.plots"],
    ]
    for cmd in steps:
        print(f"$ {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"!!! buoc that bai (bo qua, tiep tuc): {' '.join(cmd)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--methods", nargs="+", default=METHOD_ORDER)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--longrun", action="store_true",
                         help="Them dot GNN-MAPPO/Easy 3 seed x 2.000.000 buoc (Giai doan 6)")
    parser.add_argument("--longrun-out-dir", default="results_longrun_easy_gnn_mappo")
    parser.add_argument("--seeds", type=int, default=None,
                         help="Ghi de so seed/cau hinh (mac dinh: doc tu training/configs/<scenario>.yaml)")
    parser.add_argument("--steps-override", type=int, default=None,
                         help="Ghi de total_env_steps cho MOI run (dung de smoke-test nhanh)")
    parser.add_argument("--max-jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                         help="So tien trinh song song toi da (mac dinh: so_loi_CPU - 2)")
    parser.add_argument("--force", action="store_true", help="Chay lai ca cac cau hinh da co eval_metrics.json")
    parser.add_argument("--dry-run", action="store_true", help="Chi in ke hoach, khong chay gi")
    parser.add_argument("--no-postprocess", action="store_true",
                         help="Bo qua buoc ablation/zero-shot/plots sau khi huan luyen xong")
    parser.add_argument("--seed-offset", type=int, default=0,
                         help="Seed bat dau (vd 5 de chay lap lai doc lap voi seed 5..9)")
    parser.add_argument("--legacy-env", action="store_true",
                         help="Tai lap moi truong + trainer cua Giai doan 6/7 (truoc review 03/09/2026)")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    plan = build_plan(args.scenarios, args.methods, args.out_dir, args.seeds, args.longrun,
                       args.longrun_out_dir, legacy_env=args.legacy_env, seed_offset=args.seed_offset)
    if args.steps_override:
        for t in plan:
            t.total_env_steps = args.steps_override

    if not args.force:
        pending = [t for t in plan if not t.is_done()]
        skipped = len(plan) - len(pending)
    else:
        pending, skipped = plan, 0

    print(f"=== run_all_experiments: {len(plan)} cau hinh trong ke hoach, "
          f"{skipped} da xong (bo qua), {len(pending)} se chay, max_jobs={args.max_jobs} ===", flush=True)
    for t in pending:
        steps_str = f", steps={t.total_env_steps}" if t.total_env_steps else ""
        print(f"  - {t.name} -> {t.out_dir}{steps_str}")

    if args.dry_run or not pending:
        if not pending:
            print("Khong co gi de chay (tat ca da xong). Dung --force de chay lai.")
        return

    t0 = time.time()
    results: list[tuple[Task, int, float]] = []
    log_dirs = {os.path.join(t.out_dir, "logs") for t in pending}
    for d in log_dirs:
        os.makedirs(d, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.max_jobs) as pool:
        futures = {pool.submit(run_task, t, args.python, os.path.join(t.out_dir, "logs")): t for t in pending}
        for i, fut in enumerate(as_completed(futures), 1):
            task, code, elapsed = fut.result()
            status = "OK" if code == 0 else f"FAIL(code={code})"
            print(f"[{i}/{len(pending)}] {task.name}: {status} ({elapsed:.0f}s, "
                  f"tong {time.time() - t0:.0f}s) — log: {os.path.join(task.out_dir, 'logs', task.name + '.log')}",
                  flush=True)
            results.append((task, code, elapsed))

    failed = [t for t, code, _ in results if code != 0]
    print(f"\n=== HOAN TAT: {len(results)} run trong {time.time() - t0:.0f}s, {len(failed)} that bai ===")
    if failed:
        print("Cac run that bai (xem log de biet chi tiet):")
        for t in failed:
            print(f"  - {t.name} (log: {os.path.join(t.out_dir, 'logs', t.name + '.log')})")

    summary_path = os.path.join(args.out_dir, "run_all_experiments_summary.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(plan), "skipped": skipped, "ran": len(pending),
            "failed": [t.name for t in failed],
            "elapsed_seconds": time.time() - t0,
        }, f, indent=2)
    print(f"Tom tat da ghi vao {summary_path}")

    if not args.no_postprocess and args.out_dir == "results" and not failed:
        run_postprocess(args.python, legacy_env=args.legacy_env)
    elif not args.no_postprocess and args.out_dir != "results":
        print(f"\n(Bo qua post-processing tu dong vi --out-dir != 'results'. "
              f"evaluation/ablation.py, zero_shot.py, plots.py dang hardcode out_dir='results' — "
              f"chay thu cong hoac sua lai out_dir trong cac file do neu can.)")


if __name__ == "__main__":
    main()
