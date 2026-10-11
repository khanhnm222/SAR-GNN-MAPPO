"""Zero-shot scalability (RQ3) cho moi checkpoint trong mot thu muc ket qua.

Chay tat ca (kien truc x seed x N) roi gop thanh mot file
`<out_dir>/<scenario>/zero_shot_all.json` (fixed-map) hoac
`zero_shot_density_all.json` (density-controlled), dang
method -> seed -> N -> metrics, la dang `scripts.export_web_data` doc.

Vi du:
    OMP_NUM_THREADS=1 python -m scripts.run_zero_shot_all --out-dir results_v4 --mode fixed_map --seeds 0-9
    OMP_NUM_THREADS=1 python -m scripts.run_zero_shot_all --out-dir results_v4 --mode density --seeds 0-9 --workers 3

Idempotent: ket qua tung (scenario, method, seed) duoc cache trong
`<out_dir>/<scenario>/zero_shot_cache/<mode>_<method>_seed<seed>.json`; chay lai
chi lam nhung to hop chua co. Bo `--force` de tinh lai.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, ".")
from evaluation.zero_shot import zero_shot_eval  # noqa: E402

# N kiem thu theo kich ban (gom ca N < N huan luyen); N huan luyen duoc them tu dong.
TEST_N = {"easy": [2, 6, 8, 12], "medium": [4, 6, 10, 12, 16], "hard": [8, 12, 20, 24]}
METHODS = ["mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo"]
OUT_NAME = {"fixed_map": "zero_shot_all.json", "density": "zero_shot_density_all.json"}


def parse_seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def job(args):
    out_dir, sc, m, seed, mode, n_episodes, cache = args
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            return sc, m, seed, json.load(f)
    r = zero_shot_eval(out_dir, sc, m, seed, test_n_values=TEST_N[sc], n_episodes=n_episodes, mode=mode)
    payload = {"by_n": r["by_n"], "env_by_n": r["env_by_n"], "mode": mode}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return sc, m, seed, payload


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results_v4")
    ap.add_argument("--scenarios", nargs="*", default=list(TEST_N))
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--seeds", default="0-4", help="vd 0-4 hoac 0,1,2 hoac 0-9")
    ap.add_argument("--mode", choices=list(OUT_NAME), default="fixed_map")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)

    tasks = []
    for sc in a.scenarios:
        for m in a.methods:
            for s in seeds:
                if os.path.exists(os.path.join(a.out_dir, sc, m, f"seed_{s}", "policy.pt")):
                    cache = os.path.join(a.out_dir, sc, "zero_shot_cache", f"{a.mode}_{m}_seed{s}.json")
                    if a.force and os.path.exists(cache):
                        os.remove(cache)
                    tasks.append((a.out_dir, sc, m, s, a.mode, a.episodes, cache))
    print(f"{len(tasks)} tac vu zero-shot ({a.mode}), {a.workers} worker", flush=True)

    out: dict = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(job, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            sc, m, seed, payload = f.result()
            out.setdefault(sc, {}).setdefault(m, {})[str(seed)] = payload["by_n"]
            out[sc].setdefault("_env_by_n", payload["env_by_n"])
            print(f"[{i}/{len(tasks)}] {sc}/{m}/seed{seed}", flush=True)

    for sc, data in out.items():
        env_by_n = data.pop("_env_by_n", None)
        path = os.path.join(a.out_dir, sc, OUT_NAME[a.mode])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if env_by_n:
            with open(path.replace(".json", "_env.json"), "w", encoding="utf-8") as f:
                json.dump(env_by_n, f, indent=2)
        print(f"ghi {path}", flush=True)


if __name__ == "__main__":
    main()
