"""Chay lai TOAN BO ma tran thuc nghiem tren moi truong da sua (REVIEW_2026-09-03.md).

Easy 500k buoc + Medium 800k buoc, 7 phuong phap x 5 seed, ghi vao results_v2/.
Hard bi bo qua co chu y: o 200k buoc khong phuong phap nao hoi tu, nen chay lai
chi tai tao ket luan "khong ai hoc duoc" — do la gioi han tai nguyen chu khong
phai phat hien khoa hoc. Ngan sach do duoc don sang Medium (400k -> 800k), dung
cho ma gia thuyet "do thi chi co loi khi N lon" can duoc kiem chung.

Chay:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.run_v2_matrix

Idempotent: cac cau hinh da co eval_metrics.json se bi bo qua, nen ngat giua
chung roi chay lai van tiep tuc duoc.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

OUT_DIR = "results_v2"
PHASES = [
    ("easy", None),        # dung total_env_steps trong training/configs/easy.yaml (500k)
    ("medium", 800_000),   # gap doi ngan sach cu
]


def main():
    max_jobs = sys.argv[1] if len(sys.argv) > 1 else "6"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()

    for scenario, steps in PHASES:
        cmd = [sys.executable, "-m", "scripts.run_all_experiments",
               "--scenarios", scenario,
               "--out-dir", OUT_DIR,
               "--max-jobs", max_jobs,
               "--no-postprocess"]
        if steps:
            cmd += ["--steps-override", str(steps)]
        print(f"\n{'=' * 70}\n=== {scenario} (t+{(time.time() - t0) / 3600:.1f}h) "
              f"===\n{' '.join(cmd)}\n{'=' * 70}", flush=True)
        rc = subprocess.run(cmd, env=env).returncode
        print(f"=== {scenario} ket thuc, rc={rc}, t+{(time.time() - t0) / 3600:.1f}h ===", flush=True)

    print(f"\nTOAN BO XONG sau {(time.time() - t0) / 3600:.1f} gio.", flush=True)
    print(f"Buoc tiep theo:\n"
          f"  python -m scripts.export_web_data --results-dir {OUT_DIR}\n"
          f"  python -m evaluation.zero_shot easy gnn_mappo --out-dir {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
