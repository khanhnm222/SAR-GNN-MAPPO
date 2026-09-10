"""Ma tran thuc nghiem CUOI CUNG: ca ba kich ban, moi truong da sua het sau lan
ra soat 03-05/09/2026, ghi vao results_v4/.

Khac voi results_v3_rcomm/:
  * truncated BPTT (`bptt_len=8`) cho encoder hoi tiep, nen GNN-MAPPO (DGAT co
    GRU) lan dau tien duoc huan luyen dung cach. Truoc do BPTT dai dung 1 buoc
    va GRU LAM GIAM hieu nang co y nghia thong ke (Easy 0,890 vs 0,910,
    p = 0,0116) — tuc kien truc de xuat bi danh gia bat cong.
  * chay ca Hard (16 UAV, 200x200) de du ba kich ban theo de cuong.

Ngan sach theo do kho: Easy 500k, Medium 800k, Hard 1M buoc; 7 phuong phap x
5 seed moi kich ban = 81 luot.

Chay:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.run_v4_matrix

Idempotent: cau hinh da co eval_metrics.json se bi bo qua.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

OUT_DIR = "results_v4"
PHASES = [
    ("easy", None),          # 500k tu training/configs/easy.yaml
    ("medium", 800_000),
    ("hard", 1_000_000),
]


def main():
    max_jobs = sys.argv[1] if len(sys.argv) > 1 else "6"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()

    for scenario, steps in PHASES:
        cmd = [sys.executable, "-m", "scripts.run_all_experiments",
               "--scenarios", scenario, "--out-dir", OUT_DIR,
               "--max-jobs", max_jobs, "--no-postprocess"]
        if steps:
            cmd += ["--steps-override", str(steps)]
        print(f"\n{'=' * 70}\n=== {scenario} (t+{(time.time() - t0) / 3600:.1f}h)\n"
              f"{' '.join(cmd)}\n{'=' * 70}", flush=True)
        rc = subprocess.run(cmd, env=env).returncode
        print(f"=== {scenario} xong, rc={rc}, t+{(time.time() - t0) / 3600:.1f}h ===", flush=True)

    print(f"\nTOAN BO XONG sau {(time.time() - t0) / 3600:.1f} gio.", flush=True)


if __name__ == "__main__":
    main()
