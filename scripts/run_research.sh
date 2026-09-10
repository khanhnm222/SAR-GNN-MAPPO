#!/usr/bin/env bash
# Giai doan 6 (EXPERIMENT_LOG.md): 5 trainable methods x 3 scenarios x 5 seeds
# (nang tu 2-3 len 5 de kiem dinh thong ke co du luc phan biet kien truc),
# + 5 seed cho heuristic (truoc day chi 1, khong du de so sanh thong ke),
# + mot dot rieng GNN-MAPPO/Easy voi ngan sach lon hon nhieu (2M buoc, 3 seed)
# de kiem tra khoang cach voi MAPPO-GCN co thu hep khi co du du lieu hon.
#
# 108 lan chay tong cong -> dung job-pool (toi da MAX_JOBS song song) thay vi
# bung het cung luc, tranh oversubscription qua nang tren 16 loi.
# KHONG dung set -e: voi 108 lan chay, 1 lan that bai (vd loi thoang qua) khong
# duoc phep lam huy toan bo batch qua "wait -n" tra ve ma khac 0.
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p results/logs results_longrun_easy_gnn_mappo

MAX_JOBS=14
PIDS=()

launch() {
  local log_file="$1"; shift
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    wait -n
  done
  "$@" > "${log_file}" 2>&1 &
  PIDS+=($!)
}

declare -A SEEDS=( [easy]=5 [medium]=5 [hard]=5 )
TRAINABLE_METHODS=(maddpg mappo_mlp mappo_gcn mappo_gat gnn_mappo)
HEURISTICS=(random_walk greedy)

for scenario in easy medium hard; do
  n_seeds=${SEEDS[$scenario]}
  for method in "${HEURISTICS[@]}"; do
    for ((seed=0; seed<n_seeds; seed++)); do
      log_file="results/logs/${scenario}_${method}_seed${seed}.log"
      echo "[launch] scenario=${scenario} method=${method} seed=${seed}"
      launch "${log_file}" python -m training.train --method "${method}" --scenario "${scenario}" --seed "${seed}" \
        --out_dir results
    done
  done
  for method in "${TRAINABLE_METHODS[@]}"; do
    for ((seed=0; seed<n_seeds; seed++)); do
      log_file="results/logs/${scenario}_${method}_seed${seed}.log"
      echo "[launch] scenario=${scenario} method=${method} seed=${seed}"
      launch "${log_file}" python -m training.train --method "${method}" --scenario "${scenario}" --seed "${seed}" \
        --out_dir results
    done
  done
done

for seed in 0 1 2; do
  log_file="results/logs/longrun_easy_gnn_mappo_seed${seed}.log"
  echo "[launch] LONGRUN scenario=easy method=gnn_mappo seed=${seed} steps=2000000"
  launch "${log_file}" python -m training.train --method gnn_mappo --scenario easy --seed "${seed}" \
    --out_dir results_longrun_easy_gnn_mappo --total_env_steps 2000000
done

echo "Launched ${#PIDS[@]} runs total (job-pool max ${MAX_JOBS} concurrent)"
wait
echo "=== run_research (Giai doan 6) DONE: ${#PIDS[@]} runs finished ==="
