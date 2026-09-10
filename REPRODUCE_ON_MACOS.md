# Hướng dẫn chạy lại thực nghiệm trên MacBook (M3 Pro) để kiểm chứng

Mục tiêu: chạy lại đúng ma trận thực nghiệm chính thức của đề tài (Giai đoạn 6
trong [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), số liệu đang dùng trong
[CHUONG5_THUC_NGHIEM.md](CHUONG5_THUC_NGHIEM.md)) trên một máy khác — 2 heuristic
(Random Walk, Greedy) + 4 baseline học được (MADDPG, MAPPO-MLP, MAPPO-GCN,
MAPPO-GAT) + mô hình đề xuất (GNN-MAPPO/DGAT) — để đối chiếu xem kết luận có
đứng vững trên phần cứng khác hay không.

**Điểm cần biết trước:** toàn bộ pipeline chạy trên **CPU** (mục 5.1.1 —
không dùng CUDA/GPU), và code không có cờ chọn thiết bị (`device` mặc định
`"cpu"` cứng trong `algorithms/mappo.py`). Vì vậy trên MacBook M3 Pro, thực
nghiệm cũng sẽ chạy CPU-only, **không** dùng GPU/MPS của Apple Silicon — đây
là điều đúng ý muốn (so sánh công bằng, cùng điều kiện phần cứng loại "CPU"
như bản gốc), không phải hạn chế cần khắc phục.

---

## Bước 0 — Đưa mã nguồn sang MacBook

Kiểm tra `git status` cho thấy **phần lớn code hiện chưa được commit**
(`algorithms/`, `training/`, `sar_env/`, `models/`, `evaluation/`, `scripts/`,
`web/`, `requirements.txt`... đều là `??` — untracked). Nghĩa là nếu chỉ
`git clone` remote hiện tại, máy Mac sẽ **không** nhận được các thư mục này.
Chọn 1 trong 2 cách:

### Cách A — Commit + push rồi clone (khuyến nghị, có version control)

```bash
git add algorithms evaluation models sar_env scripts training web \
        requirements.txt README.md PLAN.md EXPERIMENT_LOG.md CHUONG5_THUC_NGHIEM.md \
        .gitignore
git commit -m "Add source code for cross-machine reproduction"
git push origin main
```

`.gitignore` (đã tạo sẵn) loại bỏ `results*/`, `web/node_modules/` (~812MB) —
những thứ tái sinh được, không cần đưa lên git. Trên MacBook:

```bash
git clone https://github.com/khanhnm222/SAR-GNN-MAPPO.git
cd SAR-GNN-MAPPO
```

### Cách B — Copy trực tiếp (nhanh, không cần đụng git)

Nén cả thư mục dự án (trừ `web/node_modules` và `results*/` nếu muốn nhẹ) rồi
chuyển qua AirDrop / USB / Google Drive / `scp`:

```bash
# Trên máy Windows (PowerShell, dùng 7-Zip hoặc tar có sẵn trong Git Bash)
tar --exclude='web/node_modules' --exclude='results*' --exclude='.git' \
    -czf sar-gnn-mappo.tar.gz .
```

Rồi giải nén trên Mac: `tar -xzf sar-gnn-mappo.tar.gz -C ~/SAR-GNN-MAPPO`.

---

## Bước 1 — Cài môi trường trên macOS

Yêu cầu: Python 3.10–3.12 (bản dùng khi phát triển: 3.12), không cần Xcode
Command Line Tools đặc biệt nào ngoài mức mặc định của Homebrew/pyenv.

```bash
# Cài Python (nếu chưa có) qua Homebrew
brew install python@3.12

# Tạo virtualenv riêng cho dự án
cd ~/SAR-GNN-MAPPO
python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Lưu ý về `torch` / `torch-geometric` trên Apple Silicon: bản `torch-geometric`
hiện dùng (2.8.x) không còn cần cài riêng `torch-scatter`/`torch-sparse` (PyG
có backend thuần PyTorch từ bản 2.3+), nên `pip install torch-geometric` chạy
thẳng, không cần tìm wheel index đặc biệt cho arm64. `pip install torch` trên
Mac sẽ tự lấy bản hỗ trợ Apple Silicon (dùng CPU trong pipeline này, như đã
nói ở trên).

---

## Bước 2 — Kiểm chứng cài đặt trước khi chạy full

### 2.1. Kiểm chứng môi trường mô phỏng

```bash
python -m sar_env.validation
```

Kỳ vọng (đối chiếu mục 5.2 CHUONG5_THUC_NGHIEM.md) — cả 3 kịch bản đều
`obs_valid=True, energy_non_increasing=True, illegal_move_blocked=True,
belief_converges=True, graph_symmetric=True, episode_terminates=True`
(độ dài episode có thể lệch chút do random walk chính sách mặc định của
validation, không phải dấu hiệu lỗi).

### 2.2. Smoke test nhanh (vài giây, không tốn thời gian)

```bash
python -m scripts.run_all_experiments \
  --scenarios easy --methods greedy gnn_mappo \
  --seeds 1 --steps-override 2000 \
  --out-dir results_smoketest --no-postprocess
```

Chạy xong không lỗi, và `results_smoketest/easy/gnn_mappo/seed_0/eval_metrics.json`
tồn tại → pipeline hoạt động đúng, có thể xoá `results_smoketest/` và chạy full.

---

## Bước 3 — Chạy toàn bộ ma trận thực nghiệm (1 lệnh)

Đã viết sẵn [scripts/run_all_experiments.py](scripts/run_all_experiments.py) —
bản Python cross-platform thay cho `scripts/run_research.sh` (script bash cũ
dùng `declare -A` associative array, cần bash ≥4, trong khi macOS mặc định đi
kèm bash 3.2 nên **sẽ báo lỗi cú pháp nếu chạy trực tiếp** trên Mac).

### Chạy đúng như Giai đoạn 6 (108 run: 3 scenario × (2 heuristic×5 seed + 5
phương pháp học×5 seed) + 3 seed longrun GNN-MAPPO/Easy 2 triệu bước)

```bash
python -m scripts.run_all_experiments --longrun --max-jobs 8
```

- `--max-jobs 8`: M3 Pro có 11 hoặc 12 lõi (5–6 hiệu năng + 6 tiết kiệm điện).
  Để lại 2–4 lõi cho hệ thống, bắt đầu với `8`; nếu máy quá nóng/chậm giảm
  xuống `5`–`6` (tương ứng số lõi hiệu năng), nếu ổn có thể tăng dần.
- Script **tự bỏ qua** cấu hình đã có `eval_metrics.json` (an toàn khi chạy
  lại sau khi bị ngắt) — dùng `--force` nếu muốn chạy lại từ đầu.
- Log của từng run: `results/logs/<scenario>_<method>_seed<seed>.log`.
- Sau khi xong, tóm tắt được ghi vào `results/run_all_experiments_summary.json`
  (liệt kê run nào thất bại nếu có).

### Ước tính thời gian

Theo comment trong `training/configs/*.yaml` (đo trên máy phát triển gốc,
đơn luồng): Easy ~290 bước/s, Medium ~133 bước/s, Hard ~59 bước/s. Tổng
CPU-time toàn bộ 108 run (bao gồm longrun) ước khoảng **70–75 giờ CPU**; với
`--max-jobs 8` chạy song song, thời gian thực tế rơi vào khoảng **9–12 giờ**
(phụ thuộc hiệu năng lõi thực tế của M3 Pro — nên chạy thử Bước 2.2 trước để
có cảm nhận tốc độ máy bạn, rồi ngoại suy). Có thể chạy qua đêm hoặc chia nhỏ
theo `--scenarios`/`--methods` để kiểm tra dần:

```bash
# Chạy trước scenario easy để xem sớm, rồi mới chạy medium/hard
python -m scripts.run_all_experiments --scenarios easy --max-jobs 8
python -m scripts.run_all_experiments --scenarios medium hard --max-jobs 8
python -m scripts.run_all_experiments --longrun --max-jobs 4
```

Muốn xem trước kế hoạch mà chưa chạy gì: thêm `--dry-run`.

---

## Bước 4 — Hậu xử lý (tự động nếu dùng `--out-dir results` mặc định)

`run_all_experiments.py` tự chạy 3 bước sau khi training xong (bỏ qua bằng
`--no-postprocess`, hoặc chạy tay nếu bạn dùng `--out-dir` khác `results`):

```bash
python -m evaluation.ablation      # so sánh kiến trúc encoder -> results/<scenario>/architecture_ablation.json
python -m evaluation.zero_shot easy gnn_mappo
python -m evaluation.zero_shot medium gnn_mappo
python -m evaluation.zero_shot hard gnn_mappo
python -m evaluation.plots         # -> results/figures/*.png
```

Tuỳ chọn — dựng lại web dashboard để xem trực quan (không bắt buộc cho việc
kiểm chứng số liệu):

```bash
python -m scripts.export_web_data
cd web && npm install && npm run dev
# mở http://localhost:3000
```

---

## Bước 5 — So sánh kết quả giữa 2 máy

Sau khi có `results/` trên cả Windows (máy gốc) và MacBook (`results` trên
Mac — đổi tên/copy về máy gốc thành ví dụ `results_macos/` để so sánh cạnh
nhau), dùng [scripts/compare_runs.py](scripts/compare_runs.py):

```bash
python -m scripts.compare_runs \
  --a results --b results_macos \
  --scenarios easy medium hard \
  --out-json compare_windows_vs_macos.json
```

Script in ra, theo từng (kịch bản, phương pháp, chỉ số): mean của 2 máy,
p-value và Cohen's d — dùng đúng kiểm định Welch's t-test / Mann-Whitney U có
sẵn trong `evaluation/statistics.py` (giống cách `evaluation/ablation.py` so
sánh baseline).

### Cách đọc kết quả so sánh — **quan trọng**

CPU x86 (Windows) và CPU ARM (Apple Silicon) dùng thư viện BLAS và thứ tự
vector hoá khác nhau, nên **kết quả sẽ không bit-exact dù cùng seed** — đây là
điều bình thường của mọi pipeline PyTorch chạy CPU trên kiến trúc khác nhau,
không phải lỗi. Tiêu chí "kiểm chứng thành công":

- GNN-MAPPO vẫn thắng các baseline trên **cả 2 máy** theo cùng chiều so sánh,
  dù con số tuyệt đối có thể xê dịch vài phần trăm — bấy nhiêu là đủ để coi
  là kiểm chứng thành công.
- Một vài ô có p<0.05 riêng lẻ không đáng lo với n=5 seed/nhóm (thống kê còn
  yếu) — chỉ đáng lo khi **chiều so sánh đảo ngược** (vd: GNN-MAPPO tốt hơn
  MAPPO-GCN trên Windows nhưng tệ hơn trên Mac) hoặc |Cohen's d| > 1.

---

## Xử lý sự cố thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `run_research.sh: declare: -A: invalid option` | Đang chạy script bash cũ trên bash 3.2 mặc định của macOS. Dùng `scripts/run_all_experiments.py` thay thế (đã xử lý ở trên), hoặc `brew install bash` rồi gọi `/opt/homebrew/bin/bash scripts/run_research.sh`. |
| `ModuleNotFoundError: torch_geometric` | Chưa `pip install -r requirements.txt` trong đúng venv đang active — kiểm tra `which python` trỏ vào `.venv/bin/python`. |
| Máy quá nóng / quạt kêu to khi chạy `--max-jobs` cao | Giảm `--max-jobs` xuống bằng đúng số lõi hiệu năng (P-core) của chip, ví dụ `5` hoặc `6`. |
| Một vài run trong log báo lỗi giữa chừng | Xem file log tương ứng trong `results/logs/`; chạy lại riêng run đó (script tự bỏ qua run đã xong, không cần `--force` toàn bộ): `python -m training.train --method <method> --scenario <scenario> --seed <seed> --out_dir results`. |
| Muốn dừng giữa chừng rồi chạy tiếp sau | An toàn — chỉ cần chạy lại đúng lệnh cũ, script tự bỏ qua các cấu hình đã có `eval_metrics.json`. |
