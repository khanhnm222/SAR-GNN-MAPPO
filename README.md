# GNN-MAPPO cho phối hợp đàn UAV tìm kiếm cứu nạn (SAR)

Cài đặt thực nghiệm cho đề cương luận văn ThS. *"Xây dựng và đánh giá
framework GNN-MAPPO cho bài toán phối hợp đàn UAV tìm kiếm và cứu nạn trong
môi trường mô phỏng"*. Xem [PLAN.md](PLAN.md) để biết kế hoạch chi tiết và
phạm vi rút gọn (POC) của lần chạy này.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Kiểm chứng môi trường

```bash
python -m sar_env.validation
```

## Huấn luyện một cấu hình

```bash
python -m training.train --method gnn_mappo --scenario easy --seed 0
```

`--method` ∈ `{random_walk, greedy, maddpg, mappo_mlp, mappo_gcn, mappo_gat, gnn_mappo}`,
`--scenario` ∈ `{easy, medium, hard}`.

## Chạy toàn bộ ma trận thực nghiệm POC

```bash
python -m scripts.run_poc --scenarios easy medium hard
```

## Sinh hình ảnh kết quả (PNG)

```bash
python -m evaluation.plots
```
→ `results/figures/*.png` (learning curves, so sánh baseline, quỹ đạo UAV).

## Ablation kiến trúc / Zero-shot scalability

```bash
python -m evaluation.ablation
python -m evaluation.zero_shot easy gnn_mappo
```

## Xuất dữ liệu cho website & chạy website

```bash
python -m scripts.export_web_data
cd web
npm install
npm run dev
```

Mở `http://localhost:3000` — các trang: Tổng quan, Huấn luyện, So sánh
Baseline, Ablation, Zero-shot, Phát lại mô phỏng.

---

## Sau đợt review 03/09/2026

Xem [REVIEW_2026-09-03.md](REVIEW_2026-09-03.md). Tóm tắt các thay đổi ảnh
hưởng tới cách chạy:

**Giao thức đánh giá đã đổi.** Trước đây chính sách được chạy bằng `argmax`,
biến chính sách entropy cao thành một hành động duy nhất (FAST_FORWARD) — mọi
UAV bay thẳng vào biên bản đồ rồi đứng yên. Nay mặc định là lấy mẫu từ phân
phối chính sách. Đánh giá lại checkpoint cũ mà không cần huấn luyện lại:

```bash
python -m scripts.reevaluate --out-dir results --max-jobs 6
python -m scripts.export_web_data
```

**Môi trường có 5 cờ thiết kế mới** (`sar_env/scenarios.py`), mặc định BẬT:
`local_belief`, `include_neighbor_obs=False`, `per_agent_reward`,
`normalize_obs`, `normalize_edge_attr`. Chúng thay đổi `obs_dim` (116 → 86) nên
**checkpoint cũ không tương thích** — dùng `--legacy-env` để tái lập Giai đoạn
6/7:

```bash
python -m training.train --method gnn_mappo --scenario easy --seed 0 --legacy-env
```

**Kiểm thử các sửa lỗi này**:

```bash
python -m scripts.selftest_review_fixes
```
