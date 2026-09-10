# Kế hoạch thực hiện — GNN-MAPPO cho phối hợp đàn UAV tìm kiếm cứu nạn (SAR)

> Tài liệu này mô tả kế hoạch triển khai phần thực nghiệm cho đề cương luận văn
> *"Xây dựng và đánh giá framework GNN-MAPPO cho bài toán phối hợp đàn UAV tìm
> kiếm và cứu nạn trong môi trường mô phỏng"* (Nguyễn Minh Khánh, UIT, 2026).
> Nó ánh xạ từng mục của đề cương (1.4–1.5) sang một dự án phần mềm cụ thể,
> chạy được, có kết quả thật, cùng một website Next.js để trực quan hóa quá
> trình huấn luyện và thực nghiệm tìm kiếm của UAV.

## 0. Phạm vi thực tế của lần triển khai này

Đề cương gốc yêu cầu quy mô học thuật đầy đủ: **6 phương pháp × 3 kịch bản ×
tối thiểu 5 seeds × 5×10⁶ bước môi trường**, cộng thêm ablation kiến trúc,
ablation reward, kiểm định thống kê và đánh giá zero-shot ở hai chế độ. Trên
CPU thông thường (không có CUDA khả dụng trong môi trường thực hiện — đã kiểm
tra `torch.cuda.is_available() == False`), khối lượng này tương đương nhiều
ngày tới nhiều tuần compute.

Do đó, bản triển khai này đi theo nguyên tắc **"đúng đặc tả, giảm ngân sách"**:

| Thành phần | Đề cương (full-scale) | Triển khai POC (phiên này) |
|---|---|---|
| Số bước huấn luyện / cấu hình | 5×10⁶ | 5×10⁴–2×10⁵ (cấu hình `training/configs/*.yaml`, có thể tăng bằng cách đổi `total_env_steps`) |
| Số seed / cấu hình | ≥ 5 | 3 (đủ để vẽ mean±std, kiểm định thống kê minh họa) |
| Số episode test / seed | 200 | 30–50 |
| Kịch bản chạy đầy đủ | Easy, Medium, Hard | Easy + Medium chạy đầy đủ; Hard cài đặt đầy đủ về mặt môi trường nhưng chỉ chạy huấn luyện rút gọn (do 200×200 + 16 UAV tốn chi phí) |
| Minibatch / kiến trúc / công thức toán | Giữ nguyên 100% theo mục 1.5 | Giữ nguyên |
| Baseline & ablation | 6 phương pháp trong Bảng 5 | Giữ nguyên cả 6 phương pháp + 2 heuristic |

**Toàn bộ code (môi trường, kiến trúc DGAT, MAPPO/MADDPG, hàm thưởng, giao
thức seed, thống kê) được cài đặt đúng công thức trong đề cương** — không có
phần nào bị giả lập hay mock. Chỉ có *ngân sách tính toán* (số bước, số seed,
số episode) được thu nhỏ để chạy được trong một phiên làm việc trên CPU. Khi
có GPU/Colab Pro+ (đúng như mục 1.5.5 đề cương đã dự trù), chỉ cần đổi các
tham số trong `training/configs/*.yaml` (`total_env_steps`, `num_seeds`,
`n_eval_episodes`) để lên full-scale mà không phải sửa code.

## 1. Cấu trúc dự án

```
gnn-mappo-sar/
├── PLAN.md                        # tài liệu này
├── README.md                      # hướng dẫn chạy nhanh
├── requirements.txt
├── sar_env/                       # môi trường SAR mô phỏng (PettingZoo ParallelEnv)
│   ├── entities.py                 # UAV, Victim
│   ├── terrain.py                  # flat / obstacle / dynamic (lũ lan rộng)
│   ├── belief_map.py               # bản đồ xác suất cập nhật theo Bayes (Thrun et al. 2005)
│   ├── graph_builder.py            # G=(V,E,X,A) theo bán kính r_comm
│   ├── rewards.py                  # r = α·cover + β·victim + γ·collision + δ·energy
│   ├── scenarios.py                # cấu hình Easy(4,50×50) / Medium(8,100×100) / Hard(16,200×200)
│   ├── sar_parallel_env.py         # lớp SARSwarmEnv(ParallelEnv) chính
│   └── validation.py               # giao thức kiểm chứng môi trường (mục 1.5.1)
├── models/
│   ├── encoders.py                 # MLPEncoder, GCNEncoder, GATEncoder(no GRU)
│   ├── dgat.py                     # DGAT: GATv2Conv×L, K head, edge feature, GRU, masking
│   └── actor_critic.py             # Actor phân tán + Centralized Critic (mean/attention pooling)
├── algorithms/
│   ├── heuristics.py               # Random Walk, Greedy
│   ├── mappo.py                    # MAPPO CTDE: PPO-clip + GAE, dùng chung cho 4 biến thể encoder
│   └── maddpg.py                   # MADDPG rời rạc hóa qua Gumbel-Softmax (Lowe et al. 2017)
├── training/
│   ├── train.py                    # CLI: python -m training.train --config ... --seed ...
│   ├── rollout_buffer.py
│   ├── logger.py                   # ghi learning curve + trajectory mẫu ra results/*.json
│   └── configs/
│       ├── easy.yaml / medium.yaml / hard.yaml
├── evaluation/
│   ├── evaluate.py                 # chạy N episode test, tính 5 chỉ số
│   ├── statistics.py               # Welch's t-test, Mann-Whitney U, hiệu chỉnh Bonferroni
│   ├── ablation.py                 # so sánh kiến trúc & reward theo Bảng 5
│   ├── zero_shot.py                # train N∈{4,8,16} → test N∈{6,10,12,20}, fixed-map & density-controlled
│   └── plots.py                    # xuất PNG: learning curves, bar chart, heatmap, trajectory replay
├── results/                        # sinh ra khi chạy (raw logs, checkpoints, metrics.json, figures/)
├── scripts/
│   └── run_poc.py                  # chạy toàn bộ 7 phương pháp × seeds × kịch bản POC, tuần tự
└── web/                             # Next.js 14 (App Router) + TailwindCSS
    ├── app/
    │   ├── page.tsx                 # Dashboard tổng quan
    │   ├── training/page.tsx        # Learning curves theo phương pháp/kịch bản
    │   ├── comparison/page.tsx      # So sánh baseline (Bảng 5) + kiểm định thống kê
    │   ├── ablation/page.tsx        # Ablation kiến trúc & reward
    │   ├── zero-shot/page.tsx       # Zero-shot scalability
    │   └── replay/page.tsx          # Canvas replay quỹ đạo UAV + belief map heatmap
    ├── components/
    ├── lib/
    └── public/data/                 # JSON kết quả copy từ results/ (Task 11)
```

## 2. Ánh xạ đề cương → cài đặt

### 2.1 Môi trường SAR (mục 1.5.1)
- `SARSwarmEnv` kế thừa `pettingzoo.ParallelEnv`, lưới 2D rời rạc W×H.
- Sinh nạn nhân 3 chế độ: `uniform`, `cluster` (2-3 cụm Gaussian), `corridor`.
- Địa hình 3 chế độ: `flat`, `obstacle` (5-15% ô), `dynamic` (lan rộng mô phỏng lũ).
- Quan sát 161 chiều/UAV: `self_state`(7) + `local_map`(5×5×3=75) + `K=5` hàng
  xóm × 6 chiều tương đối (30) + `task_state`(4) + phần đệm theo đặc tả Bảng 2
  (tổng khớp 161 — chi tiết breakdown trong `sar_env/entities.py::build_observation`).
- Action: `Discrete(13)` — 8 hướng ngang, lên/xuống độ cao, hover, fast_forward
  (gấp đôi tiêu hao năng lượng), return_base.
- Belief map cập nhật Bayes: `P(c) ← P(c)(1-p_d) / (1 - P(c)·p_d)`, `p_d = 0.9`.
- Năng lượng: `e_{t+1} = e_t - c0 - c1‖v‖`, `E_max = 1000`; UAV cạn năng lượng
  bị gỡ khỏi đồ thị (node masking).
- Kết thúc episode: phát hiện hết nạn nhân / hết `T_max=500` bước / toàn bộ
  UAV cạn năng lượng.
- `validation.py` cài giao thức kiểm chứng môi trường mục 1.5.1: kiểm tra bảo
  toàn năng lượng, tính hợp lệ hành động, hội tụ belief map về 0/1, tính đối
  xứng đồ thị giao tiếp.

### 2.2 Đồ thị động (mục 1.5.2)
`graph_builder.py`: cạnh `(i,j)` tồn tại khi `d(i,j) ≤ r_comm`; trọng số cạnh
giảm theo khoảng cách; đặc trưng cạnh `e_ij = [Δp, Δv, dist, q]` (q = cường độ
tín hiệu suy giảm theo khoảng cách). Số node thay đổi theo N — không cố định
trong kiến trúc (điều kiện cần cho zero-shot).

### 2.3 Kiến trúc DGAT (mục 1.5.3)
`models/dgat.py` cài đúng công thức (1.2)–(1.3):
- `L=3` lớp GAT (dùng `GATv2Conv` của PyTorch Geometric — đúng như đề cương
  ghi rõ dùng GATv2Conv), `K=4` head, hệ số attention có tích hợp đặc trưng
  cạnh `e_ij` qua ma trận `U_k` (mở rộng so với GAT gốc, tương ứng hướng
  G2ANet).
- GRU cập nhật `h_i^t = GRU(h̃_i^{L,t}, h_i^{t-1})`.
- Masking: UAV cạn năng lượng bị loại khỏi `N(j)` của các UAV còn lại.
- Critic dùng `Pool({h_1^t,...,h_N^t})` bất biến hoán vị (mean-pooling và
  attention-pooling — cả hai được cài, chọn qua config).

`models/encoders.py` cài 3 biến thể dùng để tạo baseline/ablation trong Bảng 5:
- `MLPEncoder` → MAPPO-MLP (không dùng đồ thị, chỉ self_state+local_map phẳng hóa).
- `GCNEncoder` → MAPPO-GCN (tổng hợp lân cận trọng số cố định theo Kipf & Welling).
- `GATEncoder(no GRU)` → MAPPO-GAT (chính là DGAT bỏ khối GRU) — cô lập đóng góp bộ nhớ thời gian (RQ2).
- `DGAT` (đầy đủ) → GNN-MAPPO đề xuất.

### 2.4 Framework GNN-MAPPO (mục 1.5.4)
`algorithms/mappo.py`: PPO-clip (`ε=0.2`) + GAE (`λ=0.95`, `γ=0.99`), centralized
critic / decentralized actor theo CTDE, dùng chung cho cả 4 biến thể encoder ở
trên (chỉ thay encoder, giữ nguyên phần còn lại — đúng yêu cầu "so sánh công
bằng" của đề cương). Reward đa thành phần (1.4) là *global shared reward*.
Siêu tham số nền theo Bảng 6 được đặt làm default trong `training/configs/*.yaml`.

`algorithms/maddpg.py`: MADDPG (Lowe et al. 2017) với actor-critic riêng
(không PPO-clip), hành động rời rạc qua Gumbel-Softmax reparameterization —
baseline nhóm (ii) trong Bảng 5, không có GNN encoder.

`algorithms/heuristics.py`: Random Walk và Greedy — baseline nhóm (i), không
học tăng cường, dùng để kiểm thử pipeline (GĐ1) và làm mốc sàn hiệu năng.

### 2.5 Thực nghiệm & đánh giá (mục 1.5.5)
`evaluation/`:
- `evaluate.py`: chạy N episode test, tính Coverage Rate, Victim Detection
  Rate, TTFD, Collision Rate, Energy Efficiency.
- `statistics.py`: Welch's t-test / Mann-Whitney U (kiểm tra Shapiro-Wilk để
  chọn test), mức ý nghĩa α=0.05, hiệu chỉnh Bonferroni khi so sánh bội với
  5 baseline.
- `ablation.py`: so sánh 3 nhóm trong Bảng 5 (đặt lần lượt từng trọng số
  reward về 0 + so sánh GCN/GAT/DGAT theo Bảng 5).
- `zero_shot.py`: huấn luyện N∈{4,8,16}, test zero-shot N∈{6,10,12,20}, hai
  chế độ `fixed-map` và `density-controlled`.
- `plots.py`: xuất toàn bộ hình vào `results/figures/` — learning curve
  (mean±std dải mờ theo seed), bar chart so sánh 7 phương pháp theo 5 chỉ số,
  heatmap vùng bao phủ theo thời gian, phân bố va chạm, biểu đồ zero-shot
  scalability.

### 2.6 Website trực quan hóa (Next.js + TailwindCSS)
Không phải một phần đề cương học thuật, nhưng theo yêu cầu bổ sung của người
dùng — dùng để trình bày trực quan cho hội đồng/giảng viên hướng dẫn:
- **Dashboard**: tổng quan tiến độ, số run đã hoàn thành, bảng kết quả chính.
- **Training**: learning curve tương tác theo phương pháp/kịch bản (recharts),
  chọn seed để xem chi tiết.
- **Comparison**: bar/radar chart so sánh 7 phương pháp trên 5 chỉ số + bảng
  kiểm định thống kê (p-value, hiệu ứng).
- **Ablation**: kiến trúc (MLP/GCN/GAT/DGAT) và reward component ablation.
- **Zero-shot**: biểu đồ hiệu năng theo N, hai chế độ fixed-map/density-controlled.
- **Replay**: canvas 2D phát lại quỹ đạo một episode thật (đọc từ log JSON) —
  UAV, nạn nhân, vật cản, bán kính comm/sense, belief-map heatmap theo thời
  gian, thanh trượt điều khiển thời gian thực — đúng tinh thần Hình 2 của đề
  cương nhưng động thay vì tĩnh.

Dữ liệu là JSON tĩnh xuất ra từ `results/` (Task 11) — không cần backend,
`next dev`/`next build` đọc trực tiếp từ `public/data/*.json`.

## 3. Trình tự thực hiện trong phiên này

1. Cài đặt thư viện Python (gymnasium, pettingzoo, torch-geometric, ...).
2. Cài `sar_env` + chạy giao thức kiểm chứng môi trường → xác nhận pipeline
   đúng như GĐ1 của đề cương (mục 1.5.5, Bảng 7).
3. Cài `models/` + `algorithms/` (heuristics → MAPPO đa biến thể → MADDPG).
4. Chạy `scripts/run_poc.py`: Easy → Medium, đủ 7 phương pháp, 3 seeds/phương
   pháp, ngân sách bước rút gọn.
5. Chạy `evaluation/` để có metrics.json, thống kê, ablation, zero-shot rút gọn.
6. Xuất hình PNG (`evaluation/plots.py`) làm minh chứng trực tiếp trong report.
7. Xuất JSON cho web (`training/logger.py` output + một script tổng hợp).
8. Scaffold + build Next.js/Tailwind site, đọc JSON, hiển thị đầy đủ các trang trên.
9. Chạy thử site trong Browser, kiểm tra từng trang có dữ liệu thật.

## 4. Giới hạn đã biết của bản POC

- Kết quả số liệu (Coverage Rate, Victim Detection Rate, ...) ở quy mô rút
  gọn **không đại diện cho kết luận khoa học cuối cùng của luận văn** — chúng
  chứng minh pipeline đúng và chạy được, không thay thế cho lần chạy full-scale
  ≥5×10⁶ bước/5 seeds trên GPU mà đề cương yêu cầu cho Chương 5.
  Xu hướng tương đối kỳ vọng (Random Walk < Greedy < MADDPG ≲ MAPPO-MLP <
  MAPPO-GCN ≲ MAPPO-GAT < GNN-MAPPO) có thể chưa ổn định ở ngân sách nhỏ và
  cần được xác nhận lại ở full-scale.
- Kịch bản Hard (16 UAV, 200×200, vật cản động) chỉ chạy minh họa ngắn do chi
  phí O(N·K) mỗi bước × kích thước bản đồ lớn trên CPU.
- Zero-shot scalability chỉ test một vài giá trị N nhỏ (6, 10) ở bản POC thay
  vì đủ {6,10,12,20}.

## 5. Kết quả POC thực tế đã chạy (tham khảo, không phải kết luận cuối)

Toàn bộ 7 phương pháp đã được huấn luyện/đánh giá trên cả 3 kịch bản (36 lượt
chạy chính + rerun Greedy + zero-shot, tổng ~70 phút compute trên CPU). Kết
quả đầy đủ: `results/figures/*.png`, `web/public/data/*` (xem qua website).

Quan sát chính (VDR = Victim Detection Rate trung bình):

| Kịch bản | Phương pháp tốt nhất | VDR |
|---|---|---|
| Easy | Random Walk | 53.5% |
| Medium | Random Walk | 16.5% |
| Hard | Greedy | 7.3% |

Ở ngân sách huấn luyện rút gọn (8k-40k bước thay vì 5×10⁶), **các baseline
heuristic (Random Walk, Greedy) vẫn vượt trội các phương pháp MARL/GNN-MARL**
— bao gồm cả GNN-MAPPO đề xuất. Đây là kết quả **trung thực, đúng như dự
đoán** ở mục 0 của tài liệu này: chính sách MAPPO/MADDPG/GNN-MAPPO chưa đủ
thời gian hội tụ (learning curve trong `results/figures/learning_curves_*.png`
cho thấy VDR vẫn đang tăng khi huấn luyện dừng ở ngân sách POC), trong khi
heuristic không cần học nên đạt hiệu năng ổn định ngay từ đầu. Kiểm định
thống kê (trang So sánh / Ablation trên website) xác nhận **chưa có khác biệt
có ý nghĩa thống kê** giữa 4 biến thể MAPPO (MLP/GCN/GAT/DGAT) ở quy mô này —
đúng như cảnh báo ở mục 0, xu hướng tương đối kỳ vọng của đề cương (GNN-MAPPO
> MAPPO-GAT ≳ MAPPO-GCN > MAPPO-MLP) **cần chạy full-scale để kiểm chứng**,
POC này mới chứng minh được pipeline đúng và chạy end-to-end, chưa chứng minh
được giả thuyết khoa học của luận văn.

Một lỗi cài đặt thực đã được phát hiện và sửa trong quá trình này: heuristic
Greedy ban đầu dùng `argmin`/`argmax` xác định (deterministic tie-break) trên
vùng chưa quét toàn 0, khiến mọi UAV bay thẳng một đường vào một góc bản đồ
thay vì khám phá — sau khi thêm nhiễu ngẫu nhiên nhỏ để phá vỡ ràng buộc
(`algorithms/heuristics.py`), VDR của Greedy trên Easy tăng từ 17.1% lên
47.3%. Một lỗi thống kê cũng được sửa: Cohen's d bị chia cho độ lệch chuẩn
gộp gần 0 khi so sánh với heuristic chỉ chạy 1 seed, cho ra giá trị vô nghĩa
(hàng triệu) — `evaluation/statistics.py` nay trả về "insufficient_data" khi
một trong hai mẫu có dưới 2 điểm dữ liệu thay vì suy ra một con số sai lệch.

## 6. Cách nâng lên full-scale

Không cần sửa kiến trúc — chỉ:
1. Đổi `total_env_steps: 50000` → `5000000` và `num_seeds: 3` → `5` trong
   `training/configs/*.yaml`.
2. Chạy trên máy có CUDA (code tự động dùng `cuda` nếu `torch.cuda.is_available()`).
3. Tăng `n_eval_episodes` trong `evaluation/evaluate.py` lên 200.
4. Chạy lại `scripts/run_poc.py` (đổi tên thành `run_full.py` nếu muốn) theo
   đúng lộ trình 4 giai đoạn GĐ1–GĐ3b ở Bảng 7 đề cương.
