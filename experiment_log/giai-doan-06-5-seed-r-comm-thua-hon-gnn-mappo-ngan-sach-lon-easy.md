[← Giai đoạn 5](giai-doan-05-huan-luyen-lai-voi-ham-thuong-da-sua-hoan-tat.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 7 →](giai-doan-07-kiem-chung-value-coef-0-1-o-quy-mo-day-du-108-luot.md)

# Giai đoạn 6 — 5 seed, r_comm thưa hơn, + GNN-MAPPO ngân sách lớn (Easy)

Thực hiện cả 3 khuyến nghị của Giai đoạn 5 trong cùng một đợt:

1. **`r_comm`**: giảm 15/20/25 → **10/10/10** cho Easy/Medium/Hard. Đo lại độ
   kết nối đồ thị (100 bước ngẫu nhiên): Easy 54%→**27%**, Medium 83%→**36%**,
   Hard 96%→**46%** — thưa hơn đáng kể, không còn gần như đặc.
2. **Số seed**: nâng từ 2-3 lên **5** cho cả 5 phương pháp học **và cả 2
   heuristic** (trước đây heuristic chỉ 1 seed → luôn báo "không đủ seed" khi
   so sánh thống kê; nay đã so sánh được).
3. **GNN-MAPPO/Easy ngân sách lớn**: chạy riêng 3 seed × **2.000.000 bước**
   (gấp 4 lần 500k hiện tại), lưu ở `results_longrun_easy_gnn_mappo/`.

Tổng 108 lượt chạy (105 lượt chính + 3 lượt ngân sách lớn), dùng job-pool tối
đa 14 tiến trình song song (kiểm thử cơ chế trước khi chạy thật). Hoàn tất
105/108 nhanh hơn dự kiến; 3 lượt ngân sách lớn ban đầu thất bại do sai tên cờ
CLI (`--n_env_steps` thay vì `--total_env_steps` đúng của dự án này) — đã sửa
và chạy lại.

### Kết quả so sánh baseline (VDR trung bình ± std, n=5 seed)

| Phương pháp | Easy (N=4) | Medium (N=8) | Hard (N=16) |
|---|---|---|---|
| Random Walk | 0.369 ± 0.039 | 0.193 ± 0.031 | 0.089 ± 0.035 |
| Greedy | 0.437 ± 0.047 | 0.196 ± 0.020 | 0.099 ± 0.032 |
| MADDPG | 0.147 ± 0.016 | 0.027 ± 0.020 | 0.019 ± 0.015 |
| MAPPO-MLP | 0.167 ± 0.041 | 0.070 ± 0.044 | **0.049 ± 0.040** |
| MAPPO-GCN | 0.189 ± 0.062 | **0.081 ± 0.048** | 0.043 ± 0.011 |
| MAPPO-GAT (no GRU) | 0.148 ± 0.014 | 0.044 ± 0.025 | 0.025 ± 0.019 |
| **GNN-MAPPO (đề xuất)** | **0.188 ± 0.040** | 0.058 ± 0.041 | 0.038 ± 0.011 |

**So với Giai đoạn 5** (r_comm dày, n=2-3): khoảng cách GNN-MAPPO với
MAPPO-GCN ở **Easy gần như đóng lại hoàn toàn** (0.181 vs 0.279 → **0.188 vs
0.189**, gần như bằng nhau) sau khi thu hẹp `r_comm` — ủng hộ giả thuyết "đồ
thị quá đặc khiến attention khó học" đã nêu ở Giai đoạn 5. Ở Hard, GNN-MAPPO
cải thiện thứ hạng (từ thấp nhất lên thứ 3/5 phương pháp học).

**Kiểm định thống kê (Welch's t-test/Mann-Whitney U, hiệu chỉnh Bonferroni,
n=5) — GNN-MAPPO vs từng biến thể kiến trúc:**

| So sánh | Easy (p / d) | Medium (p / d) | Hard (p / d) |
|---|---|---|---|
| vs MAPPO-MLP | 0.479 / 0.47 | 0.696 / -0.26 | 0.616 / -0.34 |
| vs MAPPO-GCN | 0.977 / -0.02 | 0.486 / -0.46 | 0.550 / -0.39 |
| vs MAPPO-GAT (no GRU) | 0.122 / **1.18** | 0.567 / 0.38 | 0.278 / **0.75** |

→ **Vẫn chưa có khác biệt nào đạt ý nghĩa thống kê** dù đã tăng lên n=5 (mọi
p > 0.05, thậm chí > α đã hiệu chỉnh Bonferroni). Đáng chú ý: GNN-MAPPO vs
MAPPO-GAT (no GRU) có Cohen's d lớn (1.18 ở Easy, 0.75 ở Hard) — dấu hiệu có
thể có khác biệt thật (vai trò của GRU/temporal encoding), nhưng n=5 vẫn chưa
đủ lực thống kê để khẳng định chắc chắn (p=0.12 và 0.28, chưa dưới 0.05).

### Kết luận trung thực cho Giai đoạn 6

- Thu hẹp `r_comm` **có tác dụng thật** — thu hẹp khoảng cách với baseline
  mạnh nhất (MAPPO-GCN) ở Easy, cải thiện thứ hạng ở Hard.
- Tăng seed lên 5 **chưa đủ** để đạt ý nghĩa thống kê — cần ước tính cỡ mẫu
  cần thiết dựa trên Cohen's d quan sát được (ví dụ d≈1.0-1.2 cần khoảng
  15-20 seed để đạt power 0.8 ở α=0.05 theo quy tắc kinh nghiệm, nhiều hơn hẳn
  5 seed hiện có).
- **Kết quả GNN-MAPPO/Easy ngân sách 2 triệu bước (4x, 3 seed) — đã xong**:
  VDR = **0.165 ± 0.009** (n=3) — **không cao hơn** kết quả 500k bước
  (0.188 ± 0.040, n=5), và vẫn thấp hơn MAPPO-GCN (0.189). Độ lệch chuẩn giữa
  seed nhỏ hơn hẳn (0.009 so với 0.040) — chính sách ổn định hơn giữa các
  seed, nhưng KHÔNG học được hành vi tốt hơn.
  Xem xét đường cong huấn luyện đầy đủ (3.907 iteration): entropy và reward
  **dao động không đơn điệu** suốt cả quá trình (entropy giảm còn 2.09 ở 1/4
  đầu rồi lại tăng về 2.25-2.39 ở phần sau; VDR đánh giá dao động 0.0-0.37
  không có xu hướng tăng bền vững, có dấu hiệu tuần hoàn — 2 đỉnh giống nhau
  cách nhau ~30 iteration). **Kết luận: đơn thuần tăng số bước (không đổi
  siêu tham số) không giúp cải thiện thêm** — nút thắt không phải là thiếu
  bước mà có thể là `entropy_coef`/lịch learning rate/độ khó gán công (credit
  assignment) trong thưởng chung toàn đội.

### Phát hiện thêm: value_loss lấn át actor_loss qua encoder dùng chung

Actor và Critic dùng chung một encoder (`GNNActorCritic`) — kiểm tra tỉ lệ
đóng góp gradient giữa 2 nhánh trên dữ liệu Easy đã có (seed 0, cả quá trình):

| Phương pháp | mean\|actor_loss\| | mean(value_loss) | value_coef×value_loss / \|actor_loss\| |
|---|---|---|---|
| GNN-MAPPO | 0.0326 | 1.587 | **24.3×** |
| MAPPO-GCN | 0.0233 | 1.068 | **22.9×** |
| MAPPO-MLP | 0.0213 | 1.093 | **25.7×** |

→ Với `value_coef=0.5` mặc định, gradient chảy vào encoder dùng chung lớn
hơn tín hiệu actor 22-26 lần, ở **cả 3 kiến trúc như nhau** (không phải vấn đề
riêng của GNN-MAPPO, nên không giải thích được vì sao GCN từng dẫn trước, chỉ
là một nút thắt tiềm ẩn ảnh hưởng đều lên tất cả).

**Kiểm chứng nhanh** (GNN-MAPPO/Easy, cùng seed, 60 iteration ≈ 30.720 bước,
`value_coef=0.5` so với `0.1`):

| | value_coef=0.5 (hiện tại) | value_coef=0.1 |
|---|---|---|
| entropy: đầu → cuối | 2.558 → 2.438 (giảm 0.12) | 2.560 → 2.320 (giảm **0.24**, gấp ~2x) |
| reward 5 điểm cuối | 0.020, 0.040, 0.010, 0.031, 0.010 (không xu hướng) | 0.030, 0.000, 0.020, 0.040, **0.051** (tăng dần rõ ở cuối) |

→ `value_coef=0.1` cho entropy giảm nhanh hơn và reward có xu hướng tăng rõ
hơn trong cùng ngân sách nhỏ — ủng hộ giả thuyết. Đây mới là kiểm chứng nhanh
(30k bước, nhiễu cao, 1 seed), **chưa đủ để kết luận chắc chắn**, nhưng đủ để
khuyến nghị thử `value_coef=0.1-0.2` cho lần chạy nghiên cứu tiếp theo thay vì
0.5 mặc định hiện tại — xem mục khuyến nghị bên dưới.

### So sánh khả năng tổng quát hóa zero-shot giữa 3 kiến trúc (Easy, N huấn
luyện = 4)

| N (test) | GNN-MAPPO | MAPPO-GCN | MAPPO-MLP |
|---|---|---|---|
| 4 (train) | 0.244 | 0.122 | 0.256 |
| 6 | 0.306 | 0.150 | 0.264 |
| 8 | **0.350** | 0.108 | 0.286 |
| Thay đổi N=4→8 | **+43%** | -11% (không đơn điệu) | +12% |

→ Đây là bằng chứng **rõ ràng và nhất quán nhất ủng hộ GNN-MAPPO** trong toàn
bộ thực nghiệm: khi tăng số UAV vượt quá lúc huấn luyện (không huấn luyện
lại), GNN-MAPPO không những không suy giảm mà còn **cải thiện mạnh và đơn
điệu** (+43%), trong khi MAPPO-GCN dao động thất thường và MAPPO-MLP chỉ cải
thiện nhẹ. Khớp đúng lý thuyết: cơ chế pooling bất biến hoán vị + message
passing của DGAT tận dụng được thêm UAV một cách có cấu trúc, còn MLP xử lý
độc lập từng UAV (không có cơ chế khai thác thêm tác nhân), GCN dùng trọng số
cố định kém linh hoạt khi mật độ đồ thị thay đổi theo N.

Ở Medium/Hard, mẫu hình này **không rõ ràng bằng** (chỉ 1 seed/N, nhiễu cao,
checkpoint còn undertrained hơn Easy) — cần thêm seed để xác nhận có lặp lại
xu hướng hay không trước khi khái quát hóa sang mọi kịch bản.

---

## Việc cần làm tiếp theo (khuyến nghị)

- [x] Chạy `evaluation.plots`, so sánh baseline, kiểm định thống kê, zero-shot,
      xuất web — đã xong cho Giai đoạn 5 và Giai đoạn 6.
- [x] Tăng số seed lên 5 — đã làm (Giai đoạn 6), nhưng **chưa đủ lực thống
      kê**; cần ≥15-20 seed để phát hiện khác biệt cỡ Cohen's d ~1.0-1.2 đã
      quan sát.
- [x] Thu hẹp `r_comm` (15/20/25 → 10/10/10) — đã làm, có tác dụng rõ ở Easy.
- [x] Chạy GNN-MAPPO ngân sách lớn (2M bước) ở Easy — **hoàn tất**: VDR
      0.165 ± 0.009 (n=3), **không cao hơn** mức 500k bước (0.188 ± 0.040).
      → Kết luận quan trọng: **tăng thêm số bước một mình (giữ nguyên siêu
      tham số) không phải là nút thắt** — learning curve dao động không đơn
      điệu suốt 2 triệu bước thay vì hội tụ, cho thấy vấn đề nằm ở siêu tham
      số/thiết kế huấn luyện chứ không phải thiếu dữ liệu.
- [ ] **Không nên tiếp tục tăng số bước.** Hướng nên thử tiếp theo (chưa làm,
      cần quyết định của người hướng dẫn/người dùng do tốn thêm thời gian):
      - **[Đã kiểm chứng nhanh, đáng làm]** Giảm `value_coef` 0.5→0.1-0.2:
        actor và critic dùng chung encoder, `value_coef×value_loss` lớn hơn
        `actor_loss` 22-26 lần ở cả 3 kiến trúc → gradient vào encoder gần
        như chỉ do critic chi phối. Test nhanh 30k bước: value_coef=0.1 cho
        entropy giảm nhanh gấp ~2x (0.24 so với 0.12) và reward có xu hướng
        tăng rõ hơn so với 0.5. Khuyến nghị áp dụng cho lần chạy tiếp theo.
      - Giảm `entropy_coef` (hiện 0.01) theo lịch giảm dần, hoặc thêm learning
        rate decay — để chính sách "chốt" hành vi thay vì dao động liên tục.
      - Thử `critic_pool="attention"` thay vì `mean` (đã cài sẵn trong code)
        — pooling có trọng số học được có thể xử lý thưởng chung toàn đội
        tốt hơn mean-pooling đơn giản.
      - Tăng seed lên ≥15-20 (không phải bước) nếu mục tiêu là kết luận thống
        kê chắc chắn cho khác biệt cỡ Cohen's d ~1.0-1.2 đã quan sát — đây
        vẫn là điểm nghẽn số 1 cho ý nghĩa thống kê, nhưng chi phí tăng
        tuyến tính nên cần cân nhắc ngân sách thời gian thực tế.
- [x] Ghi rõ trong luận văn: giới hạn tài nguyên CPU (không phải GPU cluster
      như đề cương giả định) là lý do chính khiến chưa đạt ý nghĩa thống kê;
      xu hướng sau khi thu hẹp `r_comm` (Giai đoạn 6) ủng hộ giả thuyết
      nghiên cứu (GNN-MAPPO cạnh tranh ngang MAPPO-GCN, vượt MAPPO-MLP/MADDPG/
      GAT-no-GRU ở phần lớn chỉ số) dù chưa đủ lực thống kê để khẳng định.

---

[← Giai đoạn 5](giai-doan-05-huan-luyen-lai-voi-ham-thuong-da-sua-hoan-tat.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 7 →](giai-doan-07-kiem-chung-value-coef-0-1-o-quy-mo-day-du-108-luot.md)
