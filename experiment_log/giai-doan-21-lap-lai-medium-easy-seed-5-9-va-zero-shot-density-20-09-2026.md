[← Giai đoạn 20](giai-doan-20-hard-n-10-dot-lap-lai-seed-5-9-hoan-tat-10-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 22 →](giai-doan-22-hard-ngan-sach-2x-gcn-dgat-2m-buoc-22-09-2026.md)

# Giai đoạn 21 — Lặp lại Medium và Easy (seed 5–9), zero-shot density-controlled (20/09/2026)

### Lý do

Sau khi đề cương được hiệu chỉnh (họ encoder, MAPPO-GCN là phiên bản đề xuất) và
bài báo được soạn, ba khoảng trống còn lại so với những gì bài báo/đề cương
khẳng định:

1. **Medium chỉ có n = 5** cho DGAT (GAT n = 10, MLP/GCN n = 12–15 nhờ gộp các
   đợt mã cũ `results_v3_rcomm` / `results_v4_unseeded`). Đây là kịch bản duy
   nhất không có đợt lặp lại độc lập và không có kiểm định theo cặp seed; Bảng 4
   của bài báo vì thế có thiết kế không đồng nhất giữa ba kịch bản.
2. **Easy: DGAT/GAT chỉ n = 5** gieo seed (DGAT không gộp được với v3 vì BPTT
   khác), nên kết luận RQ2 ở Easy (p = 0,12) mới dựa trên một đợt.
3. **Zero-shot chỉ có chế độ fixed-map, 3 seed**; đề cương hứa cả
   density-controlled.

Quyết định (20/09, có xác nhận của người dùng): chạy seed 5–9 đủ 7 phương pháp
cho Medium rồi Easy (một chuỗi, 6 tiến trình song song), sau đó chạy zero-shot
cả hai chế độ với seed 0–9 cho mọi kịch bản. Hard ngân sách 2× **chưa** chạy
(35–40 giờ) — để sau nếu máy rảnh.

### Cấu hình

- Medium: `run_all_experiments --scenarios medium --seeds 5 --seed-offset 5
  --steps-override 800000 --out-dir results_v4 --max-jobs 6 --no-postprocess`
  (35 lượt). Easy: như trên, không `--steps-override` (500k), 35 lượt. Cùng
  mã nguồn và cấu hình với seed 0–4; khối bản đồ test 95.000–99.999.
- Ước tính: Medium ~11–13 giờ, Easy ~5–6 giờ (tổng tuần tự ~17 giờ).
- Zero-shot: `scripts/run_zero_shot_all.py` (mới, thay `zs4.py` trong
  scratchpad): `--mode fixed_map` và `--mode density`, `--seeds 0-9`,
  10 episode/seed/N, cache theo (kịch bản, phương pháp, seed) nên chạy lại chỉ
  làm phần thiếu. Chuỗi nền tự chạy sau khi huấn luyện xong, rồi
  `export_web_data`.

### Chế độ density-controlled (mới, `evaluation/zero_shot.py::density_overrides`)

Giữ mật độ UAV và mật độ nạn nhân không đổi khi N thay đổi: cạnh bản đồ nhân
√(N/N_train), số nạn nhân nhân N/N_train; r_comm, r_sense, max_steps giữ nguyên
(bậc trung bình kỳ vọng của đồ thị chỉ phụ thuộc mật độ nên không đổi). Khối seed
test riêng (80.000 + N) để hai chế độ không dùng chung bản đồ. Kiểm tra nhanh
Easy/GCN/seed 0: N = 8 → bản đồ 71×71, nạn nhân 6–10, VDR 0,929 (2 episode).

Kết quả `zero_shot_density_all.json` (+ `_env.json` ghi kích thước bản đồ theo
N) đặt cạnh `zero_shot_all.json`; web đọc `zero_shot_density.json`.

### Trạng thái

- 20/09 13:24: khởi chạy chuỗi Medium → Easy. Heuristic Medium xong sau 2 phút.
- 21/09 16:55: Medium 30/35 (còn 5 seed DGAT, 29–99 %, ETA ~3,5 h). Chậm hơn dự kiến rất nhiều vì máy
  vào **Modern Standby ~18 giờ** trong đêm 20→21/09 (Kernel-Power 506/507): MLP seed 6–9 và GCN seed 5–6 có
  một khoảng 100k bước kéo dài 17,95 h trong learning curve; kết quả không bị ảnh hưởng (huấn luyện tiếp tục
  sau khi thức, mọi RNG gieo seed) — chỉ tốn thời gian. Đã yêu cầu app giữ máy thức (keep-awake) cho các
  đêm tiếp theo; bộ chờ zero-shot đầu tiên (hạn 30 h) sẽ hết hạn 19:26, đã cắm bộ chờ thứ hai (48 h).
- Số tạm thời Medium (VDR, đợt 1 → đợt 2): MLP 0,705 → 0,612 (Δ −0,094, p = 0,01); GCN 0,717 → 0,687;
  GAT 0,676 → 0,668; Greedy 0,498 → 0,485. Gộp n = 10: GCN vs MLP Δ +0,043, Welch p = 0,089, d = +0,81,
  theo cặp seed p = 0,018, thắng 8/10 — Medium đợt 2 khó hơn cho MLP nhiều hơn cho các encoder đồ thị; chờ
  DGAT để kết luận RQ2 ở Medium.
- 22/09 01:46: toàn bộ chuỗi hoàn tất — Easy 35/35 (xong ~00:30), zero-shot fixed-map 120 tác vụ + density 120 tác vụ
  (4 kiến trúc × 10 seed × 3 kịch bản), web đã xuất lại; 01:52 bộ chờ báo cáo sinh lại bảng/hình dưới đây,
  `evaluation.plots` và hình Chương 5. Không có lần ngủ máy nào sau 16:55 ngày 21/09. `results_v4/` nay có
  **210 lượt** (7 phương pháp × seed 0–9 × 3 kịch bản), mọi kịch bản đều là thiết kế hai đợt độc lập n = 10.

### Kết luận Giai đoạn 21

1. **RQ1 giữ vững ở cả ba kịch bản với thiết kế đồng nhất n = 10** (GCN vs MLP, VDR): Easy +0,047 (Welch p = 0,0024,
   d = +1,59, theo cặp 9/10, **đạt Bonferroni**); Medium +0,043 (Welch p = 0,089, d = +0,81, theo cặp 8/10 p = 0,018,
   Coverage p = 0,012 đạt Bonferroni); Hard +0,078 (p = 0,0012, d = +1,75, 10/10, đạt). Kết luận Medium trước đây
   (gộp 3 đợt mã cũ, p = 0,025) được thay bằng thiết kế sạch: Welch chưa qua 0,05 nhưng kiểm định theo cặp — phù hợp
   hơn vì cùng seed = cùng bản đồ — có ý nghĩa; đợt B khó hơn cho MLP (−0,094, p = 0,01) nhưng gần như không đổi cho
   GCN/GAT/DGAT/Greedy.
2. **DGAT thua GCN có ý nghĩa ở cả ba kịch bản** (VDR: Easy −0,050 p = 0,0007; Medium −0,064 p = 0,0088; Hard −0,103
   p < 0,0001; Coverage đạt Bonferroni ở cả ba). Thứ hạng GCN > GAT > DGAT lặp lại ở 3 kịch bản × 2 đợt.
3. **RQ2 (GRU, DGAT vs GAT)**: Hard gây hại rõ (−0,072, p = 0,0003); Easy −0,027 (Welch p = 0,055, theo cặp 2/10
   p = 0,040); Medium −0,034 (p = 0,149, theo cặp 3/10 p = 0,073). Dấu âm nhất quán ở cả ba kịch bản, ý nghĩa
   thống kê chỉ ở Hard (và biên ở Easy theo cặp). Phát biểu đúng: GRU không đóng góp ở kịch bản nào, gây hại có ý
   nghĩa khi quy mô lớn.
4. **RQ3 fixed-map (10 seed)**: mọi kiến trúc tăng VDR khi N tăng (bản đồ đông hơn); ở Hard GCN dẫn đầu mọi N
   (0,286 → 0,558), DGAT cuối bảng mọi N. **RQ3 density-controlled (mới)**: khi giữ mật độ không đổi, đường VDR
   gần phẳng — đúng kỳ vọng của thiết kế bất biến hoán vị; GCN cao nhất ở mọi N tại Easy (0,89–0,92) và Hard
   (0,40–0,51 so với MLP 0,29–0,43); Medium các kiến trúc đan xen. Không có bằng chứng DGAT tổng quát hoá tốt hơn
   ở chế độ nào.
5. Hai đợt độc lập không khác nhau có ý nghĩa ở phần lớn cấu hình (ngoại lệ: MLP ở Medium, GAT ở Easy, Random Walk
   ở Medium, MADDPG ở Easy — đều theo hướng đợt B khó hơn), nên gộp là hợp lệ và bảng "đợt A/B/gộp" bên dưới là
   bằng chứng tái lập cho từng con số.

Việc tiếp theo: cập nhật Chương 5/Chương 4 luận văn, Bảng 4 bài báo và báo cáo CBHD theo thiết kế "seed 0–9, hai
đợt" cho cả ba kịch bản (bỏ gộp `results_v3_rcomm`/`results_v4_unseeded`), thêm mục zero-shot density.

- 23/09 20:25: **Chương 5 đã cập nhật** theo thiết kế "seed 0–9, hai đợt" (`CHUONG5_THUC_NGHIEM.md`, 37 chỗ sửa; bản
  cũ lưu `thesis proposals/CHUONG5_v3_stage20_OBSOLETE.md`): Bảng 5.8/5.9 n = 10, Bảng 5.12 thay gộp mã cũ bằng bảng
  n = 10 có cột theo cặp, Bảng 5.13/5.15 n = 10, Bảng 5.14 theo đợt A/B/gộp cho cả ba kịch bản, Bảng 5.17 10 seed +
  Bảng 5.17b density + Hình 5.8, Bảng 5.18c/d (Medium/Easy theo đợt), RQ1–RQ3 và mục 5.8 viết lại; mục 5.4.4 thêm
  kết quả tạm GCN 2M Hard (phần DGAT bổ sung khi GD22 xong). Docx dựng lại (37 trang PDF, đã soát bảng 5.12/5.14/5.17b/
  Hình 5.8). Còn lại: Bảng 4 bài báo, báo cáo CBHD.
- 23/09 20:50: **Bản draft luận văn đã cập nhật** cùng thiết kế (69 chỗ sửa trong `thesis/src/01,02,03,04,05,07` và
  `thesis/front_matter.py`): Chương 4 bảng 4.8/4.9/4.12/4.13/4.14/4.15 n = 10, thêm Bảng 4.18 (density) + Hình zero-shot
  density, Bảng 4.21/4.22 (Medium/Easy theo đợt), RQ1–RQ3 và 4.8 viết lại, 4.4.4 thêm GCN 2M Hard tạm; Chương 3 giao
  thức seed/zero-shot/ngân sách 2×; RQ3 hai chế độ; Chương 5 kết luận/hạn chế/hướng phát triển; Phụ lục B thêm GĐ21–22;
  Tóm tắt/Abstract 210 lượt. Dựng `thesis/out/LuanVan_NguyenMinhKhanh_draft.docx` (37 bảng, 19 hình, không xuất PDF
  theo yêu cầu).

<!-- AUTO:rep-medium -->
### Kết quả tự động — medium (cập nhật 22/09/2026 01:52)

Trạng thái dữ liệu: đợt A = seed 0–4, đợt B = seed 5–9; đợt B đã có Random Walk 5/5, Greedy 5/5, MADDPG 5/5, MAPPO-MLP 5/5, MAPPO-GCN 5/5, MAPPO-GAT 5/5, MAPPO-DGAT 5/5 — **đầy đủ**.

**VDR theo đợt** (mean ± sd liên-seed; p = Welch đợt A vs B):

| Phương pháp | Đợt A | Đợt B | Δ (B−A) | p | Gộp |
|---|---|---|---|---|---|
| Random Walk | 0,183 ± 0,018 | 0,131 ± 0,032 (n=5) | -0,051 | 0,02 | 0,157 ± 0,037 (n=10) |
| Greedy | 0,498 ± 0,042 | 0,485 ± 0,031 (n=5) | -0,012 | 0,61 | 0,492 ± 0,035 (n=10) |
| MADDPG | 0,209 ± 0,117 | 0,192 ± 0,087 (n=5) | -0,017 | 0,80 | 0,200 ± 0,098 (n=10) |
| MAPPO-MLP | 0,705 ± 0,041 | 0,612 ± 0,046 (n=5) | -0,094 | 0,01 | 0,659 ± 0,064 (n=10) |
| MAPPO-GCN | 0,717 ± 0,030 | 0,687 ± 0,046 (n=5) | -0,030 | 0,27 | 0,702 ± 0,040 (n=10) |
| MAPPO-GAT | 0,676 ± 0,052 | 0,668 ± 0,040 (n=5) | -0,008 | 0,79 | 0,672 ± 0,044 (n=10) |
| MAPPO-DGAT | 0,644 ± 0,053 | 0,633 ± 0,062 (n=5) | -0,010 | 0,78 | 0,638 ± 0,055 (n=10) |

**Coverage theo đợt** (mean ± sd liên-seed; p = Welch đợt A vs B):

| Phương pháp | Đợt A | Đợt B | Δ (B−A) | p | Gộp |
|---|---|---|---|---|---|
| Random Walk | 0,149 ± 0,002 | 0,150 ± 0,005 (n=5) | +0,001 | 0,72 | 0,150 ± 0,004 (n=10) |
| Greedy | 0,516 ± 0,010 | 0,514 ± 0,011 (n=5) | -0,002 | 0,77 | 0,515 ± 0,010 (n=10) |
| MADDPG | 0,206 ± 0,119 | 0,217 ± 0,070 (n=5) | +0,011 | 0,86 | 0,212 ± 0,092 (n=10) |
| MAPPO-MLP | 0,661 ± 0,021 | 0,652 ± 0,010 (n=5) | -0,009 | 0,42 | 0,656 ± 0,016 (n=10) |
| MAPPO-GCN | 0,677 ± 0,019 | 0,675 ± 0,014 (n=5) | -0,002 | 0,85 | 0,676 ± 0,015 (n=10) |
| MAPPO-GAT | 0,638 ± 0,019 | 0,666 ± 0,021 (n=5) | +0,028 | 0,06 | 0,652 ± 0,024 (n=10) |
| MAPPO-DGAT | 0,636 ± 0,018 | 0,636 ± 0,021 (n=5) | +0,000 | 1,00 | 0,636 ± 0,019 (n=10) |

**Kiểm định gộp** (Welch t / Mann–Whitney / Cohen's d; theo cặp seed = cùng bộ bản đồ test; α' Bonferroni = 0,0167 cho 3 so sánh với MLP):

| Chỉ số | So sánh | n | Δ | p Welch | p MW | d | Theo cặp: thắng / p | Bonferroni |
|---|---|---|---|---|---|---|---|---|
| VDR | MAPPO-GCN vs MAPPO-MLP | 10/10 | +0,043 | 0,0888 | 0,1620 | +0,81 | 8/10 / 0,0178 | không |
| VDR | MAPPO-GAT vs MAPPO-MLP | 10/10 | +0,014 | 0,5914 | 0,5708 | +0,25 | 4/10 / 0,5337 | không |
| VDR | MAPPO-DGAT vs MAPPO-MLP | 10/10 | -0,020 | 0,4603 | 0,4274 | -0,34 | 4/10 / 0,4209 | không |
| VDR | MAPPO-DGAT vs MAPPO-GAT | 10/10 | -0,034 | 0,1487 | 0,1620 | -0,68 | 3/10 / 0,0725 | không |
| VDR | MAPPO-GAT vs MAPPO-GCN | 10/10 | -0,030 | 0,1288 | 0,1620 | -0,71 | 2/10 / 0,0915 | không |
| VDR | MAPPO-DGAT vs MAPPO-GCN | 10/10 | -0,064 | 0,0088 | 0,0140 | -1,33 | 1/10 / 0,0010 | đạt |
| VDR | MAPPO-GCN vs Greedy | 10/10 | +0,210 | 0,0000 | 0,0002 | +5,60 | 10/10 / 0,0000 | đạt |
| VDR | MAPPO-GAT vs Greedy | 10/10 | +0,180 | 0,0000 | 0,0002 | +4,52 | 10/10 / 0,0000 | đạt |
| VDR | MAPPO-MLP vs Greedy | 10/10 | +0,167 | 0,0000 | 0,0002 | +3,22 | 10/10 / 0,0000 | đạt |
| VDR | MAPPO-DGAT vs Greedy | 10/10 | +0,147 | 0,0000 | 0,0002 | +3,19 | 10/10 / 0,0000 | đạt |
| Coverage | MAPPO-GCN vs MAPPO-MLP | 10/10 | +0,020 | 0,0116 | 0,0140 | +1,26 | 9/10 / 0,0272 | đạt |
| Coverage | MAPPO-GAT vs MAPPO-MLP | 10/10 | -0,004 | 0,6634 | 0,7337 | -0,20 | 5/10 / 0,7175 | không |
| Coverage | MAPPO-DGAT vs MAPPO-MLP | 10/10 | -0,020 | 0,0194 | 0,0211 | -1,15 | 2/10 / 0,0407 | * |
| Coverage | MAPPO-DGAT vs MAPPO-GAT | 10/10 | -0,016 | 0,1183 | 0,1212 | -0,74 | 3/10 / 0,0631 | không |
| Coverage | MAPPO-GAT vs MAPPO-GCN | 10/10 | -0,024 | 0,0188 | 0,0211 | -1,18 | 2/10 / 0,0224 | * |
| Coverage | MAPPO-DGAT vs MAPPO-GCN | 10/10 | -0,040 | 0,0001 | 0,0008 | -2,33 | 0/10 / 0,0002 | đạt |
| Coverage | MAPPO-GCN vs Greedy | 10/10 | +0,161 | 0,0000 | 0,0002 | +12,53 | 10/10 / 0,0000 | đạt |
| Coverage | MAPPO-GAT vs Greedy | 10/10 | +0,138 | 0,0000 | 0,0002 | +7,47 | 10/10 / 0,0000 | đạt |
| Coverage | MAPPO-MLP vs Greedy | 10/10 | +0,142 | 0,0000 | 0,0002 | +10,69 | 10/10 / 0,0000 | đạt |
| Coverage | MAPPO-DGAT vs Greedy | 10/10 | +0,122 | 0,0000 | 0,0002 | +8,22 | 10/10 / 0,0000 | đạt |

Độ khó khối bản đồ test theo seed (VDR trung bình 5 phương pháp không ngẫu nhiên): seed 0: 0,653, seed 1: 0,658, seed 2: 0,695, seed 3: 0,585, seed 4: 0,649, seed 5: 0,637, seed 6: 0,602, seed 7: 0,630, seed 8: 0,575, seed 9: 0,641

![Hai đợt seed và gộp — medium](assets/giai-doan-21/medium/fig_batches_medium.png)

![Giá trị từng seed — medium](assets/giai-doan-21/medium/fig_perseed_medium.png)

![Learning curve — medium](assets/giai-doan-21/medium/fig_curves_medium.png)

Dữ liệu thô: `assets/giai-doan-21/medium/per_seed_medium.csv`, `summary_medium_*.csv`, `tests_medium.csv`.
<!-- /AUTO:rep-medium -->

<!-- AUTO:rep-easy -->
### Kết quả tự động — easy (cập nhật 22/09/2026 01:52)

Trạng thái dữ liệu: đợt A = seed 0–4, đợt B = seed 5–9; đợt B đã có Random Walk 5/5, Greedy 5/5, MADDPG 5/5, MAPPO-MLP 5/5, MAPPO-GCN 5/5, MAPPO-GAT 5/5, MAPPO-DGAT 5/5 — **đầy đủ**.

**VDR theo đợt** (mean ± sd liên-seed; p = Welch đợt A vs B):

| Phương pháp | Đợt A | Đợt B | Δ (B−A) | p | Gộp |
|---|---|---|---|---|---|
| Random Walk | 0,410 ± 0,078 | 0,382 ± 0,019 (n=5) | -0,028 | 0,47 | 0,396 ± 0,056 (n=10) |
| Greedy | 0,678 ± 0,031 | 0,656 ± 0,065 (n=5) | -0,022 | 0,52 | 0,667 ± 0,049 (n=10) |
| MADDPG | 0,149 ± 0,016 | 0,101 ± 0,029 (n=5) | -0,048 | 0,02 | 0,125 ± 0,033 (n=10) |
| MAPPO-MLP | 0,881 ± 0,038 | 0,868 ± 0,028 (n=5) | -0,013 | 0,55 | 0,874 ± 0,032 (n=10) |
| MAPPO-GCN | 0,930 ± 0,030 | 0,911 ± 0,019 (n=5) | -0,019 | 0,27 | 0,921 ± 0,026 (n=10) |
| MAPPO-GAT | 0,916 ± 0,027 | 0,878 ± 0,016 (n=5) | -0,039 | 0,03 | 0,897 ± 0,029 (n=10) |
| MAPPO-DGAT | 0,884 ± 0,032 | 0,857 ± 0,020 (n=5) | -0,027 | 0,15 | 0,870 ± 0,029 (n=10) |

**Coverage theo đợt** (mean ± sd liên-seed; p = Welch đợt A vs B):

| Phương pháp | Đợt A | Đợt B | Δ (B−A) | p | Gộp |
|---|---|---|---|---|---|
| Random Walk | 0,395 ± 0,006 | 0,394 ± 0,011 (n=5) | -0,001 | 0,87 | 0,394 ± 0,008 (n=10) |
| Greedy | 0,608 ± 0,027 | 0,625 ± 0,029 (n=5) | +0,017 | 0,35 | 0,617 ± 0,028 (n=10) |
| MADDPG | 0,121 ± 0,000 | 0,121 ± 0,000 (n=5) | +0,000 | 0,36 | 0,121 ± 0,000 (n=10) |
| MAPPO-MLP | 0,747 ± 0,027 | 0,785 ± 0,024 (n=5) | +0,038 | 0,05 | 0,766 ± 0,032 (n=10) |
| MAPPO-GCN | 0,757 ± 0,021 | 0,804 ± 0,019 (n=5) | +0,047 | 0,01 | 0,781 ± 0,031 (n=10) |
| MAPPO-GAT | 0,751 ± 0,014 | 0,790 ± 0,019 (n=5) | +0,039 | 0,01 | 0,770 ± 0,026 (n=10) |
| MAPPO-DGAT | 0,731 ± 0,026 | 0,782 ± 0,034 (n=5) | +0,051 | 0,03 | 0,757 ± 0,039 (n=10) |

**Kiểm định gộp** (Welch t / Mann–Whitney / Cohen's d; theo cặp seed = cùng bộ bản đồ test; α' Bonferroni = 0,0167 cho 3 so sánh với MLP):

| Chỉ số | So sánh | n | Δ | p Welch | p MW | d | Theo cặp: thắng / p | Bonferroni |
|---|---|---|---|---|---|---|---|---|
| VDR | MAPPO-GCN vs MAPPO-MLP | 10/10 | +0,047 | 0,0024 | 0,0036 | +1,59 | 9/10 / 0,0017 | đạt |
| VDR | MAPPO-GAT vs MAPPO-MLP | 10/10 | +0,023 | 0,1132 | 0,1041 | +0,74 | 6/10 / 0,0705 | không |
| VDR | MAPPO-DGAT vs MAPPO-MLP | 10/10 | -0,004 | 0,7860 | 0,8205 | -0,12 | 4/10 / 0,7654 | không |
| VDR | MAPPO-DGAT vs MAPPO-GAT | 10/10 | -0,027 | 0,0554 | 0,0539 | -0,92 | 2/10 / 0,0401 | không |
| VDR | MAPPO-GAT vs MAPPO-GCN | 10/10 | -0,024 | 0,0711 | 0,0757 | -0,86 | 3/10 / 0,1045 | không |
| VDR | MAPPO-DGAT vs MAPPO-GCN | 10/10 | -0,050 | 0,0007 | 0,0036 | -1,82 | 2/10 / 0,0012 | đạt |
| VDR | MAPPO-GCN vs Greedy | 10/10 | +0,254 | 0,0000 | 0,0002 | +6,43 | 10/10 / 0,0000 | đạt |
| VDR | MAPPO-GAT vs Greedy | 10/10 | +0,230 | 0,0000 | 0,0002 | +5,69 | 10/10 / 0,0000 | đạt |
| VDR | MAPPO-MLP vs Greedy | 10/10 | +0,207 | 0,0000 | 0,0002 | +4,98 | 10/10 / 0,0000 | đạt |
| VDR | MAPPO-DGAT vs Greedy | 10/10 | +0,203 | 0,0000 | 0,0002 | +5,03 | 10/10 / 0,0000 | đạt |
| Coverage | MAPPO-GCN vs MAPPO-MLP | 10/10 | +0,014 | 0,3251 | 0,3847 | +0,45 | 7/10 / 0,1037 | không |
| Coverage | MAPPO-GAT vs MAPPO-MLP | 10/10 | +0,004 | 0,7592 | 0,9698 | +0,14 | 4/10 / 0,6488 | không |
| Coverage | MAPPO-DGAT vs MAPPO-MLP | 10/10 | -0,010 | 0,5521 | 0,6232 | -0,27 | 4/10 / 0,5320 | không |
| Coverage | MAPPO-DGAT vs MAPPO-GAT | 10/10 | -0,014 | 0,3710 | 0,3847 | -0,41 | 5/10 / 0,2600 | không |
| Coverage | MAPPO-GAT vs MAPPO-GCN | 10/10 | -0,010 | 0,4373 | 0,5708 | -0,36 | 4/10 / 0,1841 | không |
| Coverage | MAPPO-DGAT vs MAPPO-GCN | 10/10 | -0,024 | 0,1503 | 0,1620 | -0,67 | 4/10 / 0,0715 | không |
| Coverage | MAPPO-GCN vs Greedy | 10/10 | +0,164 | 0,0000 | 0,0002 | +5,58 | 10/10 / 0,0000 | đạt |
| Coverage | MAPPO-GAT vs Greedy | 10/10 | +0,154 | 0,0000 | 0,0002 | +5,76 | 10/10 / 0,0000 | đạt |
| Coverage | MAPPO-MLP vs Greedy | 10/10 | +0,150 | 0,0000 | 0,0002 | +5,06 | 10/10 / 0,0000 | đạt |
| Coverage | MAPPO-DGAT vs Greedy | 10/10 | +0,140 | 0,0000 | 0,0002 | +4,14 | 10/10 / 0,0000 | đạt |

Độ khó khối bản đồ test theo seed (VDR trung bình 5 phương pháp không ngẫu nhiên): seed 0: 0,841, seed 1: 0,887, seed 2: 0,858, seed 3: 0,855, seed 4: 0,849, seed 5: 0,832, seed 6: 0,846, seed 7: 0,816, seed 8: 0,842, seed 9: 0,834

![Hai đợt seed và gộp — easy](assets/giai-doan-21/easy/fig_batches_easy.png)

![Giá trị từng seed — easy](assets/giai-doan-21/easy/fig_perseed_easy.png)

![Learning curve — easy](assets/giai-doan-21/easy/fig_curves_easy.png)

Dữ liệu thô: `assets/giai-doan-21/easy/per_seed_easy.csv`, `summary_easy_*.csv`, `tests_easy.csv`.
<!-- /AUTO:rep-easy -->

<!-- AUTO:zero-shot -->
### Kết quả tự động — zero-shot (cập nhật 22/09/2026 01:52)

**Zero-shot fixed-map — easy** (VDR trung bình theo N; N huấn luyện = 4):

| Phương pháp | seed | N=2 | N=4 | N=6 | N=8 | N=12 |
|---|---|---|---|---|---|---|
| MAPPO-MLP | 10 | 0,655 | 0,882 | 0,939 | 0,986 | 0,995 |
| MAPPO-GCN | 10 | 0,726 | 0,924 | 0,989 | 0,981 | 0,994 |
| MAPPO-GAT | 10 | 0,733 | 0,925 | 0,965 | 0,983 | 0,989 |
| MAPPO-DGAT | 10 | 0,740 | 0,898 | 0,967 | 0,982 | 0,982 |
Bản đồ theo N: N=2: 50×50, nạn nhân [3, 5], N=4: 50×50, nạn nhân [3, 5], N=6: 50×50, nạn nhân [3, 5], N=8: 50×50, nạn nhân [3, 5], N=12: 50×50, nạn nhân [3, 5]

**Zero-shot fixed-map — medium** (VDR trung bình theo N; N huấn luyện = 8):

| Phương pháp | seed | N=4 | N=6 | N=8 | N=10 | N=12 | N=16 |
|---|---|---|---|---|---|---|---|
| MAPPO-MLP | 10 | 0,418 | 0,630 | 0,675 | 0,774 | 0,847 | 0,871 |
| MAPPO-GCN | 10 | 0,436 | 0,601 | 0,769 | 0,795 | 0,836 | 0,861 |
| MAPPO-GAT | 10 | 0,446 | 0,629 | 0,682 | 0,766 | 0,795 | 0,867 |
| MAPPO-DGAT | 10 | 0,424 | 0,559 | 0,664 | 0,786 | 0,848 | 0,848 |
Bản đồ theo N: N=4: 100×100, nạn nhân [5, 8], N=6: 100×100, nạn nhân [5, 8], N=8: 100×100, nạn nhân [5, 8], N=10: 100×100, nạn nhân [5, 8], N=12: 100×100, nạn nhân [5, 8], N=16: 100×100, nạn nhân [5, 8]

**Zero-shot fixed-map — hard** (VDR trung bình theo N; N huấn luyện = 16):

| Phương pháp | seed | N=8 | N=12 | N=16 | N=20 | N=24 |
|---|---|---|---|---|---|---|
| MAPPO-MLP | 10 | 0,266 | 0,273 | 0,340 | 0,426 | 0,465 |
| MAPPO-GCN | 10 | 0,286 | 0,335 | 0,446 | 0,506 | 0,558 |
| MAPPO-GAT | 10 | 0,260 | 0,319 | 0,386 | 0,521 | 0,520 |
| MAPPO-DGAT | 10 | 0,210 | 0,259 | 0,329 | 0,385 | 0,428 |
Bản đồ theo N: N=8: 200×200, nạn nhân [8, 12], N=12: 200×200, nạn nhân [8, 12], N=16: 200×200, nạn nhân [8, 12], N=20: 200×200, nạn nhân [8, 12], N=24: 200×200, nạn nhân [8, 12]

![Zero-shot fixed-map](assets/giai-doan-21/zero_shot/fig_zeroshot_fixed_map.png)

**Zero-shot density-controlled — easy** (VDR trung bình theo N; N huấn luyện = 4):

| Phương pháp | seed | N=2 | N=4 | N=6 | N=8 | N=12 |
|---|---|---|---|---|---|---|
| MAPPO-MLP | 10 | 0,785 | 0,860 | 0,884 | 0,862 | 0,849 |
| MAPPO-GCN | 10 | 0,890 | 0,917 | 0,916 | 0,895 | 0,905 |
| MAPPO-GAT | 10 | 0,875 | 0,884 | 0,889 | 0,881 | 0,870 |
| MAPPO-DGAT | 10 | 0,830 | 0,870 | 0,858 | 0,857 | 0,844 |
Bản đồ theo N: N=2: 35×35, nạn nhân [2, 2], N=4: 50×50, nạn nhân [3, 5], N=6: 61×61, nạn nhân [4, 8], N=8: 71×71, nạn nhân [6, 10], N=12: 87×87, nạn nhân [9, 15]

**Zero-shot density-controlled — medium** (VDR trung bình theo N; N huấn luyện = 8):

| Phương pháp | seed | N=4 | N=6 | N=8 | N=10 | N=12 | N=16 |
|---|---|---|---|---|---|---|---|
| MAPPO-MLP | 10 | 0,632 | 0,647 | 0,697 | 0,711 | 0,678 | 0,626 |
| MAPPO-GCN | 10 | 0,632 | 0,669 | 0,696 | 0,760 | 0,713 | 0,625 |
| MAPPO-GAT | 10 | 0,587 | 0,627 | 0,649 | 0,724 | 0,689 | 0,568 |
| MAPPO-DGAT | 10 | 0,627 | 0,647 | 0,644 | 0,708 | 0,705 | 0,548 |
Bản đồ theo N: N=4: 71×71, nạn nhân [2, 4], N=6: 87×87, nạn nhân [4, 6], N=8: 100×100, nạn nhân [5, 8], N=10: 112×112, nạn nhân [6, 10], N=12: 122×122, nạn nhân [8, 12], N=16: 141×141, nạn nhân [10, 16]

**Zero-shot density-controlled — hard** (VDR trung bình theo N; N huấn luyện = 16):

| Phương pháp | seed | N=8 | N=12 | N=16 | N=20 | N=24 |
|---|---|---|---|---|---|---|
| MAPPO-MLP | 10 | 0,318 | 0,293 | 0,354 | 0,352 | 0,430 |
| MAPPO-GCN | 10 | 0,404 | 0,402 | 0,401 | 0,399 | 0,509 |
| MAPPO-GAT | 10 | 0,419 | 0,330 | 0,394 | 0,405 | 0,472 |
| MAPPO-DGAT | 10 | 0,327 | 0,269 | 0,334 | 0,312 | 0,409 |
Bản đồ theo N: N=8: 141×141, nạn nhân [4, 6], N=12: 173×173, nạn nhân [6, 9], N=16: 200×200, nạn nhân [8, 12], N=20: 224×224, nạn nhân [10, 15], N=24: 245×245, nạn nhân [12, 18]

![Zero-shot density-controlled](assets/giai-doan-21/zero_shot/fig_zeroshot_density.png)
<!-- /AUTO:zero-shot -->

---

[← Giai đoạn 20](giai-doan-20-hard-n-10-dot-lap-lai-seed-5-9-hoan-tat-10-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 22 →](giai-doan-22-hard-ngan-sach-2x-gcn-dgat-2m-buoc-22-09-2026.md)
