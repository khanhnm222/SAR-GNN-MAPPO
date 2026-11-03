[← Giai đoạn 18](giai-doan-18-hai-kiem-chung-cuoi-08-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 20 →](giai-doan-20-hard-n-10-dot-lap-lai-seed-5-9-hoan-tat-10-09-2026.md)

# Giai đoạn 19 — Lặp lại Hard (seed 5–9) và bản nháp Chương 5 (08/09/2026)

### Lặp lại Hard

Vì mọi RNG nay đã gieo seed, chạy lại seed 0–4 sẽ cho **đúng cùng số**, không
phải một mẫu độc lập. Đợt lặp lại dùng **seed 5–9** (thêm `--seed-offset` cho
`scripts/run_all_experiments.py`), 7 phương pháp × 5 seed = 35 lượt, 1M bước,
ghi vào cùng `results_v4/hard/`. Khi xong, Hard có n = 10 và Bảng 5.11–5.12 của
Chương 5 được bổ sung cột gộp. Ước tính ~21 giờ.

### Chương 5

`CHUONG5_THUC_NGHIEM.md` được viết lại toàn bộ theo `results_v4/` cuối
(105 lượt gieo seed, heuristic 5 seed trên cùng tập test): 19 bảng, 8 hình, mọi
số liệu và kiểm định thay bằng bộ cuối; bổ sung mục 5.2.4 (bốn lỗi giao thức),
5.4.4 (kiểm chứng ngân sách 2×), 5.6 (biến thiên giữa đợt); sửa các phát biểu
đã bị bác ở Giai đoạn 16–18 (Hard vs Greedy, "Bonferroni ở Medium", "lợi ích
giảm theo N", "DGAT tổng quát hoá tốt nhất"). Bản cũ lưu ở
`thesis proposals/CHUONG5_v2_stage15_OBSOLETE.md`. Hình 5.3/5.4/5.7 tái tạo
bằng `scripts/make_chapter5_figures.py` từ dữ liệu cuối.


### Mốc trung gian Giai đoạn 19 (09/09/2026): GCN ở Hard đứng vững với n = 10

GCN, MLP, Greedy, MADDPG đã đủ seed 5–9 (25/35 lượt). Đợt 2 **khó hơn** đợt 1
cho mọi phương pháp không dùng đồ thị — Greedy 0,403 → 0,357, MLP 0,382 →
0,341 — trong khi GCN giữ nguyên (0,438 → 0,443). Gộp n = 10:

| So sánh (Hard, n = 10 mỗi bên) | VDR | p | Cohen's d |
|---|---|---|---|
| MAPPO-GCN vs MAPPO-MLP | 0.440 ± 0.038 vs 0.362 ± 0.051 | **0.0012** | +1.75 |
| MAPPO-GCN vs Greedy | 0.440 vs 0.380 ± 0.041 | **0.0031** | +1.53 |
| MAPPO-MLP vs Greedy | 0.362 vs 0.380 | 0.3855 | -0.40 |

Kết luận Hard của Giai đoạn 15–18 ("GCN vượt MLP, p = 0,016, cần lặp lại")
**đứng vững và mạnh lên** khi lặp lại: p = 0.0012, d = +1.75. Với n = 10,
GCN cũng là kiến trúc học duy nhất vượt Greedy có ý nghĩa ở Hard. GAT và DGAT
đang chạy; Chương 5 sẽ được bổ sung cột n = 10 khi đủ 35/35.

---

[← Giai đoạn 18](giai-doan-18-hai-kiem-chung-cuoi-08-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 20 →](giai-doan-20-hard-n-10-dot-lap-lai-seed-5-9-hoan-tat-10-09-2026.md)
