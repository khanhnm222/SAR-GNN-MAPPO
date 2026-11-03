[← Giai đoạn 22](giai-doan-22-hard-ngan-sach-2x-gcn-dgat-2m-buoc-22-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md)

# Giai đoạn 23 — Ablation hàm phần thưởng (Medium, MAPPO-GCN, bỏ từng thành phần) (26/09/2026)

### Lý do

Dòng duy nhất còn "Thiếu" trong bảng đối chiếu đề cương (Bảng 5.4): ablation hàm phần thưởng
r = α·r_cover + β·r_victim + κ·r_collision + δ·r_energy, bỏ lần lượt từng thành phần. Mục đích: cho thấy mỗi thành phần
đóng góp gì vào hành vi đã học (không phải để chọn lại trọng số — trọng số (20; 5; −0,05; −0,01) đã được chốt ở Giai
đoạn 4/9 qua chẩn đoán reward hacking). Chạy với phiên bản chính của framework (MAPPO-GCN) ở Medium — kịch bản có vật
cản tĩnh (để κ có nghĩa) và chưa bão hoà như Easy.

### Cấu hình

- Mã: thêm `--reward-zero {cover,victim,collision,energy}` vào `training/train.py` (đặt trọng số thành phần đó = 0, mọi
  thứ khác giữ nguyên; `make_env` nhận `reward_weights`); `run_all_experiments --reward-zero ...` sinh một biến thể/thành
  phần vào `results_v4_reward_ablation/no_<comp>/medium/mappo_gcn/seed_k`. Smoke test 3k bước: OK, meta ghi `reward_zero`.
- Lệnh: `run_all_experiments --scenarios medium --methods mappo_gcn --seeds 5 --out-dir results_v4_reward_ablation
  --reward-zero cover victim collision energy --max-jobs 6 --no-postprocess` — 20 lượt, seed 0–4, 800k bước, cùng bản
  đồ test 90.000 + seed·1.000 với `results_v4/medium/mappo_gcn` (đối chứng đầy đủ, n = 10; so theo cặp trên seed 0–4).
- Kỳ vọng trước khi chạy (ghi để kiểm sau): bỏ β (nạn nhân) → VDR sụp mạnh nhưng Coverage có thể giữ (chính sách quét
  đơn thuần); bỏ α (bao phủ) → tín hiệu thưa, VDR/Coverage giảm; bỏ κ (va chạm) → Collision Rate tăng rõ, VDR ít đổi;
  bỏ δ (năng lượng) → gần như không đổi (δ nhỏ, UAV vẫn cạn pin theo mô hình).
- Ước tính: GCN Medium ≈ 3 h/lượt, 6 song song → ~12 h nếu máy không ngủ. Log: scratchpad `reward_ablation.log`,
  kết thúc `RWABL_DONE`.

### Trạng thái

- 26/09 05:41: khởi chạy 20 lượt.
- 26/09 14:58: 19/20 lượt OK (2,6–2,8 h/lượt với 6 song song; máy không ngủ). **1 lượt chết**: `no_energy` seed 4 lúc
  ~06:41 sau 3.622 s, mã 0xC000070A — cùng mã với vụ Restart Manager 23/09, nhưng lần này chỉ 1/6 tiến trình chết và
  Event Log (System/Application, 06:30–06:50) không có sự kiện nào; nguyên nhân chưa rõ. Chạy lại riêng seed 4 lúc 15:00
  (`reward_ablation2.log`, `RWABL2_DONE`, rồi `reward_ablation_report` tự chạy). Báo cáo tạm (n = 4 cho δ) đã chèn.
- 26/09 15:58: seed 4 `no_energy` chạy lại OK (3.523 s khi chạy một mình). **20/20 lượt xong**, báo cáo tự động bên dưới
  là số cuối (đối chứng = `results_v4/medium/mappo_gcn` seed 0–4, cùng bản đồ test, kiểm định theo cặp).

### Kết luận Giai đoạn 23

1. **Thưởng bao phủ α là thành phần quyết định.** Bỏ α: VDR 0,717 → 0,649 (−0,067, p = 0,017, d = −2,45, xấu đi 5/5
   seed), Coverage 0,677 → 0,617 (−0,060, p = 0,006); entropy cuối 1,46 nat (đối chứng 0,90) — không có tín hiệu dày
   đặc, chính sách không "chốt" được hành vi. Đây chính là cơ chế của lỗi 4 (α = 1 làm r_cover ≈ 0, Giai đoạn 4/9).
2. **Thưởng nạn nhân β không cần thiết ở Medium.** Bỏ β: VDR 0,727 (+0,010, p = 0,62) — không đổi — và Coverage
   **tăng** 0,677 → 0,737 (+0,060, p = 0,007, d = +3,72, tốt hơn ở 5/5 seed); entropy cuối 0,60 (chốt hơn). Nạn nhân
   được tìm thấy như hệ quả của quét đều; thưởng thưa β (chỉ khi phát hiện) không thêm thông tin và còn kéo chính sách
   khỏi hành vi quét tối ưu (dừng/quẩn quanh cụm nạn nhân). Kỳ vọng "bỏ β → VDR sụp" ghi trước khi chạy **sai**.
3. **Phạt va chạm κ không tác động qua Collision Rate mà qua shaping.** Bỏ κ: Collision Rate không tăng (0,023 so với
   0,025 — vốn đã rất thấp vì mô hình va chạm chỉ là cờ theo bước và UAV được huấn luyện phân tán), nhưng VDR giảm
   0,717 → 0,641 (−0,075, p = 0,020, d = −1,45, 5/5 seed) và Coverage −0,025 (p = 0,061); entropy 1,16. Tác dụng thật
   của κ là giữ các UAV tách nhau và né vật cản sớm, tức một tín hiệu phân tán bổ trợ cho α; mất nó, đàn phủ kém đi.
4. **Phạt năng lượng δ gần như trung tính** (VDR −0,027, p = 0,14; Coverage −0,024, p = 0,15; 4/5 seed xấu đi nhẹ):
   δ = 0,01 nhỏ và UAV vẫn cạn pin theo mô hình năng lượng bất kể thưởng; giữ lại vì rẻ và có xu hướng dương.
5. Không thành phần nào làm TTFD hay Collision khác biệt có ý nghĩa (n = 5). Hàm thưởng tối giản cho Medium là
   α·r_cover + κ·r_collision; β và δ có thể bỏ mà không mất hiệu năng. **Lưu ý phạm vi:** chỉ kiểm ở Medium (nạn nhân
   cụm, không ưu tiên); ở Hard (nạn nhân hành lang, ưu tiên 1–3) β mang thông tin về ưu tiên mà α không có, nên kết
   luận 2 không được ngoại suy sang Hard khi chưa chạy.

Kỳ vọng ghi trước khi chạy: đúng 2/4 (α, δ), sai 2/4 (β, κ) — thêm một ví dụ cho quy tắc "kiểm bằng ablation, không
bằng trực giác". Chi phí: 21 lượt (kể cả một lần chạy lại), ~10,3 h wall-clock.

Việc tiếp theo: đưa vào Chương 5 mục 5.4.5 / luận văn mục 4.4.5 (đã làm 26/09); Bảng 5.4 dòng ablation → Đạt.

<!-- AUTO:reward-ablation -->
### Kết quả tự động — ablation hàm thưởng medium (cập nhật 26/09/2026 15:58)

Trạng thái: bỏ α (bao phủ) 5/5, bỏ β (nạn nhân) 5/5, bỏ κ (va chạm) 5/5, bỏ δ (năng lượng) 5/5 — **đầy đủ**.

**VDR: đối chứng đầy đủ so với từng biến thể bỏ một thành phần** (cùng seed, cùng bản đồ test; p = paired t-test; d = Cohen; 'seed xấu đi' = số seed biến thể kém đối chứng):

| Biến thể | n | Đầy đủ | Bỏ thành phần | Δ (bỏ − đầy đủ) | p theo cặp | d | seed xấu đi |
|---|---|---|---|---|---|---|---|
| bỏ α (bao phủ) | 5 | 0,717 ± 0,030 | 0,649 ± 0,025 | −0,067 | 0,017 | −2,45 | 5/5 |
| bỏ β (nạn nhân) | 5 | 0,717 ± 0,030 | 0,727 ± 0,015 | +0,010 | 0,615 | +0,41 | 3/5 |
| bỏ κ (va chạm) | 5 | 0,717 ± 0,030 | 0,641 ± 0,067 | −0,075 | 0,020 | −1,45 | 5/5 |
| bỏ δ (năng lượng) | 5 | 0,717 ± 0,030 | 0,690 ± 0,034 | −0,027 | 0,144 | −0,83 | 4/5 |

**Coverage: đối chứng đầy đủ so với từng biến thể bỏ một thành phần** (cùng seed, cùng bản đồ test; p = paired t-test; d = Cohen; 'seed xấu đi' = số seed biến thể kém đối chứng):

| Biến thể | n | Đầy đủ | Bỏ thành phần | Δ (bỏ − đầy đủ) | p theo cặp | d | seed xấu đi |
|---|---|---|---|---|---|---|---|
| bỏ α (bao phủ) | 5 | 0,677 ± 0,019 | 0,617 ± 0,037 | −0,060 | 0,006 | −2,03 | 5/5 |
| bỏ β (nạn nhân) | 5 | 0,677 ± 0,019 | 0,737 ± 0,013 | +0,060 | 0,007 | +3,72 | 0/5 |
| bỏ κ (va chạm) | 5 | 0,677 ± 0,019 | 0,652 ± 0,008 | −0,025 | 0,061 | −1,78 | 5/5 |
| bỏ δ (năng lượng) | 5 | 0,677 ± 0,019 | 0,653 ± 0,024 | −0,024 | 0,151 | −1,13 | 4/5 |

**Collision: đối chứng đầy đủ so với từng biến thể bỏ một thành phần** (cùng seed, cùng bản đồ test; p = paired t-test; d = Cohen; 'seed xấu đi' = số seed biến thể kém đối chứng):

| Biến thể | n | Đầy đủ | Bỏ thành phần | Δ (bỏ − đầy đủ) | p theo cặp | d | seed xấu đi |
|---|---|---|---|---|---|---|---|
| bỏ α (bao phủ) | 5 | 0,025 ± 0,007 | 0,016 ± 0,004 | −0,010 | 0,023 | −1,58 | 0/5 |
| bỏ β (nạn nhân) | 5 | 0,025 ± 0,007 | 0,025 ± 0,007 | −0,001 | 0,854 | −0,12 | 2/5 |
| bỏ κ (va chạm) | 5 | 0,025 ± 0,007 | 0,023 ± 0,004 | −0,002 | 0,396 | −0,39 | 1/5 |
| bỏ δ (năng lượng) | 5 | 0,025 ± 0,007 | 0,024 ± 0,004 | −0,002 | 0,557 | −0,31 | 2/5 |

**TTFD (bước): đối chứng đầy đủ so với từng biến thể bỏ một thành phần** (cùng seed, cùng bản đồ test; p = paired t-test; d = Cohen; 'seed xấu đi' = số seed biến thể kém đối chứng):

| Biến thể | n | Đầy đủ | Bỏ thành phần | Δ (bỏ − đầy đủ) | p theo cặp | d | seed xấu đi |
|---|---|---|---|---|---|---|---|
| bỏ α (bao phủ) | 5 | 23,747 ± 6,117 | 31,873 ± 12,309 | +8,127 | 0,252 | +0,84 | 3/5 |
| bỏ β (nạn nhân) | 5 | 23,747 ± 6,117 | 23,387 ± 6,053 | −0,360 | 0,938 | −0,06 | 3/5 |
| bỏ κ (va chạm) | 5 | 23,747 ± 6,117 | 30,520 ± 11,839 | +6,773 | 0,201 | +0,72 | 3/5 |
| bỏ δ (năng lượng) | 5 | 23,747 ± 6,117 | 32,000 ± 6,166 | +8,253 | 0,101 | +1,34 | 4/5 |

Entropy chính sách cuối (nat, trung bình 5 % iteration cuối, trung bình seed): đầy đủ: 0,90, bỏ α (bao phủ): 1,46, bỏ β (nạn nhân): 0,60, bỏ κ (va chạm): 1,16, bỏ δ (năng lượng): 1,06

![Ablation hàm thưởng — medium](assets/giai-doan-23/reward_ablation_medium/fig_reward_ablation_medium.png)

Dữ liệu thô: `assets/giai-doan-23/reward_ablation_medium/reward_ablation_medium.csv`; checkpoint tại `results_v4_reward_ablation/no_<thành phần>/medium/`.
<!-- /AUTO:reward-ablation -->

---

[← Giai đoạn 22](giai-doan-22-hard-ngan-sach-2x-gcn-dgat-2m-buoc-22-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md)
