[← Giai đoạn 21](experiment_log\giai-doan-21-lap-lai-medium-easy-seed-5-9-va-zero-shot-density-20-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md)

# Giai đoạn 22 — Hard ngân sách 2× (GCN và DGAT, 2M bước, 5 seed) (22/09/2026)

### Lý do

Điểm yếu duy nhất còn lại mà bài báo và Chương 5 tự nêu ở mục hạn chế: Hard **chưa hội tụ** ở 1M bước
(entropy cuối 1,5–1,8 nat, learning curve còn tăng), nên phản biện "DGAT thua chỉ vì chưa đủ bước cho
nhiều tham số hơn" chưa được đóng ở kịch bản quan trọng nhất. Kiểm chứng ngân sách 2× mới làm ở Easy
(Giai đoạn 18: khoảng cách DGAT − GCN không đóng lại, −0,046 → −0,055). Đợt này lặp lại đúng kiểm chứng
đó ở Hard.

### Cấu hình

- `run_all_experiments --scenarios hard --methods mappo_gcn gnn_mappo --seeds 5 --steps-override 2000000
  --out-dir results_v4_budget2x --max-jobs 6 --no-postprocess` — 10 lượt, seed 0–4, cùng mã nguồn và cấu
  hình với `results_v4/hard` chỉ khác tổng số bước (2M thay 1M; lịch giảm LR/entropy trải theo 2M).
- Ước tính wall-clock: GCN ~16 h/lượt, DGAT ~21 h/lượt ở 6 tiến trình song song → ~37–42 h nếu máy không
  ngủ. Khối bản đồ test 90.000 + seed·1.000 (cùng đợt A của `results_v4/hard`) để so trực tiếp từng seed.
- Câu hỏi cần trả lời: (i) GCN và DGAT tăng bao nhiêu khi gấp đôi bước; (ii) khoảng cách DGAT − GCN có đóng
  lại không; (iii) thứ hạng so với MLP 1M và Greedy có đổi không.

### Trạng thái

- 22/09 21:37: khởi chạy (6 tiến trình: 5 GCN + 1 DGAT; 4 DGAT còn lại vào đợt sau).
- 23/09 07:29: máy không ngủ trong đêm. GCN seed 0–4 đều 1,32M/2M (66 %, 37–41 bước/s, ETA ~5 h → ~12:30);
  DGAT seed 0 0,92M/2M (46 %, 25 bước/s, ETA ~12 h); DGAT seed 1–4 sẽ vào khi GCN xong (~20 h/lượt ở tốc độ
  25–30 bước/s) → toàn đợt dự kiến xong sáng 24/09.
- 23/09 19:56: **GCN 5/5 xong** (2M bước, ~15,8 h/lượt, 33–35 bước/s). **DGAT thất bại cả 5 seed**: seed 1–4 chết ngay
  lúc khởi tạo lúc ~13:25 (mã 0xC0000142 STATUS_DLL_INIT_FAILED — không thấy sự kiện hệ thống tương ứng; khả năng
  phiên đăng nhập bị khoá/không tương tác khi tiến trình mới được sinh); seed 0 huấn luyện đủ 2,00M bước (22,0 h) nhưng
  bị giết ở bước đánh giá cuối (mã 0x40010004 DBG_TERMINATE_PROCESS) vì **máy được khởi động lại từ Start menu lúc
  19:38:50** (User32 1074 → Winlogon 7002 → Windows Update / TrustedInstaller reboot 19:44, hệ thống lên lại 19:48).
  `train.py` chỉ lưu `policy.pt` sau đánh giá cuối nên seed 0 không có checkpoint để đánh giá lại — mất 22 giờ máy.
  Khắc phục: thêm lưu `policy_latest.pt` ở mỗi lần đánh giá trung gian (không ảnh hưởng RNG/kết quả), rồi chạy lại
  DGAT seed 0–4 (5 tiến trình song song, ~22–24 h).
- 23/09 19:58: **GCN 2M so với GCN 1M (cùng 5 seed, cùng bản đồ test)**: VDR 0,438 → 0,465 (Δ +0,027, theo cặp
  p = 0,31, 3/5 seed tăng; std tăng 0,016 → 0,039), Coverage 0,312 → 0,341 (+0,028, p = 0,017, 5/5). Entropy cuối
  1,28 nat (1M: 1,47) — vẫn chưa hội tụ hẳn nhưng đã "chốt" hơn. Tham chiếu: MLP-1M n = 10 0,362, Greedy 0,380 → GCN
  2M vẫn trên cả hai. Câu hỏi chính (DGAT có bắt kịp không) chờ 5 seed DGAT chạy lại.
- 23/09 19:59: chạy lại DGAT seed 0–4 (5 tiến trình song song, `training/train.py` nay lưu `policy_latest.pt` sau mỗi
  lần đánh giá trung gian). ETA ~22–24 h → tối 24/09 nếu máy không khởi động lại/ngủ.
- 23/09 20:45:50: **DGAT seed 0–4 chết lần hai, đồng thời cả 5 tiến trình** sau 2.792 s (~55k/2M bước, 0,8 h), mã thoát
  0xC000070A; launcher (`run_all_experiments`) vẫn sống và ghi 5 FAIL, log stdout con rỗng (bị giết trước khi flush).
  Bằng chứng Event Log: Application/RestartManager 10001 "Ending session 0 started 2026-09-23T13:39:08Z" (= 20:39:08
  giờ máy) lúc 20:46:36, kèm Security-SPP 16394 "Offline downlevel migration succeeded" 20:49 — tức một phiên Restart
  Manager của Windows servicing (hoàn tất cập nhật sau lần khởi động lại 19:44) đã đóng các tiến trình đang chạy đúng
  thời điểm đó. Không có sự kiện ngủ/tắt máy nào trong khoảng 20:00–08:00; máy chỉ vào Modern Standby 08:12–09:03 ngày
  24/09 (sau khi các lượt đã chết). Bộ chờ báo cáo thấy `B2X_DGAT_DONE` lúc 20:50 và chạy `budget_report` → khối
  AUTO dưới đây vẫn là số **tạm thời** (chỉ GCN). `policy_latest.pt` 20:34 có nhưng `train.py` không hỗ trợ resume nên
  chạy lại từ đầu.
- 24/09 10:15: **chạy lại DGAT seed 0–4 lần ba** (cùng lệnh, 5 tiến trình song song, RunLogger ghi đè file cũ trong thư mục seed; yêu cầu
  giữ máy thức qua ứng dụng). Tốc độ đo được lần hai: 55k bước/0,8 h ≈ 19 bước/s/tiến trình → ETA ~28–29 h → khoảng
  trưa–chiều 25/09 nếu không bị gián đoạn. Log: scratchpad `budget2x_hard_dgat2.log`, kết thúc `B2X_DGAT_DONE2`, sau
  đó `budget_report` tự chạy (`REPORT3_DONE`).
- 25/09 07:46: **DGAT lần ba vẫn sống**, 5/5 seed ở 1,05–1,09M/2M bước (53–54 %) sau 21,6 h; tốc độ 13,5–14,0 bước/s/tiến
  trình (chậm hơn ước tính 19 vì 5 DGAT chạy song song tranh CPU; lúc chạy 1 DGAT cạnh 5 GCN đạt 25). Máy vào Modern
  Standby 3 lần tối 24/09 (18:08–18:58, 18:58–19:12, 23:01 tức thời) ≈ 64 phút, ngoài phiên làm việc nên keep-awake
  của ứng dụng đã hết hiệu lực; một phiên RestartManager lúc 16:21 (24/09) không giết tiến trình. VDR đánh giá trung
  gian (4 episode, nhiễu) tại ~1,05M: 0,465 / 0,583 / 0,388 / 0,402 / 0,148 (trung bình ≈ 0,40) — chỉ để theo dõi,
  không so được với số cuối. ETA: còn ~0,93M bước ≈ 19 h → khoảng 02:30–03:30 ngày 26/09, cộng đánh giá cuối; sau đó
  `budget_report` tự chạy.
- 25/09 19:01: 5/5 seed ở 1,48–1,52M/2M (74–76 %) sau 32,8 h. Máy vào Modern Standby thêm hai đợt trong ngày
  (08:13–09:01 và 16:22–18:49, ≈ 3,2 h) nên tốc độ wall-clock tụt còn 12,5–12,9 bước/s; tốc độ khi thức ≈ 15 bước/s
  (435k bước trong ~8 h thức từ 07:46). VDR trung gian tại ~1,5M: 0,413 / 0,415 / 0,433 / 0,418 / 0,384 (trung bình
  ≈ 0,41, phương sai giữa seed đã hẹp lại so với mốc 1,05M). Còn ~0,5M bước ≈ 9–10 h thức → ETA 04:30–05:00 ngày 26/09
  nếu không ngủ, muộn hơn nếu máy ngủ đêm; sau đó đánh giá cuối và `budget_report` tự chạy.
- 26/09 04:16: **DGAT 5/5 xong** (2M bước; 41,2–42,1 h/lượt wall-clock với 5 tiến trình song song và ~4 h Modern
  Standby tổng cộng; không sự cố). `budget_report` tự chạy → khối AUTO bên dưới là số cuối. Kết quả từng seed (0–4),
  VDR 1M → 2M: 0,365 → 0,402; 0,349 → 0,331; 0,345 → 0,380; 0,391 → 0,438; 0,277 → 0,373. Coverage: 0,245 → 0,287;
  0,261 → 0,276; 0,258 → 0,271; 0,228 → 0,270; 0,224 → 0,272.

### Kết luận Giai đoạn 22

1. **Gấp đôi ngân sách giúp cả hai kiến trúc, DGAT nhỉnh hơn về mức tăng nhưng không bắt kịp.** VDR: GCN 0,438 → 0,465
   (+0,027, theo cặp p = 0,31, 3/5 seed), DGAT 0,345 → 0,385 (+0,040, p = 0,093, 4/5). Coverage tăng có ý nghĩa ở cả hai
   (GCN +0,028, p = 0,017; DGAT +0,032, p = 0,012; 5/5 seed mỗi bên). Entropy cuối: GCN 1,46 → 1,20, DGAT 1,96 → 1,13 —
   DGAT-2M "chốt" hành vi hơn cả GCN-1M.
2. **Khoảng cách DGAT − GCN không đóng lại ở Hard.** Trên cùng 5 seed và cùng bản đồ test: −0,092 ở 1M → −0,080 ở 2M
   (Welch p = 0,012, d = −2,04, DGAT thua 0/5 cặp); Coverage −0,069 → −0,066 (p < 0,0001, 0/5). Mức thu hẹp 0,012 nhỏ
   hơn std liên-seed của chính DGAT-2M (0,039). Quan trọng hơn: **DGAT với 2M bước (0,385) vẫn thấp hơn GCN với 1M bước
   (0,438 cùng seed; 0,440 n = 10)**, tức gấp đôi tính toán cho DGAT không mua được hiệu năng mà GCN đạt với một nửa.
   DGAT-2M chỉ ngang Greedy (0,380) và MLP-1M (0,362).
3. **Phản biện "DGAT thua vì chưa đủ bước" bị bác bỏ ở cả Easy (GĐ18) và Hard.** Thứ hạng GCN > DGAT là do kiến trúc.
   Với RQ2: DGAT-2M (0,385) cũng thấp hơn GAT-1M trên cùng 5 seed (0,417; thua 4/5 cặp, p = 0,22) — thêm bước cho khối GRU không đưa nó lên mức
   của biến thể không GRU ở một nửa ngân sách, nên kết luận "GRU không đóng góp, gây hại ở Hard" đứng vững cả khi xét
   ngân sách.
4. Hạn chế còn lại về hội tụ: cả hai learning curve 2M vẫn dốc nhẹ ở cuối (GCN đánh giá trung gian ~0,50, DGAT ~0,40);
   thứ hạng ở 5M bước như đề cương chưa kiểm chứng, nhưng không có dấu hiệu nào cho thấy nó sẽ đảo.

Chi phí: 10 lượt 2M ≈ 5 × 15,8 h (GCN) + 5 × 42 h (DGAT, tính cả 2 lần chết) ≈ 290 h tiến trình; wall-clock 22/09
21:37 → 26/09 04:16 với hai lần chạy lại DGAT (khởi động lại Windows 23/09 19:44; Restart Manager 23/09 20:46).

Việc tiếp theo: đưa Bảng/Hình vào Chương 5 mục 5.4.4 và luận văn mục 4.4.4 (đã làm 26/09), Bảng 4 bài báo, báo cáo CBHD.

<!-- AUTO:budget-hard -->
### Kết quả tự động — ngân sách 2× hard (cập nhật 26/09/2026 04:16)

Trạng thái: ngân sách 2× đã có MAPPO-GCN 5/5, MAPPO-DGAT 5/5 — **đầy đủ**.

**VDR: 1M → 2M bước** (cùng seed, cùng bản đồ test 90.000 + seed·1.000):

| Kiến trúc | n | 1× | 2× | Δ (2× − 1×) | p theo cặp | seed tăng |
|---|---|---|---|---|---|---|
| MAPPO-GCN | 5 | 0,438 ± 0,016 | 0,465 ± 0,039 | +0,027 | 0,313 | 3/5 |
| MAPPO-DGAT | 5 | 0,345 ± 0,042 | 0,385 ± 0,039 | +0,040 | 0,093 | 4/5 |

**Coverage: 1M → 2M bước** (cùng seed, cùng bản đồ test 90.000 + seed·1.000):

| Kiến trúc | n | 1× | 2× | Δ (2× − 1×) | p theo cặp | seed tăng |
|---|---|---|---|---|---|---|
| MAPPO-GCN | 5 | 0,312 ± 0,013 | 0,341 ± 0,007 | +0,028 | 0,017 | 5/5 |
| MAPPO-DGAT | 5 | 0,243 ± 0,017 | 0,275 ± 0,007 | +0,032 | 0,012 | 5/5 |

**Collision: 1M → 2M bước** (cùng seed, cùng bản đồ test 90.000 + seed·1.000):

| Kiến trúc | n | 1× | 2× | Δ (2× − 1×) | p theo cặp | seed tăng |
|---|---|---|---|---|---|---|
| MAPPO-GCN | 5 | 0,046 ± 0,031 | 0,044 ± 0,046 | −0,002 | 0,934 | 2/5 |
| MAPPO-DGAT | 5 | 0,033 ± 0,029 | 0,051 ± 0,023 | +0,018 | 0,207 | 3/5 |

**Khoảng cách MAPPO-DGAT − MAPPO-GCN** (câu hỏi chính: có đóng lại khi gấp đôi bước không?):

| Chỉ số | Ngân sách | n | GCN | DGAT | Khoảng cách | Welch p | d | theo cặp thắng / p |
|---|---|---|---|---|---|---|---|---|
| VDR | 1M | 10 | 0,440 | 0,337 | −0,103 | 0,0000 | −2,74 | 0/10 / 0,000 |
| VDR | 2M | 5 | 0,465 | 0,385 | −0,080 | 0,0123 | −2,04 | 0/5 / 0,054 |
| Coverage | 1M | 10 | 0,320 | 0,248 | −0,072 | 0,0000 | −4,15 | 0/10 / 0,000 |
| Coverage | 2M | 5 | 0,341 | 0,275 | −0,066 | 0,0000 | −9,47 | 0/5 / 0,000 |

Tham chiếu (ngân sách 1×, n = 10): MAPPO-MLP 0,362, Greedy 0,380.

Entropy chính sách cuối (nat, trung bình seed): MAPPO-GCN 1M: 1,46, MAPPO-GCN 2M: 1,20, MAPPO-DGAT 1M: 1,96, MAPPO-DGAT 2M: 1,13

![Từng seed 1× → 2× — hard](assets/giai-doan-22/budget2x_hard/fig_budget2x_pairs_hard.png)

![Learning curve 2× — hard](assets/giai-doan-22/budget2x_hard/fig_budget2x_curves_hard.png)

Dữ liệu thô: `assets/giai-doan-22/budget2x_hard/budget2x_hard.csv`; checkpoint 2× tại `results_v4_budget2x/hard/`.
<!-- /AUTO:budget-hard -->

---

[← Giai đoạn 21](experiment_log\giai-doan-21-lap-lai-medium-easy-seed-5-9-va-zero-shot-density-20-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 23 →](giai-doan-23-ablation-ham-thuong-medium-gcn-26-09-2026.md)
