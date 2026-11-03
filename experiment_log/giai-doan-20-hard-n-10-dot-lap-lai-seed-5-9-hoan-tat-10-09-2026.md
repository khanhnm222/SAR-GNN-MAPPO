[← Giai đoạn 19](giai-doan-19-lap-lai-hard-seed-5-9-va-ban-nhap-chuong-5-08-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 21 →](giai-doan-21-lap-lai-medium-easy-seed-5-9-va-zero-shot-density-20-09-2026.md)

# Giai đoạn 20 — Hard n = 10: đợt lặp lại seed 5–9 hoàn tất (10/09/2026)

### Kết quả chạy

35/35 lượt (7 phương pháp × seed 5–9, 1M bước) xong lúc 01:15 ngày 10/09 sau
42 giờ wall-clock (bắt đầu 07:07 ngày 08/09; 6 tiến trình song song, DGAT
seed 9 chạy cuối cùng một mình). Không lượt nào lỗi; kiểm tra sức khoẻ 4 seed
DGAT cuối trong lúc chạy (phần thưởng cuộn theo mốc bước, lịch LR/entropy, KL,
clip-fraction) đều nằm trong dải của 6 seed đã xong. `results_v4/hard/` nay có
70 lượt, `results_v4/` tổng 140 lượt, toàn bộ gieo seed.

Hậu xử lý: `evaluation.ablation.save_ablation_report('results_v4','hard')`
(bản n = 5 cũ lưu `architecture_ablation_n5_seed0-4.json`),
`evaluation.plots --out-dir results_v4`, `scripts.export_web_data
--results-dir results_v4` (web/public/data/hard có 10 giá trị/phương pháp),
`scripts.make_chapter5_figures` (fig5_4 nay vẽ Hard n = 10; tiêu đề đã sửa).

### Bảng Hard n = 10 (mean ± sd liên-seed)

| Phương pháp | Coverage | VDR | TTFD | Collision |
|---|---|---|---|---|
| Random Walk | 0,040 ± 0,002 | 0,080 ± 0,030 | 342,8 | 0,139 |
| Greedy | 0,286 ± 0,018 | 0,380 ± 0,041 | 52,5 | 0,330 |
| MADDPG | 0,050 ± 0,047 | 0,052 ± 0,044 | 367,1 | 0,622 |
| MAPPO-MLP | 0,278 ± 0,022 | 0,362 ± 0,051 | 36,8 | 0,072 |
| **MAPPO-GCN** | **0,320 ± 0,018** | **0,440 ± 0,038** | 31,9 | 0,052 |
| MAPPO-GAT | 0,299 ± 0,014 | 0,409 ± 0,034 | 36,4 | 0,034 |
| GNN-MAPPO (DGAT) | 0,248 ± 0,017 | 0,337 ± 0,038 | 45,7 | 0,040 |

### Kiểm định n = 10 (Welch; α' = 0,0167 cho 3 so sánh vs MLP)

| So sánh | VDR p / d | Coverage p / d | Thắng theo cặp seed (VDR) |
|---|---|---|---|
| GCN vs MLP | **0,0012 / +1,75** | **0,0002 / +2,06** | 10/10 (paired p = 0,0001) |
| GAT vs MLP | 0,027 / +1,09 | 0,026 / +1,11 | 9/10 (paired p = 0,0094) |
| DGAT vs MLP | 0,23 / −0,56 | **0,0031 / −1,54** | 5/10 |
| DGAT vs GAT (GRU) | **0,0003 / −2,01** | **< 0,0001 / −3,31** | 2/10 (paired p = 0,0040) |
| GAT vs GCN | 0,068 / −0,87 | **0,0084 / −1,33** | 1/10 (paired p = 0,022) |
| GCN vs Greedy | **0,0031 / +1,53** | **0,0005 / +1,89** | 9/10 |
| GAT vs Greedy | 0,10 / +0,77 | 0,091 / +0,80 | 6/10 |
| MLP vs Greedy | 0,39 / −0,40 | 0,41 / −0,38 | 4/10 |
| DGAT vs Greedy | 0,024 / −1,11 | **0,0001 / −2,18** | 2/10 |

RQ2 GRU theo đợt: đợt 1 −0,071 (p = 0,0151, d = −2,10); đợt 2 −0,073 (p =
0,0205, d = −1,84); gộp −0,072 (p = 0,0003, d = −2,01). **Tác hại lặp lại độc
lập với cùng dấu và cỡ.**

### Biến thiên giữa đợt và độ khó bản đồ

Đợt 1 → đợt 2 (VDR): Greedy 0,403 → 0,357; MLP 0,382 → 0,341; GCN 0,438 →
0,443; GAT 0,417 → 0,401; DGAT 0,345 → 0,328. Không chênh lệch nào có ý nghĩa
(p ≥ 0,07). Nguồn chính là **độ khó khối bản đồ test theo seed**: seed 8 và 9
(khối 98k, 99k) khó hơn cho *mọi* phương pháp — VDR trung bình 5 phương pháp
0,333 / 0,342 so với 0,385–0,409 ở tám seed còn lại; GCN cũng tụt còn
0,381/0,386 ở hai seed đó nhưng đợt 2 của nó được bù bởi seed 5–7 (0,476–0,488).

**Rút lại nhận định trung gian ngày 09/09** ("hai kiến trúc đồ thị gần như
không dao động giữa đợt → bền hơn với phân bố bản đồ"): dữ liệu đủ 10 seed cho
thấy độ ổn định giữa đợt của GCN/GAT là trùng hợp giữa seed khó (8, 9) và seed
dễ (5–7) trong cùng đợt, không phải tính chất của kiến trúc. Nhận định đúng là:
kiểm định theo cặp seed (cùng bản đồ) loại bỏ nguồn biến thiên này và cho cùng
thứ hạng với Welch. Đây là kết luận trung gian thứ tư bị chính dữ liệu bác bỏ
(sau ba kết luận ở Giai đoạn 16–18).

### Kết luận Hard cuối cùng

Mọi kết luận Hard từ đợt 1 **đứng vững** khi lặp lại: GCN > MLP (p 0,016 →
0,0012, qua Bonferroni), GRU gây hại (p 0,015 → 0,0003), thứ hạng GCN > GAT >
DGAT. Một kết luận **mạnh lên**: GCN vượt Greedy từ p = 0,078 thành p = 0,0031
— GCN là phương pháp học duy nhất vượt heuristic ở Hard. Một kết luận **mới**:
DGAT phủ ít diện tích hơn cả MLP có ý nghĩa (Coverage p = 0,0031, d = −1,54).
Không kết luận nào bị đảo.

### Cập nhật tài liệu

`CHUONG5_THUC_NGHIEM.md` (18 chỗ, bản trước sao lưu ở scratchpad): banner và
5.1.1 (140 lượt, 109 giờ), 5.1.5 (n = 10 ở Hard, kiểm định theo cặp), Bảng 5.4,
Bảng 5.10 → n = 10, Hình 5.4, 5.3.2 (GCN vượt Greedy), Bảng 5.12 thêm 4 dòng
Hard, "Trả lời RQ1", Bảng 5.13 dòng Hard, Bảng 5.14 thêm đợt 2 và gộp, "Trả lời
RQ2", Bảng 5.15 dòng Hard (+ TTFD/Collision Hard), 5.6 thêm Bảng 5.18b và ba
nhận xét, Bảng 5.19 RQ1/RQ2, nhận định tổng thể 5.7, hạn chế 5.8 mục 1–2.
`scripts/make_chapter5_figures.py`: docstring và tiêu đề fig5_4.
Bản `.docx` trong `thesis proposals/` (07/09) đã lạc hậu, cần dựng lại.


### Bổ sung Giai đoạn 20 (10/09/2026): đề cương hiệu chỉnh và báo cáo CBHD

- `thesis proposals/Nguyễn Minh Khánh - ĐCLV ThS - Hiệu chỉnh 10-09-2026.docx`:
  bản đề cương 02/09 với nội dung cũ gạch ngang (đỏ) và nội dung đề xuất (xanh)
  chèn tại chỗ; Times New Roman 13; thêm Hình 1b (kiến trúc hiệu chỉnh). Tái định
  vị đóng góp từ “DGAT là lõi” sang “họ encoder đồ thị chọn bằng ablation có kiểm
  soát”; RQ2 viết lại có giả thuyết đối; Bảng 2/3/4/5/6/7 và cấu trúc luận văn
  cập nhật theo giao thức đã chứng minh cần thiết. Sinh bằng
  `thesis proposals/tools/revise_proposal.py` (python-docx, thao tác trực tiếp
  trên file gốc để giữ định dạng/số mục).
- `thesis proposals/Bao cao CBHD - Thuc nghiem so voi de cuong va de xuat dieu
  chinh (10-09-2026).docx` (13 trang): đối chiếu cam kết – thực hiện, danh mục 11
  lỗi, kết quả RQ1–RQ3, ba phương án điều chỉnh (A khuyến nghị), thí nghiệm bổ
  sung ưu tiên, định hướng công bố, rủi ro và 5 điểm xin ý kiến. Sinh bằng
  `thesis proposals/tools/build_report.py`.
- Hình kiến trúc hiệu chỉnh: `thesis proposals/figures_proposal_v2/
  fig1b_kien_truc_hieu_chinh.png` (`tools/draw_fig1b.py`).
- `thesis proposals/Chuong5_Thuc_Nghiem_Va_Danh_Gia.docx` dựng lại từ `CHUONG5_THUC_NGHIEM.md`
  (10/09/2026, 28 trang, Hard n = 10) bằng `thesis proposals/tools/md_to_docx.py`;
  bản cũ 07/09 lưu `Chuong5_v2_stage15_OBSOLETE.docx`.

- 10/09 (tối): rà lại trích dẫn cho phần Giới thiệu đã sửa trong đề cương hiệu chỉnh — mọi nhận định nay có
  tài liệu đỡ: tái dùng [3], [4], [5], [7], [8], [9], [12], [14], [26], [27] của đề cương; bổ sung [28]–[34]
  (cùng số hiệu với Chương 5; 7 bài đều đã xác minh qua trang AAAI/ICLR/NeurIPS/arXiv). Sửa tên xuất bản của [29]
  ở Chương 5. Xuất ra `... - Hiệu chỉnh 10-09-2026 v2.docx` vì file cũ đang mở trong Word.
- 13/09: vẽ lại Hình 1 của đề cương theo đúng phong cách hình gốc (8 bước, hai pha, nét liền/đứt) nhưng nội dung
  hiệu chỉnh — `figures_proposal_v2/fig1_kien_truc_ctde_hieu_chinh.png` (`tools/draw_fig1_revised.py`); thay thẳng
  vào vị trí Hình 1 trong `... v2.docx`, chú thích cũ gạch đỏ, chú thích mới xanh; bỏ Hình 1b khỏi đề cương (vẫn dùng
  trong báo cáo CBHD).
- 20/09: web cập nhật theo đề cương hiệu chỉnh — nhãn theo họ encoder (MAPPO-MLP / MAPPO-GCN (đề xuất) /
  MAPPO-GAT (GATv2, không GRU) / MAPPO-DGAT (GATv2 + GRU)), nhóm ba loại phương pháp, kiểm định lấy MAPPO-GCN
  làm mốc (`proposed_vs_baselines`) và thêm cặp GRU (`gru_ablation`) trong `scripts/export_web_data.py`
  (`WEB_LABELS`, tách khỏi nhãn hình luận văn ở `evaluation/plots.py`); `web/lib/types.ts` có METHOD_SHORT,
  PROPOSED_METHOD, METHOD_GROUP*; route `/comparision` (lỗi chính tả) tái xuất `/comparison`.

<!-- AUTO:rep-hard -->
### Kết quả tự động — hard (cập nhật 21/09/2026 18:52)

Trạng thái dữ liệu: đợt A = seed 0–4, đợt B = seed 5–9; đợt B đã có Random Walk 5/5, Greedy 5/5, MADDPG 5/5, MAPPO-MLP 5/5, MAPPO-GCN 5/5, MAPPO-GAT 5/5, MAPPO-DGAT 5/5 — **đầy đủ**.

**VDR theo đợt** (mean ± sd liên-seed; p = Welch đợt A vs B):

| Phương pháp | Đợt A | Đợt B | Δ (B−A) | p | Gộp |
|---|---|---|---|---|---|
| Random Walk | 0,081 ± 0,024 | 0,079 ± 0,037 (n=5) | -0,002 | 0,94 | 0,080 ± 0,030 (n=10) |
| Greedy | 0,403 ± 0,032 | 0,357 ± 0,037 (n=5) | -0,046 | 0,07 | 0,380 ± 0,041 (n=10) |
| MADDPG | 0,035 ± 0,035 | 0,069 ± 0,049 (n=5) | +0,034 | 0,24 | 0,052 ± 0,044 (n=10) |
| MAPPO-MLP | 0,382 ± 0,033 | 0,341 ± 0,061 (n=5) | -0,041 | 0,23 | 0,362 ± 0,051 (n=10) |
| MAPPO-GCN | 0,438 ± 0,016 | 0,443 ± 0,054 (n=5) | +0,005 | 0,86 | 0,440 ± 0,038 (n=10) |
| MAPPO-GAT | 0,417 ± 0,023 | 0,401 ± 0,044 (n=5) | -0,016 | 0,50 | 0,409 ± 0,034 (n=10) |
| MAPPO-DGAT | 0,345 ± 0,042 | 0,328 ± 0,035 (n=5) | -0,017 | 0,50 | 0,337 ± 0,038 (n=10) |

**Coverage theo đợt** (mean ± sd liên-seed; p = Welch đợt A vs B):

| Phương pháp | Đợt A | Đợt B | Δ (B−A) | p | Gộp |
|---|---|---|---|---|---|
| Random Walk | 0,041 ± 0,002 | 0,040 ± 0,002 (n=5) | -0,001 | 0,25 | 0,040 ± 0,002 (n=10) |
| Greedy | 0,287 ± 0,011 | 0,285 ± 0,025 (n=5) | -0,002 | 0,89 | 0,286 ± 0,018 (n=10) |
| MADDPG | 0,034 ± 0,045 | 0,066 ± 0,047 (n=5) | +0,033 | 0,29 | 0,050 ± 0,047 (n=10) |
| MAPPO-MLP | 0,286 ± 0,019 | 0,271 ± 0,025 (n=5) | -0,014 | 0,34 | 0,278 ± 0,022 (n=10) |
| MAPPO-GCN | 0,312 ± 0,013 | 0,327 ± 0,020 (n=5) | +0,015 | 0,21 | 0,320 ± 0,018 (n=10) |
| MAPPO-GAT | 0,302 ± 0,009 | 0,296 ± 0,018 (n=5) | -0,006 | 0,56 | 0,299 ± 0,014 (n=10) |
| MAPPO-DGAT | 0,243 ± 0,017 | 0,252 ± 0,018 (n=5) | +0,009 | 0,45 | 0,248 ± 0,017 (n=10) |

**Kiểm định gộp** (Welch t / Mann–Whitney / Cohen's d; theo cặp seed = cùng bộ bản đồ test; α' Bonferroni = 0,0167 cho 3 so sánh với MLP):

| Chỉ số | So sánh | n | Δ | p Welch | p MW | d | Theo cặp: thắng / p | Bonferroni |
|---|---|---|---|---|---|---|---|---|
| VDR | MAPPO-GCN vs MAPPO-MLP | 10/10 | +0,078 | 0,0012 | 0,0028 | +1,75 | 10/10 / 0,0001 | đạt |
| VDR | MAPPO-GAT vs MAPPO-MLP | 10/10 | +0,047 | 0,0273 | 0,0376 | +1,09 | 9/10 / 0,0094 | * |
| VDR | MAPPO-DGAT vs MAPPO-MLP | 10/10 | -0,025 | 0,2278 | 0,1859 | -0,56 | 5/10 / 0,2209 | không |
| VDR | MAPPO-DGAT vs MAPPO-GAT | 10/10 | -0,072 | 0,0003 | 0,0017 | -2,01 | 2/10 / 0,0040 | đạt |
| VDR | MAPPO-GAT vs MAPPO-GCN | 10/10 | -0,031 | 0,0676 | 0,0640 | -0,87 | 1/10 / 0,0217 | không |
| VDR | MAPPO-DGAT vs MAPPO-GCN | 10/10 | -0,103 | 0,0000 | 0,0006 | -2,74 | 0/10 / 0,0001 | đạt |
| VDR | MAPPO-GCN vs Greedy | 10/10 | +0,060 | 0,0031 | 0,0058 | +1,53 | 9/10 / 0,0037 | đạt |
| VDR | MAPPO-GAT vs Greedy | 10/10 | +0,029 | 0,1047 | 0,0890 | +0,77 | 6/10 / 0,1229 | không |
| VDR | MAPPO-MLP vs Greedy | 10/10 | -0,018 | 0,3855 | 0,5708 | -0,40 | 4/10 / 0,2421 | không |
| VDR | MAPPO-DGAT vs Greedy | 10/10 | -0,043 | 0,0237 | 0,0312 | -1,11 | 2/10 / 0,0207 | * |
| Coverage | MAPPO-GCN vs MAPPO-MLP | 10/10 | +0,042 | 0,0002 | 0,0006 | +2,06 | 9/10 / 0,0011 | đạt |
| Coverage | MAPPO-GAT vs MAPPO-MLP | 10/10 | +0,020 | 0,0257 | 0,0452 | +1,11 | 9/10 / 0,0107 | * |
| Coverage | MAPPO-DGAT vs MAPPO-MLP | 10/10 | -0,031 | 0,0031 | 0,0046 | -1,54 | 2/10 / 0,0141 | đạt |
| Coverage | MAPPO-DGAT vs MAPPO-GAT | 10/10 | -0,051 | 0,0000 | 0,0002 | -3,31 | 0/10 / 0,0000 | đạt |
| Coverage | MAPPO-GAT vs MAPPO-GCN | 10/10 | -0,021 | 0,0084 | 0,0091 | -1,33 | 1/10 / 0,0039 | đạt |
| Coverage | MAPPO-DGAT vs MAPPO-GCN | 10/10 | -0,072 | 0,0000 | 0,0002 | -4,15 | 0/10 / 0,0000 | đạt |
| Coverage | MAPPO-GCN vs Greedy | 10/10 | +0,034 | 0,0005 | 0,0013 | +1,89 | 9/10 / 0,0009 | đạt |
| Coverage | MAPPO-GAT vs Greedy | 10/10 | +0,013 | 0,0910 | 0,1041 | +0,80 | 7/10 / 0,0280 | không |
| Coverage | MAPPO-MLP vs Greedy | 10/10 | -0,008 | 0,4120 | 0,4727 | -0,38 | 4/10 / 0,1528 | không |
| Coverage | MAPPO-DGAT vs Greedy | 10/10 | -0,038 | 0,0001 | 0,0008 | -2,18 | 0/10 / 0,0002 | đạt |

Độ khó khối bản đồ test theo seed (VDR trung bình 5 phương pháp không ngẫu nhiên): seed 0: 0,392, seed 1: 0,392, seed 2: 0,407, seed 3: 0,409, seed 4: 0,385, seed 5: 0,404, seed 6: 0,399, seed 7: 0,393, seed 8: 0,333, seed 9: 0,342

![Hai đợt seed và gộp — hard](assets/giai-doan-20/hard/fig_batches_hard.png)

![Giá trị từng seed — hard](assets/giai-doan-20/hard/fig_perseed_hard.png)

![Learning curve — hard](assets/giai-doan-20/hard/fig_curves_hard.png)

Dữ liệu thô: `assets/giai-doan-20/hard/per_seed_hard.csv`, `summary_hard_*.csv`, `tests_hard.csv`.
<!-- /AUTO:rep-hard -->

---

[← Giai đoạn 19](giai-doan-19-lap-lai-hard-seed-5-9-va-ban-nhap-chuong-5-08-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 21 →](giai-doan-21-lap-lai-medium-easy-seed-5-9-va-zero-shot-density-20-09-2026.md)
