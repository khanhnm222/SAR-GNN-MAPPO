[← Giai đoạn 10](giai-doan-10-loi-thu-sau-do-thi-giao-tiep-gan-nhu-rong-04-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 12 →](giai-doan-12-easy-voi-r-comm-26-do-thi-co-tac-dung-nhung-khong-phai-dgat-.md)

# Giai đoạn 11 — Medium với `r_comm=34`: gỡ bỏ nhiễu, câu trả lời không đổi (05/09/2026)

27 lượt, Medium 800k bước, 7 phương pháp × 5 seed, 12,3 giờ wall. Khác biệt duy
nhất so với `results_v2/medium` là `r_comm` 10 → 34.

### Kết quả

| Phương pháp | n | VDR | Coverage | TTFD | Va chạm |
|---|---|---|---|---|---|
| Random Walk | 1 | 0,231 | 0,146 | 267,2 | 0,026 |
| Greedy | 1 | 0,564 | 0,527 | 49,5 | 0,044 |
| MADDPG | 5 | 0,159 ± 0,072 | 0,178 | 284,2 | 0,496 |
| MAPPO-MLP | 5 | 0,667 ± 0,038 | 0,654 | 25,1 | 0,026 |
| **MAPPO-GCN** | 5 | **0,689 ± 0,035** | **0,673** | **21,7** | 0,026 |
| MAPPO-GAT | 5 | 0,649 ± 0,047 | 0,660 | 33,9 | 0,024 |
| GNN-MAPPO | 5 | 0,675 ± 0,023 | 0,636 | 31,1 | 0,029 |

### Hiệu ứng của `r_comm` (mọi thứ khác giữ nguyên)

| Phương pháp | r_comm=10 | r_comm=34 | Chênh |
|---|---|---|---|
| MAPPO-MLP | 0,642 | 0,667 | **+0,025** |
| MAPPO-GCN | 0,638 | 0,689 | **+0,051** |
| MAPPO-GAT | 0,646 | 0,649 | +0,003 |
| GNN-MAPPO | 0,632 | 0,675 | **+0,043** |
| Greedy | 0,587 | 0,564 | −0,023 |
| MADDPG | 0,200 | 0,159 | −0,042 |

Điểm đáng chú ý: **MAPPO-MLP cũng cải thiện (+0,025) dù nó không dùng đồ thị.**
Lý do là `r_comm` điều khiển việc hợp nhất bản đồ niềm tin **trong động lực học
môi trường** (`_merge_beliefs_over_comm_graph`), độc lập với kiến trúc chính
sách. Lợi ích của việc mở rộng tầm liên lạc đến qua môi trường, không qua GNN.

### Câu trả lời cho RQ1: vẫn âm tính sau khi đã gỡ nhiễu

Gộp ba kiến trúc có đồ thị so với MLP:

| Cấu hình | Có đồ thị (n=15) | MLP (n=5) | p | Cohen's d |
|---|---|---|---|---|
| r_comm=10 (đồ thị rỗng 66% số bước) | 0,639 | 0,642 | 0,80 | −0,08 |
| **r_comm=34 (đồ thị rỗng 32%)** | **0,671** | **0,667** | **0,84** | **+0,11** |

GNN-MAPPO so với từng kiến trúc ở `r_comm=34`: vs MLP p=0,71 (d=+0,24),
vs GCN p=0,46 (d=−0,50), vs GAT p=0,32 (d=+0,68).

**Gỡ bỏ nhiễu "đồ thị rỗng" không làm thay đổi kết luận.** Cỡ hiệu ứng đi từ
−0,08 lên +0,11 — cả hai đều không phân biệt được với 0. Đây giờ là một kết quả
âm tính có kiểm soát tốt: giao thức đánh giá đã sửa, sáu lỗi môi trường/huấn
luyện đã sửa, đồ thị đã được kiểm chứng là không suy biến, 5 seed, hai kịch bản,
cộng thêm zero-shot.

### Quyết định về kịch bản Hard

Điều kiện đã thống nhất trước khi chạy Giai đoạn 11 là: *"chỉ chạy Hard nếu
bước 2 cho thấy đồ thị bắt đầu tạo ra khác biệt"*. **Điều kiện đó KHÔNG đạt**
(p = 0,84, d = +0,11). Vì vậy **không chạy Hard** cho mục đích trả lời RQ1.

Chi phí Hard nếu chạy vì lý do khác (đủ ba kịch bản theo đề cương), đo tại
`r_comm=46`, máy rảnh: mlp 848 s/100k · gcn 1.331 · dgat 2.204.

| Cấu hình Hard | CPU-giờ | Wall @6 job |
|---|---|---|
| 1M bước × 5 seed × 5 phương pháp | 91 | ~22–24 giờ |
| 1M bước × 3 seed × 2 phương pháp (MLP vs GNN) | 25 | ~9 giờ |

---

[← Giai đoạn 10](giai-doan-10-loi-thu-sau-do-thi-giao-tiep-gan-nhu-rong-04-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 12 →](giai-doan-12-easy-voi-r-comm-26-do-thi-co-tac-dung-nhung-khong-phai-dgat-.md)
