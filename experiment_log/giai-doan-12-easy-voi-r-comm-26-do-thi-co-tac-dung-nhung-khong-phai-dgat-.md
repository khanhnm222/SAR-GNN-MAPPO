[← Giai đoạn 11](giai-doan-11-medium-voi-r-comm-34-go-bo-nhieu-cau-tra-loi-khong-doi-05-09.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 13 →](giai-doan-13-bptt-that-cho-gru-va-loi-thu-bay-seed-05-09-2026.md)

# Giai đoạn 12 — Easy với `r_comm=26`: đồ thị CÓ tác dụng, nhưng không phải DGAT (05/09/2026)

27 lượt, Easy 500k bước, 4,5 giờ. Bộ `results_v3_rcomm/` nay nhất quán: Easy ở
`r_comm=26`, Medium ở `r_comm=34`, mọi thứ khác giống hệt nhau.

### Kiểm chứng đồ thị không suy biến (chính sách đã huấn luyện, 4 episode)

| Kịch bản | Mật độ | Bậc TB | % bước rỗng | % bước đầy đủ |
|---|---|---|---|---|
| Easy (r=26) | 0,45–0,56 | 1,3–1,5 | 0–2% | 8–17% |
| Medium (r=34) | 0,28–0,35 | 1,7–1,8 | 8–10% | 1–6% |

Cả hai đều là đồ thị **động và một phần** — không rỗng (lỗi Giai đoạn 10) và
cũng không gần đầy đủ (nếu đầy đủ thì "đồ thị có ích" sẽ chỉ là "thông tin toàn
cục có ích"). Phép so sánh vì thế hợp lệ.

### Kết quả

| Phương pháp | Easy VDR | Easy Cov | Easy TTFD | Medium VDR | Medium Cov |
|---|---|---|---|---|---|
| Random Walk | 0,433 | 0,400 | 134,8 | 0,231 | 0,146 |
| Greedy | 0,673 | 0,647 | 22,5 | 0,564 | 0,527 |
| MADDPG | 0,152 ± 0,023 | 0,126 | 278,9 | 0,159 ± 0,072 | 0,178 |
| MAPPO-MLP | 0,877 ± 0,020 | 0,739 | 11,5 | 0,667 ± 0,038 | 0,654 |
| **MAPPO-GCN** | **0,942 ± 0,021** | **0,761** | 10,3 | **0,689 ± 0,035** | **0,673** |
| MAPPO-GAT | 0,910 ± 0,005 | 0,754 | **9,3** | 0,649 ± 0,047 | 0,660 |
| GNN-MAPPO | 0,890 ± 0,011 | 0,750 | 11,8 | 0,675 ± 0,023 | 0,636 |

### RQ1 — cấu trúc đồ thị có giúp ích không? CÓ, nhưng chỉ ở Easy

So với MAPPO-MLP (n=5 mỗi bên, Welch):

| Kiến trúc | Easy | Medium |
|---|---|---|
| MAPPO-GCN | +0,065 · **p = 0,0009** · d = +3,22 (đạt cả sau Bonferroni) | +0,022 · p = 0,36 · d = +0,61 |
| MAPPO-GAT | +0,033 · **p = 0,0171** · d = +2,32 (không đạt sau Bonferroni) | −0,018 · p = 0,53 · d = −0,41 |
| GNN-MAPPO | +0,013 · p = 0,2394 · d = +0,82 | +0,008 · p = 0,71 · d = +0,24 |

**Đây là bằng chứng dương tính đầu tiên cho luận điểm trung tâm của luận văn.**
Ở Easy, MAPPO-GCN vượt MAPPO-MLP một cách chắc chắn về mặt thống kê. Nó chỉ
xuất hiện sau khi sửa `r_comm` — ở `r_comm=10` cùng phép so sánh cho p = 0,80.

Nhưng có hai điều kiện đi kèm phải báo cáo:

**(1) Kiến trúc đề xuất là kiến trúc đồ thị YẾU NHẤT.** GNN-MAPPO (DGAT) thấp
hơn MAPPO-GCN có ý nghĩa thống kê (0,890 vs 0,942, p = 0,0024, d = −3,12) và
thấp hơn MAPPO-GAT (p = 0,0116, d = −2,32). Kiến trúc đơn giản nhất — GCN với
trọng số tổng hợp cố định — thắng. Cơ chế attention có đặc trưng cạnh cộng thêm
GRU của DGAT tốn kém hơn là có lợi ở ngân sách này.

**(2) Lợi ích của đồ thị GIẢM theo N, ngược với giả thuyết.** Ở Easy (N=4)
d = +3,22 cho GCN; ở Medium (N=8) d = +0,61, không có ý nghĩa. Đề cương giả
định lợi thế của đồ thị tăng theo quy mô đàn.

### RQ2 — đóng góp của bộ nhớ thời gian (GRU): ÂM TÍNH

GNN-MAPPO = DGAT có GRU; MAPPO-GAT = cùng kiến trúc, bỏ GRU.

| Kịch bản | Có GRU | Không GRU | p | Cohen's d |
|---|---|---|---|---|
| Easy | 0,890 | **0,910** | **0,0116** | −2,32 |
| Medium | 0,675 | 0,649 | 0,32 | +0,68 |

Ở Easy, **GRU làm giảm hiệu năng có ý nghĩa thống kê**. Khớp với hạn chế đã ghi
trong `algorithms/mappo.py`: encoder hồi tiếp được cập nhật từ `h_prev` đã lưu
của từng bước, nên lan truyền ngược theo thời gian chỉ dài 1 bước — GRU có thêm
trạng thái nhưng không được huấn luyện đúng cách. Muốn kết luận công bằng về
RQ2 thì phải cài BPTT thật (lưu chuỗi theo episode, cập nhật theo đoạn).

### Hiệu ứng của `r_comm` ở Easy (10 → 26)

| Phương pháp | r=10 | r=26 | Chênh |
|---|---|---|---|
| MAPPO-MLP | 0,855 | 0,877 | +0,021 |
| MAPPO-GCN | 0,873 | 0,942 | **+0,069** |
| MAPPO-GAT | 0,870 | 0,910 | +0,040 |
| GNN-MAPPO | 0,847 | 0,890 | +0,043 |
| Greedy | 0,716 | 0,673 | −0,043 |

Các kiến trúc đồ thị hưởng lợi gấp 2–3 lần MLP từ việc mở rộng tầm liên lạc —
đúng như kỳ vọng nếu đồ thị thực sự truyền tải thông tin có ích.

---

[← Giai đoạn 11](giai-doan-11-medium-voi-r-comm-34-go-bo-nhieu-cau-tra-loi-khong-doi-05-09.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 13 →](giai-doan-13-bptt-that-cho-gru-va-loi-thu-bay-seed-05-09-2026.md)
