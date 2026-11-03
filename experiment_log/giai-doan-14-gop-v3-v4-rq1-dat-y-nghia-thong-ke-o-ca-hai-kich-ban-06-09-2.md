[← Giai đoạn 13](giai-doan-13-bptt-that-cho-gru-va-loi-thu-bay-seed-05-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 15 →](giai-doan-15-ma-tran-day-du-ba-kich-ban-results-v4-07-09-2026.md)

# Giai đoạn 14 — Gộp v3 + v4: RQ1 đạt ý nghĩa thống kê ở CẢ HAI kịch bản (06/09/2026)

`results_v4` Easy và Medium đã xong (54/81; Hard đang chạy).

### Kết quả Medium (`results_v4`, có BPTT)

| Phương pháp | VDR | Coverage | TTFD |
|---|---|---|---|
| Random Walk | 0,231 | 0,146 | 267,2 |
| Greedy | 0,564 | 0,527 | 49,5 |
| MADDPG | 0,123 ± 0,091 | 0,147 | 310,4 |
| MAPPO-MLP | 0,638 ± 0,053 | 0,656 | 33,8 |
| **MAPPO-GCN** | **0,728 ± 0,031** | **0,681** | **23,3** |
| MAPPO-GAT | 0,676 ± 0,052 | 0,638 | 30,6 |
| GNN-MAPPO | 0,644 ± 0,053 | 0,636 | 31,6 |

### Gộp v3 + v4 để tăng lực thống kê (n=10)

BPTT **chỉ** ảnh hưởng tới encoder hồi tiếp, nên với MAPPO-MLP, MAPPO-GCN và
MAPPO-GAT thì `results_v3_rcomm` và `results_v4` là **cùng một code chạy với
khởi tạo khác nhau** — tức 10 lần chạy độc lập mỗi phương pháp. Gộp lại là hợp lệ
và tăng lực thống kê đáng kể.

| Kịch bản | Phương pháp | n | VDR | p vs MLP | Cohen's d | Bonferroni |
|---|---|---|---|---|---|---|
| Easy | MAPPO-MLP | 10 | 0,868 ± 0,043 | — | — | — |
| Easy | **MAPPO-GCN** | 10 | **0,933 ± 0,020** | **0,00044** | +1,95 | **đạt** |
| Easy | **MAPPO-GAT** | 10 | **0,906 ± 0,019** | **0,00726** | +1,14 | **đạt** |
| Medium | MAPPO-MLP | 10 | 0,653 ± 0,046 | — | — | — |
| Medium | **MAPPO-GCN** | 10 | **0,709 ± 0,037** | **0,00778** | +1,35 | **đạt** |
| Medium | MAPPO-GAT | 10 | 0,663 ± 0,049 | 0,639 | +0,21 | không |

### Trả lời RQ1: CÓ — cấu trúc đồ thị giúp ích, có ý nghĩa thống kê ở cả hai kịch bản

MAPPO-GCN vượt MAPPO-MLP ở **cả Easy lẫn Medium**, đạt ý nghĩa cả sau hiệu
chỉnh Bonferroni, cỡ hiệu ứng lớn (d = +1,95 và +1,35). Đây là kết luận dương
tính vững chắc nhất từ trước tới nay cho luận điểm trung tâm của luận văn, và
nó chỉ xuất hiện sau khi sửa đủ bảy lỗi — đặc biệt là lỗi thứ sáu (`r_comm`
khiến đồ thị rỗng).

**Nhưng thứ hạng giữa các kiến trúc đồ thị là GCN > GAT > DGAT**, ngược với giả
thuyết. Tổng hợp có trọng số cố định (GCN) thắng attention có đặc trưng cạnh
(GAT), và attention thắng attention-cộng-GRU (DGAT).

### Trả lời RQ2: bộ nhớ thời gian không đóng góp gì đo được

| Kịch bản | Có GRU | Không GRU | p | d |
|---|---|---|---|---|
| Easy | 0,889 | 0,902 | 0,62 | −0,33 |
| Medium | 0,644 | 0,676 | 0,36 | −0,61 |

Sau khi sửa BPTT, GRU không còn gây hại có ý nghĩa thống kê, nhưng cũng không
mang lại lợi ích. Cả hai kịch bản đều cho cỡ hiệu ứng âm nhẹ.

### Trạng thái seed của `results_v4`

| Kịch bản | Đã gieo seed |
|---|---|
| Easy | 0/27 |
| Medium | 13/27 |
| Hard | toàn bộ (chạy sau bản sửa) |

Đúng như dự đoán: bản sửa `seed_everything` rơi vào giữa pha Medium. Kết luận
thống kê không bị ảnh hưởng (mỗi lượt vẫn là một mẫu độc lập hợp lệ), nhưng
**cần chạy lại Easy và Medium với seed đã gieo** để có một bộ tái lập được và
để độ lệch chuẩn phản ánh đúng biến thiên giữa seed thay vì lẫn nhiễu khởi tạo.

---

[← Giai đoạn 13](giai-doan-13-bptt-that-cho-gru-va-loi-thu-bay-seed-05-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 15 →](giai-doan-15-ma-tran-day-du-ba-kich-ban-results-v4-07-09-2026.md)
