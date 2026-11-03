[← Giai đoạn 9](giai-doan-09-chay-lai-toan-bo-tren-moi-truong-da-sua-results-v2-04-09-202.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 11 →](giai-doan-11-medium-voi-r-comm-34-go-bo-nhieu-cau-tra-loi-khong-doi-05-09.md)

# Giai đoạn 10 — Lỗi thứ sáu: đồ thị giao tiếp gần như RỖNG (04/09/2026)

Khi cân nhắc có nên chạy kịch bản Hard hay không, đo lại cấu trúc đồ thị dưới
một chính sách **thực sự phân tán** — thay vì dưới chính sách ngẫu nhiên khi
UAV còn túm tụm ở điểm sinh.

### Đo trên chính sách GNN-MAPPO đã huấn luyện (`results_v2`, 3 episode/kịch bản)

| Kịch bản | Bậc trung bình | Cụm lớn nhất | UAV bị cô lập | **Bước KHÔNG có cạnh nào** |
|---|---|---|---|---|
| Easy | 0,24 | 43% | 76% | **56%** |
| Medium | 0,43 | 44% | 63% | **35%** |

Ở **56% số bước của Easy** và 35% của Medium, đồ thị giao tiếp **không có cạnh
nào**. `GATv2Conv` với `add_self_loops=True` khi đó suy biến thành một phép biến
đổi tuyến tính theo từng node. **DGAT đã là một MLP trong phần lớn thời gian.**

### Nguyên nhân: `r_comm = 10` quá nhỏ so với kích thước bản đồ

Giai đoạn 6 hạ `r_comm` từ 15/20/25 xuống 10/10/10, dựa trên phép đo cho thấy
đồ thị "gần đặc (54%/83%/96%)". Phép đo đó chạy **100 bước với chính sách ngẫu
nhiên** — lúc UAV vẫn còn nằm trong lưới xuất phát quanh tâm bản đồ. Tái lập
được y hệt: với chính sách ngẫu nhiên, mật độ đo được là 0,65/0,74/0,61. Với
chính sách phân tán (Greedy), cùng `r_comm=10`, mật độ tụt xuống 0,14/0,012/0,020.

Bậc trung bình kỳ vọng cho đàn phân tán đều là `(N−1)·π·r²/(W·H)`:

| Kịch bản | N | W×H | Bậc tại r_comm=10 | r_comm cho bậc ≈ 2,5 |
|---|---|---|---|---|
| Easy | 4 | 50×50 | 0,38 | **26** |
| Medium | 8 | 100×100 | 0,22 | **34** |
| Hard | 16 | 200×200 | 0,12 | **46** |

Công thức khớp với phép quét thực nghiệm (đo được 25 / 40 / 40–60). Giá trị gốc
của đề cương (15/20/25) cũng quá nhỏ ở Medium và Hard.

### Hệ quả cho kết luận Giai đoạn 9

Kết luận "đồ thị không mang lại lợi thế đo được" **vẫn đúng như một quan sát**,
nhưng lời giải thích phải sửa. Trước đây quy cho "môi trường đã tự cung cấp
thông tin và động cơ phối hợp". Nay có một nguyên nhân cơ học đơn giản hơn và
phải loại trừ trước: **DGAT gần như không có cạnh nào để truyền tin**.

Vì vậy Giai đoạn 9 **chưa** trả lời được RQ1. Cần chạy lại với `r_comm` đúng cỡ
trước khi kết luận bất cứ điều gì về giá trị của cấu trúc đồ thị.

### Khuyến nghị về kịch bản Hard

**Chưa chạy Hard.** Ở `r_comm=10`, Hard sẽ cho kết quả âm tính thứ ba vì cùng
một lý do cơ học, với chi phí 6,6 giờ (ngân sách 200k, chắc chắn không hội tụ)
đến 33 giờ (ngân sách 1M, đủ 5 seed × 5 phương pháp). Thứ tự đúng:

1. Sửa `r_comm` → 26/34/46, kiểm chứng lại độ liên thông dưới chính sách đã
   huấn luyện.
2. Chạy lại **Medium** trước (rẻ nhất, và hiện đang là kịch bản có đồ thị suy
   biến nặng nhất).
3. Chỉ chạy Hard nếu bước 2 cho thấy đồ thị bắt đầu tạo ra khác biệt.

### Cổng kiểm tra sớm cho `results_v3_rcomm` (04/09/2026)

Sau khi checkpoint MAPPO-MLP seed 0 hoàn tất ở môi trường mới, đo lại độ liên
thông đồ thị **dưới chính sách huấn luyện trong chính môi trường đó** (Medium,
4 episode), thay vì dưới Greedy:

| r_comm | Bậc TB | Cụm lớn nhất | UAV cô lập | Bước không có cạnh |
|---|---|---|---|---|
| 34 (mới) | **1,13** | 61% | 49% | **32%** |
| 10 (cũ) | 0,17 | 30% | 84% | 66% |

Cải thiện thật: số cạnh tăng 6,6 lần, tỷ lệ bước đồ thị rỗng giảm từ 66% xuống
32%. Nhưng **chưa đạt mục tiêu** bậc ≈ 2,5 và < 20% cô lập mà công thức dự
đoán. Hai lý do:

1. Công thức giả định đàn phân tán **đều** trên W×H. Thực tế UAV lan ra từ tâm
   và tụ lại dọc theo vùng biên chưa quét, không phủ đều.
2. UAV cạn năng lượng từ khoảng bước 260–330, nên nửa sau episode số node giảm
   dần và đồ thị thưa đi theo.

**Quyết định: tiếp tục ở r_comm=34, không nâng thêm.** Nâng lên ~55–60 để đạt
bậc 2,5 sẽ khiến `r_comm` vượt quá nửa chiều rộng bản đồ 100×100 — đổi một chế
độ suy biến (đồ thị rỗng) lấy một chế độ suy biến khác (đồ thị gần đầy đủ, mọi
UAV nối với mọi UAV, attention không còn gì để phân biệt). Ở mức 34, đồ thị vẫn
**động** thật sự (cụm lớn nhất chỉ chứa 61% số UAV) và có cạnh ở 68% số bước —
đủ để trả lời câu hỏi nếu cấu trúc đồ thị thực sự có ích.

Chi phí thực tế cao hơn dự kiến: throughput đo được tại r_comm=34 là 134 bước/s
cho MLP và 66 bước/s cho GCN (so với ~10× ít cạnh hơn ở r_comm=10), nên toàn bộ
Medium mất ~19 giờ thay vì 10,7 giờ. Đây là chi phí thật của việc có một đồ thị
mang nội dung.

---

[← Giai đoạn 9](giai-doan-09-chay-lai-toan-bo-tren-moi-truong-da-sua-results-v2-04-09-202.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 11 →](giai-doan-11-medium-voi-r-comm-34-go-bo-nhieu-cau-tra-loi-khong-doi-05-09.md)
