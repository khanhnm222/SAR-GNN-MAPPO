[← Giai đoạn 16](giai-doan-16-gieo-seed-toan-bo-va-mot-ket-luan-phai-dieu-chinh-07-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 18 →](giai-doan-18-hai-kiem-chung-cuoi-08-09-2026.md)

# Giai đoạn 17 — Rà soát lần cuối: những gì còn lại (07/09/2026)

Sau khi `results_v4/` đã gieo seed đủ 81/81, rà soát lại toàn bộ với câu hỏi
"còn gì có thể làm sai lệch kết luận không". Tìm thêm bốn điểm.

### ⑧ Số tham số chênh nhau 3 lần giữa các kiến trúc — nhiễu cho thứ hạng GCN > GAT > DGAT

| Kiến trúc | Encoder | Actor + Critic | So với MLP |
|---|---|---|---|
| MAPPO-MLP | 13.888 | 41.102 | 1,0× |
| MAPPO-GCN | 18.432 | 50.190 | 1,2× |
| MAPPO-GAT | 32.448 | 78.222 | 1,9× |
| GNN-MAPPO (DGAT) | 57.408 | **128.142** | **3,1×** |

Thứ hạng hiệu năng GCN > GAT > DGAT **trùng khớp ngược chiều với số tham số**.
Ở cùng ngân sách bước, kiến trúc nhiều tham số hơn có thể chỉ đơn giản là chưa
được huấn luyện đủ. Đây là lời phản biện tự nhiên nhất cho kết luận "DGAT là
kiến trúc đồ thị yếu nhất" và phải được loại trừ trước khi viết vào luận văn.

**Kiểm chứng đang chạy** (`results_v4_budget2x/`): GCN và DGAT ở Easy với **1M
bước** (gấp đôi), 5 seed. Nếu DGAT bắt kịp GCN → thứ hạng là do ngân sách, không
do kiến trúc. Nếu cả hai cùng tăng một lượng tương đương → ngân sách ảnh hưởng
đều, thứ hạng là thật. Nếu DGAT đứng yên → kiến trúc.

### ⑨ Heuristic được đánh giá trên tập bản đồ KHÁC với các phương pháp học

`train.py` chấm heuristic với `seed_start = 10_000 + seed*1000` nhưng chấm các
phương pháp học với `90_000 + seed*1000`. Mọi so sánh heuristic-vs-học từ trước
tới nay là **trên hai tập bản đồ khác nhau**. Đã sửa về cùng 90_000; heuristic
cũ lưu ở `results_v4_heur10k/`, đang chạy lại.

### ⑩ Heuristic chỉ có 1 seed

Runner đặt `n = 1` cho heuristic, nên std = 0 và không kiểm định thống kê được
với chúng. Random Walk ngẫu nhiên, Greedy có phá ràng buộc ngẫu nhiên — cần đủ
5 seed như mọi phương pháp khác. Đã sửa; chi phí chỉ vài phút.

### ⑪ Seed đánh giá trung gian trượt qua tập test

`seed_start = 50_000 + seed*1000 + global_step` tăng theo bước huấn luyện và
đi xuyên qua khối test `90_000..94_029` khi `global_step` vượt ~40k. Không có
gradient nào đến từ các episode này nên **kết quả không bị ảnh hưởng**, nhưng
tập test đã bị "nhìn thấy" trong quá trình phát triển — lỗi vệ sinh giao thức.
Đã sửa thành khối bị chặn `50_000 + seed*1000 + eval_idx*10`.

### Ba đặc tính thiết kế cần ghi vào phần hạn chế (không phải lỗi)

Đo dưới chính sách MAPPO-GCN đã huấn luyện, 3 episode:

| Kịch bản | Bước/episode | UAV còn sống khi kết thúc | Kết thúc do |
|---|---|---|---|
| Easy | **81** | 67% | tìm hết nạn nhân |
| Medium | 128 | 33% | hỗn hợp |
| Hard | 361 | **4%** | **cạn năng lượng** |

- **Easy bão hoà**: episode kết thúc sau ~81 bước vì đã tìm hết nạn nhân. Không
  gian phân biệt giữa các kiến trúc bị nén (mọi phương pháp học đều ≥ 0,88).
- **Hard bị năng lượng chi phối**: 96% UAV chết trước khi episode kết thúc.
  `E_max = 1000` với chi phí ~3/bước cho ~330 bước bay; `T_max = 500` chưa bao
  giờ đạt tới. Hiệu năng ở Hard đo "tìm được bao nhiêu trước khi hết pin", và
  hành động `RETURN_BASE` không có mục đích vì không có cơ chế sạc.
- **Đồ thị ở cả ba kịch bản đều lành mạnh** sau sửa `r_comm`: mật độ
  0,67 / 0,28 / 0,52, bậc trung bình 1,96 / 1,80 / 1,54, tỷ lệ bước rỗng
  0% / 12% / 14%.

### Zero-shot sau gieo seed — mẫu hình không đổi

VDR tuyệt đối trung bình tại N > N huấn luyện:

| | Easy | Medium | Hard |
|---|---|---|---|
| MAPPO-MLP | 0,976 | 0,834 | 0,458 |
| MAPPO-GCN | **0,983** | 0,820 | 0,535 |
| MAPPO-GAT | 0,965 | 0,826 | **0,577** |
| GNN-MAPPO | 0,960 | **0,869** | 0,433 |

GNN-MAPPO vẫn chỉ dẫn đầu ở Medium.

### Bảng tổng hợp các sai lệch so với đề cương (để đưa vào Chương 5)

| Tham số / cơ chế | Đề cương | Thực nghiệm | Lý do |
|---|---|---|---|
| Giao thức đánh giá | (không nói rõ) | Lấy mẫu từ π | `argmax` thoái hoá với không gian hành động này |
| `r_comm` | 15 / 20 / 25 | **26 / 34 / 46** | Đồ thị rỗng ở giá trị cũ; chọn theo bậc kỳ vọng ≈ 2,5 |
| Bản đồ niềm tin | (ngụ ý toàn cục) | Riêng từng UAV, hợp nhất theo thành phần liên thông | Để giao tiếp mang thông tin |
| Quan sát hàng xóm (30 chiều) | có trong Bảng 2 | **bỏ** | Rò rỉ thông tin vào baseline MLP |
| Kênh `prob`/`visited` cục bộ | 5×5 độ phân giải 1 | 5×5, mỗi ô gộp khối 5×5 | Cửa sổ cũ nằm trong đĩa cảm biến → hằng số |
| Hàm thưởng | toàn cục | theo từng UAV, pha 30% đội | Gán công |
| `alpha` (r_cover) | 1,0 | 20,0 | Cân lại với r_victim |
| `gamma` (va chạm) | −2,0 | −0,05 | Đã sửa từ Giai đoạn 4 |
| Critic | chung encoder với actor | encoder riêng, V theo từng UAV, chuẩn hoá return | Value loss lấn át 22–26× |
| BPTT | (không nói) | 8 bước, chỉ encoder hồi tiếp | GRU trước đó không nhận gradient theo thời gian |
| Ngân sách bước | 5×10⁶ | 5×10⁵ / 8×10⁵ / 10⁶ | Giới hạn CPU |
| Seed | ≥ 5, tái lập | 5, tái lập (gieo `torch`) | Trước đó chưa gieo |

---

[← Giai đoạn 16](giai-doan-16-gieo-seed-toan-bo-va-mot-ket-luan-phai-dieu-chinh-07-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 18 →](giai-doan-18-hai-kiem-chung-cuoi-08-09-2026.md)
