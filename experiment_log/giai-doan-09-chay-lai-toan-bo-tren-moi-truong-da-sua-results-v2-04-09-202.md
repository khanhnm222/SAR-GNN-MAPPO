[← Giai đoạn 8](giai-doan-08-ra-soat-toan-bo-code-03-09-2026-loi-danh-gia-argmax.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 10 →](giai-doan-10-loi-thu-sau-do-thi-giao-tiep-gan-nhu-rong-04-09-2026.md)

# Giai đoạn 9 — Chạy lại toàn bộ trên môi trường đã sửa (`results_v2/`, 04/09/2026)

54 lượt (Easy 500k bước + Medium 800k bước, 7 phương pháp × 5 seed), 15,1 giờ
wall-clock với 6 job song song, `OMP_NUM_THREADS=1`.

### Lỗi thứ năm, phát hiện giữa lúc đang chạy: 50/86 chiều quan sát là hằng số

Đợt chạy đầu bị dừng ở lượt 14/27 vì MADDPG vẫn ở VDR 0,147 — thấp hơn Random
Walk (0,433) — dù đã huấn luyện lại hợp lệ. Truy nguyên: actor của MADDPG có
**entropy = 0,000**, dồn toàn bộ xác suất vào hành động 11. Đó không phải lỗi
MADDPG (DDPG sinh chính sách tất định theo thiết kế) mà là triệu chứng của một
lỗi môi trường sâu hơn.

`r_sense = 3` nhưng `LOCAL_MAP_RADIUS = 2`: cửa sổ niềm tin bán kính 2 nằm
**hoàn toàn bên trong** đĩa cảm biến bán kính 3 mà UAV vừa quét ở bước trước.
Đo trên 480 lượt agent-step:

| Kênh | Trạng thái |
|---|---|
| `visited` (25 chiều) | toàn bộ = 1 ở **100%** số bước |
| `prob` (25 chiều) | độ lệch chuẩn **0,0003** |

UAV không thể thấy bất kỳ ô chưa quét nào → không có cơ sở chọn hướng → rẽ chỉ
có thể là ngẫu nhiên. Điều này giải thích cùng lúc: vì sao chính sách tất định
luôn thoái hoá, vì sao MADDPG hỏng hoàn toàn, và vì sao chia sẻ độ phủ qua đồ
thị không giúp được gì (không tác nhân nào *cảm nhận* được độ phủ).

Lỗi thứ hai cùng chỗ: vùng ngoài bản đồ đệm `visited = 0` ("chưa quét"), kéo cả
đàn ra biên — đúng hành vi bám mép thấy trong mọi replay.

**Sửa** (`belief_patch_block = 5`): kênh `occupancy` giữ độ phân giải cao (bán
kính 2, để tránh vật cản); hai kênh `prob`/`visited` chuyển sang thô, mỗi ô gộp
khối 5×5 nên cửa sổ trải 25×25 ô (bán kính 12), vượt ra ngoài đĩa cảm biến. Giữ
nguyên hình dạng 5×5×3 của Bảng 2. Đệm ngoài bản đồ đổi thành `visited = 1`.
Độ lệch chuẩn kênh `visited`: 0,000 → **0,395**.

Bằng chứng độc lập rằng bản sửa là thật: **Greedy** — heuristic thuần cục bộ,
không học gì — nhảy từ 0,473 lên **0,716** ở Easy. Trước đây nó "chọn ô ít được
thăm nhất" từ một ma trận toàn số 1, tức chọn ngẫu nhiên.

Đợt chạy dở lưu ở `results_v2_constmap/`.

### Kết quả (VDR trung bình ± std, n=5 seed)

| Phương pháp | Easy cũ | **Easy v2** | Medium cũ | **Medium v2** |
|---|---|---|---|---|
| Random Walk | 0,410 | 0,433 | 0,183 | 0,231 |
| Greedy | 0,451 | **0,716** | 0,182 | **0,587** |
| MADDPG | 0,147 | 0,147 ± 0,019 | 0,027 | 0,200 ± 0,082 |
| MAPPO-MLP | 0,546 | **0,855 ± 0,010** | 0,171 | **0,642 ± 0,037** |
| MAPPO-GCN | 0,496 | **0,873 ± 0,030** | 0,166 | 0,638 ± 0,044 |
| MAPPO-GAT | 0,463 | 0,870 ± 0,038 | 0,180 | **0,646 ± 0,032** |
| GNN-MAPPO | 0,582 | 0,847 ± 0,035 | 0,200 | 0,632 ± 0,040 |

Easy: 500k bước. Medium: 800k bước (gấp đôi ngân sách cũ). Hard bỏ qua có chủ ý.

### Kết luận cho RQ1: đồ thị KHÔNG mang lại lợi thế đo được

| Kịch bản | GNN-MAPPO vs MLP | vs GCN | vs GAT |
|---|---|---|---|
| Easy | p = 0,63, d = −0,33 | p = 0,25, d = −0,79 | p = 0,35, d = −0,62 |
| Medium | p = 0,84, d = −0,25 | p = 0,82, d = −0,15 | p = 0,55, d = −0,39 |

Bốn kiến trúc nằm trong khoảng 0,026 (Easy) và 0,014 (Medium) của nhau —
chênh lệch giữa kiến trúc **nhỏ hơn** độ lệch chuẩn giữa các seed của chính
chúng. GNN-MAPPO đứng thứ tư về danh nghĩa ở cả hai kịch bản.

### Zero-shot cũng không phân biệt được (3 seed × 10 episode/mức N)

Tỷ lệ hiệu năng giữ được so với N lúc huấn luyện:

| Easy (train N=4) | N=2 | N=6 | N=8 | N=12 |
|---|---|---|---|---|
| MAPPO-MLP | 66% | 109% | 110% | 112% |
| MAPPO-GCN | 92% | 110% | 113% | 115% |
| GNN-MAPPO | 80% | 112% | 114% | 116% |

| Medium (train N=8) | N=4 | N=6 | N=10 | N=12 | N=16 |
|---|---|---|---|---|---|
| MAPPO-MLP | 77% | 84% | 119% | 133% | **149%** |
| MAPPO-GCN | 68% | 85% | 131% | 127% | 129% |
| GNN-MAPPO | 75% | 87% | 116% | 138% | 138% |

Mọi kiến trúc đều tổng quát hoá tốt và tương đương nhau; MAPPO-MLP thậm chí giữ
được nhiều nhất ở N=16. **Bảng zero-shot của Giai đoạn 6** (GNN +43%, GCN −11%)
được đo bằng `argmax` trên môi trường lỗi — **không dùng được nữa**.

### Vì sao đồ thị không giúp — giải thích thiết kế

Sau các sửa lỗi, môi trường đã tự cung cấp cả *thông tin* lẫn *động cơ* phối hợp:

- bản đồ niềm tin được hợp nhất **trong môi trường** dọc theo liên kết `r_comm`,
  nên UAV nhận được độ phủ của hàng xóm mà không cần GNN;
- thưởng theo từng UAV chỉ tính ô mới **so với bản đồ riêng của nó**, nên bay
  vào vùng hàng xóm đã quét được 0 điểm.

Chính sách chỉ cần đi theo gradient cục bộ "chỗ nào chưa quét". Phần việc còn
lại cho GNN — suy luận về *ý định* của hàng xóm — đóng góp không đo được ở quy
mô này. Củng cố: Greedy (thuần cục bộ, không học) đạt 0,716/0,587, tức 84%/91%
mức của phương pháp học tốt nhất. **Bài toán SAR như đang đặc tả bị chi phối
bởi khám phá cục bộ, không phải phối hợp.**

Đây là kết quả âm tính nhất quán qua ba phép thử độc lập với phương pháp luận
đã sạch. Nó cần được báo cáo đúng như vậy trong luận văn.

---

[← Giai đoạn 8](giai-doan-08-ra-soat-toan-bo-code-03-09-2026-loi-danh-gia-argmax.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 10 →](giai-doan-10-loi-thu-sau-do-thi-giao-tiep-gan-nhu-rong-04-09-2026.md)
