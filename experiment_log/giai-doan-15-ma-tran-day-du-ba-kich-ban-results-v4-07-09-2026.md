[← Giai đoạn 14](giai-doan-14-gop-v3-v4-rq1-dat-y-nghia-thong-ke-o-ca-hai-kich-ban-06-09-2.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 16 →](giai-doan-16-gieo-seed-toan-bo-va-mot-ket-luan-phai-dieu-chinh-07-09-2026.md)

# Giai đoạn 15 — Ma trận đầy đủ ba kịch bản (`results_v4/`, 07/09/2026)

81 lượt (Easy 500k · Medium 800k · Hard 1M bước; 7 phương pháp × 5 seed),
**52,3 giờ** wall với 6 job song song. Cộng zero-shot cho cả ba kịch bản
(4 kiến trúc × 3 seed × 4–5 mức N).

### Bảng so sánh baseline đầy đủ (VDR ± std liên-seed)

| Phương pháp | Easy | Medium | Hard |
|---|---|---|---|
| Random Walk | 0,433 | 0,231 | 0,082 |
| Greedy | 0,673 | 0,564 | 0,346 |
| MADDPG | 0,149 ± 0,016 | 0,123 ± 0,091 | 0,035 ± 0,035 |
| MAPPO-MLP | 0,860 ± 0,060 | 0,638 ± 0,053 | 0,382 ± 0,033 |
| **MAPPO-GCN** | **0,925 ± 0,016** | **0,728 ± 0,031** | **0,438 ± 0,016** |
| MAPPO-GAT | 0,902 ± 0,027 | 0,676 ± 0,052 | 0,417 ± 0,023 |
| GNN-MAPPO | 0,889 ± 0,049 | 0,644 ± 0,053 | 0,345 ± 0,042 |

### RQ1 — CÓ, cấu trúc đồ thị giúp ích, nhất quán qua cả ba quy mô

| Kịch bản | GCN vs MLP | GAT vs MLP | DGAT vs MLP |
|---|---|---|---|
| Easy | p = 0,032 · d = +1,49 | p = 0,21 · d = +0,91 | p = 0,43 · d = +0,53 |
| Medium | **p = 0,0147 · d = +2,09 · Bonf đạt** | p = 0,29 · d = +0,72 | p = 0,87 · d = +0,11 |
| Hard | **p = 0,0157 · d = +2,14 · Bonf đạt** | p = 0,096 · d = +1,21 | p = 0,16 · d = −0,98 |

Chênh lệch GCN − MLP: +0,065 / +0,090 / +0,055 — **ổn định theo N**. Nhận định
"lợi ích của đồ thị giảm theo N" ở Giai đoạn 12 dựa trên dữ liệu còn nhiễu
(`r_comm` chưa sửa ở Medium, seed chưa gieo) và **đã bị bác bỏ**.

Gộp v3 + v4 cho MLP/GCN/GAT (n=10, cùng code): Easy p = 0,00044 (d = +1,95),
Medium p = 0,00778 (d = +1,35), cả hai đạt Bonferroni.

### RQ2 — bộ nhớ thời gian không đóng góp, và gây hại ở quy mô lớn

| Kịch bản | Có GRU | Không GRU | p | d |
|---|---|---|---|---|
| Easy | 0,889 | 0,902 | 0,62 | −0,33 |
| Medium | 0,644 | 0,676 | 0,36 | −0,61 |
| **Hard** | **0,345** | **0,417** | **0,0151** | **−2,10** |

Truncated BPTT (Giai đoạn 13) đã gỡ được tác hại ở Easy (p: 0,0116 → 0,62) và
Medium, nhưng ở Hard GRU vẫn kéo hiệu năng xuống có ý nghĩa thống kê. Giả
thuyết: 12.480 tham số thêm của GRU không được đền đáp khi ngân sách trên mỗi
tham số mỏng đi (Hard có 16 lần số ô của Easy nhưng chỉ 2 lần số bước).

### RQ3 — zero-shot: lợi thế của DGAT KHÔNG lặp lại

VDR **tuyệt đối** trung bình tại các mức N vượt N huấn luyện (không dùng "% giữ
được", vì mẫu số là chính hiệu năng tại N huấn luyện nên phương pháp yếu trông
có vẻ tổng quát hoá tốt — một bẫy phương pháp luận cần tránh trong luận văn):

| | Easy (N>4) | Medium (N>8) | Hard (N>16) |
|---|---|---|---|
| MAPPO-MLP | 0,964 | 0,856 | 0,458 |
| MAPPO-GCN | **0,985** | 0,845 | 0,535 |
| MAPPO-GAT | **0,989** | 0,826 | **0,577** |
| GNN-MAPPO | 0,982 | **0,869** | 0,433 |

GNN-MAPPO chỉ dẫn đầu ở **Medium**. Ở Easy bốn kiến trúc ngang nhau, ở Hard nó
**thấp nhất**. Nhận định "DGAT tổng quát hoá tốt nhất khi quy mô đàn thay đổi"
mà tôi đưa ra khi chỉ có dữ liệu Medium **không vững** và phải rút lại.

### Kết luận cho luận văn

**Bảo vệ được:** cấu trúc đồ thị mang lại lợi thế có ý nghĩa thống kê, cỡ hiệu
ứng lớn, nhất quán qua ba quy mô đàn (4/8/16 UAV) và ba loại địa hình. Ở Hard,
MAPPO-MLP chỉ hơn Greedy 0,036 trong khi MAPPO-GCN hơn 0,092 — phần lớn giá trị
mà học tăng cường mang lại ở quy mô lớn đến **từ cấu trúc đồ thị**.

**Không bảo vệ được:** thứ hạng GCN > GAT > DGAT lặp lại ở cả ba kịch bản.
Kiến trúc đề xuất (DGAT = GATv2 + đặc trưng cạnh + GRU) là kiến trúc đồ thị
yếu nhất, và ở Hard còn thấp hơn cả MLP. Attention có đặc trưng cạnh không
thắng tổng hợp trọng số cố định; GRU không đóng góp gì và gây hại ở N=16.

**Đóng góp phương pháp luận** (có thể là đóng góp mạnh nhất): bảy lỗi được tìm
ra và sửa, mỗi lỗi đều thuộc loại làm hỏng âm thầm mọi so sánh kiến trúc trong
MARL — giao thức đánh giá tất định trên chính sách entropy cao; bản đồ niềm tin
toàn cục khiến giao tiếp vô nghĩa; thông tin hàng xóm rò rỉ vào baseline; thưởng
và advantage toàn cục; quan sát cục bộ là hằng số; bán kính liên lạc khiến đồ
thị rỗng; và seed chưa gieo. Mỗi lỗi đều kèm phép đo định lượng và một kiểm thử
tự động chống tái diễn (`scripts/selftest_review_fixes.py`, 34 kiểm tra).

---

[← Giai đoạn 14](giai-doan-14-gop-v3-v4-rq1-dat-y-nghia-thong-ke-o-ca-hai-kich-ban-06-09-2.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 16 →](giai-doan-16-gieo-seed-toan-bo-va-mot-ket-luan-phai-dieu-chinh-07-09-2026.md)
