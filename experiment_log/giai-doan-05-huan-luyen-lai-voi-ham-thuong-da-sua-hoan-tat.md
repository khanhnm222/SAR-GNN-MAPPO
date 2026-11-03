[← Giai đoạn 4](giai-doan-04-chan-doan-va-sua-loi-hieu-chinh-ham-thuong.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 6 →](giai-doan-06-5-seed-r-comm-thua-hon-gnn-mappo-ngan-sach-lon-easy.md)

# Giai đoạn 5 — Huấn luyện lại với hàm thưởng đã sửa (hoàn tất)

Cùng ngân sách như Giai đoạn 3 (500k/400k/200k bước, 3/3/2 seed), 46 lượt chạy
song song, hoàn tất toàn bộ.

**Xu hướng học (so với Giai đoạn 3 — hàm thưởng lỗi)**, GNN-MAPPO, trung bình
1/4 đầu vs 1/4 cuối mỗi lượt chạy:

| Kịch bản | Entropy: đầu → cuối | Reward: đầu → cuối |
|---|---|---|
| Easy | 2.338 → **2.145** (giảm thật) | 0.0506 → **0.0663** (+31%) |
| Medium | 2.518 → 2.502 (gần như không đổi) | 0.0147 → 0.0125 (giảm nhẹ) |
| Hard | 2.542 → 2.501 (giảm nhẹ) | 0.0032 → 0.0107 (×3, nhưng tuyệt đối rất nhỏ) |

→ Easy có tín hiệu học rõ nhất; Medium/Hard cải thiện rất ít dù hàm thưởng đã
sửa — nhiều khả năng cần nhiều bước hơn nữa ở quy mô lớn (nhiều UAV hơn = bài
toán khó hơn nhiều, không chỉ tuyến tính theo N).

### Kết quả so sánh baseline cuối cùng (VDR trung bình ± std liên-seed)

| Phương pháp | Easy (N=4) | Medium (N=8) | Hard (N=16) |
|---|---|---|---|
| Random Walk | 0.433 | 0.231 | 0.082 |
| Greedy | 0.473 | 0.197 | 0.109 |
| MADDPG | 0.117 ± 0.028 | 0.026 ± 0.025 | 0.027 ± 0.016 |
| MAPPO-MLP | 0.135 ± 0.015 | 0.082 ± 0.029 | 0.037 ± 0.000 |
| MAPPO-GCN | **0.279 ± 0.081** | **0.109 ± 0.040** | 0.030 ± 0.016 |
| MAPPO-GAT (no GRU) | 0.153 ± 0.015 | 0.045 ± 0.013 | **0.035 ± 0.002** |
| **GNN-MAPPO (đề xuất)** | 0.181 ± 0.027 | 0.075 ± 0.015 | 0.021 ± 0.007 |

**Đánh giá trung thực — chưa xác nhận được giả thuyết chính của luận văn:**

1. **GNN-MAPPO không phải phương pháp học tốt nhất ở bất kỳ kịch bản nào.**
   MAPPO-GCN (đồ thị, trọng số cố định, KHÔNG có attention) đứng đầu ở
   Easy/Medium; MAPPO-GAT (không GRU) nhỉnh hơn ở Hard. GNN-MAPPO (DGAT đầy đủ
   = attention + GRU) luôn ở giữa hoặc thấp nhất.
2. **Không có khác biệt nào có ý nghĩa thống kê** (kiểm định Welch's
   t-test/Mann-Whitney U, hiệu chỉnh Bonferroni, xem
   `results/*/architecture_ablation.json`) — ví dụ GNN-MAPPO vs MAPPO-MLP ở
   Easy: p=0.125 (chưa hiệu chỉnh), Cohen's d=1.70 (lớn) nhưng **không đủ ý
   nghĩa** với chỉ 3 seed. Với n=2-3 seed, phương sai giữa các seed quá lớn so
   với khác biệt trung bình quan sát được — **chưa thể kết luận kiến trúc nào
   tốt hơn kiến trúc nào** ở dữ liệu hiện tại, dù thứ tự trung bình có vẻ khác
   nhau.
3. **Đồ thị giao tiếp gần như đặc** (đo thực nghiệm, 100 bước ngẫu nhiên):
   Easy 54% số cạnh khả dĩ được kết nối, Medium 83%, **Hard 96%** — với đồ thị
   gần đầy đủ như vậy, cơ chế attention của GAT/DGAT phải học cách phân biệt
   giữa rất nhiều "hàng xóm" gần như ngang nhau, là bài toán tối ưu khó hơn hẳn
   so với GCN (trọng số cố định theo khoảng cách, không cần học attention) hay
   MLP (không cần học quan hệ liên tác nhân). Đây là giả thuyết hợp lý nhất cho
   việc GNN-MAPPO (nhiều tham số + attention + GRU) học chậm hơn các baseline
   đơn giản hơn **ở cùng ngân sách bước** — không nhất thiết là kiến trúc kém,
   mà là cần nhiều dữ liệu huấn luyện hơn để phần attention "đáng" so với chi
   phí học thêm nó.
4. Heuristic (Random Walk, Greedy) vẫn dẫn đầu ở cả 3 kịch bản — nhất quán với
   nhận định trong đề cương rằng cần ngân sách rất lớn (đề cương: 5×10⁶
   bước/cấu hình) để các phương pháp học vượt qua heuristic; 200k-500k bước
   (dù đã hơn ~50-60 lần so với POC ban đầu) vẫn có thể còn xa mức đó.

---

[← Giai đoạn 4](giai-doan-04-chan-doan-va-sua-loi-hieu-chinh-ham-thuong.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 6 →](giai-doan-06-5-seed-r-comm-thua-hon-gnn-mappo-ngan-sach-lon-easy.md)
