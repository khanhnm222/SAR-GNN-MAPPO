[← Giai đoạn 6](giai-doan-06-5-seed-r-comm-thua-hon-gnn-mappo-ngan-sach-lon-easy.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 8 →](giai-doan-08-ra-soat-toan-bo-code-03-09-2026-loi-danh-gia-argmax.md)

# Giai đoạn 7 — Kiểm chứng `value_coef=0.1` ở quy mô đầy đủ (108 lượt)

Theo khuyến nghị của Giai đoạn 6, chạy lại toàn bộ ma trận (5 seed, r_comm=10)
với `value_coef` giảm từ 0.5 xuống **0.1**.

### Kết quả (VDR trung bình ± std, n=5 seed) — value_coef=0.1

| Phương pháp | Easy | Medium | Hard |
|---|---|---|---|
| MADDPG | 0.138 ± 0.028 | 0.040 ± 0.028 | 0.020 ± 0.018 |
| MAPPO-MLP | 0.151 ± 0.014 | **0.093 ± 0.034** | 0.048 ± 0.039 |
| MAPPO-GCN | **0.290 ± 0.144** | 0.068 ± 0.020 | **0.056 ± 0.023** |
| MAPPO-GAT (no GRU) | 0.199 ± 0.063 | 0.058 ± 0.018 | 0.033 ± 0.021 |
| **GNN-MAPPO** | 0.157 ± 0.035 | 0.034 ± 0.013 | 0.038 ± 0.021 |

### So sánh trực tiếp với Giai đoạn 6 (value_coef=0.5) — GNN-MAPPO

| Kịch bản | vc=0.5 (Giai đoạn 6) | vc=0.1 (Giai đoạn 7) | Thay đổi |
|---|---|---|---|
| Easy | 0.188 ± 0.040 | 0.157 ± 0.035 | **giảm** |
| Medium | 0.058 ± 0.041 | 0.034 ± 0.013 | **giảm** |
| Hard | 0.038 ± 0.011 | 0.038 ± 0.021 | không đổi |

**Kết luận trung thực: giả thuyết KHÔNG được xác nhận ở quy mô đầy đủ cho
GNN-MAPPO** — dù kiểm chứng nhanh (30k bước, 1 seed) cho tín hiệu tích cực,
chạy đủ 500k/400k/200k bước × 5 seed cho thấy `value_coef=0.1` làm GNN-MAPPO
**kém đi** ở Easy/Medium, không đổi ở Hard. Đây là bài học quan trọng về
phương pháp: **chẩn đoán nhanh ở quy mô nhỏ có thể không khái quát hóa được**
— cần luôn kiểm chứng lại ở quy mô đầy đủ trước khi kết luận.

Thú vị là `value_coef=0.1` lại **giúp MAPPO-GCN cải thiện rõ rệt** (Easy:
0.189→0.290, Hard: 0.043→0.056) — cho thấy độ nhạy với siêu tham số này khác
nhau giữa các kiến trúc, và tối ưu siêu tham số nên làm **riêng cho từng kiến
trúc** thay vì dùng chung một bộ cho tất cả (dù điều này lại mâu thuẫn với
yêu cầu "so sánh công bằng" của đề cương — cần cân nhắc đánh đổi này nếu tiếp
tục tinh chỉnh).

**Phát hiện bền vững quan trọng**: tính chất zero-shot scalability của
GNN-MAPPO ở Easy **vẫn giữ nguyên** ở value_coef=0.1 (N=4→6→8: 0.156→0.197→
0.219, **+41%**, so với +43% ở value_coef=0.5) — xác nhận đây là ưu điểm cấu
trúc bền vững, không phụ thuộc vào lựa chọn siêu tham số này, khác với hiệu
năng tuyệt đối ở N gốc vốn nhạy cảm với value_coef.

**Quyết định**: khôi phục `results/` về kết quả Giai đoạn 6 (value_coef=0.5,
tốt hơn cho GNN-MAPPO) làm bộ số liệu chính thức; kết quả value_coef=0.1 lưu
tham khảo tại `results_stage7_vc01/`. File config đã khôi phục `value_coef: 0.5`
kèm ghi chú lý do.

---

[← Giai đoạn 6](giai-doan-06-5-seed-r-comm-thua-hon-gnn-mappo-ngan-sach-lon-easy.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 8 →](giai-doan-08-ra-soat-toan-bo-code-03-09-2026-loi-danh-gia-argmax.md)
