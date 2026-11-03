[← Giai đoạn 2](giai-doan-02-loi-va-cham-tai-diem-sinh-uav.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 4 →](giai-doan-04-chan-doan-va-sua-loi-hieu-chinh-ham-thuong.md)

# Giai đoạn 3 — Nâng ngân sách lên quy mô nghiên cứu → phát hiện lỗi mới

Theo yêu cầu "muốn thực nghiệm nghiên cứu thay vì POC", ngân sách được nâng:

| Kịch bản | Trước | Sau | Số seed |
|---|---|---|---|
| Easy | 8.000 | **500.000** | 3 |
| Medium | 10.000 | **400.000** | 3 |
| Hard | 4.000 | **200.000** | 2 |

Đo tốc độ thực tế (đơn luồng, 1 tiến trình/lõi để chạy song song 46 lượt trên
16 lõi): Easy ≈ 290 bước/s, Medium ≈ 133 bước/s, Hard ≈ 59 bước/s. Chạy song
song 46 lượt (5 phương pháp học × 3 kịch bản × seed + 2 heuristic), tổng thời
gian ~45 phút thực tế cho phiên đầu tiên.

**Kết quả bất ngờ**: sau khi huấn luyện xong, **không một phương pháp nào** —
kể cả GNN-MAPPO — cho thấy xu hướng học rõ rệt trong suốt 500k/400k/200k bước.
Learning curve gần như đi ngang từ đầu đến cuối (xem
`results_broken_reward/figures/easy/learning_curves.png`).

Kiểm định định lượng trên toàn bộ 977 iteration của GNN-MAPPO/Easy:

| Chỉ số | Hồi quy tuyến tính (mỗi 1.000 bước) | Trung bình 1/4 đầu | Trung bình 1/4 cuối |
|---|---|---|---|
| mean_reward | -0.000012 (≈0) | 0.0435 | 0.0314 (giảm) |
| entropy | -0.00014 (≈0) | 2.297 | 2.318 (không đổi) |
| value_loss | -0.00105 (≈0) | — | — |

→ Entropy đứng yên gần `ln(13) = 2.565` (gần ngẫu nhiên đều trên 13 hành động)
suốt toàn bộ quá trình huấn luyện. Đây **không phải** vấn đề "chưa đủ bước" mà
là dấu hiệu tín hiệu học có vấn đề căn bản.

**Kiểm tra loại trừ nguyên nhân**: chạy 20 iteration, so sánh tham số mạng actor
trước/sau — tham số **có** thay đổi (drift trung bình 0.005-0.007 cho các lớp
actor), tức gradient không phải bằng 0 tuyệt đối (loại trừ khả năng optimizer
không chạy).

---

[← Giai đoạn 2](giai-doan-02-loi-va-cham-tai-diem-sinh-uav.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 4 →](giai-doan-04-chan-doan-va-sua-loi-hieu-chinh-ham-thuong.md)