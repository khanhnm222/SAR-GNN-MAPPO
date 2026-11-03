[← Giai đoạn 1](giai-doan-01-poc-ban-dau.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 3 →](giai-doan-03-nang-ngan-sach-len-quy-mo-nghien-cuu-phat-hien-loi-moi.md)

# Giai đoạn 2 — Lỗi va chạm tại điểm sinh UAV

**Phát hiện**: người dùng lưu ý kết quả không khả quan → kiểm tra lại toàn bộ
pipeline. Viết script kiểm chứng thực nghiệm: reset 50 episode/kịch bản, đếm số
episode có ≥1 cặp UAV vi phạm khoảng cách an toàn (`SAFETY_DIST = 1.0`) **ngay
tại bước reset** (trước khi bất kỳ hành động nào được thực hiện).

| Kịch bản | % episode va chạm ngay lúc reset | Số cặp va chạm trung bình |
|---|---|---|
| Easy (N=4) | 48% (24/50) | 1.46 |
| Medium (N=8) | **100%** (50/50) | 4.38 |
| Hard (N=16) | **100%** (50/50) | 18.54 |

**Nguyên nhân**: `_spawn_uavs()` sinh UAV bằng nhiễu ngẫu nhiên độc lập
`uniform(-2,2)` cho từng UAV quanh cùng một tâm — với N≥8, xác suất có ít nhất
một cặp cách nhau < 1.0 gần như chắc chắn.

**Sửa** (`sar_env/sar_parallel_env.py::_spawn_uavs`): thay bằng lưới cách đều
`spacing = max(1.5, SAFETY_DIST × 2.0)` quanh tâm, cộng thêm nhiễu nhỏ ±0.3 để
vẫn đa dạng giữa các episode.

**Kiểm chứng lại**: 0/50 episode va chạm ở cả 3 kịch bản sau khi sửa.
`sar_env.validation` vẫn pass toàn bộ 7 tiêu chí.

**Việc khác làm cùng đợt này**:
- Tái cấu trúc `results/figures/` từ dạng phẳng (`comparison_easy.png`) sang
  thư mục con theo kịch bản (`results/figures/easy/comparison.png`, ...).
- Cập nhật hình quỹ đạo (PNG tĩnh) và canvas phát lại (web) đúng ký hiệu Hình 2
  của đề cương: UAV hình thoi, nạn nhân hình tròn màu theo mức ưu tiên
  (vàng/cam/đỏ = 1/2/3), vật cản tĩnh ô vuông xám — trước đó dùng tam giác/sao
  và không vẽ vật cản.
- Thêm trường `terrain_occupancy` vào `trajectory_sample.json` (trước đó không
  xuất bản đồ vật cản ra dữ liệu phát lại).

**Huấn luyện lại ở quy mô POC** (cùng ngân sách 8k/10k/4k bước): `collision_rate`
trở nên hợp lý hơn nhiều — ví dụ MAPPO-MLP/MAPPO-GCN ở Easy = 0.000 thay vì
luôn dương do đội hình xuất phát đã chồng lấn. Kết quả lưu tại
`results_poc_scale/` (đã archive).

---

[← Giai đoạn 1](giai-doan-01-poc-ban-dau.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 3 →](giai-doan-03-nang-ngan-sach-len-quy-mo-nghien-cuu-phat-hien-loi-moi.md)
