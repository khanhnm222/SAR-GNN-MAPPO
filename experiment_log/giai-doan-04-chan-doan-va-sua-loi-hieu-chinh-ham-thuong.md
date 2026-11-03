[← Giai đoạn 3](giai-doan-03-nang-ngan-sach-len-quy-mo-nghien-cuu-phat-hien-loi-moi.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 5 →](giai-doan-05-huan-luyen-lai-voi-ham-thuong-da-sua-hoan-tat.md)

# Giai đoạn 4 — Chẩn đoán và sửa lỗi hiệu chỉnh hàm thưởng

**Giả thuyết**: hàm thưởng có thành phần lấn át. Kiểm chứng trực tiếp bằng cách
cho Greedy và Random Walk (không cần huấn luyện) chạy trên cùng environment và
so sánh **tổng reward** với **hiệu năng nhiệm vụ thật**:

| Seed | Phương pháp | Tổng reward (trước sửa) | Coverage | Nạn nhân tìm được |
|---|---|---|---|---|
| 1 | Greedy | **-9.584** | 0.456 | 2/4 |
| 1 | Random Walk | **+5.417** | 0.456 | 1/4 |

**Greedy tìm được NHIỀU nạn nhân hơn nhưng tổng reward THẤP HƠN HẲN Random** —
hàm thưởng đang dạy sai mục tiêu.

**Nguyên nhân gốc** (`sar_env/rewards.py`): thành phần va chạm được tính bằng
`float(n_collisions)` — **tổng số cặp UAV vi phạm cùng lúc, không giới hạn
trên** — nhân với trọng số `γ = -2.0`, rồi **cộng dồn theo từng bước suốt cả
episode**. Với Greedy/Easy: 10 bước có va chạm × -2.0 = **-20**, trong khi tổng
thưởng tối đa có thể từ nạn nhân chỉ là 4.0 (4 nạn nhân × ưu tiên 1 ở Easy).
Với N lớn hơn (Hard, 16 UAV), số cặp vi phạm đồng thời có thể lên hàng chục,
khiến phạt một bước duy nhất đã vượt xa mọi tín hiệu nhiệm vụ khác.

Đề cương gốc mô tả trọng số này là "đủ mạnh... nhưng không quá lớn gây ra hành
vi quá thận trọng" — giá trị -2.0 khi cộng dồn không giới hạn đã vi phạm chính
tinh thần đó.

**Sửa 2 chỗ**:

1. `sar_env/rewards.py` — thành phần va chạm đổi từ **tổng số cặp vi phạm**
   sang **cờ 0/1** (`1.0 nếu có ≥1 vi phạm, ngược lại 0.0`), đồng thời giảm
   trọng số γ từ **-2.0 → -0.05** (giữ mức phạt rõ ràng hơn thưởng bao phủ mỗi
   bước ~15-50 lần, nhưng không để cả episode tích lũy vượt quá thưởng phát
   hiện nạn nhân).
2. `evaluation/evaluate.py` — chỉ số `collision_rate` đổi từ "tổng số cặp vi
   phạm / số bước" (có thể > 1, không phải "tỉ lệ") sang đúng định nghĩa đề
   cương mục 1.5.5: **"tỉ lệ BƯỚC có va chạm trên tổng số bước"** (đếm số bước
   có ≥1 vi phạm, không cộng dồn số cặp).

**Kiểm chứng lại** (sau sửa, nhiều seed, Easy):

| Seed | Greedy | Random Walk |
|---|---|---|
| 1 | +9.916 | +5.417 |
| 2 | +10.181 | +10.263 |
| 3 | +5.237 | +10.410 |

→ Tổng reward nay dương và tương quan hợp lý hơn nhiều với hiệu năng thật (còn
dao động giữa seed là bình thường do chưa huấn luyện, nạn nhân sinh ngẫu nhiên
khác nhau mỗi seed).

**Kiểm tra nhanh tác động lên huấn luyện** (GNN-MAPPO/Easy, 100 iteration ≈
46.592 bước, cùng seed/cấu hình như trước):

| | Bước 512 | Bước 46.592 |
|---|---|---|
| mean_reward | 0.0292 | 0.0695 (**×2,4**) |
| entropy | 2.5559 | 2.4735 (bắt đầu giảm) |

So với hàm thưởng cũ, nơi cả reward lẫn entropy **không đổi suốt 500.000
bước**, đây là khác biệt rõ rệt trong một ngân sách nhỏ hơn 10 lần — xác nhận
hàm thưởng mới cho tín hiệu học thật.

---

[← Giai đoạn 3](giai-doan-03-nang-ngan-sach-len-quy-mo-nghien-cuu-phat-hien-loi-moi.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 5 →](giai-doan-05-huan-luyen-lai-voi-ham-thuong-da-sua-hoan-tat.md)
