[↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 2 →](giai-doan-02-loi-va-cham-tai-diem-sinh-uav.md)

# Giai đoạn 1 — POC ban đầu

**Cấu hình**: Easy 8.000 bước × 3 seed, Medium 10.000 bước × 2 seed, Hard 4.000
bước × 1 seed. 7 phương pháp: Random Walk, Greedy, MADDPG, MAPPO-MLP,
MAPPO-GCN, MAPPO-GAT (no GRU), GNN-MAPPO.

**Kết quả (Easy, VDR)**: Random Walk 0.535 · Greedy 0.171 (*) · MADDPG 0.096 ·
MAPPO-MLP 0.108 · MAPPO-GCN 0.079 · MAPPO-GAT 0.049 · GNN-MAPPO 0.110.

**Quan sát**: heuristic vượt trội tất cả phương pháp học — hợp lý ở quy mô POC
(ngân sách quá nhỏ để hội tụ), chưa có gì bất thường tại thời điểm này.

(*) Sau đó phát hiện `GreedyPolicy` có lỗi tie-break xác định trên `argmin`
khiến UAV bay thẳng một đường vào góc bản đồ khi vùng chưa quét toàn 0 — sửa
bằng cách thêm nhiễu ngẫu nhiên nhỏ phá vỡ ràng buộc, VDR của Greedy/Easy tăng
lên 0.473.

---

[↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 2 →](giai-doan-02-loi-va-cham-tai-diem-sinh-uav.md)
