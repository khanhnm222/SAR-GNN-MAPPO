[← Giai đoạn 15](giai-doan-15-ma-tran-day-du-ba-kich-ban-results-v4-07-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 17 →](giai-doan-17-ra-soat-lan-cuoi-nhung-gi-con-lai-07-09-2026.md)

# Giai đoạn 16 — Gieo seed toàn bộ, và một kết luận phải điều chỉnh (07/09/2026)

Chạy lại 41 lượt chưa gieo seed (Easy 27 + Medium 14; Hard đã gieo đủ 27/27 nên
giữ nguyên), 8,5 giờ. **`results_v4/` nay có 81/81 lượt đều tái lập được.** Bản
cũ lưu ở `results_v4_unseeded/`.

### Bảng cuối cùng (VDR ± std liên-seed, n=5, toàn bộ đã gieo seed)

| Phương pháp | Easy | Medium | Hard |
|---|---|---|---|
| Random Walk | 0,433 | 0,231 | 0,082 |
| Greedy | 0,673 | 0,564 | 0,346 |
| MADDPG | 0,149 ± 0,016 | 0,209 ± 0,117 | 0,035 ± 0,035 |
| MAPPO-MLP | 0,881 ± 0,038 | 0,705 ± 0,041 | 0,382 ± 0,033 |
| **MAPPO-GCN** | **0,930 ± 0,030** | **0,717 ± 0,030** | **0,438 ± 0,016** |
| MAPPO-GAT | 0,916 ± 0,027 | 0,676 ± 0,052 | 0,417 ± 0,023 |
| GNN-MAPPO | 0,884 ± 0,032 | 0,644 ± 0,053 | 0,345 ± 0,042 |

### Kết luận Giai đoạn 15 về Medium PHẢI ĐIỀU CHỈNH

MAPPO-MLP ở Medium đi từ 0,638 (chưa gieo) lên **0,705** (đã gieo). Phép so sánh
GCN vs MLP vì thế đi từ p = 0,0147 (đạt Bonferroni) xuống **p = 0,63 (không còn
khác biệt)**. Kết quả "có ý nghĩa" ở Giai đoạn 15 phần lớn do lần rút thăm khởi
tạo không may của MLP, không phải do đồ thị.

### Biến thiên GIỮA CÁC ĐỢT CHẠY — phát hiện quan trọng về phương pháp

Cùng code, chỉ khác khởi tạo, 5 seed mỗi đợt:

| Kịch bản | Phương pháp | v3 | v4 chưa gieo | v4 đã gieo | Biên độ |
|---|---|---|---|---|---|
| Easy | MAPPO-MLP | 0,877 | 0,860 | 0,881 | 0,021 |
| Easy | MAPPO-GCN | 0,942 | 0,925 | 0,930 | 0,017 |
| **Medium** | **MAPPO-MLP** | 0,667 | 0,638 | **0,705** | **0,067** |
| Medium | MAPPO-GCN | 0,689 | 0,736 | 0,717 | 0,047 |

**Ở Medium, biên độ giữa các đợt (0,067) LỚN HƠN độ lệch chuẩn giữa seed
(0,041–0,050).** Nghĩa là n=5 ở Medium cho ước lượng trung bình không ổn định:
chạy lại cùng cấu hình có thể đảo ngược kết luận thống kê. Ở Easy thì ổn định
(biên độ 0,017–0,021). Đây là điều phải ghi trong phần hạn chế của luận văn.

### Ước lượng có lực thống kê cao nhất — gộp cả ba đợt

BPTT chỉ ảnh hưởng encoder hồi tiếp, nên với MLP/GCN/GAT thì ba đợt là ba mẫu
độc lập của cùng một cấu hình. Gộp lại:

| Kịch bản | Phương pháp | n | VDR | p vs MLP | Cohen's d | Bonferroni |
|---|---|---|---|---|---|---|
| Easy | MAPPO-MLP | 15 | 0,872 ± 0,040 | — | — | — |
| Easy | **MAPPO-GCN** | 15 | **0,932 ± 0,023** | **0,00005** | +1,83 | **đạt** |
| Easy | **MAPPO-GAT** | 15 | **0,909 ± 0,021** | **0,00497** | +1,14 | **đạt** |
| Medium | MAPPO-MLP | 15 | 0,670 ± 0,050 | — | — | — |
| Medium | MAPPO-GCN | 12 | 0,709 ± 0,034 | 0,0252 | +0,90 | không |
| Medium | MAPPO-GAT | 10 | 0,663 ± 0,049 | 0,717 | −0,15 | không |
| Hard | MAPPO-GCN | 5 | 0,438 ± 0,016 | 0,0157 | +2,14 | (1 đợt) |
| Hard | MAPPO-GAT | 5 | 0,417 ± 0,023 | 0,0964 | +1,21 | (1 đợt) |

### RQ1 — phát biểu đã hiệu chỉnh

Cấu trúc đồ thị **có** mang lợi thế, nhưng bằng chứng mạnh yếu khác nhau theo
kịch bản, và chỉ vững khi gộp đủ mẫu:

- **Easy (n=15)**: chắc chắn. GCN +0,060 (p = 0,00005), GAT +0,037 (p = 0,005),
  cả hai đạt Bonferroni.
- **Medium (n=12–15)**: bằng chứng vừa phải cho GCN (p = 0,025, d = +0,90),
  không đạt sau hiệu chỉnh. GAT không khác MLP.
- **Hard (n=5)**: GCN +0,056 (p = 0,016, d = +2,14) nhưng mới một đợt chạy;
  với biến thiên giữa đợt đã quan sát được ở Medium, con số này **cần một đợt
  lặp lại** trước khi trích dẫn như kết quả chắc chắn.

Phát biểu ở Giai đoạn 15 ("có ý nghĩa thống kê ở cả ba kịch bản, đạt Bonferroni
ở Medium và Hard") **quá mạnh** và phải thay bằng bảng trên.

### Không đổi

Thứ hạng **GCN > GAT > DGAT** vẫn lặp lại ở cả ba kịch bản. GNN-MAPPO không
khác MLP ở Easy (p = 0,89), thấp hơn MLP ở Medium (p = 0,077, d = −1,30) và ở
Hard (p = 0,16). RQ2: GRU vẫn không đóng góp, và gây hại có ý nghĩa ở Hard
(p = 0,0151, d = −2,10).

---

[← Giai đoạn 15](giai-doan-15-ma-tran-day-du-ba-kich-ban-results-v4-07-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 17 →](giai-doan-17-ra-soat-lan-cuoi-nhung-gi-con-lai-07-09-2026.md)
