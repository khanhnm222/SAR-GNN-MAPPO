[← Giai đoạn 17](giai-doan-17-ra-soat-lan-cuoi-nhung-gi-con-lai-07-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 19 →](giai-doan-19-lap-lai-hard-seed-5-9-va-ban-nhap-chuong-5-08-09-2026.md)

# Giai đoạn 18 — Hai kiểm chứng cuối (08/09/2026)

### ⑧ Kiểm chứng ngân sách 2×: thứ hạng GCN > DGAT là do KIẾN TRÚC, không do ngân sách

Easy, 1M bước (gấp đôi), 5 seed (`results_v4_budget2x/`):

| Kiến trúc | 500k | 1M | Chênh |
|---|---|---|---|
| MAPPO-GCN | 0,930 ± 0,030 | **0,966 ± 0,020** | +0,035 |
| GNN-MAPPO | 0,884 ± 0,032 | 0,911 ± 0,044 | +0,027 |

Khoảng cách DGAT − GCN tại 500k: −0,046 (p = 0,048). Tại 1M: **−0,055
(p = 0,049)**. Gấp đôi ngân sách giúp cả hai tăng gần bằng nhau; **DGAT không
bắt kịp, khoảng cách còn hơi rộng ra**. Giả thuyết "DGAT thua vì có 3× tham số
nên chưa được huấn luyện đủ" **bị bác bỏ** ở Easy. Thứ hạng là thật.

### ⑨⑩ Heuristic 5 seed trên cùng tập bản đồ test — Hard đổi câu chuyện

| Heuristic | Easy | Medium | Hard |
|---|---|---|---|
| Random Walk | 0,410 ± 0,078 | 0,183 ± 0,018 | 0,081 ± 0,024 |
| Greedy | 0,678 ± 0,031 | 0,498 ± 0,042 | **0,403 ± 0,032** |

Greedy ở Hard là **0,403** trên tập test chung (trước đây 0,346 trên tập 10k).
Điều đó đảo lại một phát biểu của Giai đoạn 15:

| Hard, so với Greedy 0,403 | VDR | p | d |
|---|---|---|---|
| MAPPO-MLP | 0,382 | 0,34 | −0,64 |
| MAPPO-GCN | 0,438 | 0,078 | +1,35 |
| GNN-MAPPO | 0,345 | **0,043** | **−1,54** |

**Ở Hard, không phương pháp học nào vượt Greedy có ý nghĩa thống kê.** MAPPO-MLP
thấp hơn Greedy; GNN-MAPPO thấp hơn có ý nghĩa; chỉ MAPPO-GCN nhỉnh hơn về danh
nghĩa (+0,035, p = 0,078). Phát biểu "ở Hard phần lớn giá trị của học tăng cường
đến từ đồ thị" ở Giai đoạn 15 **phải sửa thành**: ở Hard với 16 UAV và 1M bước,
học tăng cường chưa vượt được một heuristic cục bộ; cấu trúc đồ thị là thứ duy
nhất kéo nó lên ngang mức đó.

Ở Easy và Medium mọi phương pháp MAPPO vượt Greedy áp đảo (p ≤ 0,0016,
d ≥ +3,0).

### `results_v4/` — trạng thái cuối

105/105 lượt (7 phương pháp × 5 seed × 3 kịch bản), **toàn bộ đã gieo seed**,
toàn bộ đánh giá trên cùng tập bản đồ test, zero-shot đủ ba kịch bản.

### Tóm tắt trả lời ba câu hỏi nghiên cứu — bản cuối

**RQ1 (cấu trúc đồ thị có giúp ích không):** Có, với bằng chứng mạnh yếu khác
nhau theo quy mô. Easy (gộp n=15): GCN và GAT đều vượt MLP, đạt Bonferroni
(p = 0,00005 và 0,005). Medium (n=12–15): GCN p = 0,025, d = +0,90, không đạt
Bonferroni. Hard (n=5, một đợt): GCN p = 0,016, d = +2,14, cần lặp lại.

**RQ2 (bộ nhớ thời gian):** Không đóng góp ở Easy/Medium; **gây hại có ý nghĩa
ở Hard** (p = 0,015, d = −2,10), ngay cả sau khi cài BPTT thật.

**RQ3 (zero-shot):** Mọi kiến trúc đều tổng quát hoá tốt khi N tăng. DGAT chỉ
dẫn đầu ở Medium, thấp nhất ở Hard. Không có bằng chứng DGAT tổng quát hoá tốt
hơn.

**Về kiến trúc đề xuất:** GCN > GAT > DGAT ở cả ba kịch bản; khoảng cách không
đóng lại khi gấp đôi ngân sách. Attention có đặc trưng cạnh không thắng tổng hợp
trọng số cố định theo cường độ tín hiệu; GRU không thêm giá trị.

---

[← Giai đoạn 17](giai-doan-17-ra-soat-lan-cuoi-nhung-gi-con-lai-07-09-2026.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 19 →](giai-doan-19-lap-lai-hard-seed-5-9-va-ban-nhap-chuong-5-08-09-2026.md)
