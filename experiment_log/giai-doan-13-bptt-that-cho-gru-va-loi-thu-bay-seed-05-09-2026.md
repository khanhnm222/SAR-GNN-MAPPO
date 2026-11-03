[← Giai đoạn 12](giai-doan-12-easy-voi-r-comm-26-do-thi-co-tac-dung-nhung-khong-phai-dgat-.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 14 →](giai-doan-14-gop-v3-v4-rq1-dat-y-nghia-thong-ke-o-ca-hai-kich-ban-06-09-2.md)

# Giai đoạn 13 — BPTT thật cho GRU, và lỗi thứ bảy: seed (05/09/2026)

### Sửa BPTT — để kiến trúc đề xuất được đánh giá công bằng

GNN-MAPPO = DGAT **có GRU**; GRU chính là đóng góp riêng của nó so với
MAPPO-GAT. Nhưng nó được cập nhật từ `h_prev` đã lưu của từng bước với các bước
bị xáo trộn — **BPTT dài đúng 1 bước**. GRU mang trạng thái nhưng chưa từng nhận
gradient cho việc mang trạng thái đó.

`bptt_len=8`: cắt rollout thành đoạn liên tục trong cùng episode, xáo trộn theo
**đoạn**, chạy encoder xuyên đoạn mang hidden state **có gradient**. Chỉ encoder
hồi tiếp đi đường này; MLP/GCN/GAT giữ nguyên đường phẳng nên so sánh Bảng 5 vẫn
công bằng. Chi phí thêm cho DGAT: **11%** (68 so với 76 bước/giây).

**RQ2 — kết quả sau khi sửa (Easy, n=5):**

| | Có GRU | Không GRU | p | Cohen's d |
|---|---|---|---|---|
| BPTT=1 (v3) | 0,890 | 0,910 | **0,0116** | −2,32 |
| BPTT=8 (v4) | 0,889 | 0,902 | 0,62 | −0,33 |

**BPTT loại bỏ được tác hại nhưng không tạo ra lợi ích.** Câu trả lời trung
thực cho RQ2 nay là *"bộ nhớ thời gian không đóng góp gì đo được"* thay vì
*"bộ nhớ thời gian gây hại"* — một phát biểu công bằng hơn hẳn với đề xuất.

### Lỗi thứ bảy: `torch.manual_seed` chưa từng được gọi

So sánh `results_v3_rcomm` với `results_v4` ở Easy cho thấy các phương pháp
**không đổi một dòng code nào** vẫn ra kết quả khác:

| Phương pháp | v3 | v4 | Chênh |
|---|---|---|---|
| MAPPO-MLP | 0,877 | 0,860 | −0,017 |
| MAPPO-GCN | 0,942 | 0,925 | −0,017 |
| MAPPO-GAT | 0,910 | 0,902 | −0,009 |

Nguyên nhân: `torch.manual_seed` không được gọi ở bất kỳ đâu trong dự án. Tham
số `--seed` chỉ điều khiển môi trường và phép xáo trộn minibatch của numpy;
**khởi tạo trọng số mạng và mọi `dist.sample()` trong rollout đều lấy từ RNG
toàn cục chưa gieo**. Các lần chạy vì thế **không tái lập được**, trái với giao
thức seed ở mục 1.5.5. Độ lớn của nhiễu này (0,017) **ngang với độ lệch chuẩn
giữa các seed** (0,016–0,060) — nghĩa là một phần đáng kể của "±std liên-seed"
đang báo cáo thực ra là nhiễu khởi tạo chưa kiểm soát.

**Sửa** (`training/train.py::seed_everything`, gọi TRƯỚC `build_encoder`):
gieo `random`, `numpy` và `torch`; truyền `seed` cho controller đánh giá.
Kiểm chứng: hai lần chạy cùng seed cho kết quả **trùng khớp từng episode**.

**Xuất xứ của `results_v4/`** (phải ghi rõ trong luận văn): Easy chạy xong
trước khi có bản sửa này nên chưa gieo seed; Medium chạy vắt qua thời điểm sửa
nên hỗn hợp; Hard chạy hoàn toàn sau khi sửa nên đã gieo. Muốn có một bộ tái
lập được hoàn toàn thì chạy lại Easy (~4,7 giờ) và Medium (~12,7 giờ) sau khi
Hard xong.

### Kết quả Easy của `results_v4` (27/27)

| Phương pháp | VDR | Coverage | TTFD |
|---|---|---|---|
| Random Walk | 0,433 | 0,400 | 134,8 |
| Greedy | 0,673 | 0,647 | 22,5 |
| MADDPG | 0,149 ± 0,016 | 0,121 | 268,6 |
| MAPPO-MLP | 0,860 ± 0,060 | 0,743 | 12,5 |
| **MAPPO-GCN** | **0,925 ± 0,016** | **0,780** | **9,9** |
| MAPPO-GAT | 0,902 ± 0,027 | 0,756 | 12,7 |
| GNN-MAPPO | 0,889 ± 0,049 | 0,756 | 10,7 |

RQ1 (Easy): GCN vượt MLP p = 0,032 (d = +1,49) nhưng **không còn đạt sau
Bonferroni** — độ lệch chuẩn của MLP tăng từ 0,020 lên 0,060 do nhiễu khởi tạo
chưa gieo seed, làm giảm lực thống kê. Đây là lý do trực tiếp để chạy lại Easy
với seed đã gieo.

---

[← Giai đoạn 12](giai-doan-12-easy-voi-r-comm-26-do-thi-co-tac-dung-nhung-khong-phai-dgat-.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 14 →](giai-doan-14-gop-v3-v4-rq1-dat-y-nghia-thong-ke-o-ca-hai-kich-ban-06-09-2.md)
