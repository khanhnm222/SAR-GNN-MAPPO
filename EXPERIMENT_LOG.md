# Nhật ký thực nghiệm — GNN-MAPPO cho SAR

> Ghi lại toàn bộ các lần chạy thực nghiệm, lỗi phát hiện, điều chỉnh và kết quả
> theo đúng trình tự thời gian, để phục vụ Chương 5 (thực nghiệm & đánh giá) và
> làm minh chứng cho quá trình debug/hiệu chỉnh — không chỉ báo cáo con số cuối
> cùng mà cả lý do vì sao con số thay đổi giữa các lần chạy.

**Cấu trúc:** mỗi giai đoạn là một file riêng trong [`experiment_log/`](experiment_log/),
đặt tên `giai-doan-NN-....md`. File này chỉ là mục lục — không chứa chi tiết kỹ
thuật, bảng số hay đoạn code. Các file giai đoạn có link điều hướng (← trước ·
↑ mục lục · sau →) ở đầu và cuối để đọc tuần tự mà không cần quay lại đây. Toàn
bộ nội dung của bản gộp một file trước đây (đến 10/09/2026) được bảo toàn
nguyên vẹn, chỉ chia lại theo giai đoạn — không có đoạn nào bị bỏ sót.

## Tóm tắt các giai đoạn

| # | Giai đoạn | Ngày | Kết quả / vấn đề chính | Trạng thái |
|---|---|---|---|---|
| [1](experiment_log/giai-doan-01-poc-ban-dau.md) | POC ban đầu | — | Ngân sách quá nhỏ (4–10k bước) để bất kỳ phương pháp học nào vượt heuristic; phát hiện lỗi tie-break của Greedy | Xong |
| [2](experiment_log/giai-doan-02-loi-va-cham-tai-diem-sinh-uav.md) | Lỗi va chạm tại điểm sinh UAV | — | UAV sinh quá gần nhau → va chạm giả ngay bước đầu | Xong |
| [3](experiment_log/giai-doan-03-nang-ngan-sach-len-quy-mo-nghien-cuu-phat-hien-loi-moi.md) | Nâng ngân sách lên quy mô nghiên cứu | — | 200k–500k bước; không phương pháp học nào vượt Random Walk → phát hiện hàm thưởng có lỗi | Xong (phát hiện lỗi mới) |
| [4](experiment_log/giai-doan-04-chan-doan-va-sua-loi-hieu-chinh-ham-thuong.md) | Chẩn đoán và sửa lỗi hiệu chỉnh hàm thưởng | — | Phạt va chạm át tín hiệu nhiệm vụ (reward hacking); cân lại trọng số | Xong |
| [5](experiment_log/giai-doan-05-huan-luyen-lai-voi-ham-thuong-da-sua-hoan-tat.md) | Huấn luyện lại với hàm thưởng đã sửa | — | GNN-MAPPO chưa vượt trội baseline; n=2–3 seed chưa đủ lực thống kê | Xong |
| [6](experiment_log/giai-doan-06-5-seed-r-comm-thua-hon-gnn-mappo-ngan-sach-lon-easy.md) | 5 seed + r_comm thưa hơn + GNN-MAPPO 2M bước (Easy) | — | Khoảng cách với MAPPO-GCN thu hẹp; thêm bước không giúp cải thiện; vẫn chưa đạt ý nghĩa thống kê ở n=5 | Xong (108/108) |
| [7](experiment_log/giai-doan-07-kiem-chung-value-coef-0-1-o-quy-mo-day-du-108-luot.md) | Kiểm chứng `value_coef=0.1` ở quy mô đầy đủ | — | `value_coef=0.1` KHÔNG cải thiện GNN-MAPPO — khôi phục 0.5; zero-shot vẫn bền vững | Xong (108/108) |
| [8](experiment_log/giai-doan-08-ra-soat-toan-bo-code-03-09-2026-loi-danh-gia-argmax.md) | Rà soát toàn bộ code | 03/09/2026 | **Lỗi 1**: đánh giá bằng `argmax` trên chính sách entropy cao → mọi chính sách "thoái hoá" như nhau khi đo | Xong |
| [9](experiment_log/giai-doan-09-chay-lai-toan-bo-tren-moi-truong-da-sua-results-v2-04-09-202.md) | Chạy lại toàn bộ trên môi trường đã sửa (`results_v2/`) | 04/09/2026 | 54 lượt Easy+Medium; giữa chừng phát hiện **lỗi 5**: 50/86 chiều quan sát là hằng số | Xong |
| [10](experiment_log/giai-doan-10-loi-thu-sau-do-thi-giao-tiep-gan-nhu-rong-04-09-2026.md) | Lỗi thứ sáu: đồ thị giao tiếp gần như RỖNG | 04/09/2026 | Đo dưới chính sách phân tán thật: `r_comm` cũ khiến đồ thị rỗng 35–56% số bước | Xong |
| [11](experiment_log/giai-doan-11-medium-voi-r-comm-34-go-bo-nhieu-cau-tra-loi-khong-doi-05-09.md) | Medium với `r_comm=34` | 05/09/2026 | Gỡ bỏ nhiễu đồ thị rỗng; câu trả lời RQ1 không đổi so với trước | Xong |
| [12](experiment_log/giai-doan-12-easy-voi-r-comm-26-do-thi-co-tac-dung-nhung-khong-phai-dgat-.md) | Easy với `r_comm=26` | 05/09/2026 | Đồ thị **CÓ** tác dụng (GCN > MLP) nhưng **không phải DGAT** là kiến trúc hưởng lợi nhiều nhất | Xong |
| [13](experiment_log/giai-doan-13-bptt-that-cho-gru-va-loi-thu-bay-seed-05-09-2026.md) | BPTT thật cho GRU, lỗi thứ bảy: seed | 05/09/2026 | GRU trước đó chưa từng được huấn luyện đúng (BPTT=1); **lỗi 7**: torch chưa gieo seed | Xong |
| [14](experiment_log/giai-doan-14-gop-v3-v4-rq1-dat-y-nghia-thong-ke-o-ca-hai-kich-ban-06-09-2.md) | Gộp v3 + v4: RQ1 đạt ý nghĩa ở cả hai kịch bản | 06/09/2026 | Gộp các đợt chạy độc lập cùng mã nguồn để nâng n; GCN vs MLP đạt ý nghĩa ở Easy và Medium | Xong |
| [15](experiment_log/giai-doan-15-ma-tran-day-du-ba-kich-ban-results-v4-07-09-2026.md) | Ma trận đầy đủ ba kịch bản (`results_v4/`) | 07/09/2026 | 81 lượt (Easy/Medium/Hard), 52,3 giờ; cộng zero-shot cả ba kịch bản | Xong |
| [16](experiment_log/giai-doan-16-gieo-seed-toan-bo-va-mot-ket-luan-phai-dieu-chinh-07-09-2026.md) | Gieo seed toàn bộ, một kết luận phải điều chỉnh | 07/09/2026 | Chạy lại 41 lượt chưa gieo seed; kết luận "GCN đạt Bonferroni ở Medium" đổi khi gieo seed lại | Xong |
| [17](experiment_log/giai-doan-17-ra-soat-lan-cuoi-nhung-gi-con-lai-07-09-2026.md) | Rà soát lần cuối: những gì còn lại | 07/09/2026 | Bốn lỗi giao thức: nhiễu số tham số, heuristic chấm sai tập test, 1 seed, seed trượt vào tập test | Xong |
| [18](experiment_log/giai-doan-18-hai-kiem-chung-cuoi-08-09-2026.md) | Hai kiểm chứng cuối | 08/09/2026 | Ngân sách 2× ở Easy: khoảng cách GCN–DGAT không đóng lại → do kiến trúc, không do ngân sách | Xong |
| [19](experiment_log/giai-doan-19-lap-lai-hard-seed-5-9-va-ban-nhap-chuong-5-08-09-2026.md) | Lặp lại Hard (seed 5–9), bản nháp Chương 5 | 08/09/2026 | Bắt đầu đợt lặp lại độc lập 35 lượt cho Hard; viết bản nháp đầy đủ Chương 5 | Xong (khởi chạy) |
| [20](experiment_log/giai-doan-20-hard-n-10-dot-lap-lai-seed-5-9-hoan-tat-10-09-2026.md) | Hard n = 10: đợt lặp lại hoàn tất | 10/09/2026 | 35/35 lượt xong; mọi kết luận Hard đứng vững hoặc mạnh lên khi lặp lại độc lập | Xong |
| [21](experiment_log/giai-doan-21-lap-lai-medium-easy-seed-5-9-va-zero-shot-density-20-09-2026.md) | Lặp lại Medium + Easy (seed 5–9), zero-shot density-controlled | 20/09/2026 | 70 lượt xong → mọi kịch bản n = 10 hai đợt độc lập; RQ1 giữ vững (GCN > MLP ở cả ba, Bonferroni ở Easy/Hard), GCN > GAT > DGAT lặp lại 3×2; zero-shot fixed-map + density, 10 seed | Xong |
| [22](experiment_log/giai-doan-22-hard-ngan-sach-2x-gcn-dgat-2m-buoc-22-09-2026.md) | Hard ngân sách 2× (GCN, DGAT, 2M bước, 5 seed) | 22–26/09/2026 | Gấp đôi bước: GCN 0,438 → 0,465, DGAT 0,345 → 0,385; khoảng cách −0,092 → −0,080 (p = 0,012, 0/5 cặp) **không đóng lại**; DGAT-2M vẫn dưới GCN-1M và GAT-1M → thứ hạng là do kiến trúc. Hai lần DGAT chết vì Windows Update/Restart Manager | Xong |
| [23](experiment_log/giai-doan-23-ablation-ham-thuong-medium-gcn-26-09-2026.md) | Ablation hàm phần thưởng (Medium, GCN, bỏ từng thành phần, 5 seed) | 26/09/2026 | Bỏ α: VDR −0,067, Coverage −0,060 (5/5 seed); bỏ β: VDR không đổi, Coverage **+0,060** (p = 0,007); bỏ κ: VDR −0,075 qua shaping, Collision không tăng; bỏ δ: trung tính. Dòng "Thiếu" cuối của Bảng 5.4 đã đóng | Xong |

## Tổng quan kết quả cuối cùng

Bộ số liệu chính thức trích dẫn trong luận văn là `results_v4/` (210 lượt: 7
phương pháp × seed 0–9 × 3 kịch bản — hai đợt seed độc lập ở mọi kịch bản; zero-shot
fixed-map và density-controlled, 10 seed). Xem
[Giai đoạn 21](experiment_log/giai-doan-21-lap-lai-medium-easy-seed-5-9-va-zero-shot-density-20-09-2026.md)
cho kết luận cuối và [`CHUONG5_THUC_NGHIEM.md`](CHUONG5_THUC_NGHIEM.md) cho bản
trình bày đầy đủ dùng cho luận văn.

Mười một lỗi được phát hiện và sửa xuyên suốt Giai đoạn 8–17 (bảy lỗi làm hỏng
âm thầm mọi so sánh kiến trúc, bốn lỗi giao thức) — danh mục đầy đủ nằm ở
[Giai đoạn 8](experiment_log/giai-doan-08-ra-soat-toan-bo-code-03-09-2026-loi-danh-gia-argmax.md)
(Bảng lỗi 1–7) và
[Giai đoạn 17](experiment_log/giai-doan-17-ra-soat-lan-cuoi-nhung-gi-con-lai-07-09-2026.md)
(lỗi 8–11), cũng được tóm tắt lại ở Chương 5 mục 5.2.

## Phụ lục: thư mục kết quả theo từng giai đoạn (Giai đoạn 1–7)

Bảng dưới ghi lại thư mục kết quả cho các đợt chạy sớm (trước khi chuyển sang
quy ước đặt tên `results_v2/` → `results_v4/` từ Giai đoạn 9 trở đi); xem từng
file giai đoạn để biết thư mục kết quả của Giai đoạn 8 trở về sau.

| Thư mục | Giai đoạn | Ghi chú |
|---|---|---|
| `results_poc_scale/` | 1-2 | POC, đã sửa lỗi va chạm điểm sinh |
| `results_broken_reward/` | 3 | Quy mô nghiên cứu, **hàm thưởng còn lỗi** — không dùng để kết luận |
| `results_stage5_r_comm_dense/` | 5 | Hàm thưởng đã sửa, r_comm dày (15/20/25), n=2-3 seed |
| `results/` | 6 | r_comm thưa (10/10/10), n=5 seed, value_coef=0.5 — kết quả chính thức tại thời điểm đó |
| `results_longrun_easy_gnn_mappo/` | 6 | GNN-MAPPO/Easy, 2M bước, 3 seed (value_coef=0.5) |
| `results_stage7_vc01/` | 7 | Như `results/` nhưng value_coef=0.1 — tham khảo, không dùng làm số liệu chính |
| `results_stage7_vc01_longrun/` | 7 | GNN-MAPPO/Easy 2M bước với value_coef=0.1 |
