[← Giai đoạn 7](giai-doan-07-kiem-chung-value-coef-0-1-o-quy-mo-day-du-108-luot.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 9 →](giai-doan-09-chay-lai-toan-bo-tren-moi-truong-da-sua-results-v2-04-09-202.md)

# Giai đoạn 8 — Rà soát toàn bộ code (03/09/2026): lỗi đánh giá `argmax`

Người thực hiện đặt lại hai câu hỏi: (1) vì sao mô hình đồ thị vẫn chưa vượt
học tăng cường thông thường, (2) vì sao mô phỏng trên web không thể hiện được
việc tìm nạn nhân, tránh vật cản hay giao tiếp giữa UAV. Rà soát toàn bộ
pipeline cho thấy **cả hai có chung một nguyên nhân**.

### Phát hiện: `MAPPOController.act` chạy `argmax`, không lấy mẫu

`evaluation/controllers.py` roll-out chính sách bằng `logits.argmax(dim=-1)`
trong khi PPO tối ưu chính sách ngẫu nhiên. Đo trên
`results_stage7_argmax_eval/easy/gnn_mappo/seed_0/policy.pt`:

| | |
|---|---|
| Entropy chính sách | 1,449 nats (đều = 2,565) |
| Hành động `argmax` chọn | **11 (FAST_FORWARD) ở 480/480 lượt** |
| Phân bố khi lấy mẫu | a11 59%, a7 16%, còn lại trải đều |

`FAST_FORWARD` bay theo `heading` đã lưu và không bao giờ đổi `heading`, nên
`argmax` biến chính sách thành trạng thái hấp thụ: cả đàn bay thẳng một hướng,
đâm vào biên bản đồ, rồi đứng yên tới hết episode. Quỹ đạo đã lưu xác nhận:
Easy toạ độ không đổi từ t=50 đến t=150 (coverage đứng ở 0,121); Medium không
đổi từ t=50 đến t=500 (coverage 0,064).

Đây cũng là lý do replay không thể hiện hành vi tìm kiếm: **trong quỹ đạo đó
thực sự không có hành vi tìm kiếm nào**.

### Sửa và đánh giá lại (không huấn luyện lại)

`scripts/reevaluate.py` nạp lại toàn bộ checkpoint và tính lại theo giao thức
lấy mẫu. 90/90 lượt có checkpoint đã chạy lại (15 lượt MADDPG bị bỏ qua vì
`train.py` chưa từng lưu checkpoint cho MADDPG).

**VDR trung bình ± std (n=5 seed)**:

| Phương pháp | Easy (argmax → lấy mẫu) | Medium | Hard |
|---|---|---|---|
| Random Walk | 0,369 → 0,410 ± 0,078 | 0,183 ± 0,018 | 0,081 ± 0,024 |
| Greedy | 0,437 → 0,451 ± 0,027 | 0,182 ± 0,025 | 0,087 ± 0,028 |
| MADDPG (*) | 0,147 (chưa chạy lại) | 0,027 | 0,019 |
| MAPPO-MLP | 0,167 → **0,546 ± 0,177** | 0,171 ± 0,071 | **0,109 ± 0,072** |
| MAPPO-GCN | 0,189 → **0,496 ± 0,084** | 0,166 ± 0,046 | 0,079 ± 0,031 |
| MAPPO-GAT | 0,148 → **0,463 ± 0,089** | 0,180 ± 0,014 | 0,082 ± 0,019 |
| **GNN-MAPPO** | 0,188 → **0,582 ± 0,052** | **0,200 ± 0,066** | 0,066 ± 0,019 |

(*) MADDPG chưa so sánh được — cần huấn luyện lại, lưu checkpoint đã bật.

Thứ hạng đảo ngược hoàn toàn. Trước đây **mọi phương pháp học đều thua Random
Walk** — dấu hiệu lẽ ra phải được coi là bằng chứng lỗi pipeline chứ không
phải "chưa hội tụ". Ở Easy, GNN-MAPPO nay dẫn đầu cả VDR, Coverage (0,556) và
TTFD (47,4 bước so với 113,1 của Greedy), vượt cả hai heuristic ở mức có ý
nghĩa thống kê (Welch: p = 0,002 vs Greedy, p = 0,005 vs Random Walk, Cohen's
d = +3,18 và +2,61). So với MAPPO-MLP vẫn chưa tách được (p = 0,68).

Ở Medium GNN-MAPPO cao nhất nhưng không tách khỏi bất kỳ đối chứng nào
(p ≥ 0,31). Ở Hard thấp hơn MAPPO-MLP (0,066 vs 0,109), chênh lệch không có ý
nghĩa (p ≥ 0,22) — ngân sách 200k bước trên 200×200 với 16 UAV là quá nhỏ.

Kết quả cũ lưu tại `results_stage7_argmax_eval/`.

### Bốn lỗi thiết kế khiến so sánh GNN vs MLP không có ý nghĩa khoa học

Ngay cả sau khi sửa khâu đánh giá, thí nghiệm vẫn chưa trả lời được RQ1:

1. **Bản đồ niềm tin toàn cục dùng chung.** Chỉ có một `BeliefMap` cho cả đàn;
   `BeliefMap.sync` chưa từng được gọi ở bất kỳ đâu trong repo. Đồ thị giao
   tiếp không mang thông tin nào mà một UAV chưa có sẵn.
2. **Quan sát đã chứa sẵn 30 chiều hàng xóm K=5**, kể cả cho MAPPO-MLP —
   baseline mà Bảng 1 định nghĩa là "không có cấu trúc đồ thị".
3. **Thưởng toàn cục + advantage toàn cục.** Mọi UAV nhận cùng một advantage ở
   mỗi bước, nên không có gradient nào phân biệt UAV vừa tìm ra nạn nhân với
   UAV đang lơ lửng ở góc đã quét. Khớp với quan sát Giai đoạn 6: tăng lên 2
   triệu bước không cải thiện gì.
4. **Quan sát/edge feature không chuẩn hoá + actor và critic dùng chung
   encoder** (`value_coef × value_loss` lớn hơn `|actor_loss|` 22–26× — chính
   Giai đoạn 6 đã đo được; Giai đoạn 7 thử hạ `value_coef` nhưng cách sửa đúng
   là tách encoder và chuẩn hoá return).

### Đã cài đặt (có cờ bật/tắt, `--legacy-env` tái lập Giai đoạn 6/7)

`local_belief`, `include_neighbor_obs=False`, `per_agent_reward`,
`normalize_obs`, `normalize_edge_attr`, `per_agent_critic`,
`separate_critic_encoder`, `normalize_returns`, `anneal_lr`, GAE theo từng UAV,
hidden state riêng cho critic, `alpha` 1,0 → 20,0, khởi tạo actor gain 0,01.
`obs_dim` đổi 116 → 86.

Hiệu ứng đo được (Easy, MAPPO-MLP): entropy giảm còn **1,33 sau 21k bước**, so
với code cũ vẫn ở 2,32 sau 500k bước; `mean_reward` tăng đơn điệu 0,012 →
0,085. Chi tiết: [REVIEW_2026-09-03.md](REVIEW_2026-09-03.md).

Kiểm thử: `python -m scripts.selftest_review_fixes` (25 kiểm tra, tất cả pass).

### Dữ liệu replay

`get_render_state()` nay xuất thêm `comm_edges` (kèm cường độ tín hiệu),
`action`/`blocked` từng UAV, `detected_by`/`detected_at_step` từng nạn nhân,
và `terrain_new` (ô mới bị chặn do lũ lan ở kịch bản Hard).
`web/components/ReplayCanvas.tsx` vẽ liên kết truyền tin, vòng đỏ khi UAV bị
vật cản chặn, xung sáng + đường nối tới UAV phát hiện nạn nhân, và có 5 nút
bật/tắt lớp hiển thị.

### Chạy thử trên môi trường đã sửa (`results_v2_reviewed/`)

Easy, 150k bước (30% ngân sách cũ), 3 seed, tất cả cờ sửa lỗi bật:

| Phương pháp | VDR cũ (500k bước) | VDR mới (150k bước) | Coverage | TTFD |
|---|---|---|---|---|
| MAPPO-MLP | 0,546 | **0,764 ± 0,054** | 0,692 | 19,6 |
| MAPPO-GCN | 0,496 | **0,793 ± 0,024** | 0,705 | 12,5 |
| GNN-MAPPO | 0,582 | **0,789 ± 0,037** | 0,669 | 19,1 |

Hai kết luận, cả hai cần báo cáo trung thực:

1. **Các sửa lỗi cấu trúc cải thiện rất mạnh cho mọi kiến trúc** — VDR tăng từ
   0,50–0,58 lên 0,76–0,79 với ngân sách nhỏ hơn 3,3 lần.
2. **Nhưng GNN-MAPPO vẫn không tách khỏi MLP ở Easy** (p = 0,55, d = +0,54;
   so với GCN p = 0,90). Lần này MAPPO-MLP thực sự không có thông tin hàng xóm
   nào (`include_neighbor_obs=False`) và bản đồ niềm tin chỉ lan qua liên kết
   vô tuyến — tức đúng thiết lập mà đồ thị lẽ ra phải thắng.

Cách đọc hợp lý: Easy (4 UAV, 50×50) là bài toán phối hợp quá dễ. Giả thuyết
cần kiểm chứng — chưa phải kết luận — là lợi thế của đồ thị chỉ xuất hiện ở N
lớn hơn (Medium/Hard) và trong thử nghiệm zero-shot. **Cần chạy Medium và Hard
đủ 5 seed trước khi kết luận về RQ1.**

### Kết quả âm tính: đưa `heading` vào quan sát (`results_v3_heading/`)

`FAST_FORWARD` bay theo `heading` đã lưu, không đổi hướng được, và `heading`
không có trong quan sát. Giả thuyết: đó là lý do `argmax` luôn thoái hoá. Đã
cài cờ `include_heading_obs` (obs_dim 86 → 88) và huấn luyện lại MAPPO-MLP
Easy 150k bước × 2 seed:

| | VDR | Coverage |
|---|---|---|
| Không heading | 0,764 | 0,692 |
| Có heading | 0,777 | 0,667 |

Chênh lệch trong nhiễu giữa seed, và roll-out `argmax` của chính sách mới
**vẫn** chọn hành động 11 ở 800/800 lượt, vẫn đứng yên ở biên (coverage 0,121,
0/5 nạn nhân). **Giả thuyết bị bác bỏ.**

Lý do sâu hơn: `FAST_FORWARD` quét 2 ô/bước nên là hành động giá trị cao nhất ở
gần như mọi trạng thái, còn rẽ chỉ đáng giá thỉnh thoảng — điều một chính sách
tất định không biểu diễn được. Cách sửa đúng nằm ở **không gian hành động**
(cho 8 hành động di chuyển một biến thể "nhanh", hoặc bỏ hành động 11), là sai
lệch so với `Discrete(13)` của Bảng 3 nên cần trao đổi với người hướng dẫn.
Cờ được giữ nhưng **mặc định tắt**.

---

[← Giai đoạn 7](giai-doan-07-kiem-chung-value-coef-0-1-o-quy-mo-day-du-108-luot.md) · [↑ Mục lục](../EXPERIMENT_LOG.md) · [Giai đoạn 9 →](giai-doan-09-chay-lai-toan-bo-tren-moi-truong-da-sua-results-v2-04-09-202.md)
