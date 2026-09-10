"""Sinh hình minh hoạ cho Chương 5 (kiến trúc, luồng thông tin, kết quả v4).

    python -m scripts.make_chapter5_figures  ->  thesis proposals/figures_ch5/*.png

Số liệu kết quả được đọc trực tiếp từ `web/public/data/*/comparison.json`
(bản xuất từ results_v4 ngày 10/09/2026, Giai đoạn 20 — Hard n = 10) và
`results_v4/*/zero_shot_all.json`; các con số của mục "tiến trình sửa lỗi"
được chép từ EXPERIMENT_LOG.md (Giai đoạn 6, 8, 9, 12, 13, 15) vì các bộ
kết quả trung gian nằm ở nhiều thư mục khác nhau.
"""
from __future__ import annotations

import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join("thesis proposals", "figures_ch5")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

METHOD_LABEL = {"random_walk": "Random Walk", "greedy": "Greedy", "maddpg": "MADDPG",
                "mappo_mlp": "MAPPO-MLP", "mappo_gcn": "MAPPO-GCN",
                "mappo_gat": "MAPPO-GAT\n(không GRU)", "gnn_mappo": "GNN-MAPPO\n(DGAT, đề xuất)"}
COLOR = {"random_walk": "#9e9e9e", "greedy": "#616161", "maddpg": "#c98a2b", "mappo_mlp": "#e07b39",
         "mappo_gcn": "#1f77b4", "mappo_gat": "#4c9fd6", "gnn_mappo": "#0b3d91"}
SC_LABEL = {"easy": "Easy (N=4, 50×50)", "medium": "Medium (N=8, 100×100)", "hard": "Hard (N=16, 200×200)"}


def box(ax, x, y, w, h, text, fc="#f4f6fa", ec="#2b3a55", lw=1.2, fs=8.2, style="round,pad=0.02,rounding_size=0.02", bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, dashed=False, color="#2b3a55", lw=1.3):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=lw,
                        color=color, linestyle="--" if dashed else "-")
    ax.add_patch(a)


# ----------------------------------------------------------------- Hình 5.1
def fig_architecture():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(27, 97.5, "PHA THỰC THI PHÂN TÁN\n(lặp cho mỗi UAV i, mỗi bước t — chỉ dùng thông tin cục bộ)",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0b3d91")
    ax.text(77, 97.5, "PHA HUẤN LUYỆN TẬP TRUNG\n(chỉ tồn tại khi huấn luyện — mũi tên nét đứt)",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#8a1c1c")
    ax.plot([53, 53], [1, 93], ls=":", color="#888")

    F = 7.6
    box(ax, 5, 84, 45, 8, "Môi trường SAR (PettingZoo ParallelEnv): lưới W×H, M nạn nhân, vật cản,\n"
        "bản đồ niềm tin RIÊNG của từng UAV, hợp nhất theo thành phần liên thông của đồ thị r_comm",
        fc="#eef7ee", fs=F)
    box(ax, 5, 70, 21, 10, "Quan sát cục bộ oᵢᵗ (86 chiều)\nself_state 7 · local_map 5×5×3\ntask_state 4 — chuẩn hoá [-1,1]\nKHÔNG chứa thông tin hàng xóm", fs=F)
    box(ax, 29, 70, 21, 10, "Đồ thị động Gᵗ = (V, E, X, A)\ncạnh (i,j) ⇔ d(i,j) ≤ r_comm\nr_comm = 26 / 34 / 46\neᵢⱼ = [Δp, Δv, dist, q] (6 chiều)", fs=F)
    arrow(ax, 15.5, 84, 15.5, 80); arrow(ax, 39.5, 84, 39.5, 80)

    box(ax, 5, 36, 45, 30, "", fc="#fdf6e3", ec="#b7791f", lw=1.6)
    ax.text(27.5, 63.5, "Bộ mã hoá DGAT (mục 1.5.3) — 57.408 tham số", ha="center", fontsize=8.6,
            fontweight="bold", color="#7a4f00")
    box(ax, 8, 56, 39, 5.5, "Linear(86 → 64)", fc="white", fs=F)
    box(ax, 8, 46, 39, 8.5, "3 × [ GATv2Conv(64, K=4 head, edge_dim=6) → ELU → +residual → LayerNorm ]\n"
        "αᵢⱼ⁽ᵏ⁾ = softmax( aₖᵀ [ Wₖhᵢ ‖ Wₖhⱼ ‖ Uₖeᵢⱼ ] )      — công thức (1.2), có đặc trưng cạnh",
        fc="white", fs=7.2)
    box(ax, 8, 37.5, 39, 7, "GRUCell(64): hᵢᵗ = GRU(h̃ᵢᴸ'ᵗ, hᵢᵗ⁻¹) — công thức (1.3), 24.960 tham số\n"
        "bỏ khối này ⇒ biến thể MAPPO-GAT (không GRU) của Bảng 5", fc="white", fs=7.2)
    arrow(ax, 15.5, 70, 22, 66); arrow(ax, 39.5, 70, 33, 66)
    arrow(ax, 27.5, 56, 27.5, 54.5); arrow(ax, 27.5, 46, 27.5, 44.5)
    ax.plot([47, 49, 49], [42.5, 42.5, 39.5], color="#7a4f00", lw=1)
    ax.annotate("", xy=(47, 39.5), xytext=(49, 39.5), arrowprops=dict(arrowstyle="-|>", color="#7a4f00"))
    ax.text(49.6, 41, "hᵢᵗ⁻¹", fontsize=7, color="#7a4f00")

    box(ax, 13, 22, 29, 8, "Actor πθ(aᵢ | hᵢᵗ) = MLP(64 → 64 → 13)\nlớp cuối khởi tạo gain 0,01 (chính sách ban đầu gần đều)", fs=F)
    arrow(ax, 27.5, 36, 27.5, 30)
    box(ax, 13, 8, 29, 8, "aᵢᵗ ~ Categorical(13): 8 hướng · lên/xuống · hover\nfast_forward · return_base — LẤY MẪU khi đánh giá", fc="#eef7ee", fs=F)
    arrow(ax, 27.5, 22, 27.5, 16)
    arrow(ax, 2.5, 12, 2.5, 84, color="#3a7d44")
    ax.text(1.0, 48, "thực thi trong môi trường", rotation=90, va="center", ha="center", fontsize=7.2, color="#3a7d44")
    ax.plot([13, 2.5], [12, 12], color="#3a7d44", lw=1.2)

    R = "#8a1c1c"; RF = "#fbeeee"
    box(ax, 57, 72, 40, 8, "Bộ mã hoá RIÊNG cho critic (cùng kiến trúc, tham số riêng,\n"
        "trạng thái ẩn riêng h'ᵢ) — tách khỏi actor để value loss không lấn át", fc=RF, ec=R, fs=F)
    arrow(ax, 50, 76, 57, 76, dashed=True, color=R)
    box(ax, 57, 58, 40, 9.5, "Critic tập trung, bất biến hoán vị, theo từng UAV\nVᵢ = f_φ( [ h'ᵢ ‖ mean-Pool{h'₁ … h'ₙ} ] )\n"
        "một giá trị cho MỖI UAV; N tuỳ ý ⇒ chạy được zero-shot theo N", fc=RF, ec=R, fs=F)
    arrow(ax, 77, 72, 77, 67.5, dashed=True, color=R)
    box(ax, 57, 44, 40, 9.5, "Thưởng theo từng UAV: 0,7·cá nhân + 0,3·trung bình đội\n"
        "r = 20·r_cover + 5·r_victim − 0,05·r_collision − 0,01·r_energy\n"
        "GAE (γ=0,99, λ=0,95) theo từng UAV · chuẩn hoá return (Welford)", fc=RF, ec=R, fs=F)
    arrow(ax, 77, 58, 77, 53.5, dashed=True, color=R)
    box(ax, 57, 30, 40, 9.5, "Cập nhật MAPPO: PPO-clip ε=0,2 · 4 epoch · minibatch 256\n"
        "Adam lr 3·10⁻⁴ → 0 và entropy 0,01 → 0,001 (giảm tuyến tính)\n"
        "truncated BPTT dài 8 bước cho encoder hồi tiếp (chỉ DGAT)", fc=RF, ec=R, fs=F)
    arrow(ax, 77, 44, 77, 39.5, dashed=True, color=R)
    arrow(ax, 42, 26, 57, 33, dashed=True, color=R); ax.text(48.5, 31, "log πθ(aᵢ|hᵢ)", fontsize=7, color=R)
    arrow(ax, 57, 31, 50, 48, dashed=True, color=R, lw=1.1); ax.text(51.2, 36.5, "∇θ, ∇φ", fontsize=7, color=R, rotation=65)

    box(ax, 57, 8, 40, 15.5, "Bốn biến thể của Bảng 5 dùng CHUNG trainer này, chỉ khác encoder\n"
        "MAPPO-MLP  : Linear×3, không đồ thị — 13.888 tham số\n"
        "MAPPO-GCN  : GCNConv×3, trọng số cạnh = q — 18.432\n"
        "MAPPO-GAT  : DGAT bỏ GRU — 32.448\n"
        "GNN-MAPPO : DGAT đầy đủ — 57.408", fc="#f4f6fa", fs=F)
    fig.savefig(os.path.join(OUT, "fig5_1_kien_truc_gnn_mappo.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- Hình 5.2
def fig_info_flow():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, title in zip(axes, ["(a) Trước rà soát (Giai đoạn 1–7)", "(b) Sau rà soát (Giai đoạn 8–15, results_v4)"]):
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold", color="#8a1c1c" if "Trước" in title else "#0b3d91")
    F = 6.9; R = "#8a1c1c"; RF = "#fbeeee"; G = "#3a7d44"; GF = "#eef7ee"
    a = axes[0]
    box(a, 15, 80, 70, 12, "MỘT bản đồ niềm tin TOÀN CỤC dùng chung cho cả đàn\n(BeliefMap.sync chưa từng được gọi)", fc=RF, ec=R, fs=F)
    for x in (14, 38, 62, 86):
        box(a, x - 7, 52, 14, 10, "UAV", fc="white", fs=F)
        arrow(a, x, 80, x, 62, color=R)
    a.text(50, 70, "đồ thị không mang thông tin nào mà UAV chưa có sẵn", ha="center", fontsize=7.5, color=R, style="italic")
    box(a, 3, 30, 94, 16, "quan sát 116 chiều: đã chứa sẵn 30 chiều về K=5 hàng xóm\n(kể cả với MAPPO-MLP — baseline 'không đồ thị')\n"
        "một thưởng vô hướng r chung cho cả đàn\n⇒ cùng một advantage cho mọi UAV, không gán công", fc=RF, ec=R, fs=F)
    box(a, 3, 4, 94, 22, "đánh giá bằng argmax ⇒ 100% chọn FAST_FORWARD\n⇒ cả đàn bay thẳng, đứng yên ở biên bản đồ\n"
        "r_comm = 10 ⇒ 56% (Easy) / 35% (Medium) số bước đồ thị KHÔNG có cạnh\n"
        "50 / 86 chiều quan sát là hằng số (cửa sổ niềm tin nằm trong đĩa cảm biến)\n"
        "torch.manual_seed chưa từng được gọi ⇒ không tái lập được", fc=RF, ec=R, fs=F)

    b = axes[1]
    for x in (14, 38, 62, 86):
        box(b, x - 9, 76, 18, 16, "UAV\nbản đồ niềm tin\nriêng", fc=GF, ec=G, fs=F)
    b.plot([23, 29], [84, 84], color="#0b3d91", lw=2.2); b.text(26, 87, "≤ r_comm", ha="center", fontsize=6.8, color="#0b3d91")
    b.plot([47, 53], [84, 84], color="#0b3d91", lw=2.2)
    b.text(74, 84, "✕", ha="center", va="center", fontsize=9, color="#888"); b.text(74, 87.5, "ngoài tầm", ha="center", fontsize=6.8, color="#888")
    b.text(50, 69, "hợp nhất bản đồ theo thành phần liên thông của Gᵗ (một hop mỗi bước)\n"
           "⇒ thông tin về vùng đã quét CHỈ đến qua liên kết vô tuyến", ha="center", fontsize=7.5, color="#0b3d91", style="italic")
    box(b, 3, 34, 94, 26, "quan sát 86 chiều, chuẩn hoá; KHÔNG có khối hàng xóm\n⇒ MAPPO-MLP thật sự là baseline 'không đồ thị'\n"
        "local_map: kênh occupancy mịn (bán kính 2) + kênh prob/visited thô,\nmỗi ô gộp khối 5×5 ⇒ cửa sổ 25×25 vượt ra ngoài đĩa cảm biến\n"
        "thưởng theo từng UAV (ô CHÍNH nó quét, nạn nhân CHÍNH nó tìm) + 30% trung bình đội\n"
        "critic Vᵢ theo từng UAV · encoder riêng · return chuẩn hoá · LR & entropy giảm tuyến tính", fc=GF, ec=G, fs=F)
    box(b, 3, 4, 94, 26, "đánh giá bằng LẤY MẪU a ~ πθ — đúng phân phối mà PPO đã tối ưu\n"
        "r_comm = 26 / 34 / 46 (bậc kỳ vọng ≈ 2,5 với đàn phân tán đều)\n⇒ đồ thị động: không rỗng, không đầy đủ\n"
        "truncated BPTT 8 bước cho GRU · torch / numpy / random đều được gieo seed\n"
        "34 kiểm thử tự động chống tái diễn (scripts/selftest_review_fixes.py)", fc=GF, ec=G, fs=F)
    fig.savefig(os.path.join(OUT, "fig5_2_luong_thong_tin_truoc_sau.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- Hình 5.4 (bar)
def load_comparison(sc):
    return json.load(open(f"web/public/data/{sc}/comparison.json", encoding="utf-8"))


def fig_bars():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    order = ["random_walk", "greedy", "maddpg", "mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo"]
    for ax, sc in zip(axes, ["easy", "medium", "hard"]):
        d = load_comparison(sc)["methods"]
        means = [d[m]["metrics"]["victim_detection_rate"]["mean"] for m in order]
        stds = [d[m]["metrics"]["victim_detection_rate"]["std"] if len(d[m]["metrics"]["victim_detection_rate"]["values"]) > 1 else 0 for m in order]
        xs = range(len(order))
        bars = ax.bar(xs, means, yerr=stds, capsize=3, color=[COLOR[m] for m in order], edgecolor="black", lw=0.5)
        for xi, mu in zip(xs, means):
            ax.text(xi, mu + 0.02, f"{mu:.3f}", ha="center", fontsize=7)
        ax.set_xticks(list(xs)); ax.set_xticklabels([METHOD_LABEL[m].replace("\n", "\n") for m in order], rotation=45, ha="right", fontsize=7)
        ax.set_title(SC_LABEL[sc], fontsize=9.5); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Victim Detection Rate (mean ± std liên-seed)")
    fig.suptitle("So sánh baseline — results_v4 (Easy 500k · Medium 800k · Hard 1M bước; 5 seed Easy/Medium, 10 seed Hard; toàn bộ đã gieo seed)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_4_so_sanh_vdr.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- Hình 5.6 (zero-shot)
def fig_zero_shot():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    train_n = {"easy": 4, "medium": 8, "hard": 16}
    for ax, sc in zip(axes, ["easy", "medium", "hard"]):
        d = json.load(open(f"results_v4/{sc}/zero_shot_all.json", encoding="utf-8"))
        for m in ["mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo"]:
            seeds = d[m]
            Ns = sorted({int(n) for s in seeds.values() for n in s})
            ys = [st.mean(seeds[s][str(N)]["victim_detection_rate"]["mean"] for s in seeds if str(N) in seeds[s]) for N in Ns]
            ax.plot(Ns, ys, marker="o", ms=4, color=COLOR[m], label=METHOD_LABEL[m].replace("\n", " "), lw=1.6)
        ax.axvline(train_n[sc], color="#888", ls="--", lw=1); ax.text(train_n[sc], 0.02, f" N huấn luyện = {train_n[sc]}", fontsize=7, color="#555")
        ax.set_title(SC_LABEL[sc], fontsize=9.5); ax.set_xlabel("Số UAV khi kiểm tra (N), bản đồ giữ nguyên"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    axes[0].set_ylabel("VDR (3 seed × 10 episode)"); axes[2].legend(fontsize=7, loc="lower right")
    fig.suptitle("Zero-shot fixed-map: chính sách huấn luyện ở N gốc, đánh giá ở N khác mà không huấn luyện lại", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_7_zero_shot.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- Hình 5.3 (audit trail)
def fig_progress():
    stages = ["GĐ6\nargmax,\nmôi trường cũ", "GĐ8\nlấy mẫu\n(cùng ckpt)", "GĐ8\n+4 sửa lỗi\n(150k bước)",
              "GĐ9 v2\n+obs hằng số\n(500k)", "GĐ12 v3\nr_comm 10→26", "GĐ13/15 v4\n+BPTT,\nseed 1 phần", "Chạy lại\nseed đầy đủ\n(07/09)"]
    series = {
        "mappo_mlp": [0.167, 0.546, 0.764, 0.855, 0.877, 0.860, 0.881],
        "mappo_gcn": [0.189, 0.496, 0.793, 0.873, 0.942, 0.925, 0.930],
        "gnn_mappo": [0.188, 0.582, 0.789, 0.847, 0.890, 0.889, 0.884],
        "greedy":    [0.437, 0.451, None,  0.716, 0.673, 0.673, 0.678],
    }
    fig, ax = plt.subplots(figsize=(11, 4.2))
    xs = range(len(stages))
    for m, ys in series.items():
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=COLOR[m], lw=1.8, label=METHOD_LABEL[m].replace("\n", " "))
        for x, y in pts:
            ax.text(x, y + (0.025 if m != "gnn_mappo" else -0.05), f"{y:.3f}", ha="center", fontsize=6.8, color=COLOR[m])
    ax.set_xticks(list(xs)); ax.set_xticklabels(stages, fontsize=7.5)
    ax.set_ylabel("VDR — kịch bản Easy"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Dấu vết kiểm toán: VDR của cùng một phương pháp thay đổi theo từng lỗi được sửa (Easy, trung bình liên-seed)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_3_tien_trinh_sua_loi.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture(); fig_info_flow(); fig_progress(); fig_bars(); fig_zero_shot()
    print("written to", OUT, sorted(os.listdir(OUT)))
