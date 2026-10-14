"""Sinh hình minh hoạ cho Chương 5 (kiến trúc, luồng thông tin, kết quả v4).

    python -m scripts.make_chapter5_figures  ->  thesis proposals/figures_ch5/*.png

Số liệu kết quả được đọc trực tiếp từ `web/public/data/*/comparison.json`
(bản xuất từ results_v4 ngày 10/09/2026, Giai đoạn 20 — Hard n = 10) và
`results_v4/*/zero_shot_all.json`; các con số của mục "tiến trình sửa lỗi"
được chép từ EXPERIMENT_LOG.md (Giai đoạn 6, 8, 9, 12, 13, 15) vì các bộ
kết quả trung gian nằm ở nhiều thư mục khác nhau.

Chuẩn trình bày (14/09/2026): vẽ ở bề rộng FIG_W = 6,3 in (bề rộng vùng chữ
trang A4), nhúng không thu nhỏ, chữ chính 12–13 pt, chữ phụ ≥ 10,5 pt; các
panel xếp dọc; biểu đồ 7 phương pháp dùng thanh ngang. Hai hình sơ đồ (5.1,
5.2) được rút gọn nội dung — chi tiết đã có ở Bảng 5.5 và mục 5.1.3 — để đủ
chỗ cho chữ 12 pt trong một trang.
"""
from __future__ import annotations

import json
import os
import statistics as st
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from evaluation.plots import METHOD_COLORS as COLOR, VI_TICKS, _vi, hbar_panel

OUT = os.path.join("thesis proposals", "figures_ch5")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
FIG_W = 6.3
DPI = 200

METHOD_LABEL = {"random_walk": "Random Walk", "greedy": "Greedy", "maddpg": "MADDPG",
                "mappo_mlp": "MAPPO-MLP", "mappo_gcn": "MAPPO-GCN",
                "mappo_gat": "MAPPO-GAT (không GRU)", "gnn_mappo": "MAPPO-DGAT"}
SC_TITLE = {"easy": "Easy (N = 4, 50×50, 500k bước)", "medium": "Medium (N = 8, 100×100, 800k bước)",
            "hard": "Hard (N = 16, 200×200, 1M bước)"}
NAVY, RED, GOLD, GREEN = "#0b3d91", "#8a1c1c", "#7a4f00", "#3a7d44"


def _fit_text(ax, w, h, text, fs, bold=False, min_fs=11.0):
    fig_w, fig_h = ax.figure.get_size_inches()
    bb = ax.get_position()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    w_in = w * fig_w * bb.width / (x1 - x0)
    h_in = h * fig_h * bb.height / (y1 - y0)
    f = fs
    while True:
        cpl = max(8, int(w_in * 0.96 / (f * 0.58 / 72) / (1.03 if bold else 1.0)))
        lines = []
        for ln in text.split("\n"):
            lines += textwrap.wrap(ln, cpl, break_long_words=False) or [""]
        need = len(lines) * f * 1.5 / 72
        if need <= h_in * 0.95 or f <= min_fs:
            return "\n".join(lines), f
        f -= 0.3


def box(ax, x, y, w, h, text, fc="#f4f6fa", ec="#2b3a55", lw=1.3, fs=12,
        style="round,pad=0.02,rounding_size=0.02", bold=False, color="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    if text:
        text, fs = _fit_text(ax, w, h, text, fs, bold)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=color, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, dashed=False, color="#2b3a55", lw=1.5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=lw,
                                 color=color, linestyle="--" if dashed else "-"))


# ----------------------------------------------------------------- Hình 5.1 (kiến trúc DGAT trong MAPPO, rút gọn)
def fig_architecture():
    fig, ax = plt.subplots(figsize=(FIG_W, 8.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 132); ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
    ax.text(50, 129.5, "PHA THỰC THI PHÂN TÁN — nét liền, mỗi UAV i, mỗi bước t",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color=NAVY)
    box(ax, 6, 117, 90, 9, "Môi trường SAR (PettingZoo ParallelEnv)\nbản đồ niềm tin riêng từng UAV, hợp nhất trong r_comm",
        fc="#eef7ee")
    box(ax, 6, 101, 42, 12.5, "Quan sát oᵢᵗ (86 chiều)\nself 7 · map 5×5×3 · task 4\nchuẩn hoá; không có hàng xóm", fs=11.5)
    box(ax, 54, 101, 42, 12.5, "Đồ thị động Gᵗ\ncạnh khi d(i, j) ≤ r_comm\neᵢⱼ = [Δp, Δv, dist, q]", fs=11.5)
    arrow(ax, 27, 117, 27, 114); arrow(ax, 75, 117, 75, 114)

    box(ax, 6, 62, 90, 35, "", fc="#fdf6e3", ec="#b7791f", lw=1.6)
    ax.text(51, 94, "Bộ mã hoá DGAT — 57.408 tham số", ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=GOLD)
    box(ax, 10, 85, 82, 5.5, "Linear(86 → 64)", fc="white")
    box(ax, 10, 74, 82, 9, "3 × GATv2Conv(64, K = 4 head, edge_dim = 6)\n+ ELU, residual, LayerNorm — công thức (1.2)", fc="white", fs=11.5)
    box(ax, 10, 63.5, 82, 9, "GRUCell(64): hᵢᵗ = GRU(h̃ᵢᵗ, hᵢᵗ⁻¹) — công thức (1.3)\nbỏ khối này ⇒ MAPPO-GAT (không GRU)", fc="white", fs=11.5)
    arrow(ax, 27, 101, 40, 97); arrow(ax, 75, 101, 62, 97)
    arrow(ax, 51, 85, 51, 83.3); arrow(ax, 51, 74, 51, 72.8)
    ax.plot([92, 94.5, 94.5], [70, 70, 65.5], color=GOLD, lw=1.2)
    ax.annotate("", xy=(92, 65.5), xytext=(94.5, 65.5), arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.2))
    ax.text(97.6, 68, "hᵢᵗ⁻¹", fontsize=11, color=GOLD, ha="center")

    box(ax, 18, 51, 64, 7.5, "Actor πθ(aᵢ | hᵢᵗ) = MLP(64 → 64 → 13)")
    arrow(ax, 51, 62, 51, 58.7)
    box(ax, 18, 41, 64, 7.5, "aᵢᵗ ~ Categorical(13) — lấy mẫu khi đánh giá", fc="#eef7ee")
    arrow(ax, 51, 51, 51, 48.7)
    ax.plot([18, 2.8, 2.8], [44.75, 44.75, 121.5], color=GREEN, lw=1.5)
    ax.annotate("", xy=(6, 121.5), xytext=(2.8, 121.5), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5))
    ax.text(1.5, 83, "thực thi trong môi trường", rotation=90, va="center", ha="center", fontsize=11, color=GREEN)

    ax.plot([2, 98], [37, 37], ls=":", color="#888", lw=1.3)
    ax.text(47, 34.3, "PHA HUẤN LUYỆN TẬP TRUNG — nét đứt, chỉ khi huấn luyện",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color=RED)
    RF = "#fbeeee"
    box(ax, 6, 16.5, 90, 14.5, "Encoder riêng cho critic — nhận Gᵗ, {oᵢᵗ} của toàn đàn\n"
        "Critic theo từng UAV: Vᵢ = f_φ([ h'ᵢ ‖ mean-Pool{h'} ])\n"
        "bất biến hoán vị ⇒ chạy được với N bất kỳ (zero-shot)", fc=RF, ec=RED, fs=11.5)
    box(ax, 6, 0.3, 90, 14.5, "Thưởng theo từng UAV (0,7 cá nhân + 0,3 đội) · GAE theo UAV\n"
        "chuẩn hoá return · PPO-clip ε = 0,2 · lr và entropy giảm dần\n"
        "BPTT 8 bước (DGAT) · ∇θ → encoder, actor · ∇φ → critic", fc=RF, ec=RED, fs=11.5)
    arrow(ax, 51, 16.5, 51, 15.2, dashed=True, color=RED)
    ax.plot([82, 98, 98], [54.75, 54.75, 7.5], ls="--", color=RED, lw=1.3)
    ax.annotate("", xy=(96, 7.5), xytext=(98, 7.5), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3))
    ax.text(99.4, 25, "log πθ", rotation=90, va="center", ha="center", fontsize=11, color=RED)
    fig.savefig(os.path.join(OUT, "fig5_1_kien_truc_gnn_mappo.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- Hình 5.2 (luồng thông tin trước/sau, rút gọn)
def fig_info_flow():
    fig, (a, b) = plt.subplots(2, 1, figsize=(FIG_W, 8.6), gridspec_kw={"height_ratios": [0.44, 0.56]})
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.005, top=0.97, hspace=0.12)
    RF = "#fbeeee"; GF = "#eef7ee"
    a.set_xlim(0, 100); a.set_ylim(0, 60); a.axis("off")
    a.set_title("(a) Trước rà soát (Giai đoạn 1–7)", fontsize=12.5, fontweight="bold", color=RED, loc="left")
    box(a, 12, 49, 76, 10, "MỘT bản đồ niềm tin toàn cục dùng chung\n(BeliefMap.sync không bao giờ được gọi)", fc=RF, ec=RED)
    for x in (14, 38, 62, 86):
        box(a, x - 7, 33, 14, 8, "UAV", fc="white")
        arrow(a, x, 49, x, 41, color=RED)
    a.text(50, 45, "đồ thị không mang thông tin nào mà UAV chưa có", ha="center", fontsize=11.5, color=RED, style="italic")
    box(a, 3, 16.5, 94, 13.5, "quan sát 116 chiều đã chứa 30 chiều về K = 5 hàng xóm\n(kể cả MAPPO-MLP — baseline 'không đồ thị')\n"
        "một thưởng chung ⇒ cùng một advantage cho mọi UAV", fc=RF, ec=RED, fs=11.5)
    box(a, 3, 0.5, 94, 13.5, "đánh giá bằng argmax ⇒ 100 % FAST_FORWARD, đàn đứng yên ở biên\n"
        "r_comm = 10 ⇒ đồ thị rỗng ở 56 % (Easy) / 35 % (Medium) số bước\n"
        "50 / 86 chiều quan sát là hằng số · seed chưa được gieo", fc=RF, ec=RED, fs=11.5)

    b.set_xlim(0, 100); b.set_ylim(0, 76); b.axis("off")
    b.set_title("(b) Sau rà soát (Giai đoạn 8–20, results_v4)", fontsize=12.5, fontweight="bold", color=NAVY, loc="left")
    for x in (14, 38, 62, 86):
        box(b, x - 10, 63, 20, 11, "UAV\nbản đồ riêng", fc=GF, ec=GREEN)
    b.plot([24, 28], [68.5, 68.5], color=NAVY, lw=2.6); b.text(26, 59.5, "≤ r_comm", ha="center", fontsize=11, color=NAVY)
    b.plot([48, 52], [68.5, 68.5], color=NAVY, lw=2.6)
    b.text(74, 68.5, "✕", ha="center", va="center", fontsize=13, color="#888"); b.text(74, 59.5, "ngoài tầm", ha="center", fontsize=11, color="#888")
    b.text(50, 52.5, "hợp nhất theo thành phần liên thông của Gᵗ (một hop mỗi bước)\n"
           "⇒ thông tin về vùng đã quét chỉ đến qua liên kết vô tuyến", ha="center", va="center", fontsize=11.5,
           color=NAVY, style="italic", linespacing=1.4)
    box(b, 3, 26, 94, 20, "quan sát 86 chiều, chuẩn hoá, KHÔNG có khối hàng xóm\n"
        "local_map thô 25×25 (gộp khối 5×5) vượt ra ngoài đĩa cảm biến\n"
        "thưởng theo từng UAV (0,7 cá nhân + 0,3 đội)\n"
        "critic Vᵢ theo UAV · encoder riêng · return chuẩn hoá · lr, entropy giảm", fc=GF, ec=GREEN, fs=11.5)
    box(b, 3, 2, 94, 20, "đánh giá bằng LẤY MẪU a ~ πθ (đúng phân phối PPO tối ưu)\n"
        "r_comm = 26 / 34 / 46 ⇒ đồ thị động, không rỗng, không đầy đủ\n"
        "BPTT 8 bước cho GRU · torch / numpy / random đều gieo seed\n"
        "47 kiểm thử hồi quy chống tái diễn", fc=GF, ec=GREEN, fs=11.5)
    fig.savefig(os.path.join(OUT, "fig5_2_luong_thong_tin_truoc_sau.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- Hình 5.4 (VDR ba kịch bản, thanh ngang, xếp dọc)
def load_comparison(sc):
    return json.load(open(f"web/public/data/{sc}/comparison.json", encoding="utf-8"))


def fig_bars():
    order = ["random_walk", "greedy", "maddpg", "mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo"]
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 8.8))
    for ax, sc in zip(axes, ["easy", "medium", "hard"]):
        d = load_comparison(sc)["methods"]
        means, stds, ns = [], [], []
        for m in order:
            vals = np.array(d[m]["metrics"]["victim_detection_rate"]["values"], dtype=float)
            means.append(float(vals.mean())); stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0); ns.append(len(vals))
        hbar_panel(ax, order, means, stds, f"{sc.capitalize()} (N = {dict(easy=4, medium=8, hard=16)[sc]}) — n = {max(ns)} seed", label_fs=12)
        ax.set_xlim(0, 1.5)
    axes[-1].set_xlabel("VDR (trung bình ± std liên-seed; 10 seed cho mọi phương pháp, hai đợt độc lập)", fontsize=11.5)
    fig.tight_layout(h_pad=1.3)
    fig.savefig(os.path.join(OUT, "fig5_4_so_sanh_vdr.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- Hình 5.7 (zero-shot, xếp dọc)
def _fig_zero_shot(fname, out_name, xlabel):
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 8.8))
    train_n = {"easy": 4, "medium": 8, "hard": 16}
    for k, (ax, sc) in enumerate(zip(axes, ["easy", "medium", "hard"])):
        d = json.load(open(f"results_v4/{sc}/{fname}", encoding="utf-8"))
        n_seeds = 0
        for m in ["mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo"]:
            seeds = d[m]; n_seeds = max(n_seeds, len(seeds))
            Ns = sorted({int(n) for s in seeds.values() for n in s})
            ys = [st.mean(seeds[s][str(N)]["victim_detection_rate"]["mean"] for s in seeds if str(N) in seeds[s]) for N in Ns]
            sd = [st.stdev(seeds[s][str(N)]["victim_detection_rate"]["mean"] for s in seeds if str(N) in seeds[s]) if len(seeds) > 1 else 0 for N in Ns]
            ax.errorbar(Ns, ys, yerr=sd, marker="o", ms=6, color=COLOR[m], label=METHOD_LABEL[m], lw=2.0, capsize=3, elinewidth=1)
        ax.axvline(train_n[sc], color="#666", ls="--", lw=1.3)
        ax.text(train_n[sc], 0.04, f"  N huấn luyện = {train_n[sc]}", fontsize=11, color="#444")
        ax.set_title(SC_TITLE[sc], fontsize=12.5, loc="left")
        ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.tick_params(labelsize=11.5)
        ax.yaxis.set_major_formatter(VI_TICKS)
        ax.set_ylabel("VDR", fontsize=12)
        if k == 0:
            ax.legend(fontsize=11, loc="lower right", frameon=True)
    axes[-1].set_xlabel(xlabel.format(n=n_seeds), fontsize=11.5)
    fig.tight_layout(h_pad=1.3)
    fig.savefig(os.path.join(OUT, out_name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_zero_shot():
    _fig_zero_shot("zero_shot_all.json", "fig5_7_zero_shot.png",
                   "Số UAV khi kiểm tra N (fixed-map: bản đồ giữ nguyên; {n} seed × 10 episode)")
    if os.path.exists("results_v4/hard/zero_shot_density_all.json"):
        _fig_zero_shot("zero_shot_density_all.json", "fig5_8_zero_shot_density.png",
                       "Số UAV khi kiểm tra N (density-controlled: diện tích và số nạn nhân scale theo N; {n} seed × 10 episode)")


# ----------------------------------------------------------------- Hình 5.3 (dấu vết kiểm toán)
def fig_progress():
    stages = ["GĐ6\nargmax", "GĐ8\nlấy mẫu", "GĐ8\n+4 lỗi", "GĐ9\n+obs", "GĐ12\nr_comm", "GĐ13–15\n+BPTT", "GĐ16\nseed"]
    series = {
        "mappo_mlp": [0.167, 0.546, 0.764, 0.855, 0.877, 0.860, 0.881],
        "mappo_gcn": [0.189, 0.496, 0.793, 0.873, 0.942, 0.925, 0.930],
        "gnn_mappo": [0.188, 0.582, 0.789, 0.847, 0.890, 0.889, 0.884],
        "greedy":    [0.437, 0.451, None,  0.716, 0.673, 0.673, 0.678],
    }
    fig, ax = plt.subplots(figsize=(FIG_W, 6.0))
    xs = np.arange(len(stages))
    for m, ys in series.items():
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=6.5, color=COLOR[m], lw=2.2, label=METHOD_LABEL[m])
    min_gap = 0.058
    for x in xs:
        items = sorted([(ys[x], m) for m, ys in series.items() if ys[x] is not None])
        lab = [y for y, _ in items]
        for i in range(1, len(lab)):
            if lab[i] - lab[i - 1] < min_gap:
                lab[i] = lab[i - 1] + min_gap
        shift = np.mean(lab) - np.mean([y for y, _ in items])
        for (y, m), ly in zip(items, lab):
            ax.text(x + 0.1, ly - shift, _vi(y), ha="left", va="center", fontsize=10, color=COLOR[m])
    ax.set_xticks(xs); ax.set_xticklabels(stages, fontsize=10.5)
    ax.set_xlim(-0.4, len(stages) - 0.25)
    ax.set_ylabel("VDR — kịch bản Easy (trung bình liên-seed)", fontsize=12)
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=11, loc="lower right")
    ax.yaxis.set_major_formatter(VI_TICKS); ax.tick_params(axis="y", labelsize=11.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_3_tien_trinh_sua_loi.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture(); fig_info_flow(); fig_progress(); fig_bars(); fig_zero_shot()
    print("written to", OUT, sorted(os.listdir(OUT)))
