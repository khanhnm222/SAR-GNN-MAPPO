"""Exports PNG figures into <out_dir>/figures/<scenario>/ from the JSON logs
written by training/train.py: learning curves, baseline comparison bars, and
trajectory visualizations built from a saved trajectory sample.

Bố cục (sửa 14/09/2026): mọi hình được vẽ ở đúng bề rộng in trong luận văn
(≈ 6,3 in, khổ A4 lề 3,5/2 cm) với cỡ chữ 9–11 pt, các panel xếp THEO CHIỀU
DỌC và biểu đồ so sánh 7 phương pháp dùng THANH NGANG để tên phương pháp đọc
được mà không phải xoay chữ. Trước đó comparison.png là 5 panel nằm ngang
(3375×675 px) — thu về bề rộng trang thì chữ chỉ còn ~3 pt.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})

FIG_W = 6.3          # inch — bề rộng vùng chữ của trang luận văn
DPI = 200

METHOD_LABELS = {
    "random_walk": "Random Walk", "greedy": "Greedy", "maddpg": "MADDPG",
    "mappo_mlp": "MAPPO-MLP", "mappo_gcn": "MAPPO-GCN", "mappo_gat": "MAPPO-GAT (không GRU)",
    "gnn_mappo": "MAPPO-DGAT",
}
METHOD_ORDER = ["random_walk", "greedy", "maddpg", "mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo"]
METRIC_LABELS = {
    "coverage_rate": "Coverage Rate", "victim_detection_rate": "Victim Detection Rate",
    "ttfd": "Time to First Detection (steps)", "collision_rate": "Collision Rate",
    "energy_efficiency": "Energy Efficiency",
}
# tiêu đề panel tiếng Việt + quy ước hướng tốt hơn
METRIC_TITLES_VI = {
    "coverage_rate": "Coverage Rate — tỷ lệ bao phủ (↑ tốt hơn)",
    "victim_detection_rate": "VDR — tỷ lệ phát hiện nạn nhân (↑ tốt hơn)",
    "ttfd": "TTFD — bước tới phát hiện đầu tiên (↓ tốt hơn)",
    "collision_rate": "Collision Rate — tỷ lệ bước va chạm (↓ tốt hơn)",
    "energy_efficiency": "Energy Efficiency ×10⁻⁴ — nạn nhân/năng lượng (↑ tốt)",
}
METRIC_SCALE = {"energy_efficiency": 1e4}
METRIC_FMT = {"ttfd": "{:.1f}", "energy_efficiency": "{:.2f}"}

# cùng bảng màu với scripts/make_chapter5_figures.py: xám = heuristic,
# cam = không đồ thị, xanh = kiến trúc đồ thị (đậm dần theo độ phức tạp)
METHOD_COLORS = {"random_walk": "#9e9e9e", "greedy": "#616161", "maddpg": "#c98a2b", "mappo_mlp": "#e07b39",
                 "mappo_gcn": "#1f77b4", "mappo_gat": "#4c9fd6", "gnn_mappo": "#0b3d91"}

# muc 1.5.1 Hinh 2: nan nhan hinh tron mau theo do uu tien (vang/cam/do = 1/2/3)
PRIORITY_COLORS = {1: "#eab308", 2: "#f97316", 3: "#ef4444"}


def _vi(x: float, fmt: str = "{:.3f}") -> str:
    """Định dạng số theo quy ước tiếng Việt (dấu phẩy thập phân)."""
    return fmt.format(x).replace(".", ",")


VI_TICKS = FuncFormatter(lambda x, _pos: _vi(x, "{:g}"))


def _priority_color(p: int) -> str:
    return PRIORITY_COLORS.get(int(p), "#94a3b8")


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _results_root(out_dir: str, scenario: str, method: str):
    return os.path.join(out_dir, scenario, method)


def _seed_dirs(out_dir: str, scenario: str, method: str):
    root = _results_root(out_dir, scenario, method)
    if not os.path.isdir(root):
        return []
    return sorted(os.path.join(root, d) for d in os.listdir(root) if d.startswith("seed_"))


def _scenario_fig_dir(out_dir: str, scenario: str) -> str:
    return os.path.join(out_dir, "figures", scenario)


# --------------------------------------------------------------------------- learning curves
def plot_learning_curves(out_dir: str, scenario: str, methods: list[str], fig_dir: str):
    fig, ax = plt.subplots(figsize=(FIG_W, 4.6))
    any_data = False
    n_seeds = {}
    for method in methods:
        if method in ("random_walk", "greedy"):
            continue
        curves = []
        for seed_dir in _seed_dirs(out_dir, scenario, method):
            data = _load_json(os.path.join(seed_dir, "learning_curve.json"))
            if data and data["curve"]:
                steps = [c["global_step"] for c in data["curve"] if "eval_vdr" in c]
                vals = [c["eval_vdr"] for c in data["curve"] if "eval_vdr" in c]
                if steps:
                    curves.append((steps, vals))
        if not curves:
            continue
        any_data = True
        n_seeds[method] = len(curves)
        max_len = max(len(c[1]) for c in curves)
        steps_ref = max(curves, key=lambda c: len(c[1]))[0]
        padded = np.full((len(curves), max_len), np.nan)
        for i, (_, vals) in enumerate(curves):
            padded[i, :len(vals)] = vals
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        col = METHOD_COLORS.get(method)
        ax.plot(steps_ref, mean, label=METHOD_LABELS.get(method, method), color=col, lw=2.0)
        ax.fill_between(steps_ref, mean - std, mean + std, alpha=0.18, color=col)

    ax.set_xlabel("Số bước môi trường", fontsize=12.5)
    ax.set_ylabel("VDR đánh giá trung gian", fontsize=12.5)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=11.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x/1000:.0f}k" if x < 1e6 else f"{x/1e6:g}M"))
    ax.yaxis.set_major_formatter(VI_TICKS)
    ax.grid(alpha=0.3)
    if any_data:
        n = sorted(set(n_seeds.values()))
        n_txt = f"n = {n[0]}" if len(n) == 1 else "n = " + "/".join(map(str, n))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=11, frameon=False,
                  title=f"trung bình ± std liên-seed ({n_txt} seed)", title_fontsize=11)
    os.makedirs(fig_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "learning_curves.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- comparison bars
def collect_final_metrics(out_dir: str, scenario: str, methods: list[str]) -> dict:
    """method -> {metric -> list of per-seed means}"""
    result = {}
    for method in methods:
        per_metric = {m: [] for m in METRIC_LABELS}
        for seed_dir in _seed_dirs(out_dir, scenario, method):
            data = _load_json(os.path.join(seed_dir, "eval_metrics.json"))
            if not data:
                continue
            metrics = data["metrics"]
            for m in METRIC_LABELS:
                if m in metrics:
                    per_metric[m].append(metrics[m]["mean"])
        if any(per_metric[m] for m in METRIC_LABELS):
            result[method] = per_metric
    return result


def hbar_panel(ax, methods: list[str], means: list[float], stds: list[float], title: str,
               fmt: str = "{:.3f}", label_fs: float = 11):
    """Một panel thanh ngang: phương pháp trên trục y (thứ tự Bảng 5 từ trên
    xuống), giá trị mean ± std ghi ở đầu thanh."""
    y = np.arange(len(methods))[::-1]
    colors = [METHOD_COLORS.get(m, "#888") for m in methods]
    ax.barh(y, means, xerr=stds, color=colors, edgecolor="black", lw=0.5, height=0.68,
            error_kw=dict(ecolor="#333", capsize=3, lw=1))
    xmax = max((mu + sd) for mu, sd in zip(means, stds)) if means else 1.0
    ax.set_xlim(0, xmax * 1.42)
    for yi, mu, sd in zip(y, means, stds):
        txt = _vi(mu, fmt) if sd == 0 else f"{_vi(mu, fmt)} ± {_vi(sd, fmt)}"
        ax.text(mu + sd + xmax * 0.02, yi, txt, va="center", ha="left", fontsize=label_fs - 0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=label_fs)
    ax.set_title(title, fontsize=max(label_fs + 0.5, 12), loc="left", pad=6)
    ax.xaxis.set_major_formatter(VI_TICKS)
    ax.tick_params(axis="x", labelsize=label_fs - 1)
    ax.grid(axis="x", alpha=0.3); ax.grid(axis="y", alpha=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_comparison_bars(out_dir: str, scenario: str, methods: list[str], fig_dir: str):
    data = collect_final_metrics(out_dir, scenario, methods)
    if not data:
        return
    metrics = list(METRIC_LABELS.keys())
    # 5 panel xếp dọc, mỗi panel chiếm hết bề rộng trang
    fig, axes = plt.subplots(len(metrics), 1, figsize=(FIG_W, 1.8 * len(metrics)))
    present = [m for m in METHOD_ORDER if m in data]
    n_seeds = {m: len(data[m]["victim_detection_rate"]) for m in present}
    for ax, metric in zip(axes, metrics):
        scale = METRIC_SCALE.get(metric, 1.0)
        means, stds = [], []
        for method in present:
            vals = np.array(data[method][metric], dtype=float) * scale
            means.append(float(vals.mean()))
            stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
        hbar_panel(ax, present, means, stds, METRIC_TITLES_VI[metric], fmt=METRIC_FMT.get(metric, "{:.3f}"))
    ns = sorted(set(n_seeds.values()))
    note = ("trung bình ± std liên-seed; số seed: " +
            ", ".join(f"{METHOD_LABELS[m]} n = {n_seeds[m]}" for m in present) if len(ns) > 1
            else f"trung bình ± std liên-seed, n = {ns[0]} seed mỗi phương pháp")
    fig.text(0.01, -0.005, note, fontsize=10.5, color="#444", ha="left", va="top", wrap=True)
    fig.tight_layout(h_pad=1.2)
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "comparison.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- trajectory
def plot_trajectory(out_dir: str, scenario: str, method: str, fig_dir: str, seed: int = 0):
    """Quy trinh minh hoa mot episode, dung dung ky hieu Hinh 2 cua de cuong:
    UAV hinh thoi, nan nhan hinh tron mau theo do uu tien, vat can o vuong xam.
    """
    seed_dir = os.path.join(_results_root(out_dir, scenario, method), f"seed_{seed}")
    data = _load_json(os.path.join(seed_dir, "trajectory_sample.json"))
    if not data:
        return
    frames = data["frames"]
    if not frames:
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_W))

    for ox, oy in data.get("terrain_occupancy", []):
        ax.add_patch(Rectangle((ox - 0.5, oy - 0.5), 1, 1, color="#64748b", zorder=1))

    colors = sns.color_palette("tab10", n_colors=len(frames[0]["uavs"]))
    for i, uav in enumerate(frames[0]["uavs"]):
        xs = [f["uavs"][i]["x"] for f in frames if i < len(f["uavs"])]
        ys = [f["uavs"][i]["y"] for f in frames if i < len(f["uavs"])]
        ax.plot(xs, ys, "-", color=colors[i], alpha=0.6, linewidth=1, zorder=2)
        ax.scatter(xs[-1:], ys[-1:], color=colors[i], marker="D", s=70,
                   edgecolors="#0f172a", linewidths=0.8, label=f"UAV {i}", zorder=4)

    victims = frames[-1]["victims"]
    for v in victims:
        ax.scatter(v["x"], v["y"], s=90 + 15 * v["priority"],
                   facecolors=_priority_color(v["priority"]),
                   edgecolors="#4ade80" if v["detected"] else "#0f172a",
                   linewidths=1.6 if v["detected"] else 0.8,
                   alpha=0.4 if v["detected"] else 0.95, marker="o", zorder=3)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#334155",
               markeredgecolor="#0f172a", markersize=9, label="UAV (vị trí cuối)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#eab308",
               markeredgecolor="#0f172a", markersize=9, label="Nạn nhân (ưu tiên 1–3)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#eab308",
               markeredgecolor="#4ade80", markersize=9, alpha=0.6, label="Nạn nhân đã tìm thấy"),
        Rectangle((0, 0), 1, 1, color="#64748b", label="Vật cản tĩnh"),
    ]

    ax.set_title(f"{METHOD_LABELS.get(method, method)} — {scenario}, seed {seed}", fontsize=12.5)
    ax.legend(handles=legend_handles, fontsize=11, loc="upper right")
    ax.tick_params(labelsize=11.5)
    ax.set_aspect("equal")
    os.makedirs(fig_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"trajectory_{method}.png"), dpi=DPI)
    plt.close(fig)


def generate_all_figures(out_dir: str = "results", scenarios: list[str] | None = None):
    scenarios = scenarios or ["easy", "medium", "hard"]
    for scenario in scenarios:
        fig_dir = _scenario_fig_dir(out_dir, scenario)
        plot_learning_curves(out_dir, scenario, METHOD_ORDER, fig_dir)
        plot_comparison_bars(out_dir, scenario, METHOD_ORDER, fig_dir)
        for method in METHOD_ORDER:
            plot_trajectory(out_dir, scenario, method, fig_dir)
        print(f"[{scenario}] figures written to {fig_dir}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results", help="thu muc ket qua (vd results_v4)")
    ap.add_argument("--scenarios", nargs="*", default=None)
    a = ap.parse_args()
    generate_all_figures(a.out_dir, a.scenarios)
