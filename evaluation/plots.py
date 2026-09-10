"""Exports PNG figures into results/figures/<scenario>/ from the JSON logs
written by training/train.py: learning curves, baseline comparison bars, and
coverage/collision visualizations built from a saved trajectory sample.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")

METHOD_LABELS = {
    "random_walk": "Random Walk", "greedy": "Greedy", "maddpg": "MADDPG",
    "mappo_mlp": "MAPPO-MLP", "mappo_gcn": "MAPPO-GCN", "mappo_gat": "MAPPO-GAT (no GRU)",
    "gnn_mappo": "GNN-MAPPO (proposed)",
}
METHOD_ORDER = ["random_walk", "greedy", "maddpg", "mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo"]
METRIC_LABELS = {
    "coverage_rate": "Coverage Rate", "victim_detection_rate": "Victim Detection Rate",
    "ttfd": "Time to First Detection (steps)", "collision_rate": "Collision Rate",
    "energy_efficiency": "Energy Efficiency",
}
PALETTE = sns.color_palette("viridis", n_colors=len(METHOD_ORDER))
METHOD_COLORS = dict(zip(METHOD_ORDER, PALETTE))

# muc 1.5.1 Hinh 2: nan nhan hinh tron mau theo do uu tien (vang/cam/do = 1/2/3)
PRIORITY_COLORS = {1: "#eab308", 2: "#f97316", 3: "#ef4444"}


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


def plot_learning_curves(out_dir: str, scenario: str, methods: list[str], fig_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False
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
        max_len = max(len(c[1]) for c in curves)
        steps_ref = max(curves, key=lambda c: len(c[1]))[0]
        padded = np.full((len(curves), max_len), np.nan)
        for i, (_, vals) in enumerate(curves):
            padded[i, :len(vals)] = vals
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        ax.plot(steps_ref, mean, label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method))
        ax.fill_between(steps_ref, mean - std, mean + std, alpha=0.2, color=METHOD_COLORS.get(method))

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Victim Detection Rate (eval)")
    ax.set_title(f"Learning curves — scenario: {scenario}")
    if any_data:
        ax.legend()
    os.makedirs(fig_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "learning_curves.png"), dpi=150)
    plt.close(fig)


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


def plot_comparison_bars(out_dir: str, scenario: str, methods: list[str], fig_dir: str):
    data = collect_final_metrics(out_dir, scenario, methods)
    if not data:
        return
    metrics = list(METRIC_LABELS.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.5))
    for ax, metric in zip(axes, metrics):
        names, means, stds, colors = [], [], [], []
        for method in METHOD_ORDER:
            if method not in data or not data[method][metric]:
                continue
            vals = data[method][metric]
            names.append(METHOD_LABELS.get(method, method))
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            colors.append(METHOD_COLORS.get(method))
        ax.bar(names, means, yerr=stds, color=colors, capsize=4)
        ax.set_title(METRIC_LABELS[metric])
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"Baseline comparison — scenario: {scenario}")
    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "comparison.png"), dpi=150)
    plt.close(fig)


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

    fig, ax = plt.subplots(figsize=(6, 6))

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

    # legend proxy handles (khong lap qua nhieu UAV rieng le)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#334155",
               markeredgecolor="#0f172a", markersize=9, label="UAV (hình thoi)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#eab308",
               markeredgecolor="#0f172a", markersize=9, label="Nạn nhân (ưu tiên 1-3)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#eab308",
               markeredgecolor="#4ade80", markersize=9, alpha=0.6, label="Nạn nhân đã tìm thấy"),
        Rectangle((0, 0), 1, 1, color="#64748b", label="Vật cản tĩnh"),
    ]

    ax.set_title(f"{METHOD_LABELS.get(method, method)} trajectory — {scenario} (seed {seed})")
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    ax.set_aspect("equal")
    os.makedirs(fig_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"trajectory_{method}.png"), dpi=150)
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
    ap.add_argument("--out-dir", default="results", help="thu muc ket qua (vd results_v2)")
    ap.add_argument("--scenarios", nargs="*", default=None)
    a = ap.parse_args()
    generate_all_figures(a.out_dir, a.scenarios)
