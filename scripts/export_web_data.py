"""Aggregates results/ into a stable JSON schema under web/public/data/ for the
Next.js visualization site. Safe to re-run at any time (e.g. while training is
still in progress in the background) — it only includes whatever runs have
produced output so far.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, ".")

from evaluation.ablation import save_ablation_report  # noqa: E402
from evaluation.plots import (  # noqa: E402
    METHOD_LABELS, METHOD_ORDER, METRIC_LABELS, _load_json, _seed_dirs,
    collect_final_metrics,
)
from evaluation.statistics import compare_against_baselines  # noqa: E402

SCENARIOS = ["easy", "medium", "hard"]

# Nhan hien thi tren web theo de cuong hieu chinh 10/09/2026: GNN-MAPPO la ten framework,
# bo ma hoa la mot HO encoder thay the duoc; MAPPO-GCN la phien ban de xuat, GATv2+GRU la MAPPO-DGAT.
WEB_LABELS = {
    "random_walk": "Random Walk",
    "greedy": "Greedy",
    "maddpg": "MADDPG",
    "mappo_mlp": "MAPPO-MLP (không đồ thị)",
    "mappo_gcn": "MAPPO-GCN (đề xuất)",
    "mappo_gat": "MAPPO-GAT (GATv2, không GRU)",
    "gnn_mappo": "MAPPO-DGAT (GATv2 + GRU)",
}
PROPOSED = "mappo_gcn"
RESULTS_DIR = "results"          # overridden by --results-dir
WEB_DATA_DIR = os.path.join("web", "public", "data")


def _write(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def export_learning_curves(scenario: str) -> dict:
    out = {"scenario": scenario, "methods": {}}
    for method in METHOD_ORDER:
        seed_curves = []
        for seed_dir in _seed_dirs(RESULTS_DIR, scenario, method):
            data = _load_json(os.path.join(seed_dir, "learning_curve.json"))
            if not data:
                continue
            points = [{"step": c["global_step"], "vdr": c.get("eval_vdr"), "coverage": c.get("eval_coverage")}
                      for c in data["curve"] if "eval_vdr" in c]
            if points:
                seed_curves.append({"seed": data["meta"]["seed"], "points": points})
        if seed_curves:
            out["methods"][method] = {"label": WEB_LABELS.get(method, method), "seeds": seed_curves}
    return out


def export_comparison(scenario: str) -> dict:
    data = collect_final_metrics(RESULTS_DIR, scenario, METHOD_ORDER)
    out = {"scenario": scenario, "metric_labels": METRIC_LABELS, "methods": {}}
    for method, per_metric in data.items():
        out["methods"][method] = {
            "label": WEB_LABELS.get(method, method),
            "metrics": {m: {"mean": (sum(v) / len(v)) if v else None,
                             "std": (sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)) ** 0.5 if v else None,
                             "values": v}
                        for m, v in per_metric.items()},
        }
    # kiem dinh thong ke: phien ban de xuat (MAPPO-GCN) so voi tung phuong phap con lai, theo chi so
    out["proposed_method"] = PROPOSED
    if PROPOSED in data:
        comparisons = {}
        for metric in METRIC_LABELS:
            target = data[PROPOSED].get(metric, [])
            baselines = {m: data[m][metric] for m in data if m != PROPOSED and data[m].get(metric)}
            if target and baselines:
                comparisons[metric] = compare_against_baselines(target, baselines)
        out["proposed_vs_baselines"] = comparisons
    # RQ2: co GRU (MAPPO-DGAT) so voi khong GRU (MAPPO-GAT), cung kien truc GATv2
    if "gnn_mappo" in data and "mappo_gat" in data:
        gru = {}
        for metric in METRIC_LABELS:
            a, b = data["gnn_mappo"].get(metric, []), data["mappo_gat"].get(metric, [])
            if a and b:
                gru[metric] = compare_against_baselines(a, {"mappo_gat": b})["mappo_gat"]
        out["gru_ablation"] = gru
    return out


def export_trajectory(scenario: str, method: str, seed: int = 0):
    seed_dir = os.path.join(RESULTS_DIR, scenario, method, f"seed_{seed}")
    data = _load_json(os.path.join(seed_dir, "trajectory_sample.json"))
    return data


def export_ablation(scenario: str) -> dict | None:
    try:
        return save_ablation_report(RESULTS_DIR, scenario)
    except Exception as e:
        print(f"ablation export skipped for {scenario}: {e}")
        return None


def export_zero_shot(scenario: str, mode: str = "fixed_map") -> dict | None:
    """Aggregate `zero_shot_all.json` (fixed-map) hoac `zero_shot_density_all.json`
    (density-controlled) — method -> seed -> N -> metrics — into the
    cross-architecture shape the web page renders: mean +- std over seeds, per
    method, per swarm size. This is the RQ3 comparison — the earlier per-method
    report could only show one architecture at a time."""
    fname = "zero_shot_all.json" if mode == "fixed_map" else "zero_shot_density_all.json"
    path = os.path.join(RESULTS_DIR, scenario, fname)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    env_path = path.replace(".json", "_env.json")
    env_by_n = json.load(open(env_path, "r", encoding="utf-8")) if os.path.isfile(env_path) else None

    train_n = {"easy": 4, "medium": 8, "hard": 16}.get(scenario)
    ns = sorted({int(n) for m in raw for s in raw[m] for n in raw[m][s]})
    out = {"scenario": scenario, "train_n": train_n, "ns": ns, "mode": mode,
           "env_by_n": env_by_n, "eval_protocol": "sampled", "methods": {}}

    for method, by_seed in raw.items():
        series = {}
        for metric in ("victim_detection_rate", "coverage_rate", "collision_rate"):
            means, stds = [], []
            for n in ns:
                vals = [by_seed[s][str(n)][metric]["mean"] for s in by_seed if str(n) in by_seed[s]]
                if vals:
                    mu = sum(vals) / len(vals)
                    var = sum((v - mu) ** 2 for v in vals) / len(vals)
                    means.append(mu)
                    stds.append(var ** 0.5)
                else:
                    means.append(None)
                    stds.append(None)
            series[metric] = {"mean": means, "std": stds}
        series["n_seeds"] = len(by_seed)
        out["methods"][method] = {"label": WEB_LABELS.get(method, method), **series}
    return out


def main():
    global RESULTS_DIR, WEB_DATA_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=RESULTS_DIR,
                    help="thu muc ket qua nguon (vd results_v2)")
    ap.add_argument("--web-data-dir", default=WEB_DATA_DIR)
    args = ap.parse_args()
    RESULTS_DIR = args.results_dir
    WEB_DATA_DIR = args.web_data_dir
    print(f"exporting from {RESULTS_DIR} -> {WEB_DATA_DIR}")

    manifest = {"scenarios": [], "methods": METHOD_ORDER, "method_labels": WEB_LABELS, "proposed_method": PROPOSED}
    for scenario in SCENARIOS:
        scenario_dir = os.path.join(RESULTS_DIR, scenario)
        if not os.path.isdir(scenario_dir):
            continue
        manifest["scenarios"].append(scenario)

        curves = export_learning_curves(scenario)
        _write(os.path.join(WEB_DATA_DIR, scenario, "learning_curves.json"), curves)

        comparison = export_comparison(scenario)
        _write(os.path.join(WEB_DATA_DIR, scenario, "comparison.json"), comparison)

        ablation = export_ablation(scenario)
        if ablation:
            _write(os.path.join(WEB_DATA_DIR, scenario, "ablation.json"), ablation)

        zero_shot = export_zero_shot(scenario, "fixed_map")
        if zero_shot:
            _write(os.path.join(WEB_DATA_DIR, scenario, "zero_shot.json"), zero_shot)
        zero_shot_density = export_zero_shot(scenario, "density")
        if zero_shot_density:
            _write(os.path.join(WEB_DATA_DIR, scenario, "zero_shot_density.json"), zero_shot_density)

        traj_manifest = {}
        for method in METHOD_ORDER:
            traj = export_trajectory(scenario, method)
            if traj:
                _write(os.path.join(WEB_DATA_DIR, scenario, "trajectories", f"{method}.json"), traj)
                traj_manifest[method] = True
        _write(os.path.join(WEB_DATA_DIR, scenario, "trajectories", "manifest.json"), traj_manifest)

    _write(os.path.join(WEB_DATA_DIR, "manifest.json"), manifest)
    print(f"Web data exported to {WEB_DATA_DIR}: scenarios={manifest['scenarios']}")


if __name__ == "__main__":
    main()
