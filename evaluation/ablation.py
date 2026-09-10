"""Architecture ablation report (Bang 5, nhom "Ablation kien truc"): MAPPO-MLP
vs MAPPO-GCN vs MAPPO-GAT (no GRU) vs GNN-MAPPO share the exact same MAPPO
trainer/hyperparameters (Bang 6) and only differ in encoder, so no extra
training is required beyond the runs already produced by scripts/run_poc.py —
this module just aggregates and statistically compares their eval_metrics.json.
"""
from __future__ import annotations

import json
import os

from evaluation.plots import METRIC_LABELS, _seed_dirs  # reuse path helpers
from evaluation.statistics import compare_against_baselines

ARCHITECTURE_ABLATION_ORDER = ["mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo"]


def _load_metric_series(out_dir: str, scenario: str, method: str, metric: str) -> list[float]:
    values = []
    for seed_dir in _seed_dirs(out_dir, scenario, method):
        path = os.path.join(seed_dir, "eval_metrics.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if metric in data["metrics"]:
            values.append(data["metrics"][metric]["mean"])
    return values


def architecture_ablation_report(out_dir: str, scenario: str) -> dict:
    report = {"scenario": scenario, "metrics": {}}
    for metric in METRIC_LABELS:
        target = _load_metric_series(out_dir, scenario, "gnn_mappo", metric)
        baselines = {
            m: _load_metric_series(out_dir, scenario, m, metric)
            for m in ARCHITECTURE_ABLATION_ORDER if m != "gnn_mappo"
        }
        baselines = {k: v for k, v in baselines.items() if v}
        entry = {"gnn_mappo_values": target}
        if target and baselines:
            entry["comparisons"] = compare_against_baselines(target, baselines)
        report["metrics"][metric] = entry
    return report


def save_ablation_report(out_dir: str, scenario: str):
    report = architecture_ablation_report(out_dir, scenario)
    path = os.path.join(out_dir, scenario, "architecture_ablation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Ablation report written to {path}")
    return report


if __name__ == "__main__":
    for scenario in ["easy", "medium", "hard"]:
        try:
            save_ablation_report("results", scenario)
        except Exception as e:
            print(f"skip {scenario}: {e}")
 