"""So sanh ket qua thuc nghiem giua 2 lan chay (vd: may Windows goc vs MacBook
M3 Pro kiem chung lai) — dung DE KIEM CHUNG, khong dung de tao ket qua bao cao
chinh thuc.

Vi CPU x86 (Windows) va CPU ARM (Apple Silicon) dung thu vien BLAS/vector hoa
khac nhau, ket qua se KHONG bit-exact du cung seed — day la binh thuong. Script
nay so sanh theo PHAN PHOI (mean/std qua cac seed + kiem dinh Welch/Mann-Whitney
tu evaluation/statistics.py, giong cach lam trong evaluation/ablation.py), tuc
la: hai lan chay duoc coi la "khop nhau" neu khac biet KHONG co y nghia thong ke
(p >= 0.05) va Cohen's d nho, chu khong doi hoi bang so tuyet doi.

Vi du:
    python -m scripts.compare_runs --a results --b results_macos \
        --scenarios easy medium hard --methods gnn_mappo mappo_gcn mappo_gat mappo_mlp maddpg
"""
from __future__ import annotations

import argparse
import json
import os

from evaluation.plots import METHOD_ORDER, METRIC_LABELS, _seed_dirs
from evaluation.statistics import compare_two_samples


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="out_dir thu nhat (vd: results, may goc)")
    parser.add_argument("--b", required=True, help="out_dir thu hai (vd: results_macos, may kiem chung)")
    parser.add_argument("--scenarios", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--methods", nargs="+", default=METHOD_ORDER)
    parser.add_argument("--out-json", default=None, help="Neu dat, ghi bao cao day du ra file JSON nay")
    args = parser.parse_args()

    report = {}
    n_flagged = 0
    n_total = 0
    for scenario in args.scenarios:
        report[scenario] = {}
        print(f"\n=== {scenario} ===")
        for method in args.methods:
            report[scenario][method] = {}
            row_printed = False
            for metric in METRIC_LABELS:
                a = _load_metric_series(args.a, scenario, method, metric)
                b = _load_metric_series(args.b, scenario, method, metric)
                if not a or not b:
                    continue
                cmp = compare_two_samples(a, b)
                report[scenario][method][metric] = cmp
                n_total += 1
                if not row_printed:
                    print(f"  -- {method} --")
                    row_printed = True
                flag = ""
                if cmp["test"] != "insufficient_data" and cmp["significant"]:
                    flag = "  <-- KHAC BIET CO Y NGHIA THONG KE (p<0.05)"
                    n_flagged += 1
                mean_a = cmp["mean_a"]
                mean_b = cmp["mean_b"]
                p = cmp["p_value"]
                d = cmp["cohens_d"]
                if p is None:
                    print(f"    {metric:24s} a(n={len(a)})={mean_a:.4f}  b(n={len(b)})={mean_b:.4f}  "
                          f"(khong du du lieu de kiem dinh)")
                else:
                    print(f"    {metric:24s} a(n={len(a)})={mean_a:.4f}  b(n={len(b)})={mean_b:.4f}  "
                          f"p={p:.3f}  d={d:+.2f} [{cmp['test']}]{flag}")

    print(f"\n=== TOM TAT: {n_flagged}/{n_total} (scenario, method, metric) co khac biet "
          f"co y nghia thong ke (p<0.05, chua hieu chinh Bonferroni) ===")
    if n_flagged:
        print("Luu y: mot vai khac biet co y nghia la binh thuong voi n=5 seed/nhom (thong ke yeu) va do")
        print("khac biet phan cung CPU (x86 vs Apple Silicon). Chi dang lo ngai neu CHIEU so sanh doi")
        print("(vd: GNN-MAPPO > baseline tren may A nhung < baseline tren may B) hoac lech qua lon (|d|>1).")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nBao cao day du: {args.out_json}")


if __name__ == "__main__":
    main()
