# -*- coding: utf-8 -*-
"""Kiem chung ngan sach 2x: so sanh checkpoint huan luyen 2x buoc (results_v4_budget2x) voi
1x buoc (results_v4) cho tung kien truc, cung seed va cung bo ban do test; do khoang cach
DGAT - GCN o 1x va 2x. Sinh CSV + PNG vao experiment_log/assets/<tag>/budget2x_<sc>/ va chen
khoi Markdown vao file nhat ky giua `<!-- AUTO:budget-<sc> -->` ... `<!-- /AUTO:budget-<sc> -->`.

    python -m scripts.budget_report --scenario hard --log experiment_log/giai-doan-22-*.md
    python -m scripts.budget_report --scenario easy --log experiment_log/giai-doan-18-*.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from scripts.stage_report import COLOR, FIG_W, DPI, LABEL, cohen, insert_block, vi  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
ASSETS = os.path.join("experiment_log", "assets")
ARCHS = ["mappo_gcn", "gnn_mappo"]
METRICS = [("victim_detection_rate", "VDR"), ("coverage_rate", "Coverage"), ("collision_rate", "Collision")]
BASE_STEPS = {"easy": "500k", "medium": "800k", "hard": "1M"}
DOUBLE_STEPS = {"easy": "1M", "medium": "1,6M", "hard": "2M"}


def sgn(x, nd=3):
    return f"{x:+.{nd}f}".replace(".", ",").replace("-", "−")


def load(root, sc, m):
    out = {}
    for p in glob.glob(os.path.join(root, sc, m, "seed_*", "eval_metrics.json")):
        d = json.load(open(p, encoding="utf-8"))
        out[d["meta"]["seed"]] = {k: v["mean"] for k, v in d["metrics"].items() if isinstance(v, dict)}
    return out


def curve(root, sc, m, s):
    p = os.path.join(root, sc, m, f"seed_{s}", "learning_curve.json")
    if not os.path.exists(p):
        return None
    c = json.load(open(p))["curve"]
    return [(x["global_step"], x["eval_vdr"]) for x in c if "eval_vdr" in x], c[-1].get("entropy")


def report(base_dir, dbl_dir, sc, tag):
    adir = os.path.join(ASSETS, tag, f"budget2x_{sc}"); os.makedirs(adir, exist_ok=True)
    base = {m: load(base_dir, sc, m) for m in ARCHS}
    dbl = {m: load(dbl_dir, sc, m) for m in ARCHS}
    seeds = sorted(set.intersection(*[set(dbl[m]) for m in ARCHS])) if all(dbl[m] for m in ARCHS) else []
    status = ", ".join(f"{LABEL[m]} {len(dbl[m])}/5" for m in ARCHS)
    complete = all(len(dbl[m]) >= 5 for m in ARCHS)
    lines = [f"Trạng thái: ngân sách 2× đã có {status} — " + ("**đầy đủ**." if complete else "**chưa đầy đủ, số tạm thời**.")]

    # --- per-arch 1x vs 2x
    rows_csv = [["metric", "arch", "n", "mean_1x", "sd_1x", "mean_2x", "sd_2x", "delta", "paired_p", "wins"]]
    for k, kl in METRICS:
        lines.append(f"\n**{kl}: {BASE_STEPS[sc]} → {DOUBLE_STEPS[sc]} bước** (cùng seed, cùng bản đồ test 90.000 + seed·1.000):\n")
        lines.append("| Kiến trúc | n | 1× | 2× | Δ (2× − 1×) | p theo cặp | seed tăng |\n|---|---|---|---|---|---|---|")
        for m in ARCHS:
            ss = sorted(set(base[m]) & set(dbl[m]))
            if len(ss) < 2:
                lines.append(f"| {LABEL[m]} | {len(ss)} | — | — | — | — | — |"); continue
            a = [base[m][s][k] for s in ss]; b = [dbl[m][s][k] for s in ss]
            p = stats.ttest_rel(b, a).pvalue; wins = sum(x > y for x, y in zip(b, a))
            lines.append(f"| {LABEL[m]} | {len(ss)} | {vi(st.mean(a))} ± {vi(st.stdev(a))} | {vi(st.mean(b))} ± {vi(st.stdev(b))} | "
                         f"{sgn(st.mean(b) - st.mean(a))} | {vi(p, 3)} | {wins}/{len(ss)} |")
            rows_csv.append([k, m, len(ss), st.mean(a), st.stdev(a), st.mean(b), st.stdev(b), st.mean(b) - st.mean(a), p, wins])
    with open(os.path.join(adir, f"budget2x_{sc}.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows_csv)

    # --- gap DGAT - GCN at 1x and 2x
    lines.append(f"\n**Khoảng cách MAPPO-DGAT − MAPPO-GCN** (câu hỏi chính: có đóng lại khi gấp đôi bước không?):\n")
    lines.append("| Chỉ số | Ngân sách | n | GCN | DGAT | Khoảng cách | Welch p | d | theo cặp thắng / p |\n|---|---|---|---|---|---|---|---|---|")
    for k, kl in METRICS[:2]:
        for name, src in [(BASE_STEPS[sc], base), (DOUBLE_STEPS[sc], dbl)]:
            ss = sorted(set(src["mappo_gcn"]) & set(src["gnn_mappo"]))
            if len(ss) < 2:
                lines.append(f"| {kl} | {name} | {len(ss)} | — | — | — | — | — | — |"); continue
            g = [src["mappo_gcn"][s][k] for s in ss]; d_ = [src["gnn_mappo"][s][k] for s in ss]
            t = stats.ttest_ind(d_, g, equal_var=False); pp = stats.ttest_rel(d_, g).pvalue
            lines.append(f"| {kl} | {name} | {len(ss)} | {vi(st.mean(g))} | {vi(st.mean(d_))} | {sgn(st.mean(d_) - st.mean(g))} | "
                         f"{vi(t.pvalue, 4)} | {sgn(cohen(d_, g), 2)} | {sum(x > y for x, y in zip(d_, g))}/{len(ss)} / {vi(pp, 3)} |")

    # --- reference rows
    ref = {m: load(base_dir, sc, m) for m in ["mappo_mlp", "greedy"]}
    lines.append("\nTham chiếu (ngân sách 1×, n = %d): MAPPO-MLP %s, Greedy %s." % (
        len(ref["mappo_mlp"]), vi(st.mean(v["victim_detection_rate"] for v in ref["mappo_mlp"].values())),
        vi(st.mean(v["victim_detection_rate"] for v in ref["greedy"].values()))))

    # --- figure 1: paired dots 1x -> 2x
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.9), dpi=DPI, sharey=False)
    for ax, (k, kl) in zip(axes, METRICS[:2]):
        for i, m in enumerate(ARCHS):
            ss = sorted(set(base[m]) & set(dbl[m]))
            for s in ss:
                ax.plot([i * 2.2 + 0, i * 2.2 + 1], [base[m][s][k], dbl[m][s][k]], color=COLOR[m], lw=1.0, alpha=0.7)
                ax.scatter([i * 2.2 + 0, i * 2.2 + 1], [base[m][s][k], dbl[m][s][k]], color=COLOR[m], s=28, zorder=3,
                           edgecolor="black", linewidth=0.4)
            if ss:
                ax.plot([i * 2.2, i * 2.2 + 1], [st.mean(base[m][s][k] for s in ss), st.mean(dbl[m][s][k] for s in ss)],
                        color="black", lw=2.2, zorder=4)
        ax.set_xticks([0, 1, 2.2, 3.2]); ax.set_xticklabels([f"GCN {BASE_STEPS[sc]}", f"GCN {DOUBLE_STEPS[sc]}",
                                                             f"DGAT {BASE_STEPS[sc]}", f"DGAT {DOUBLE_STEPS[sc]}"], fontsize=9.5, rotation=20)
        ax.set_title(kl, fontsize=12); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{sc.capitalize()}: từng seed khi gấp đôi ngân sách (đậm = trung bình)", fontsize=12)
    fig.tight_layout(); f1 = os.path.join(adir, f"fig_budget2x_pairs_{sc}.png"); fig.savefig(f1); plt.close(fig)

    # --- figure 2: learning curves 2x (with 1x overlaid)
    fig, ax = plt.subplots(figsize=(FIG_W, 4.4), dpi=DPI)
    ent = {}
    for m in ARCHS:
        for root, ls, lab in [(base_dir, "--", BASE_STEPS[sc]), (dbl_dir, "-", DOUBLE_STEPS[sc])]:
            cs = [curve(root, sc, m, s) for s in range(5)]; cs = [c for c in cs if c and c[0]]
            if not cs:
                continue
            n = min(len(c[0]) for c in cs)
            xs = np.array([cs[0][0][i][0] for i in range(n)]) / 1e3
            ys = np.array([[c[0][i][1] for i in range(n)] for c in cs])
            ax.plot(xs, ys.mean(0), ls, color=COLOR[m], lw=1.6, label=f"{LABEL[m]} {lab} (n={len(cs)})")
            ax.fill_between(xs, ys.mean(0) - ys.std(0), ys.mean(0) + ys.std(0), color=COLOR[m], alpha=0.08)
            ent[(m, lab)] = st.mean(c[1] for c in cs if c[1] is not None)
    ax.set_xlabel("bước môi trường (nghìn)"); ax.set_ylabel("VDR đánh giá trung gian"); ax.grid(alpha=0.3)
    ax.legend(fontsize=9); ax.set_title(f"{sc.capitalize()}: learning curve ngân sách 2× (liền) và 1× (đứt)", fontsize=12)
    fig.tight_layout(); f2 = os.path.join(adir, f"fig_budget2x_curves_{sc}.png"); fig.savefig(f2); plt.close(fig)
    if ent:
        lines.append("\nEntropy chính sách cuối (nat, trung bình seed): " + ", ".join(f"{LABEL[m]} {lab}: {vi(e, 2)}" for (m, lab), e in ent.items()))

    rel = lambda p: os.path.relpath(p, "experiment_log").replace("\\", "/")
    lines.append(f"\n![Từng seed 1× → 2× — {sc}]({rel(f1)})\n\n![Learning curve 2× — {sc}]({rel(f2)})")
    lines.append(f"\nDữ liệu thô: `{rel(adir)}/budget2x_{sc}.csv`; checkpoint 2× tại `{dbl_dir}/{sc}/`.")
    return "\n".join(lines), complete


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default="results_v4"); ap.add_argument("--double-dir", default="results_v4_budget2x")
    ap.add_argument("--scenario", default="hard"); ap.add_argument("--log", required=True); ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    logs = glob.glob(a.log); assert len(logs) == 1, logs
    tag = a.tag or os.path.basename(logs[0])[:12]
    from datetime import datetime
    body, complete = report(a.base_dir, a.double_dir, a.scenario, tag)
    insert_block(logs[0], f"budget-{a.scenario}", f"### Kết quả tự động — ngân sách 2× {a.scenario} (cập nhật {datetime.now():%d/%m/%Y %H:%M})\n\n{body}")
    print(f"{a.scenario}: {'DAY DU' if complete else 'tam thoi'} — da chen vao {logs[0]}")


if __name__ == "__main__":
    main()
