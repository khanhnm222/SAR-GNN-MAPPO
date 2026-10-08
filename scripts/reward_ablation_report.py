# -*- coding: utf-8 -*-
"""Ablation ham thuong (Giai doan 23): so sanh moi bien the bo mot thanh phan thuong
(results_v4_reward_ablation/no_<comp>/<sc>/<method>/seed_k) voi doi chung day du
(results_v4/<sc>/<method>/seed_k), cung seed va cung bo ban do test -> kiem dinh theo cap.
Sinh CSV + PNG vao experiment_log/assets/<tag>/reward_ablation_<sc>/ va chen khoi Markdown
vao nhat ky giua `<!-- AUTO:reward-ablation -->` ... `<!-- /AUTO:reward-ablation -->`.

    python -m scripts.reward_ablation_report --scenario medium --log experiment_log/giai-doan-23-*.md
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
from scripts.budget_report import sgn  # noqa: E402
from scripts.stage_report import DPI, FIG_W, cohen, insert_block, vi  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
ASSETS = os.path.join("experiment_log", "assets")
COMPONENTS = [("cover", "α (bao phủ)"), ("victim", "β (nạn nhân)"), ("collision", "κ (va chạm)"), ("energy", "δ (năng lượng)")]
METRICS = [("victim_detection_rate", "VDR"), ("coverage_rate", "Coverage"), ("collision_rate", "Collision"), ("ttfd", "TTFD (bước)")]
VCOLOR = {"full": "#333333", "cover": "#1f77b4", "victim": "#d62728", "collision": "#ff7f0e", "energy": "#2ca02c"}


def load(root, sc, m):
    out = {}
    for p in glob.glob(os.path.join(root, sc, m, "seed_*", "eval_metrics.json")):
        d = json.load(open(p, encoding="utf-8"))
        out[d["meta"]["seed"]] = {k: v["mean"] for k, v in d["metrics"].items() if isinstance(v, dict)}
    return out


def entropy_final(root, sc, m, seeds):
    vals = []
    for s in seeds:
        p = os.path.join(root, sc, m, f"seed_{s}", "learning_curve.json")
        if os.path.exists(p):
            c = json.load(open(p))["curve"]
            tail = [x["entropy"] for x in c[-max(1, len(c) // 20):] if x.get("entropy") is not None]
            if tail:
                vals.append(st.mean(tail))
    return st.mean(vals) if vals else None


def report(base_dir, abl_dir, sc, method, tag):
    adir = os.path.join(ASSETS, tag, f"reward_ablation_{sc}"); os.makedirs(adir, exist_ok=True)
    # Doi chung chi lay seed 0-4 (cung dot A) de bang, hinh va kiem dinh theo cap cung mot mau.
    full = {s: v for s, v in load(base_dir, sc, method).items() if s < 5}
    variants = {c: load(os.path.join(abl_dir, f"no_{c}"), sc, method) for c, _ in COMPONENTS}
    status = ", ".join(f"bỏ {lab} {len(variants[c])}/5" for c, lab in COMPONENTS)
    complete = all(len(v) >= 5 for v in variants.values())
    lines = [f"Trạng thái: {status} — " + ("**đầy đủ**." if complete else "**chưa đầy đủ, số tạm thời**.")]
    rows_csv = [["metric", "variant", "n", "mean_full", "sd_full", "mean_variant", "sd_variant", "delta", "paired_p", "cohen_d", "worse_seeds"]]
    summary = {}
    for k, kl in METRICS:
        lines.append(f"\n**{kl}: đối chứng đầy đủ so với từng biến thể bỏ một thành phần** (cùng seed, cùng bản đồ test; p = paired t-test; d = Cohen; "
                     f"'seed xấu đi' = số seed biến thể kém đối chứng):\n")
        lines.append("| Biến thể | n | Đầy đủ | Bỏ thành phần | Δ (bỏ − đầy đủ) | p theo cặp | d | seed xấu đi |\n|---|---|---|---|---|---|---|---|")
        for c, lab in COMPONENTS:
            ss = sorted(set(full) & set(variants[c]))
            if len(ss) < 2:
                lines.append(f"| bỏ {lab} | {len(ss)} | — | — | — | — | — | — |"); continue
            a = [full[s][k] for s in ss]; b = [variants[c][s][k] for s in ss]
            p = stats.ttest_rel(b, a).pvalue; d = cohen(b, a)
            worse = sum((y > x) if k in ("collision_rate", "ttfd") else (y < x) for x, y in zip(a, b))
            lines.append(f"| bỏ {lab} | {len(ss)} | {vi(st.mean(a))} ± {vi(st.stdev(a))} | {vi(st.mean(b))} ± {vi(st.stdev(b))} | "
                         f"{sgn(st.mean(b) - st.mean(a))} | {vi(p, 3)} | {sgn(d, 2)} | {worse}/{len(ss)} |")
            rows_csv.append([k, c, len(ss), st.mean(a), st.stdev(a), st.mean(b), st.stdev(b), st.mean(b) - st.mean(a), p, d, worse])
            summary[(k, c)] = (st.mean(a), st.mean(b), p)
    with open(os.path.join(adir, f"reward_ablation_{sc}.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows_csv)

    # entropy
    ent = {"đầy đủ": entropy_final(base_dir, sc, method, range(5))}
    for c, lab in COMPONENTS:
        ent[f"bỏ {lab}"] = entropy_final(os.path.join(abl_dir, f"no_{c}"), sc, method, range(5))
    lines.append("\nEntropy chính sách cuối (nat, trung bình 5 % iteration cuối, trung bình seed): " +
                 ", ".join(f"{k}: {vi(v, 2)}" for k, v in ent.items() if v is not None))

    # figure: 4 metric panels, bars = full + 4 variants, dots = seeds
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 6.2), dpi=DPI)
    names = ["đầy đủ"] + [f"bỏ {lab.split(' ')[0]}" for _, lab in COMPONENTS]
    keys = ["full"] + [c for c, _ in COMPONENTS]
    for ax, (k, kl) in zip(axes.flat, METRICS):
        for i, key in enumerate(keys):
            src = full if key == "full" else variants[key]
            vals = [src[s][k] for s in sorted(src)]
            if not vals:
                continue
            ax.bar(i, st.mean(vals), color=VCOLOR[key], alpha=0.85, width=0.65)
            ax.scatter([i + 0.18] * len(vals), vals, s=14, color="black", zorder=3)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(names, fontsize=9, rotation=20)
        ax.set_title(kl, fontsize=11); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{sc.capitalize()}, MAPPO-GCN: bỏ từng thành phần thưởng (chấm = seed)", fontsize=12)
    fig.tight_layout(); f1 = os.path.join(adir, f"fig_reward_ablation_{sc}.png"); fig.savefig(f1); plt.close(fig)

    rel = lambda p: os.path.relpath(p, "experiment_log").replace("\\", "/")
    lines.append(f"\n![Ablation hàm thưởng — {sc}]({rel(f1)})")
    lines.append(f"\nDữ liệu thô: `{rel(adir)}/reward_ablation_{sc}.csv`; checkpoint tại `{abl_dir}/no_<thành phần>/{sc}/`.")
    return "\n".join(lines), complete


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default="results_v4"); ap.add_argument("--abl-dir", default="results_v4_reward_ablation")
    ap.add_argument("--scenario", default="medium"); ap.add_argument("--method", default="mappo_gcn")
    ap.add_argument("--log", required=True); ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    logs = glob.glob(a.log); assert len(logs) == 1, logs
    tag = a.tag or os.path.basename(logs[0])[:12]
    from datetime import datetime
    body, complete = report(a.base_dir, a.abl_dir, a.scenario, a.method, tag)
    insert_block(logs[0], "reward-ablation", f"### Kết quả tự động — ablation hàm thưởng {a.scenario} (cập nhật {datetime.now():%d/%m/%Y %H:%M})\n\n{body}")
    print(f"{a.scenario}: {'DAY DU' if complete else 'tam thoi'} — da chen vao {logs[0]}")


if __name__ == "__main__":
    main()
