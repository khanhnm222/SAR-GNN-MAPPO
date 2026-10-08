# -*- coding: utf-8 -*-
"""Bao cao tu dong cho mot dot lap lai (seed batch A vs batch B) va zero-shot:
sinh bang so (CSV + Markdown), hinh (PNG) va chen khoi Markdown vao file nhat ky
giai doan giua hai moc `<!-- AUTO:<tag> -->` ... `<!-- /AUTO:<tag> -->` (chay lai
thi cap nhat tai cho). Moi so trong nhat ky/luan van/bai bao co the truy ve day.

Vi du:
    python -m scripts.stage_report --scenario medium --log experiment_log/giai-doan-21-*.md
    python -m scripts.stage_report --scenario hard   --log experiment_log/giai-doan-20-*.md
    python -m scripts.stage_report --zero-shot --log experiment_log/giai-doan-21-*.md
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

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
FIG_W, DPI = 6.3, 200
ASSETS = os.path.join("experiment_log", "assets")

METHODS = ["random_walk", "greedy", "maddpg", "mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo"]
LABEL = {"random_walk": "Random Walk", "greedy": "Greedy", "maddpg": "MADDPG", "mappo_mlp": "MAPPO-MLP",
         "mappo_gcn": "MAPPO-GCN", "mappo_gat": "MAPPO-GAT", "gnn_mappo": "MAPPO-DGAT"}
COLOR = {"random_walk": "#94a3b8", "greedy": "#64748b", "maddpg": "#f59e0b", "mappo_mlp": "#ef4444",
         "mappo_gcn": "#7c3aed", "mappo_gat": "#3b82f6", "gnn_mappo": "#10b981"}
METRICS = [("victim_detection_rate", "VDR"), ("coverage_rate", "Coverage"), ("collision_rate", "Collision"),
           ("ttfd", "TTFD")]
PAIRS = [("mappo_gcn", "mappo_mlp"), ("mappo_gat", "mappo_mlp"), ("gnn_mappo", "mappo_mlp"),
         ("gnn_mappo", "mappo_gat"), ("mappo_gat", "mappo_gcn"), ("gnn_mappo", "mappo_gcn"),
         ("mappo_gcn", "greedy"), ("mappo_gat", "greedy"), ("mappo_mlp", "greedy"), ("gnn_mappo", "greedy")]
TRAIN_N = {"easy": 4, "medium": 8, "hard": 16}
STEPS = {"easy": "500k", "medium": "800k", "hard": "1M"}


def vi(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


def load(out_dir, sc):
    """method -> {seed -> {metric -> mean}}"""
    data = {}
    for m in METHODS:
        for p in glob.glob(os.path.join(out_dir, sc, m, "seed_*", "eval_metrics.json")):
            d = json.load(open(p, encoding="utf-8"))
            data.setdefault(m, {})[d["meta"]["seed"]] = {k: v["mean"] for k, v in d["metrics"].items() if isinstance(v, dict)}
    return data


def vals(data, m, k, seeds):
    return [data[m][s][k] for s in seeds if m in data and s in data[m]]


def cohen(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else float("nan")


def msd(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), float("nan"))


# ----------------------------------------------------------------------------- replicate report
def replicate_report(out_dir, sc, batch_a, batch_b, tag):
    data = load(out_dir, sc)
    adir = os.path.join(ASSETS, tag, sc)
    os.makedirs(adir, exist_ok=True)
    lines = []
    nb_done = {m: len(vals(data, m, "victim_detection_rate", batch_b)) for m in METHODS}
    complete = all(nb_done[m] == len(batch_b) for m in METHODS)
    lines.append(f"Trạng thái dữ liệu: đợt A = seed {batch_a[0]}–{batch_a[-1]}, đợt B = seed {batch_b[0]}–{batch_b[-1]}; "
                 f"đợt B đã có " + ", ".join(f"{LABEL[m]} {nb_done[m]}/{len(batch_b)}" for m in METHODS) +
                 (" — **đầy đủ**." if complete else " — **chưa đầy đủ, số tạm thời**."))

    # --- per-seed CSV
    with open(os.path.join(adir, f"per_seed_{sc}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["method", "seed", "batch"] + [k for k, _ in METRICS])
        for m in METHODS:
            for s in sorted(data.get(m, {})):
                w.writerow([m, s, "A" if s in batch_a else "B"] + [data[m][s].get(k) for k, _ in METRICS])

    # --- summary table (A vs B vs pooled) per metric
    for k, kl in METRICS:
        rows = []
        for m in METHODS:
            a, b = vals(data, m, k, batch_a), vals(data, m, k, batch_b)
            ma, sa = msd(a); mb, sb = msd(b); mp, sp_ = msd(a + b)
            p_ab = stats.ttest_ind(a, b, equal_var=False).pvalue if len(a) > 1 and len(b) > 1 else float("nan")
            rows.append([LABEL[m], f"{vi(ma)} ± {vi(sa)}", f"{vi(mb)} ± {vi(sb)} (n={len(b)})", f"{mb - ma:+.3f}".replace(".", ","),
                         vi(p_ab, 2) if p_ab == p_ab else "—", f"{vi(mp)} ± {vi(sp_)} (n={len(a) + len(b)})"])
        with open(os.path.join(adir, f"summary_{sc}_{k}.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([["method", "batch_A", "batch_B", "delta", "p_A_vs_B", "pooled"]] + rows)
        if k == "victim_detection_rate" or k == "coverage_rate":
            lines.append(f"\n**{kl} theo đợt** (mean ± sd liên-seed; p = Welch đợt A vs B):\n")
            lines.append("| Phương pháp | Đợt A | Đợt B | Δ (B−A) | p | Gộp |\n|---|---|---|---|---|---|")
            lines += ["| " + " | ".join(r) + " |" for r in rows]

    # --- pairwise tests on pooled
    pooled_seeds = batch_a + batch_b
    lines.append("\n**Kiểm định gộp** (Welch t / Mann–Whitney / Cohen's d; theo cặp seed = cùng bộ bản đồ test; "
                 "α' Bonferroni = 0,0167 cho 3 so sánh với MLP):\n")
    lines.append("| Chỉ số | So sánh | n | Δ | p Welch | p MW | d | Theo cặp: thắng / p | Bonferroni |\n|---|---|---|---|---|---|---|---|---|")
    test_rows = []
    for k, kl in METRICS[:2]:
        for a_m, b_m in PAIRS:
            a, b = vals(data, a_m, k, pooled_seeds), vals(data, b_m, k, pooled_seeds)
            if len(a) < 2 or len(b) < 2:
                continue
            t = stats.ttest_ind(a, b, equal_var=False); u = stats.mannwhitneyu(a, b, alternative="two-sided")
            d = cohen(a, b)
            common = [s for s in pooled_seeds if s in data.get(a_m, {}) and s in data.get(b_m, {})]
            pa = [data[a_m][s][k] for s in common]; pb = [data[b_m][s][k] for s in common]
            wins = sum(x > y for x, y in zip(pa, pb))
            pp = stats.ttest_rel(pa, pb).pvalue if len(common) > 2 else float("nan")
            bonf = "đạt" if t.pvalue < 0.05 / 3 else ("*" if t.pvalue < 0.05 else "không")
            row = [kl, f"{LABEL[a_m]} vs {LABEL[b_m]}", f"{len(a)}/{len(b)}", f"{st.mean(a) - st.mean(b):+.3f}".replace(".", ","),
                   vi(t.pvalue, 4), vi(u.pvalue, 4), f"{d:+.2f}".replace(".", ","), f"{wins}/{len(common)} / {vi(pp, 4)}", bonf]
            test_rows.append(row); lines.append("| " + " | ".join(row) + " |")
    with open(os.path.join(adir, f"tests_{sc}.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["metric", "pair", "n", "delta", "p_welch", "p_mw", "cohen_d", "paired_wins_p", "bonferroni"]] + test_rows)

    # --- per-seed difficulty (mean over learned+greedy methods)
    diff = []
    for s in pooled_seeds:
        v = [data[m][s]["victim_detection_rate"] for m in ["greedy", "mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo"]
             if m in data and s in data[m]]
        if v:
            diff.append((s, st.mean(v)))
    lines.append("\nĐộ khó khối bản đồ test theo seed (VDR trung bình 5 phương pháp không ngẫu nhiên): " +
                 ", ".join(f"seed {s}: {vi(v)}" for s, v in diff))

    # --- figure 1: batches
    fig, ax = plt.subplots(figsize=(FIG_W, 5.0), dpi=DPI)
    ms = [m for m in METHODS if m in data]
    y = np.arange(len(ms)); h = 0.26
    for off, (seeds, lab, alpha) in enumerate([(batch_a, f"Đợt A (seed {batch_a[0]}–{batch_a[-1]})", 0.55),
                                                 (batch_b, f"Đợt B (seed {batch_b[0]}–{batch_b[-1]})", 0.85),
                                                 (pooled_seeds, "Gộp", 1.0)]):
        mu = [msd(vals(data, m, "victim_detection_rate", seeds))[0] for m in ms]
        sd = [msd(vals(data, m, "victim_detection_rate", seeds))[1] for m in ms]
        ax.barh(y + (off - 1) * h, mu, h, xerr=sd, color=[COLOR[m] for m in ms], alpha=alpha,
                edgecolor="black" if off == 2 else "none", linewidth=0.6, label=lab, error_kw=dict(lw=0.8, capsize=2))
    ax.set_yticks(y); ax.set_yticklabels([LABEL[m] for m in ms]); ax.invert_yaxis()
    ax.set_xlabel("Victim Detection Rate"); ax.grid(axis="x", alpha=0.3)
    ax.set_title(f"{sc.capitalize()} ({TRAIN_N[sc]} UAV, {STEPS[sc]} bước): hai đợt seed độc lập và gộp", fontsize=12)
    ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.tight_layout(); f1 = os.path.join(adir, f"fig_batches_{sc}.png"); fig.savefig(f1, bbox_inches="tight"); plt.close(fig)

    # --- figure 2: per-seed dots with paired GCN-MLP lines
    fig, ax = plt.subplots(figsize=(FIG_W, 4.2), dpi=DPI)
    for i, m in enumerate(["mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo", "greedy"]):
        if m not in data:
            continue
        for s in sorted(data[m]):
            ax.scatter(i + (0.12 if s in batch_b else -0.12), data[m][s]["victim_detection_rate"], color=COLOR[m],
                       marker="o" if s in batch_a else "s", s=36, edgecolor="black", linewidth=0.4, zorder=3)
    for s in pooled_seeds:
        if "mappo_mlp" in data and "mappo_gcn" in data and s in data["mappo_mlp"] and s in data["mappo_gcn"]:
            ax.plot([0 + (0.12 if s in batch_b else -0.12), 1 + (0.12 if s in batch_b else -0.12)],
                    [data["mappo_mlp"][s]["victim_detection_rate"], data["mappo_gcn"][s]["victim_detection_rate"]],
                    color="#94a3b8", lw=0.7, zorder=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(["MAPPO-MLP", "MAPPO-GCN", "MAPPO-GAT", "MAPPO-DGAT", "Greedy"], fontsize=10.5)
    ax.set_ylabel("VDR từng seed"); ax.grid(axis="y", alpha=0.3)
    ax.scatter([], [], marker="o", color="grey", label=f"đợt A (seed {batch_a[0]}–{batch_a[-1]})")
    ax.scatter([], [], marker="s", color="grey", label=f"đợt B (seed {batch_b[0]}–{batch_b[-1]})")
    ax.plot([], [], color="#94a3b8", lw=0.7, label="cặp cùng seed MLP→GCN")
    ax.legend(fontsize=9.5, loc="lower left"); ax.set_title(f"{sc.capitalize()}: giá trị từng seed", fontsize=12)
    fig.tight_layout(); f2 = os.path.join(adir, f"fig_perseed_{sc}.png"); fig.savefig(f2); plt.close(fig)

    # --- figure 3: learning curves batch B vs A
    fig, ax = plt.subplots(figsize=(FIG_W, 4.4), dpi=DPI)
    for m in ["mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo"]:
        for seeds, ls in [(batch_a, "--"), (batch_b, "-")]:
            curves = []
            for s in seeds:
                p = os.path.join(out_dir, sc, m, f"seed_{s}", "learning_curve.json")
                if not os.path.exists(p):
                    continue
                c = json.load(open(p))["curve"]
                pts = [(x["global_step"], x["eval_vdr"]) for x in c if "eval_vdr" in x]
                if pts:
                    curves.append(pts)
            if not curves:
                continue
            n = min(len(c) for c in curves)
            xs = np.array([c[i][0] for i in range(n) for c in curves[:1]])
            ys = np.array([[c[i][1] for i in range(n)] for c in curves])
            ax.plot(xs / 1e3, ys.mean(0), ls, color=COLOR[m], lw=1.6,
                    label=f"{LABEL[m]} {'đợt B' if ls == '-' else 'đợt A'} (n={len(curves)})")
            ax.fill_between(xs / 1e3, ys.mean(0) - ys.std(0), ys.mean(0) + ys.std(0), color=COLOR[m], alpha=0.08)
    ax.set_xlabel("bước môi trường (nghìn)"); ax.set_ylabel("VDR đánh giá trung gian"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, ncol=2); ax.set_title(f"{sc.capitalize()}: learning curve đợt B (liền) so với đợt A (đứt)", fontsize=12)
    fig.tight_layout(); f3 = os.path.join(adir, f"fig_curves_{sc}.png"); fig.savefig(f3); plt.close(fig)

    rel = lambda p: os.path.relpath(p, "experiment_log").replace("\\", "/")
    lines.append(f"\n![Hai đợt seed và gộp — {sc}]({rel(f1)})\n\n![Giá trị từng seed — {sc}]({rel(f2)})\n\n![Learning curve — {sc}]({rel(f3)})")
    lines.append(f"\nDữ liệu thô: `{rel(adir)}/per_seed_{sc}.csv`, `summary_{sc}_*.csv`, `tests_{sc}.csv`.")
    return "\n".join(lines), complete


# ----------------------------------------------------------------------------- zero-shot report
def zero_shot_report(out_dir, tag):
    adir = os.path.join(ASSETS, tag, "zero_shot"); os.makedirs(adir, exist_ok=True)
    lines = []
    for mode, fname, title in [("fixed_map", "zero_shot_all.json", "fixed-map"), ("density", "zero_shot_density_all.json", "density-controlled")]:
        fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 9.0), dpi=DPI)
        any_data = False
        for ax, sc in zip(axes, ["easy", "medium", "hard"]):
            p = os.path.join(out_dir, sc, fname)
            if not os.path.exists(p):
                ax.set_title(f"{sc}: chưa có dữ liệu {title}", fontsize=11); ax.axis("off"); continue
            raw = json.load(open(p, encoding="utf-8")); any_data = True
            ns = sorted({int(n) for m in raw for s in raw[m] for n in raw[m][s]})
            rows = []
            for m in ["mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo"]:
                if m not in raw:
                    continue
                mu, sd = [], []
                for n in ns:
                    v = [raw[m][s][str(n)]["victim_detection_rate"]["mean"] for s in raw[m] if str(n) in raw[m][s]]
                    mu.append(st.mean(v) if v else np.nan); sd.append(st.stdev(v) if len(v) > 1 else 0)
                ax.errorbar(ns, mu, yerr=sd, color=COLOR[m], marker="o", ms=4, lw=1.5, capsize=2, label=f"{LABEL[m]} (n={len(raw[m])} seed)")
                rows.append([LABEL[m], len(raw[m])] + [vi(x) for x in mu])
            ax.axvline(TRAIN_N[sc], color="grey", ls="--", lw=1); ax.set_title(f"{sc.capitalize()} — huấn luyện N={TRAIN_N[sc]}", fontsize=11)
            ax.set_ylabel("VDR"); ax.grid(alpha=0.3); ax.legend(fontsize=8.5)
            ax.set_xticks(ns)
            with open(os.path.join(adir, f"zeroshot_{mode}_{sc}.csv"), "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows([["method", "n_seeds"] + [f"N={n}" for n in ns]] + rows)
            lines.append(f"\n**Zero-shot {title} — {sc}** (VDR trung bình theo N; N huấn luyện = {TRAIN_N[sc]}):\n")
            lines.append("| Phương pháp | seed | " + " | ".join(f"N={n}" for n in ns) + " |\n|---|---|" + "---|" * len(ns))
            lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
            envp = p.replace(".json", "_env.json")
            if os.path.exists(envp):
                env = json.load(open(envp, encoding="utf-8"))
                lines.append("Bản đồ theo N: " + ", ".join(f"N={n}: {env[str(n)]['width']}×{env[str(n)]['height']}, nạn nhân {env[str(n)]['n_victims_range']}" for n in ns if str(n) in env))
        axes[-1].set_xlabel("Số UAV lúc kiểm thử N")
        fig.suptitle(f"Zero-shot {title}: VDR theo N (đường đứt = N huấn luyện)", fontsize=12)
        fig.tight_layout(); fp = os.path.join(adir, f"fig_zeroshot_{mode}.png"); fig.savefig(fp); plt.close(fig)
        if any_data:
            lines.append(f"\n![Zero-shot {title}]({os.path.relpath(fp, 'experiment_log').replace(chr(92), '/')})")
    return "\n".join(lines)


def insert_block(log_path, tag, body):
    start, end = f"<!-- AUTO:{tag} -->", f"<!-- /AUTO:{tag} -->"
    s = open(log_path, encoding="utf-8").read()
    block = f"{start}\n{body}\n{end}"
    if start in s and end in s:
        s = s[: s.index(start)] + block + s[s.index(end) + len(end):]
    else:
        # chen truoc footer nav ("\n---\n" cuoi cung)
        k = s.rstrip().rfind("\n---\n")
        s = s[:k] + "\n" + block + "\n" + s[k:]
    open(log_path, "w", encoding="utf-8").write(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results_v4")
    ap.add_argument("--scenario", choices=["easy", "medium", "hard"])
    ap.add_argument("--batch-a", default="0-4"); ap.add_argument("--batch-b", default="5-9")
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--tag", default=None, help="thu muc con trong experiment_log/assets (mac dinh: ten file log)")
    ap.add_argument("--log", required=True, help="file nhat ky giai doan (glob cho phep)")
    a = ap.parse_args()
    logs = glob.glob(a.log); assert len(logs) == 1, logs
    log = logs[0]
    tag = a.tag or os.path.basename(log)[:12]
    rng = lambda s: list(range(int(s.split("-")[0]), int(s.split("-")[1]) + 1))
    from datetime import datetime
    stamp = datetime.now().strftime("%d/%m/%Y %H:%M")
    if a.scenario:
        body, complete = replicate_report(a.out_dir, a.scenario, rng(a.batch_a), rng(a.batch_b), tag)
        insert_block(log, f"rep-{a.scenario}", f"### Kết quả tự động — {a.scenario} (cập nhật {stamp})\n\n{body}")
        print(f"{a.scenario}: {'DAY DU' if complete else 'tam thoi'} — da chen vao {log}")
    if a.zero_shot:
        body = zero_shot_report(a.out_dir, tag)
        insert_block(log, "zero-shot", f"### Kết quả tự động — zero-shot (cập nhật {stamp})\n{body}")
        print(f"zero-shot: da chen vao {log}")


if __name__ == "__main__":
    main()
