"""Round-2 analysis: bursty / delayed / rate-limited links, the Hard energy control,
and the frontier and Bayesian-search baselines.

    python -m comm_robustness.analyze2

Writes data/summary2.json and data/report2.txt. Statistics follow analyze.py: the
unit is the training seed (n = 10), differences are paired by seed, with a 95 %
percentile bootstrap CI, an exact sign-flip permutation p and Holm correction
inside each named family.
"""
from __future__ import annotations

import io
import json
import os

import numpy as np

from .analyze import boot_ci, holm, paired, sign_flip_p
from .run_eval import (BASELINE_CONDS, BASELINES, COMM2, CONDITIONS, ENERGY2, LEARNED, METHODS,
                       SCENARIOS, SEEDS, raw_path)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
METRICS = ["victim_detection_rate", "coverage_rate", "collision_rate"]
ALIAS = {("mappo_mlp", "message_cut"): "p0", ("mappo_mlp", "belief_cut"): "p100",
         ("greedy", "message_cut"): "p0", ("greedy", "belief_cut"): "p100",
         ("mappo_mlp", "belief_cut_nd"): "p100_nd", ("greedy", "belief_cut_nd"): "p100_nd"}
ALL_METHODS = METHODS + BASELINES


def cell(sc, m, c, metric="victim_detection_rate"):
    src = ALIAS.get((m, c), c)
    out = np.full(len(SEEDS), np.nan)
    for k, s in enumerate(SEEDS):
        p = raw_path(sc, m, s, src)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out[k] = json.load(f)["metrics"][metric]["mean"]
    return out


def alive_frac(sc, m, c):
    """Mean fraction of UAVs alive per step (round-2 cells log alive_uav_steps)."""
    from sar_env import get_scenario
    n = get_scenario(sc).n_uavs
    vals = []
    for s in SEEDS:
        p = raw_path(sc, m, s, ALIAS.get((m, c), c))
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
        steps = rec["metrics"].get("steps")
        L = rec["links"]
        if "alive_uav_steps" in L and L.get("message_steps", 0):
            vals.append(L["alive_uav_steps"] / (n * max(L["message_steps"], 1)))
    return float(np.mean(vals)) if vals else float("nan")


def fam(rows):
    adj = holm([r[-1]["p"] for r in rows])
    for r, a in zip(rows, adj):
        r[-1]["p_holm"] = a
    return rows


def summ(x):
    return {"mean": float(np.nanmean(x)), "sd": float(np.nanstd(x, ddof=1)), "n": int((~np.isnan(x)).sum())}


def main():
    out = {"comm2": {}, "comm2_vs_iid": {}, "energy": {}, "baselines": {}, "gcn_mlp2": {}}
    V = "victim_detection_rate"

    # -------- new channel models: drop vs intact, per scenario x method (family of 6)
    for sc in SCENARIOS:
        for m in ALL_METHODS:
            base = cell(sc, m, "p0")
            rows = []
            for c in COMM2:
                x = cell(sc, m, c)
                st = paired(x, base)
                st["level"] = summ(x)
                for k in ("coverage_rate", "collision_rate"):
                    st[k] = paired(cell(sc, m, c, k), cell(sc, m, "p0", k))
                rows.append([c, st])
            for c, st in fam(rows):
                out["comm2"].setdefault(sc, {}).setdefault(m, {})[c] = st
    # bursty vs independent loss at the same 75 % loss rate (family: per scenario, methods x 2)
    for sc in SCENARIOS:
        rows = []
        for m in ALL_METHODS:
            for c in ("ge75L10", "ge75L50"):
                rows.append([m, c, paired(cell(sc, m, c), cell(sc, m, "p75"))])
        for m, c, st in fam(rows):
            out["comm2_vs_iid"].setdefault(sc, {}).setdefault(m, {})[c] = st
    # GCN - MLP under the new models (family per scenario)
    for sc in SCENARIOS:
        rows = [[c, paired(cell(sc, "mappo_gcn", c), cell(sc, "mappo_mlp", c))] for c in COMM2]
        for c, st in fam(rows):
            out["gcn_mlp2"].setdefault(sc, {})[c] = st

    # -------- Hard energy control
    sc = "hard"
    base_conds = {"p0_nd": "p0", "p75_nd": "p75", "p100_nd": "p100", "belief_cut_nd": "belief_cut",
                  "r25_nd": "r25"}
    for m in METHODS:
        e = {}
        rows, did_rows = [], []
        p0n, p0 = cell(sc, m, "p0_nd"), cell(sc, m, "p0")
        e["intact_normal"] = summ(p0)
        e["intact_no_death"] = summ(p0n)
        e["no_death_gain"] = paired(p0n, p0)
        for cn, c in base_conds.items():
            if cn == "p0_nd":
                continue
            drop_nd = cell(sc, m, cn) - p0n
            drop = cell(sc, m, c) - p0
            st = {"drop_no_death": {"mean": float(np.nanmean(drop_nd)), "ci95": boot_ci(drop_nd),
                                    "p": sign_flip_p(drop_nd)},
                  "drop_normal": {"mean": float(np.nanmean(drop)), "ci95": boot_ci(drop)},
                  "level_no_death": summ(cell(sc, m, cn))}
            did = drop_nd - drop
            st["diff_in_diff"] = {"mean": float(np.nanmean(did)), "ci95": boot_ci(did), "p": sign_flip_p(did)}
            rows.append([cn, st["drop_no_death"]])
            did_rows.append([cn, st["diff_in_diff"]])
            e[cn] = st
        fam(rows)
        fam(did_rows)
        e["alive_frac"] = {c: alive_frac(sc, m, c) for c in ENERGY2}
        out["energy"][m] = e
    rows = []
    for c in ENERGY2:
        rows.append([c, paired(cell(sc, "mappo_gcn", c), cell(sc, "mappo_mlp", c))])
    out["energy"]["gcn_minus_mlp"] = {c: st for c, st in fam(rows)}

    # -------- baselines
    for sc in SCENARIOS:
        b = {}
        for m in BASELINES:
            lv = {c: {k: summ(cell(sc, m, c, k)) for k in METRICS} for c in BASELINE_CONDS + COMM2}
            rows = [[c, paired(cell(sc, m, c), cell(sc, m, "p0"))] for c in BASELINE_CONDS if c != "p0"]
            fam(rows)
            coll = {c: paired(cell(sc, m, c, "collision_rate"), cell(sc, m, "p0", "collision_rate"))
                    for c in ("message_cut", "belief_cut", "p100")}
            b[m] = {"levels": lv, "drop_vs_p0": {c: st for c, st in rows}, "collision_change": coll}
        # every learner and Greedy versus each baseline, intact and blackout (family per scenario)
        rows = []
        for m in METHODS:
            for bl in BASELINES:
                for c in ("p0", "p100"):
                    rows.append([m, bl, c, paired(cell(sc, m, c), cell(sc, bl, c))])
        fam(rows)
        b["method_minus_baseline"] = {}
        for m, bl, c, st in rows:
            b["method_minus_baseline"].setdefault(m, {}).setdefault(bl, {})[c] = st
        out["baselines"][sc] = b

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "summary2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    buf = io.StringIO()
    report(out, buf)
    with open(os.path.join(DATA_DIR, "report2.txt"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(buf.getvalue())


def fmt(st, pct=True):
    k = 100 if pct else 1
    s = f"{st['mean']*k:+.1f}[{st['ci95'][0]*k:+.1f},{st['ci95'][1]*k:+.1f}]"
    if "p" in st:
        s += f" p={st['p']:.3f}"
    if "p_holm" in st:
        s += f"/h{st['p_holm']:.3f}"
    if "wins" in st:
        s += f" {st['wins']}/{st['n']}"
    return s


def report(out, w):
    for sc in SCENARIOS:
        print(f"\n=== {sc}: new channel models, VDR change vs intact (points) ===", file=w)
        for m in ALL_METHODS:
            d = out["comm2"][sc][m]
            print(f"{m:10s} " + "  ".join(f"{c}:{fmt(st)}" for c, st in d.items()), file=w)
        print("collision change (points):", file=w)
        for m in ALL_METHODS:
            d = out["comm2"][sc][m]
            print(f"{m:10s} " + "  ".join(f"{c}:{st['collision_rate']['mean']*100:+.1f}" for c, st in d.items()), file=w)
        print("bursty minus independent at 75 % loss:", file=w)
        for m in ALL_METHODS:
            d = out["comm2_vs_iid"][sc][m]
            print(f"{m:10s} " + "  ".join(f"{c}:{fmt(st)}" for c, st in d.items()), file=w)
        print("GCN-MLP: " + "  ".join(f"{c}:{fmt(st)}" for c, st in out["gcn_mlp2"][sc].items()), file=w)
        b = out["baselines"][sc]
        for m in BASELINES:
            lv = b[m]["levels"]
            print(f"{m:8s} VDR " + " ".join(f"{c}:{lv[c]['victim_detection_rate']['mean']:.3f}" for c in lv), file=w)
            print(f"{m:8s} coll " + " ".join(f"{c}:{lv[c]['collision_rate']['mean']:.3f}" for c in lv), file=w)
            print(f"{m:8s} drop " + "  ".join(f"{c}:{fmt(st)}" for c, st in b[m]["drop_vs_p0"].items()), file=w)
        for m, d in b["method_minus_baseline"].items():
            print(f"{m:10s} - " + "  ".join(f"{bl}@{c}:{fmt(st)}" for bl, dd in d.items() for c, st in dd.items()), file=w)
    print("\n=== Hard energy control (no_death) ===", file=w)
    for m in METHODS:
        e = out["energy"][m]
        print(f"{m:10s} intact {e['intact_normal']['mean']:.3f} -> no-death {e['intact_no_death']['mean']:.3f} "
              f"gain {fmt(e['no_death_gain'])}", file=w)
        for c in ENERGY2[1:]:
            st = e[c]
            print(f"     {c:14s} drop normal {st['drop_normal']['mean']*100:+.1f}  drop no-death {fmt(st['drop_no_death'])}"
                  f"  DiD {fmt(st['diff_in_diff'])}", file=w)
        print(f"     alive frac {e['alive_frac']}", file=w)
    print("GCN-MLP no-death: " + "  ".join(f"{c}:{fmt(st)}" for c, st in out["energy"]["gcn_minus_mlp"].items()), file=w)


if __name__ == "__main__":
    main()
