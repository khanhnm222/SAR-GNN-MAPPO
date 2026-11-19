"""Round-3 analysis (15/11/2026), answering a pre-submission review of 14/11.

  * factorial: the 2 x 2 channel design per encoder and scenario, written as
    effects on VDR (negative = drop), Y00 perfect, Y10 belief cut, Y01 message cut,
    Y11 both cut:
        dB = Y10 - Y00, dM = Y01 - Y00, I = Y11 - Y10 - Y01 + Y00,
    plus the Shapley split of the total effect Y11 - Y00 into a belief part
    phiB = (dB + (Y11 - Y01)) / 2 and a message part phiM = (dM + (Y11 - Y10)) / 2.
    CIs are seed bootstraps of the per-seed values, p values exact sign-flip, Holm
    over the three tests (dB, dM, I) of an encoder and scenario.
  * msg_bounds: for every learned graph encoder and scenario, the 95 % CI of the
    message-cut effect, to say which effect sizes the data exclude.
  * ttfd / collisions under the channel conditions (review issue 7).
  * gap: map gap of MAPPO-GCN under every condition of round 3 and the seed-level
    association between the map gap and the VDR change (review issue 10).
  * hop1: one-hop-per-step map relay versus the instantaneous component merge.

    python -m comm_robustness.analyze3
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np

from .analyze import boot_ci, holm, sign_flip_p
from .run_eval import HERE, raw_path

RAW3 = os.path.join(HERE, "results", "raw3")
SC = ["easy", "medium", "hard"]
GRAPH = ["mappo_gcn", "mappo_gat", "gnn_mappo"]
SEEDS = range(10)


def vals(sc, m, cond, key="victim_detection_rate", raw=None):
    out = np.full(10, np.nan)
    for s in SEEDS:
        p = (os.path.join(raw, sc, m, f"seed_{s}", f"{cond}.json") if raw
             else raw_path(sc, m, s, cond))
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out[s] = json.load(f)["metrics"][key]["mean"]
    return out


def stat(d):
    d = np.asarray(d, float)
    return {"mean": float(np.nanmean(d)), "ci95": boot_ci(d), "p": sign_flip_p(d),
            "n": int((~np.isnan(d)).sum()), "neg": int((d < 0).sum())}


def factorial():
    out = {}
    for sc, m in itertools.product(SC, GRAPH):
        y00, y10 = vals(sc, m, "p0"), vals(sc, m, "belief_cut")
        y01, y11 = vals(sc, m, "message_cut"), vals(sc, m, "p100")
        dB, dM, dBoth = y10 - y00, y01 - y00, y11 - y00
        inter = y11 - y10 - y01 + y00
        phiB = 0.5 * (dB + (y11 - y01))
        phiM = 0.5 * (dM + (y11 - y10))
        rec = {"dB": stat(dB), "dM": stat(dM), "I": stat(inter), "both": stat(dBoth),
               "phiB": stat(phiB), "phiM": stat(phiM)}
        ph = holm([rec[k]["p"] for k in ("dB", "dM", "I")])
        for k, p in zip(("dB", "dM", "I"), ph):
            rec[k]["p_holm"] = p
        tot = np.nanmean(dBoth)
        rec["share_conditional"] = float(np.nanmean(dB) / tot)
        rec["share_shapley"] = float(np.nanmean(phiB) / tot)
        out.setdefault(sc, {})[m] = rec
    return out


def ttfd_collisions():
    out = {}
    methods = ["mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo", "greedy", "frontier", "bayes"]
    for sc, m in itertools.product(SC, methods):
        rec = {}
        for c in ["p0", "belief_cut", "message_cut", "p100"]:
            src = c
            if m in ("mappo_mlp", "greedy") and c == "message_cut":
                src = "p0"
            if m in ("mappo_mlp", "greedy") and c == "belief_cut":
                src = "p100"
            rec[c] = {"ttfd": float(np.nanmean(vals(sc, m, src, "ttfd"))),
                      "collision": float(np.nanmean(vals(sc, m, src, "collision_rate")))}
        for c in ["belief_cut", "p100"]:
            src = "p100" if (m in ("mappo_mlp", "greedy") and c == "belief_cut") else c
            d = vals(sc, m, src, "ttfd") - vals(sc, m, "p0", "ttfd")
            rec[c]["ttfd_diff"] = stat(d)
        out.setdefault(sc, {})[m] = rec
    return out


def gap():
    if not os.path.isdir(RAW3):
        return {}
    conds = ["p0", "p50", "p75", "p100", "belief_cut", "message_cut", "r75", "r50", "r25",
             "ge75L10", "ge75L50", "delay5", "delay25", "bw100", "bw25", "hop1"]
    out = {}
    for sc in SC:
        base_v = vals(sc, "mappo_gcn", "p0", raw=RAW3)
        rows, xs, ys = {}, [], []
        for c in conds:
            g = vals(sc, "mappo_gcn", c, "gap", raw=RAW3)
            w = vals(sc, "mappo_gcn", c, "wgap", raw=RAW3)
            v = vals(sc, "mappo_gcn", c, raw=RAW3)
            if np.all(np.isnan(v)):
                continue
            rows[c] = {"gap": float(np.nanmean(g)), "wgap": float(np.nanmean(w)),
                       "vdr": float(np.nanmean(v)), "d_vdr": stat(v - base_v)}
            if c != "p0":
                xs.append(w - vals(sc, "mappo_gcn", "p0", "wgap", raw=RAW3))
                ys.append(v - base_v)
        if xs:
            x, y = np.concatenate(xs), np.concatenate(ys)
            ok = ~(np.isnan(x) | np.isnan(y))
            x, y = x[ok], y[ok]
            # Spearman over seed-condition pairs (no scipy needed)
            rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
            rho = float(np.corrcoef(rx, ry)[0, 1])
            cond_x = [rows[c]["wgap"] - rows["p0"]["wgap"] for c in rows if c != "p0"]
            cond_y = [rows[c]["d_vdr"]["mean"] for c in rows if c != "p0"]
            rcx, rcy = np.argsort(np.argsort(cond_x)), np.argsort(np.argsort(cond_y))
            out[sc] = {"rows": rows, "spearman_seed_level": rho, "n_pairs": int(len(x)),
                       "spearman_condition_level": float(np.corrcoef(rcx, rcy)[0, 1]),
                       "n_conditions": len(cond_x)}
        else:
            out[sc] = {"rows": rows}
    return out


def hop1():
    if not os.path.isdir(RAW3):
        return {}
    out = {}
    for sc, m in itertools.product(SC, ["mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo", "greedy"]):
        a, b = vals(sc, m, "hop1", raw=RAW3), vals(sc, m, "p0", raw=RAW3)
        if np.all(np.isnan(a)) or np.all(np.isnan(b)):
            continue
        out.setdefault(sc, {})[m] = {"vdr_hop1": float(np.nanmean(a)), "vdr_p0": float(np.nanmean(b)),
                                     "diff": stat(a - b),
                                     "wgap_hop1": float(np.nanmean(vals(sc, m, "hop1", "wgap", raw=RAW3))),
                                     "wgap_p0": float(np.nanmean(vals(sc, m, "p0", "wgap", raw=RAW3)))}
    for sc in out:
        ms = list(out[sc])
        ph = holm([out[sc][m]["diff"]["p"] for m in ms])
        for m, p in zip(ms, ph):
            out[sc][m]["diff"]["p_holm"] = p
    return out


def main():
    res = {"factorial": factorial(), "ttfd_collisions": ttfd_collisions(), "gap": gap(), "hop1": hop1()}
    path = os.path.join(HERE, "data", "summary3.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    pp = lambda r: f"{100*r['mean']:+.1f} [{100*r['ci95'][0]:+.1f},{100*r['ci95'][1]:+.1f}]"
    print("FACTORIAL (VDR pts)")
    for sc, m in itertools.product(SC, GRAPH):
        r = res["factorial"][sc][m]
        print(f"{sc:6s} {m:10s} dB {pp(r['dB'])} pH={r['dB']['p_holm']:.3f} | dM {pp(r['dM'])} "
              f"pH={r['dM']['p_holm']:.3f} | I {pp(r['I'])} praw={r['I']['p']:.3f} pH={r['I']['p_holm']:.3f}"
              f" | both {pp(r['both'])} | share cond {r['share_conditional']:.2f} shapley {r['share_shapley']:.2f}")
    print("TTFD / COLLISIONS")
    for sc, m in itertools.product(SC, res["ttfd_collisions"]["easy"]):
        r = res["ttfd_collisions"][sc][m]
        print(f"{sc:6s} {m:10s} " + " ".join(
            f"{c}: ttfd {r[c]['ttfd']:.1f} col {100*r[c]['collision']:.1f}%" for c in r))
    for sc, g in res["gap"].items():
        print("GAP", sc, {k: g[k] for k in g if k != "rows"})
        for c, r in g["rows"].items():
            print(f"   {c:12s} gap {r['gap']:.3f} wgap {r['wgap']:.3f} vdr {r['vdr']:.3f} d {pp(r['d_vdr'])}")
    for sc, h in res["hop1"].items():
        for m, r in h.items():
            print(f"HOP1 {sc:6s} {m:10s} {r['vdr_p0']:.3f}->{r['vdr_hop1']:.3f} {pp(r['diff'])} "
                  f"pH={r['diff'].get('p_holm', float('nan')):.3f} wgap {r['wgap_p0']:.3f}->{r['wgap_hop1']:.3f}")


if __name__ == "__main__":
    main()
