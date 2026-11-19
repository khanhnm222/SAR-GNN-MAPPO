"""Aggregate results/raw/ into data/summary.json and the paper tables.

    python -m comm_robustness.analyze

Unit of analysis is the training seed (n = 10 per method and scenario): each
seed's value is the mean over its test episodes. Every comparison is paired by
seed, which is valid because a seed fixes the test maps and, through the
episode seed, the link-loss draws, so two cells with the same seed differ only
in the policy or in the condition.

Statistics: mean seed-level difference, 95 % percentile
bootstrap CI over seeds (10 000 resamples), exact two-sided sign-flip
permutation p over the 2^10 sign patterns, and Holm correction within each
family of comparisons: one degradation curve (scenario x method, nine
conditions), GCN - MLP over the ten conditions (per scenario), the channel
decomposition (per scenario) and learner - Greedy (per scenario).
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np

from .run_eval import CONDITIONS, LEARNED, METHODS, RAW_DIR, SCENARIOS, SEEDS, raw_path

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
METRICS = ["victim_detection_rate", "coverage_rate", "collision_rate", "ttfd"]

# MLP and Greedy ignore the message channel, so these cells are copies (run_eval.needed).
ALIAS = {("mappo_mlp", "message_cut"): "p0", ("mappo_mlp", "belief_cut"): "p100",
         ("greedy", "message_cut"): "p0", ("greedy", "belief_cut"): "p100"}


def load():
    """vals[scenario][method][cond][metric] -> array over seeds (NaN if missing)."""
    vals, links = {}, {}
    for sc, m, c in itertools.product(SCENARIOS, METHODS, CONDITIONS):
        src = ALIAS.get((m, c), c)
        arr = {k: np.full(len(SEEDS), np.nan) for k in METRICS}
        lk = []
        for i, s in enumerate(SEEDS):
            p = raw_path(sc, m, s, src)
            if not os.path.exists(p):
                lk.append(None)
                continue
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            for k in METRICS:
                arr[k][i] = rec["metrics"][k]["mean"]
            lk.append(rec["links"])
        vals.setdefault(sc, {}).setdefault(m, {})[c] = arr
        links.setdefault(sc, {}).setdefault(m, {})[c] = lk
    return vals, links


def sign_flip_p(d: np.ndarray) -> float:
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return float("nan")
    obs = abs(d.mean())
    signs = np.array(list(itertools.product([1, -1], repeat=n)))
    null = np.abs((signs * d).mean(axis=1))
    return float((null >= obs - 1e-12).mean())


def boot_ci(d: np.ndarray, n_boot: int = 10_000, seed: int = 12345):
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[k]))
        adj[k] = running
    return adj.tolist()


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Statistics of a - b paired by seed."""
    d = a - b
    ok = ~np.isnan(d)
    return {"mean": float(np.nanmean(d)), "ci95": boot_ci(d), "p": sign_flip_p(d),
            "wins": int((d[ok] > 0).sum()), "n": int(ok.sum())}


def summarise(x: np.ndarray) -> dict:
    return {"mean": float(np.nanmean(x)), "sd": float(np.nanstd(x, ddof=1)), "n": int((~np.isnan(x)).sum())}


def link_summary(lk: list) -> dict:
    lk = [l for l in lk if l]
    if not lk:
        return {}
    tot = {k: sum(l[k] for l in lk) for k in lk[0]}
    out = {}
    if tot["belief_links_nominal"]:
        out["belief_kept_frac"] = tot["belief_links_kept"] / tot["belief_links_nominal"]
    if tot["message_links_nominal"]:
        out["message_kept_frac"] = tot["message_links_kept"] / tot["message_links_nominal"]
    if tot["message_node_steps"]:
        out["isolated_frac"] = tot["message_isolated_node_steps"] / tot["message_node_steps"]
        out["mean_degree"] = 2 * tot["message_links_kept"] / tot["message_node_steps"]
    return out


def main():
    vals, links = load()
    V = "victim_detection_rate"
    out = {"n_seeds": len(SEEDS), "conditions": {k: list(v) for k, v in CONDITIONS.items()},
           "cells": {}, "links": {}, "drop_vs_p0": {}, "gcn_minus_mlp": {}, "channels": {},
           "learned_minus_greedy": {}}

    missing = [(sc, m, c) for sc in SCENARIOS for m in METHODS for c in CONDITIONS
               if np.isnan(vals[sc][m][c][V]).any()]
    out["missing_cells"] = len(missing)

    for sc in SCENARIOS:
        for m in METHODS:
            for c in CONDITIONS:
                out["cells"].setdefault(sc, {}).setdefault(m, {})[c] = {
                    k: summarise(vals[sc][m][c][k]) for k in METRICS}
                out["links"].setdefault(sc, {}).setdefault(m, {})[c] = link_summary(links[sc][m][c])

    # Family 1, per scenario and method: VDR(condition) - VDR(p0) over the nine degraded
    # conditions (one degradation curve is one family). Coverage and collision rate get the
    # same paired statistics without correction, as secondary metrics.
    for sc in SCENARIOS:
        for m in METHODS:
            rows = [(c, paired(vals[sc][m][c][V], vals[sc][m]["p0"][V])) for c in CONDITIONS if c != "p0"]
            adj = holm([r[1]["p"] for r in rows])
            for (c, st), pa in zip(rows, adj):
                st["p_holm"] = pa
                for k in ("coverage_rate", "collision_rate"):
                    st[k] = paired(vals[sc][m][c][k], vals[sc][m]["p0"][k])
                out["drop_vs_p0"].setdefault(sc, {}).setdefault(m, {})[c] = st

    # Family 2, per scenario: GCN - MLP at each condition (does the graph advantage survive?).
    for sc in SCENARIOS:
        rows = [(c, paired(vals[sc]["mappo_gcn"][c][V], vals[sc]["mappo_mlp"][c][V])) for c in CONDITIONS]
        adj = holm([r[1]["p"] for r in rows])
        for (c, st), pa in zip(rows, adj):
            st["p_holm"] = pa
            out["gcn_minus_mlp"].setdefault(sc, {})[c] = st

    # Family 3, per scenario: channel decomposition for the graph encoders.
    # loss_belief = p0 - belief_cut, loss_message = p0 - message_cut, loss_both = p0 - p100,
    # interaction = loss_both - loss_belief - loss_message (paired by seed).
    for sc in SCENARIOS:
        rows = []
        for m in LEARNED:
            base = vals[sc][m]["p0"][V]
            lb = base - vals[sc][m]["belief_cut"][V]
            lm = base - vals[sc][m]["message_cut"][V]
            lboth = base - vals[sc][m]["p100"][V]
            inter = lboth - lb - lm
            for name, d in (("loss_belief", lb), ("loss_message", lm), ("loss_both", lboth),
                            ("interaction", inter)):
                st = {"mean": float(np.nanmean(d)), "ci95": boot_ci(d), "p": sign_flip_p(d),
                      "n": int((~np.isnan(d)).sum())}
                rows.append((m, name, st))
        adj = holm([r[2]["p"] for r in rows])
        for (m, name, st), pa in zip(rows, adj):
            st["p_holm"] = pa
            out["channels"].setdefault(sc, {}).setdefault(m, {})[name] = st

    # Family 4, per scenario: each learner minus Greedy under p0 and full blackout p100.
    for sc in SCENARIOS:
        rows = [(m, c, paired(vals[sc][m][c][V], vals[sc]["greedy"][c][V]))
                for m in LEARNED for c in ("p0", "p100")]
        adj = holm([r[2]["p"] for r in rows])
        for (m, c, st), pa in zip(rows, adj):
            st["p_holm"] = pa
            out["learned_minus_greedy"].setdefault(sc, {}).setdefault(m, {})[c] = st

    # Family 5, per scenario: shortened range minus independent loss at a similar fraction of
    # surviving links (r50 keeps ~45 %, p50 50 %; r25 keeps ~23 %, p75 25 %).
    out["range_minus_iid"] = {}
    for sc in SCENARIOS:
        rows = [(m, f"{r}-{q}", paired(vals[sc][m][r][V], vals[sc][m][q][V]))
                for m in METHODS for r, q in (("r50", "p50"), ("r25", "p75"))]
        adj = holm([r[2]["p"] for r in rows])
        for (m, name, st), pa in zip(rows, adj):
            st["p_holm"] = pa
            out["range_minus_iid"].setdefault(sc, {}).setdefault(m, {})[name] = st

    # Mean degree of the message graph per seed (graph encoders log it), p0 vs belief_cut:
    # does the swarm stay closer together when maps are not shared?
    out["degree_belief_cut"] = {}
    for sc in SCENARIOS:
        for m in LEARNED:
            d0, d1 = [], []
            for a, b in zip(links[sc][m]["p0"], links[sc][m]["belief_cut"]):
                d0.append(2 * a["message_links_nominal"] / a["message_node_steps"])
                d1.append(2 * b["message_links_nominal"] / b["message_node_steps"])
            out["degree_belief_cut"].setdefault(sc, {})[m] = {
                "p0": float(np.mean(d0)), "belief_cut": float(np.mean(d1)),
                "diff": paired(np.array(d1), np.array(d0))}

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print_report(out)


def print_report(out):
    V = "victim_detection_rate"
    print(f"missing cells: {out['missing_cells']}")
    for sc in SCENARIOS:
        print(f"\n=== {sc} : VDR mean (sd) by condition ===")
        print("method      " + " ".join(f"{c:>12s}" for c in CONDITIONS))
        for m in METHODS:
            cells = out["cells"][sc][m]
            print(f"{m:11s} " + " ".join(f"{cells[c][V]['mean']:.3f}({cells[c][V]['sd']:.3f})" for c in CONDITIONS))
        print("links (GCN): " + "  ".join(
            f"{c}:kept={out['links'][sc]['mappo_gcn'][c].get('message_kept_frac', float('nan')):.2f}"
            f"/iso={out['links'][sc]['mappo_gcn'][c].get('isolated_frac', float('nan')):.2f}"
            f"/deg={out['links'][sc]['mappo_gcn'][c].get('mean_degree', float('nan')):.2f}"
            for c in CONDITIONS))
        print("GCN-MLP: " + "  ".join(
            f"{c}:{st['mean']:+.3f}[{st['ci95'][0]:+.3f},{st['ci95'][1]:+.3f}]p={st['p']:.3f}/h{st['p_holm']:.3f} {st['wins']}/{st['n']}"
            for c, st in out["gcn_minus_mlp"][sc].items()))
        for m in METHODS:
            dr = out["drop_vs_p0"][sc][m]
            print(f"drop {m:10s} " + "  ".join(
                f"{c}:{st['mean']:+.3f}h{st['p_holm']:.3f}/cov{st['coverage_rate']['mean']:+.3f}"
                for c, st in dr.items()))
        for m in LEARNED:
            ch = out["channels"][sc][m]
            print(f"channels {m:10s} " + "  ".join(
                f"{k}:{v['mean']:+.3f}[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]p={v['p']:.3f}/h{v['p_holm']:.3f}"
                for k, v in ch.items()))
        for m in METHODS:
            r = out["range_minus_iid"][sc][m]
            print(f"range-iid {m:10s} " + "  ".join(
                f"{k}:{st['mean']:+.3f}[{st['ci95'][0]:+.3f},{st['ci95'][1]:+.3f}]p={st['p']:.3f}/h{st['p_holm']:.3f} {st['wins']}/{st['n']}"
                for k, st in r.items()))
        for m in LEARNED:
            dg = out["degree_belief_cut"][sc][m]
            print(f"degree {m:10s} p0={dg['p0']:.2f} belief_cut={dg['belief_cut']:.2f} p={dg['diff']['p']:.3f} {dg['diff']['wins']}/10")
        for m in LEARNED:
            g = out["learned_minus_greedy"][sc][m]
            print(f"vs greedy {m:10s} " + "  ".join(
                f"{c}:{st['mean']:+.3f}p={st['p']:.3f}/h{st['p_holm']:.3f} {st['wins']}/{st['n']}" for c, st in g.items()))


if __name__ == "__main__":
    main()
