# Communication degradation at execution time

Code, raw results and figure/table scripts for the paper
*Disentangling Communication Loss in Graph-Based MARL for UAV Search and Rescue*.

The study does not train any new policy. It takes the trained MAPPO policies stored in
[`../results/`](../results) (four encoders: MAPPO-MLP, MAPPO-GCN, MAPPO-GAT and MAPPO-DGAT; 10 seeds;
three scenarios; 120 policies in total), plus the Greedy, frontier-based and Bayesian rule-based
searches. It then breaks the radio links between UAVs at test time and measures how much of the victim
detection rate (VDR) is lost. The two information channels can be cut separately:

- **belief-map sharing**: UAVs in the same connected component merge their belief maps (all methods use it);
- **GNN message passing**: neighbours exchange embeddings inside the graph encoder (GCN, GAT and DGAT only).

Nothing in `sar_env/`, `models/`, `training/` or `evaluation/` is modified. `degraded_env.py` subclasses
`SARSwarmEnv` and adds the link-loss models.

## Requirements

Install the packages in [`../requirements.txt`](../requirements.txt). Run every command below from the
**repository root**, not from this folder.

## Reproducing the results

```bash
# 0. sanity checks
python -m comm_robustness.test_merge                    # properties of the belief merge, Eq. (1)
python -m comm_robustness.run_eval --check --seeds 0    # p0 must equal results/*/eval_metrics.json episode by episode

# 1. round 1: i.i.d. link loss, channel cuts, range reduction
python -m comm_robustness.run_eval --workers 14
python -m comm_robustness.analyze                       # -> data/summary.json (report printed; saved copy: data/report.txt)
python -m comm_robustness.diag_greedy_spread            # -> data/greedy_spread.json

# 2. round 2: bursty loss, delay, bandwidth, no-depletion control, frontier and Bayes baselines
python -m comm_robustness.run_eval --round 2 --workers 14
python -m comm_robustness.analyze2                      # -> data/summary2.json, data/report2.txt
python comm_robustness/diag_hard_alive_end.py           # -> data/hard_alive_end.json

# 3. round 3: measured map gap, one-hop relaying, 2x2 channel design with Shapley split
python -m comm_robustness.round3 --workers 14           # -> results/raw3/
python -m comm_robustness.analyze3                      # -> data/summary3.json

# 4. tables and figures
python -m comm_robustness.make_tables                   # -> en/tables.tex, en/tables.md
python -m comm_robustness.make_figures ieee en          # -> figures/ieee, figures/en (link loss, range, channels)
python -m comm_robustness.make_fig_setup                # -> Fig. 1 and the schematic figures from figures/src/*.html
```

Every (scenario, method, seed, condition) cell is one JSON file under `results/raw/` (rounds 1 and 2) or
`results/raw3/` (round 3). Existing files are skipped, so a run can be stopped and resumed. All raw cells
are included in this repository, so the analysis and plotting steps (`analyze*`, `make_tables`,
`make_figures`) can be run directly without re-running the evaluation.

Notes:

- Test maps, number of episodes and episode seeds are those of the final evaluation in
  `training/train.py` (`seed_start = 90_000 + 1000 * seed`), so condition `p0` reproduces the stored
  evaluation exactly. The check log is `results/check_p0.log`.
- Rounds 1 and 2 together take about 14 hours on a 16-thread CPU, mostly for the Hard scenario. The
  `*.log` files in `results/` are the logs of the original runs.
- `make_fig_setup.py` renders HTML with headless Edge or Chrome. The figures use Times New Roman and
  Cambria Math, so render on a machine that has both fonts (e.g. Windows).
- `make_tables.py` also writes `vi/tables.md`, a Vietnamese copy of two of the tables.

## Conditions

Round 1 (`run_eval.py`, `CONDITIONS`):

| Name | p_belief | p_message | range scale | Meaning |
|---|---|---|---|---|
| `p0` | 0 | 0 | 1 | perfect communication |
| `p25`, `p50`, `p75`, `p100` | p | p | 1 | i.i.d. Bernoulli loss of each link, one draw shared by both channels |
| `belief_cut` | 1 | 0 | 1 | no belief-map merge, GNN messages intact |
| `message_cut` | 0 | 1 | 1 | belief maps merged as usual, the GNN sees no edges |
| `r75`, `r50`, `r25` | 0 | 0 | ρ | radio range reduced to ρ · r_comm, long links break first |

For MAPPO-MLP and Greedy, which do not use GNN messages, `message_cut` equals `p0` and `belief_cut` equals
`p100`, so these cells are copied rather than re-run.

Round 2 (`run_eval.py`, `CONDITIONS2`):

| Name | Meaning |
|---|---|
| `ge75L10`, `ge75L50` | Gilbert–Elliott bursty loss, 75 % of steps in the bad state, mean burst of 10 or 50 steps |
| `delay5`, `delay25` | belief channel only: map updates arrive 5 or 25 steps late |
| `bw100`, `bw25` | belief channel only: at most 100 or 25 map cells per UAV and step, nearest cells first |
| `*_nd` | Hard scenario with energy depletion disabled (UAVs are never removed) |

Round 3 (`round3.py`): the map gap (share of the swarm's scanned cells that a UAV does not know) is
measured for MAPPO-GCN under the round-1 and round-2 communication conditions. The condition `hop1`
replaces the per-component merge with a merge between direct neighbours only, so information travels one
hop per step.

## Statistics

The unit of analysis is the training seed (n = 10). Each seed's value is the mean over its test episodes,
and every comparison is paired by seed: a seed fixes the test maps and the link-loss draws. Reported are
the mean paired difference, a 95 % percentile bootstrap CI (10 000 resamples), the exact two-sided
sign-flip permutation p over the 2^10 sign patterns, and a Holm correction within each family of tests
(see the docstrings of `analyze.py`, `analyze2.py` and `analyze3.py`).

## Files

| Path | Content |
|---|---|
| `degraded_env.py` | environment subclass: two channels, link-loss, delay and bandwidth models |
| `run_eval.py` | evaluation grid for rounds 1 and 2, `--check` reproduction test |
| `baselines.py` | coordinated frontier-based exploration (after Yamauchi, 1997) and Bayesian search (after Bourgault et al., 2003) |
| `round3.py` | map-gap measurement and one-hop relaying |
| `analyze.py`, `analyze2.py`, `analyze3.py` | statistics for rounds 1, 2 and 3 |
| `test_merge.py` | unit checks of the belief merge (commutative, idempotent, path-independent) |
| `diag_greedy_spread.py` | whether Greedy UAVs spread out without shared maps |
| `diag_hard_alive_end.py` | share of UAVs still active at the end of Hard episodes under perfect communication |
| `make_tables.py`, `make_figures.py`, `make_fig_setup.py` | tables and figures, generated from `data/` so no number is copied by hand |
| `data/` | aggregated results (`summary*.json`) and text reports |
| `results/raw/`, `results/raw3/` | per-cell raw results |
| `figures/ieee/`, `figures/en/` | figures (PDF and PNG) at IEEE two-column width and at A4 single-column width |
| `figures/src/` | HTML sources of Fig. 1 and the schematic figures |
| `en/tables.tex`, `en/tables.md` | generated tables |
