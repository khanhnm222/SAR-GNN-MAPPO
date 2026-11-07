# GNN-MAPPO for UAV swarm coordination in search and rescue (SAR)

**English** | [Tiếng Việt](README.vi.md)

Experimental implementation for the master's thesis proposal *"Building and
evaluating a GNN-MAPPO framework for UAV swarm coordination in search and
rescue in a simulated environment"*. See [PLAN.md](PLAN.md) for the detailed
plan and the reduced (POC) scope of this run.

## Installation

```bash
pip install -r requirements.txt
```

## Validate the environment

```bash
python -m sar_env.validation
```

## Train one configuration

```bash
python -m training.train --method gnn_mappo --scenario easy --seed 0
```

`--method` ∈ `{random_walk, greedy, maddpg, mappo_mlp, mappo_gcn, mappo_gat, gnn_mappo}`,
`--scenario` ∈ `{easy, medium, hard}`.

## Run the full POC experiment matrix

```bash
python -m scripts.run_poc --scenarios easy medium hard
```

## Generate result figures (PNG)

```bash
python -m evaluation.plots
```
→ `results/figures/*.png` (learning curves, baseline comparison, UAV trajectories).

## Architecture ablation / zero-shot scalability

```bash
python -m evaluation.ablation
python -m evaluation.zero_shot easy gnn_mappo
```

## Export data for the website and run it

```bash
python -m scripts.export_web_data
cd web
npm install
npm run dev
```

Open `http://localhost:3200`. Pages: Overview, Training, Baseline comparison,
Ablation, Zero-shot, Simulation replay. See [web/README.md](web/README.md) for
the data layout.

---

## After the 03/09/2026 review

See [REVIEW_2026-09-03.md](REVIEW_2026-09-03.md). Summary of the changes that
affect how to run things:

**The evaluation protocol changed.** Policies used to be run with `argmax`,
which turned a high-entropy policy into a single action (FAST_FORWARD): every
UAV flew straight into the map border and stayed there. Evaluation now samples
from the policy distribution by default. To re-evaluate old checkpoints without
retraining:

```bash
python -m scripts.reevaluate --out-dir results --max-jobs 6
python -m scripts.export_web_data
```

**The environment has 5 new design flags** (`sar_env/scenarios.py`), all ON by
default: `local_belief`, `include_neighbor_obs=False`, `per_agent_reward`,
`normalize_obs`, `normalize_edge_attr`. They change `obs_dim` (116 → 86), so
**old checkpoints are not compatible**. Use `--legacy-env` to reproduce
Stages 6/7:

```bash
python -m training.train --method gnn_mappo --scenario easy --seed 0 --legacy-env
```

**Test these fixes**:

```bash
python -m scripts.selftest_review_fixes
```
