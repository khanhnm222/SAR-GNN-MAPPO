# GNN-MAPPO SAR Toolkit (web)

**English** | [Tiếng Việt](README.vi.md)

A Next.js app that visualizes training and experiment results of GNN-MAPPO for
UAV swarm coordination in search and rescue. It reads static JSON from
`public/data/` and needs no backend.

## Pages

| Route         | Content                                                        |
| ------------- | -------------------------------------------------------------- |
| `/`           | Overview                                                       |
| `/training`   | Learning curves per method and seed                            |
| `/comparison` | Baseline comparison (metrics, statistical tests)               |
| `/ablation`   | Encoder/GRU ablation                                           |
| `/zero-shot`  | Zero-shot transfer to unseen swarm sizes (fixed map, or map scaled to keep UAV density) |
| `/replay`     | Replay of recorded UAV trajectories                            |

## Run locally

```bash
npm install
npm run dev      # http://localhost:3200
```

Production build: `npm run build && npm run start`.

## Data

The data is generated from the experiment results by the Python exporter at
the repository root:

```bash
python -m scripts.export_web_data
```

Layout under `public/data/`:

```
manifest.json                 scenarios, methods, display labels
<scenario>/                   easy | medium | hard
  learning_curves.json
  comparison.json
  ablation.json
  zero_shot.json
  zero_shot_density.json
  trajectories/
    manifest.json
    <method>.json             random_walk, greedy, maddpg, mappo_mlp,
                              mappo_gcn, mappo_gat, gnn_mappo
```

## Stack

Next.js 16 (App Router), React, TypeScript, Tailwind CSS. This Next.js version
has breaking changes compared with older releases; see [AGENTS.md](AGENTS.md).
