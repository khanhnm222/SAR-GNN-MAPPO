"""Dynamic communication graph G = (V, E, X, A) for the UAV swarm (muc 1.5.2).

Two UAVs i, j are connected iff d(i,j) <= r_comm. Edge weight decays with
distance; edge features carry the relative position/velocity, distance and a
signal-strength proxy that decays with distance, so that communication
quality participates directly in the DGAT attention mechanism (muc 1.5.3).

Edge features are normalised by r_comm (2026-09-03). GATv2Conv feeds edge_attr
through a Linear alongside the node embedding, so raw metres (dx, dy, dist all
up to r_comm=10) against LayerNorm'd node features made the attention logits
depend mostly on edge geometry magnitude rather than on content. Normalising
also keeps the attention well-behaved if r_comm is changed between scenarios.
"""
from __future__ import annotations

import numpy as np
import torch

EDGE_DIM = 6


def signal_strength(dist: np.ndarray, r_comm: float) -> np.ndarray:
    """Simple inverse-distance signal decay in [0, 1], 1 at dist=0, 0 at dist=r_comm."""
    return np.clip(1.0 - dist / max(r_comm, 1e-6), 0.0, 1.0)


def build_graph(uavs, r_comm: float, normalize: bool = True):
    """Build the dynamic communication graph for a list of UAV entities.

    Returns a dict with:
      - alive_idx: indices (into `uavs`) of alive UAVs, i.e. the graph's nodes
      - edge_index: (2, E) torch.long tensor, local node indices (0..n_alive-1)
      - edge_attr: (E, 6) torch.float tensor [dx, dy, dvx, dvy, dist, signal]
      - adjacency: (n_alive, n_alive) numpy adjacency (weighted by signal strength)
    """
    alive_idx = [i for i, u in enumerate(uavs) if u.alive]
    n = len(alive_idx)
    if n == 0:
        return {"alive_idx": [], "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "edge_attr": torch.zeros((0, EDGE_DIM), dtype=torch.float32),
                "adjacency": np.zeros((0, 0), dtype=np.float32), "n_nodes": 0}

    pos = np.array([[uavs[i].x, uavs[i].y] for i in alive_idx], dtype=np.float32)
    vel = np.array([[uavs[i].vx, uavs[i].vy] for i in alive_idx], dtype=np.float32)

    # vectorised pairwise geometry: the original double Python loop was O(N^2)
    # interpreted work on every single env step (256 iterations per step at
    # N=16, 500 steps per episode).
    delta = pos[None, :, :] - pos[:, None, :]          # [a, b] -> pos[b] - pos[a]
    dvel = vel[None, :, :] - vel[:, None, :]
    dist = np.linalg.norm(delta, axis=-1)
    mask = (dist <= r_comm) & ~np.eye(n, dtype=bool)
    src, dst = np.nonzero(mask)

    d = dist[src, dst]
    sig = signal_strength(d, r_comm)
    feats = np.stack([delta[src, dst, 0], delta[src, dst, 1],
                      dvel[src, dst, 0], dvel[src, dst, 1], d, sig], axis=-1).astype(np.float32)
    if normalize:
        scale = np.array([r_comm, r_comm, 2.0, 2.0, r_comm, 1.0], dtype=np.float32)
        feats = np.clip(feats / scale, -1.0, 1.0)

    adjacency = np.zeros((n, n), dtype=np.float32)
    adjacency[src, dst] = sig

    return {
        "alive_idx": alive_idx,
        "edge_index": torch.from_numpy(np.stack([src, dst]).astype(np.int64)),
        "edge_attr": torch.from_numpy(feats),
        "adjacency": adjacency,
        "n_nodes": n,
    }


def k_nearest_neighbors(uavs, self_idx: int, k: int, r_comm: float):
    """Return up to k nearest alive neighbours of uavs[self_idx] within r_comm,
    as a list of (uid, relative_feature[6]) sorted by distance ascending.
    Relative feature = [dx, dy, dz, dvx, dvy, dist].
    """
    me = uavs[self_idx]
    if not me.alive:
        return []
    candidates = []
    for j, other in enumerate(uavs):
        if j == self_idx or not other.alive:
            continue
        dx, dy, dz = other.x - me.x, other.y - me.y, other.z - me.z
        dist = float(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        if dist <= r_comm:
            dvx, dvy = other.vx - me.vx, other.vy - me.vy
            candidates.append((dist, other.uid, np.array([dx, dy, dz, dvx, dvy, dist], dtype=np.float32)))
    candidates.sort(key=lambda t: t[0])
    return [(uid, feat) for _, uid, feat in candidates[:k]]
