"""Encoder variants used to build the three ablation baselines in Bang 5
(MAPPO-MLP, MAPPO-GCN, MAPPO-GAT) alongside the full DGAT (models/dgat.py),
all plugged into the *same* MAPPO trainer (algorithms/mappo.py) so that only
the encoder differs between methods — this isolates the contribution of graph
structure (RQ1) and of attention vs. fixed aggregation (RQ1) independently of
temporal memory (RQ2, tested via DGATEncoder(use_gru=False)).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .dgat import DGATEncoder


class MLPEncoder(nn.Module):
    """No graph structure at all: each UAV is treated as an independent agent
    (Bang 1 / Bang 5 baseline "MAPPO (MLP)")."""
    needs_graph = False

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index=None, edge_attr=None, h_prev=None):
        return self.net(x)

    def init_hidden(self, n_nodes, device=None):
        return None


class GCNEncoder(nn.Module):
    """Fixed-weight neighborhood aggregation (Kipf & Welling, 2017) — Bang 5
    ablation "MAPPO-GCN", isolates the contribution of learned attention vs.
    fixed aggregation weights when compared against MAPPO-GAT / GNN-MAPPO."""
    needs_graph = True

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

    def forward(self, x, edge_index, edge_attr=None, h_prev=None):
        # edge_attr's last column is the signal-strength weight (see graph_builder.py)
        edge_weight = edge_attr[:, -1] if edge_attr is not None and edge_attr.numel() > 0 else None
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_weight=edge_weight)
            h = norm(F.relu(h_new) + h)
        return h

    def init_hidden(self, n_nodes, device=None):
        return None


ENCODER_REGISTRY = {
    "mlp": lambda in_dim, hidden_dim: MLPEncoder(in_dim, hidden_dim),
    "gcn": lambda in_dim, hidden_dim: GCNEncoder(in_dim, hidden_dim),
    "gat": lambda in_dim, hidden_dim: DGATEncoder(in_dim, hidden_dim, use_gru=False),
    "dgat": lambda in_dim, hidden_dim: DGATEncoder(in_dim, hidden_dim, use_gru=True),
}


def build_encoder(name: str, in_dim: int, hidden_dim: int = 64) -> nn.Module:
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Choose from {list(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[name](in_dim, hidden_dim)
