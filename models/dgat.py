"""Dynamic Graph Attention Network (DGAT) — the central architectural
contribution of the thesis (muc 1.5.3), implementing equations (1.2)-(1.3):

    h_i^{l+1} = sigma( sum_{j in N(i)} alpha_ij * W [h_j^l || e_ij] )     (1.2)
    h_i^t     = GRU(h_tilde_i^{L,t}, h_i^{t-1})                          (1.3)

Cai dat: L=3 lop GATv2Conv, K=4 head, dua dac trung canh e_ij vao he so
attention (huong G2ANet), tiep theo mot GRUCell de ma hoa lich su theo thoi
gian. `use_gru=False` tai hien bien the "MAPPO-GAT (khong GRU)" trong Bang 5,
dung de co lap dong gop cua bo nho thoi gian (RQ2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class DGATEncoder(nn.Module):
    needs_graph = True

    def __init__(self, in_dim: int, hidden_dim: int = 64, edge_dim: int = 6,
                 heads: int = 4, n_layers: int = 3, use_gru: bool = True, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.hidden_dim = hidden_dim
        self.use_gru = use_gru
        self.n_layers = n_layers

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim,
                       concat=True, dropout=dropout, add_self_loops=True)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        if use_gru:
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                h_prev: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr=edge_attr)
            h = norm(F.elu(h_new) + h)  # residual + norm for training stability

        if not self.use_gru:
            return h

        if h_prev is None:
            h_prev = torch.zeros_like(h)
        return self.gru(h, h_prev)

    def init_hidden(self, n_nodes: int, device=None) -> torch.Tensor:
        return torch.zeros(n_nodes, self.hidden_dim, device=device)
