import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GNN_TBNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=10):
        super().__init__()

        # --- graph encoder ---
        self.g1 = GATConv(in_dim, hidden_dim)
        self.g2 = GATConv(hidden_dim, hidden_dim)

        # --- node MLP to infer g_k coefficients ---
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)  # 10 basis coefficients
        )

    def forward(self, x, invariants, edge_index, basis_tensors):
        # Graph message passing
        h = self.g1(x, edge_index)
        h = torch.relu(h)
        h = self.g2(h, edge_index)

        # Concatenate invariants (for TBNN)
        h = torch.cat([h, invariants], dim=1)

        # Predict TBNN coefficients
        g = self.mlp(h)

        # Assemble anisotropy
        # basis_tensors: shape (N, 10, 3, 3)
        a = torch.einsum("nk,nkij->nij", g, basis_tensors)
        return a
