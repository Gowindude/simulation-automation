"""
DeepONet for Cp(s | geometry, AoA): branch net encodes (resampled
geometry, AoA), trunk net encodes the query arc-length position s,
combined by a dot product + bias -- the standard architecture (Lu et al.
2021), chosen over FNO per the build spec's own reasoning: pressure
curves vary in length/resolution across airfoils, so a fixed-grid
representation (what FNO wants) would need lossy resampling, while
DeepONet's trunk naturally takes an arbitrary query point.
"""

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int, n_hidden_layers: int = 2) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(n_hidden_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    def __init__(self, branch_in_dim: int, trunk_in_dim: int = 1, p: int = 64, hidden: int = 64):
        super().__init__()
        self.branch = _mlp(branch_in_dim, hidden, p)
        self.trunk = _mlp(trunk_in_dim, hidden, p)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_x: torch.Tensor, trunk_x: torch.Tensor) -> torch.Tensor:
        b = self.branch(branch_x)
        t = self.trunk(trunk_x)
        return (b * t).sum(dim=-1, keepdim=True) + self.bias
