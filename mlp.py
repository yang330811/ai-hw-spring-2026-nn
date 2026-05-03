"""Shallow MLP for MNIST: flatten 28x784 -> hidden -> logits."""

import torch
import torch.nn as nn


class ShallowMLP(nn.Module):
    def __init__(self, num_classes: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.net(x)
