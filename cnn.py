"""Convolutional Neural Network (CNN) model."""

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Spatial backbone: tensor shapes N,C,H,W go 1x28x28 -> 32x14x14 -> 64x7x7 (batch N omitted).
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # Flatten 64*7*7 features, then two linear layers -> class logits (size num_classes).
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
