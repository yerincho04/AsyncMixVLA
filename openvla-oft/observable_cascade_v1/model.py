"""Neural architecture used by the frozen observable visual gate."""

import torch
import torch.nn as nn


class VisualGate(nn.Module):
    def __init__(self, in_ch=24, aux_dim=15):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.aux = nn.Sequential(nn.Linear(aux_dim, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(160, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, aux):
        return self.head(torch.cat([self.cnn(x), self.aux(aux)], dim=1)).squeeze(1)
