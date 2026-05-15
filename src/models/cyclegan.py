"""Cycle GAN with two unpaired domains: conditions (A) and dates (B)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.encoding import (
    COND_DIM,
    DATE_DIM,
    DAY_OF_MONTH_DIM,
    DAY_DIM,
    MONTH_DIM,
    LEAP_DIM,
    DECADE_DIM,
)

HIDDEN_DIM = 256


def _mlp_block(in_dim: int, out_dim: int, hidden: int = HIDDEN_DIM) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


class GeneratorAB(nn.Module):
    """Domain A → B: conditions (62-dim) → date (41-dim)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = _mlp_block(COND_DIM, DATE_DIM)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        logits = self.net(cond)
        return torch.cat([
            torch.softmax(logits[:, :DAY_OF_MONTH_DIM], dim=1),
            torch.softmax(logits[:, DAY_OF_MONTH_DIM:], dim=1),
        ], dim=1)


class GeneratorBA(nn.Module):
    """Domain B → A: date (41-dim) → conditions (62-dim)."""

    # Slice boundaries within the 62-dim condition vector
    _D = DAY_DIM
    _M = DAY_DIM + MONTH_DIM
    _L = DAY_DIM + MONTH_DIM + LEAP_DIM

    def __init__(self) -> None:
        super().__init__()
        self.net = _mlp_block(DATE_DIM, COND_DIM)

    def forward(self, date: torch.Tensor) -> torch.Tensor:
        logits = self.net(date)
        return torch.cat([
            torch.softmax(logits[:, :self._D],         dim=1),  # day-of-week
            torch.softmax(logits[:, self._D:self._M],  dim=1),  # month
            torch.softmax(logits[:, self._M:self._L],  dim=1),  # leap
            torch.softmax(logits[:, self._L:],         dim=1),  # decade
        ], dim=1)


class CycleDiscriminator(nn.Module):
    """Discriminator for a single domain (used for both D_A and D_B)."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
