"""Basic GAN for conditional date generation."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.encoding import COND_DIM, DATE_DIM, DAY_OF_MONTH_DIM

NOISE_DIM = 64


def _date_head(logits: torch.Tensor) -> torch.Tensor:
    """Apply independent softmax to the day-of-month and year-in-decade parts."""
    return torch.cat([
        torch.softmax(logits[:, :DAY_OF_MONTH_DIM], dim=1),
        torch.softmax(logits[:, DAY_OF_MONTH_DIM:], dim=1),
    ], dim=1)


class Generator(nn.Module):
    """MLP generator: noise(64) ‖ conditions(62) → date(41)."""

    def __init__(
        self,
        noise_dim: int = NOISE_DIM,
        cond_dim: int  = COND_DIM,
        out_dim: int   = DATE_DIM,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )

    def forward(self, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return _date_head(self.net(torch.cat([noise, cond], dim=1)))


class Discriminator(nn.Module):
    """MLP discriminator: date(41) ‖ conditions(62) → scalar."""

    def __init__(
        self,
        date_dim: int = DATE_DIM,
        cond_dim: int = COND_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(date_dim + cond_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, date: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([date, cond], dim=1))
