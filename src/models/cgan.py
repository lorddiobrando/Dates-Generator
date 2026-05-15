"""Conditional GAN with projection discriminator and spectral normalisation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from ..data.encoding import COND_DIM, DATE_DIM, DAY_OF_MONTH_DIM

NOISE_DIM = 64
EMBED_DIM = 64


def _date_head(logits: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        torch.softmax(logits[:, :DAY_OF_MONTH_DIM], dim=1),
        torch.softmax(logits[:, DAY_OF_MONTH_DIM:], dim=1),
    ], dim=1)


class _CondEmbedding(nn.Module):
    """Project the sparse one-hot condition vector into a dense embedding."""

    def __init__(self, cond_dim: int = COND_DIM, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.fc = nn.Linear(cond_dim, embed_dim)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.fc(cond)


class CGANGenerator(nn.Module):
    """Generator that uses a dense condition embedding instead of raw one-hot."""

    def __init__(
        self,
        noise_dim: int  = NOISE_DIM,
        embed_dim: int  = EMBED_DIM,
        out_dim: int    = DATE_DIM,
    ) -> None:
        super().__init__()
        self.noise_dim   = noise_dim
        self.cond_embed  = _CondEmbedding(embed_dim=embed_dim)
        self.body = nn.Sequential(
            nn.Linear(noise_dim + embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(128, out_dim)

    def forward(self, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        e = self.cond_embed(cond)
        return _date_head(self.head(self.body(torch.cat([noise, e], dim=1))))


class CGANDiscriminator(nn.Module):
    """Projection discriminator (Miyato & Koyama, 2018) with spectral norm.

    Output = linear(h) + <h, embed(cond)>  passed through sigmoid.
    This hard-wires the conditioning so the model cannot ignore it.
    """

    def __init__(
        self,
        date_dim: int  = DATE_DIM,
        embed_dim: int = EMBED_DIM,
    ) -> None:
        super().__init__()
        self.cond_embed = _CondEmbedding(embed_dim=embed_dim)
        self.body = nn.Sequential(
            spectral_norm(nn.Linear(date_dim, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(256, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(256, embed_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_out = spectral_norm(nn.Linear(embed_dim, 1))

    def forward(self, date: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h    = self.body(date)
        e    = self.cond_embed(cond)
        proj = (h * e).sum(dim=1, keepdim=True)
        return torch.sigmoid(self.fc_out(h) + proj)

    def intermediate(self, date: torch.Tensor) -> torch.Tensor:
        """Penultimate features for feature-matching loss."""
        return self.body(date)
