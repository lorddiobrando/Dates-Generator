"""Conditional Variational Autoencoder for date generation."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.encoding import COND_DIM, DATE_DIM, DAY_OF_MONTH_DIM

LATENT_DIM = 32


def _date_head(logits: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        torch.softmax(logits[:, :DAY_OF_MONTH_DIM], dim=1),
        torch.softmax(logits[:, DAY_OF_MONTH_DIM:], dim=1),
    ], dim=1)


class VAE(nn.Module):
    """Conditional VAE.

    Training: encode(date, cond) → (μ, σ²) → z → decode(z, cond) → date′
    Inference: sample z ~ N(0, I) → decode(z, cond) → date
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        cond_dim: int   = COND_DIM,
        date_dim: int   = DATE_DIM,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(date_dim + cond_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.fc_mu     = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, date_dim),
        )

    def encode(
        self, date: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([date, cond], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return _date_head(self.decoder(torch.cat([z, cond], dim=1)))

    def forward(
        self, date: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(date, cond)
        z          = self.reparameterize(mu, logvar)
        return self.decode(z, cond), mu, logvar

    @torch.no_grad()
    def generate(self, cond: torch.Tensor) -> torch.Tensor:
        """Sample a date from the prior for the given conditions."""
        self.eval()
        z = torch.randn(cond.size(0), self.latent_dim, device=cond.device)
        return self.decode(z, cond)
