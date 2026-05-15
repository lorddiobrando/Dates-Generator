"""Training loop for the Conditional VAE (Autoencoder)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..data.dataset import make_loaders
from ..data.encoding import DAY_OF_MONTH_DIM
from ..models.vae import VAE
from ..utils import evaluate_csr

_EPS = 1e-8


def _vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO = reconstruction (categorical NLL) + β · KL divergence."""
    dom_nll = -(target[:, :DAY_OF_MONTH_DIM] * torch.log(recon[:, :DAY_OF_MONTH_DIM] + _EPS)).sum(1).mean()
    yid_nll = -(target[:, DAY_OF_MONTH_DIM:] * torch.log(recon[:, DAY_OF_MONTH_DIM:] + _EPS)).sum(1).mean()
    recon_loss = dom_nll + yid_nll

    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_vae(
    data_path: str,
    epochs: int     = 200,
    batch_size: int = 128,
    lr: float       = 1e-3,
    beta: float     = 1.0,
    save_dir: str   = "checkpoints",
    seed: int       = 42,
    device: Optional[torch.device] = None,
) -> None:
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VAE] device={device}")

    train_loader, val_loader, _ = make_loaders(data_path, batch_size=batch_size, seed=seed)

    vae       = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=False)

    ckpt_dir = Path(save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_csr = 0.0

    for epoch in range(1, epochs + 1):
        vae.train()
        total_loss = recon_sum = kl_sum = 0.0

        for cond, real_dates in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            cond       = cond.to(device)
            real_dates = real_dates.to(device)

            recon, mu, logvar = vae(real_dates, cond)
            loss, recon_l, kl_l = _vae_loss(recon, real_dates, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            recon_sum  += recon_l.item()
            kl_sum     += kl_l.item()

        n_b = len(train_loader)
        scheduler.step(total_loss / n_b)

        vae.eval()
        csr = evaluate_csr(lambda c: vae.generate(c), val_loader, device)["all"]

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"loss={total_loss/n_b:.4f}  recon={recon_sum/n_b:.4f}  kl={kl_sum/n_b:.4f}  "
            f"val_CSR={csr:.4f}"
        )

        if csr > best_csr:
            best_csr = csr
            torch.save({"model": vae.state_dict()}, ckpt_dir / "vae_best.pt")
            print(f"  ↑ checkpoint saved (best CSR={best_csr:.4f})")

    print(f"[VAE] finished — best val CSR: {best_csr:.4f}")
