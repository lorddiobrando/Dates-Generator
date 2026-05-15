"""Training loop for the Basic GAN."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..data.dataset import make_loaders
from ..models.gan import Discriminator, Generator, NOISE_DIM
from ..utils import evaluate_csr

_BCE = nn.BCELoss()


def _real_labels(n: int, device: torch.device) -> torch.Tensor:
    return torch.full((n, 1), 0.9, device=device)   # one-sided label smoothing


def _fake_labels(n: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(n, 1, device=device)


def train_gan(
    data_path: str,
    epochs: int        = 200,
    batch_size: int    = 128,
    lr: float          = 2e-4,
    noise_dim: int     = NOISE_DIM,
    save_dir: str      = "checkpoints",
    seed: int          = 42,
    device: Optional[torch.device] = None,
) -> None:
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GAN] device={device}")

    train_loader, val_loader, _ = make_loaders(data_path, batch_size=batch_size, seed=seed)

    G = Generator(noise_dim=noise_dim).to(device)
    D = Discriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    ckpt_dir = Path(save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_csr = 0.0

    for epoch in range(1, epochs + 1):
        G.train()
        D.train()
        g_total = d_total = 0.0

        for cond, real_dates in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            cond       = cond.to(device)
            real_dates = real_dates.to(device)
            n          = cond.size(0)

            # ── Discriminator ────────────────────────────────────────────────
            with torch.no_grad():
                fake_dates = G(torch.randn(n, noise_dim, device=device), cond)
            loss_D = (
                _BCE(D(real_dates, cond), _real_labels(n, device))
                + _BCE(D(fake_dates, cond), _fake_labels(n, device))
            )
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ── Generator ────────────────────────────────────────────────────
            fake_dates = G(torch.randn(n, noise_dim, device=device), cond)
            loss_G = _BCE(D(fake_dates, cond), _real_labels(n, device))
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            g_total += loss_G.item()
            d_total += loss_D.item()

        # ── Validation CSR ───────────────────────────────────────────────────
        G.eval()
        gen_fn = lambda c: G(torch.randn(c.size(0), noise_dim, device=device), c)
        csr    = evaluate_csr(gen_fn, val_loader, device)["all"]

        n_batches = len(train_loader)
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"G={g_total/n_batches:.4f}  D={d_total/n_batches:.4f}  "
            f"val_CSR={csr:.4f}"
        )

        if csr > best_csr:
            best_csr = csr
            torch.save({"G": G.state_dict(), "D": D.state_dict()}, ckpt_dir / "gan_best.pt")
            print(f"  ↑ checkpoint saved (best CSR={best_csr:.4f})")

    print(f"[GAN] finished — best val CSR: {best_csr:.4f}")
