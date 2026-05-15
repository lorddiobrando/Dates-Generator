"""Training loop for the Conditional GAN (projection discriminator)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..data.dataset import make_loaders
from ..models.cgan import CGANDiscriminator, CGANGenerator, NOISE_DIM
from ..utils import evaluate_csr

_BCE = nn.BCELoss()


def train_cgan(
    data_path: str,
    epochs: int        = 200,
    batch_size: int    = 128,
    lr: float          = 2e-4,
    noise_dim: int     = NOISE_DIM,
    lambda_fm: float   = 10.0,
    save_dir: str      = "checkpoints",
    seed: int          = 42,
    device: Optional[torch.device] = None,
) -> None:
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cGAN] device={device}")

    train_loader, val_loader, _ = make_loaders(data_path, batch_size=batch_size, seed=seed)

    G = CGANGenerator(noise_dim=noise_dim).to(device)
    D = CGANDiscriminator().to(device)

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
            real_lbl   = torch.full((n, 1), 0.9, device=device)
            fake_lbl   = torch.zeros(n, 1, device=device)

            # ── Discriminator ────────────────────────────────────────────────
            with torch.no_grad():
                fake_d = G(torch.randn(n, noise_dim, device=device), cond)
            loss_D = _BCE(D(real_dates, cond), real_lbl) + _BCE(D(fake_d, cond), fake_lbl)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ── Generator ────────────────────────────────────────────────────
            fake_d = G(torch.randn(n, noise_dim, device=device), cond)

            adv_loss = _BCE(D(fake_d, cond), real_lbl)

            # Feature-matching: align fake intermediate features with real ones
            real_feat = D.intermediate(real_dates).detach()
            fake_feat = D.intermediate(fake_d)
            fm_loss   = nn.functional.mse_loss(fake_feat, real_feat)

            loss_G = adv_loss + lambda_fm * fm_loss
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            g_total += loss_G.item()
            d_total += loss_D.item()

        # ── Validation CSR ───────────────────────────────────────────────────
        G.eval()
        gen_fn = lambda c: G(torch.randn(c.size(0), noise_dim, device=device), c)
        csr    = evaluate_csr(gen_fn, val_loader, device)["all"]

        n_b = len(train_loader)
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"G={g_total/n_b:.4f}  D={d_total/n_b:.4f}  "
            f"val_CSR={csr:.4f}"
        )

        if csr > best_csr:
            best_csr = csr
            torch.save({"G": G.state_dict(), "D": D.state_dict()}, ckpt_dir / "cgan_best.pt")
            print(f"  ↑ checkpoint saved (best CSR={best_csr:.4f})")

    print(f"[cGAN] finished — best val CSR: {best_csr:.4f}")
