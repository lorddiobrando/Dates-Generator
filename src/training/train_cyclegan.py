"""Training loop for the Cycle GAN (conditions ↔ dates)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..data.dataset import make_loaders
from ..models.cyclegan import CycleDiscriminator, GeneratorAB, GeneratorBA
from ..utils import evaluate_csr

_BCE = nn.BCELoss()
_L1  = nn.L1Loss()


def train_cyclegan(
    data_path: str,
    epochs: int          = 200,
    batch_size: int      = 128,
    lr: float            = 2e-4,
    lambda_cycle: float  = 10.0,
    save_dir: str        = "checkpoints",
    seed: int            = 42,
    device: Optional[torch.device] = None,
) -> None:
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CycleGAN] device={device}")

    train_loader, val_loader, _ = make_loaders(data_path, batch_size=batch_size, seed=seed)

    G_AB = GeneratorAB().to(device)   # conditions → dates
    G_BA = GeneratorBA().to(device)   # dates → conditions
    D_A  = CycleDiscriminator(in_dim=62).to(device)  # real vs fake conditions
    D_B  = CycleDiscriminator(in_dim=41).to(device)  # real vs fake dates

    opt_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(0.5, 0.999)
    )
    opt_D = optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()), lr=lr, betas=(0.5, 0.999)
    )

    ckpt_dir = Path(save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_csr = 0.0

    for epoch in range(1, epochs + 1):
        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()
        g_total = d_total = 0.0

        for cond, real_dates in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            cond       = cond.to(device)
            real_dates = real_dates.to(device)
            n          = cond.size(0)
            real_lbl   = torch.ones(n, 1, device=device)
            fake_lbl   = torch.zeros(n, 1, device=device)

            # ── Generator step ───────────────────────────────────────────────
            fake_dates  = G_AB(cond)          # A → B
            recon_cond  = G_BA(fake_dates)    # B → A  (cycle A)
            fake_cond   = G_BA(real_dates)    # B → A
            recon_dates = G_AB(fake_cond)     # A → B  (cycle B)

            adv_AB   = _BCE(D_B(fake_dates), real_lbl)
            adv_BA   = _BCE(D_A(fake_cond),  real_lbl)
            cycle_A  = _L1(recon_cond,  cond)
            cycle_B  = _L1(recon_dates, real_dates)

            loss_G = adv_AB + adv_BA + lambda_cycle * (cycle_A + cycle_B)
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # ── Discriminator step ───────────────────────────────────────────
            loss_D_A = (
                _BCE(D_A(cond),             real_lbl)
                + _BCE(D_A(fake_cond.detach()), fake_lbl)
            )
            loss_D_B = (
                _BCE(D_B(real_dates),            real_lbl)
                + _BCE(D_B(fake_dates.detach()), fake_lbl)
            )
            loss_D = (loss_D_A + loss_D_B) * 0.5
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            g_total += loss_G.item()
            d_total += loss_D.item()

        # ── Validation CSR ───────────────────────────────────────────────────
        G_AB.eval()
        csr = evaluate_csr(lambda c: G_AB(c), val_loader, device)["all"]

        n_b = len(train_loader)
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"G={g_total/n_b:.4f}  D={d_total/n_b:.4f}  "
            f"val_CSR={csr:.4f}"
        )

        if csr > best_csr:
            best_csr = csr
            torch.save(
                {"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict()},
                ckpt_dir / "cyclegan_best.pt",
            )
            print(f"  ↑ checkpoint saved (best CSR={best_csr:.4f})")

    print(f"[CycleGAN] finished — best val CSR: {best_csr:.4f}")
