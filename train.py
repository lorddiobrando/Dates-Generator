#!/usr/bin/env python3
"""CLI entry point for training any of the four generative models.

Usage:
    python train.py --model gan
    python train.py --model vae  --epochs 300 --lr 5e-4
    python train.py --model cgan --batch-size 256
    python train.py --model cyclegan --lambda-cycle 5
"""

from __future__ import annotations

import argparse

import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a date-generator model")
    p.add_argument("--model",        choices=["gan", "vae", "cgan", "cyclegan", "cyclegan-joint"], required=True)
    p.add_argument("--data",         default="data/data.txt",   help="Path to data.txt")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--save-dir",     default="checkpoints")
    p.add_argument("--seed",         type=int,   default=42)
    # VAE-specific
    p.add_argument("--beta",         type=float, default=1.0,   help="KL weight for VAE")
    # cGAN-specific
    p.add_argument("--lambda-fm",    type=float, default=10.0,  help="Feature-matching weight")
    # CycleGAN-specific
    p.add_argument("--lambda-cycle", type=float, default=10.0,  help="Cycle-consistency weight")
    # CycleGAN-Joint-specific
    p.add_argument("--lambda-recon", type=float, default=5.0,   help="Soft-label reconstruction weight (cyclegan-joint)")
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = dict(
        data_path  = args.data,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        save_dir   = args.save_dir,
        seed       = args.seed,
        device     = device,
    )

    if args.model == "gan":
        from src.training.train_gan import train_gan
        train_gan(**kwargs)

    elif args.model == "vae":
        from src.training.train_vae import train_vae
        train_vae(**kwargs, beta=args.beta)

    elif args.model == "cgan":
        from src.training.train_cgan import train_cgan
        train_cgan(**kwargs, lambda_fm=args.lambda_fm)

    elif args.model == "cyclegan":
        from src.training.train_cyclegan import train_cyclegan
        train_cyclegan(**kwargs, lambda_cycle=args.lambda_cycle)

    elif args.model == "cyclegan-joint":
        from src.training.train_cyclegan_joint import train_cyclegan_joint
        train_cyclegan_joint(**kwargs, lambda_cycle=args.lambda_cycle, lambda_recon=args.lambda_recon)


if __name__ == "__main__":
    main()
