#!/usr/bin/env python3
"""Evaluate a trained model and generate dates from an input conditions file.

Usage:
    python evaluate.py --model gan      --checkpoint checkpoints/gan_best.pt
    python evaluate.py --model vae      --checkpoint checkpoints/vae_best.pt
    python evaluate.py --model cgan     --checkpoint checkpoints/cgan_best.pt
    python evaluate.py --model cyclegan --checkpoint checkpoints/cyclegan_best.pt
    python evaluate.py --model gan      --checkpoint checkpoints/gan_best.pt \\
                       --input data/example_input.txt --output-dir output
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch

from src.data.dataset import make_loaders
from src.data.encoding import (
    DAY_TOKENS,
    DECADE_MIN,
    MONTH_TOKENS,
    _DAY_END,
    _LEAP_END,
    _MONTH_END,
    constrained_decode,
    decode_date,
    encode_conditions,
    parse_conditions,
)
from src.utils import evaluate_csr

GeneratorFn = Callable[[torch.Tensor], torch.Tensor]


def _load_model(model_name: str, checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device)

    if model_name == "gan":
        from src.models.gan import Generator, NOISE_DIM
        m = Generator().to(device)
        m.load_state_dict(ckpt["G"])
        return m

    if model_name == "vae":
        from src.models.vae import VAE
        m = VAE().to(device)
        m.load_state_dict(ckpt["model"])
        return m

    if model_name == "cgan":
        from src.models.cgan import CGANGenerator, NOISE_DIM
        m = CGANGenerator().to(device)
        m.load_state_dict(ckpt["G"])
        return m

    if model_name == "cyclegan":
        from src.models.cyclegan import GeneratorAB
        m = GeneratorAB().to(device)
        m.load_state_dict(ckpt["G_AB"])
        return m

    raise ValueError(f"Unknown model: {model_name}")


def _make_gen_fn(model_name: str, model, device: torch.device) -> GeneratorFn:
    if model_name in ("gan", "cgan"):
        if model_name == "gan":
            from src.models.gan import NOISE_DIM
        else:
            from src.models.cgan import NOISE_DIM
        return lambda c: model(torch.randn(c.size(0), NOISE_DIM, device=device), c)

    if model_name == "vae":
        return lambda c: model.generate(c)

    if model_name == "cyclegan":
        return lambda c: model(c)

    raise ValueError(f"Unknown model: {model_name}")


def _decode_condition(ci: torch.Tensor) -> tuple[str, str, str, str]:
    day    = DAY_TOKENS[int(ci[:_DAY_END].argmax().item())]
    month  = MONTH_TOKENS[int(ci[_DAY_END:_MONTH_END].argmax().item())]
    leap_i = int(ci[_MONTH_END:_LEAP_END].argmax().item())
    leap   = "True" if leap_i == 1 else "False"
    decade = str(int(ci[_LEAP_END:].argmax().item()) + DECADE_MIN)
    return day, month, leap, decade


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained date-generator model")
    parser.add_argument("--model",      choices=["gan", "vae", "cgan", "cyclegan"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data",       default="data/data.txt",          help="Full dataset for test-set CSR")
    parser.add_argument("--input",      default="data/example_input.txt", help="Conditions file to generate from")
    parser.add_argument("--output-dir",  default="output")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--constrained", action="store_true",
                        help="Enforce weekday condition via constrained decoding")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model  = _load_model(args.model, args.checkpoint, device)
    model.eval()
    gen_fn = _make_gen_fn(args.model, model, device)

    # ── Test-set CSR ─────────────────────────────────────────────────────────
    _, _, test_loader = make_loaders(args.data, batch_size=256, seed=args.seed)
    csr = evaluate_csr(gen_fn, test_loader, device, constrained=args.constrained)

    mode = "constrained" if args.constrained else "unconstrained"
    print(f"\n=== Test-Set Condition Satisfaction Rates ({mode}) ===")
    for k, v in csr.items():
        print(f"  {k:<8} {v*100:6.2f}%")

    # ── Generate from input file ──────────────────────────────────────────────
    with open(args.input) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    print(f"\n=== Generating {len(lines)} dates from {args.input} ===")
    results: list[str] = []

    for line in lines:
        day, month, leap, decade = parse_conditions(line)
        cond_t = encode_conditions(day, month, leap, decade).unsqueeze(0).to(device)
        with torch.no_grad():
            date_t = gen_fn(cond_t)
        decode_fn = constrained_decode if args.constrained else decode_date
        date_str  = decode_fn(cond_t[0].cpu(), date_t[0].cpu())
        results.append(date_str)
        print(f"  {line}  ->  {date_str}")

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.model}_predictions.txt"
    out_path.write_text("\n".join(results) + "\n")
    print(f"\nPredictions saved to {out_path}")


if __name__ == "__main__":
    main()
