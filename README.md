# Dates Generator

Generative models that produce a valid calendar date satisfying four input conditions:
**day-of-week**, **month**, **leap-year flag**, and **decade**.

## Project structure

```
dates-generator/
├── data.txt                   # full dataset
├── example_input.txt          # input conditions for generation
├── checkpoints/               # saved model weights (created at training time)
├── output/                    # generated predictions (created at inference time)
├── src/
│   ├── data/
│   │   ├── encoding.py        # one-hot encode/decode utilities
│   │   └── dataset.py         # PyTorch Dataset + DataLoader factory
│   ├── models/
│   │   ├── gan.py             # Basic GAN
│   │   ├── vae.py             # Variational Autoencoder
│   │   ├── cgan.py            # Conditional GAN (projection discriminator)
│   │   └── cyclegan.py        # Cycle GAN (conditions ↔ dates)
│   ├── training/
│   │   ├── train_gan.py
│   │   ├── train_vae.py
│   │   ├── train_cgan.py
│   │   └── train_cyclegan.py
│   └── utils.py               # shared Condition Satisfaction Rate evaluator
├── solver.py                  # algorithmic solver (no ML)
├── train.py                   # unified training CLI
└── evaluate.py                # evaluation + batch generation CLI
```

## Input / Output format

**Input conditions** (one per line):
```
[MON] [DEC] [False] [196]
[THU] [DEC] [True]  [204]
```

**Output date** — any `dd-mm-yyyy` string satisfying all four conditions:
```
03-12-1962
03-12-2048
```

## Encoding

| Condition   | Encoding              | Dim |
|-------------|-----------------------|-----|
| Day of week | one-hot MON…SUN       | 7   |
| Month       | one-hot JAN…DEC       | 12  |
| Leap year   | one-hot [False, True] | 2   |
| Decade      | one-hot 180…220       | 41  |

Total condition vector: **62-dim**.  
Date (variable parts only): day-of-month (31) + year-in-decade (10) = **41-dim**.

## Models

| Model       | Key idea |
|-------------|----------|
| **Basic GAN** | MLP generator + discriminator, both conditioned via concatenation |
| **VAE** | Encoder → latent (μ, σ²); decoder samples from prior at inference |
| **Conditional GAN** | Dense condition embedding; projection discriminator + spectral norm + feature-matching loss |
| **Cycle GAN** | Domains A=conditions, B=dates; cycle consistency enforces A→B→A and B→A→B |

## Evaluation metric

Since multiple dates satisfy each condition, **exact-match accuracy is meaningless**.  
The primary metric is the **Condition Satisfaction Rate (CSR)**:

| Metric      | Description |
|-------------|-------------|
| `day`       | Generated date falls on the required weekday |
| `month`     | Generated date is in the required month |
| `leap`      | Generated year has the required leap-year status |
| `decade`    | Generated year is in the required decade |
| **`all`**   | **All four conditions satisfied simultaneously** ← headline metric |

## Setup

```bash
pip install -r requirements.txt
```

`data/data.txt` and `data/example_input.txt` are provided in the repository.

## Usage

### Algorithmic solver (no ML)
```bash
python solver.py                               # uses data/example_input.txt
python solver.py data/example_input.txt --seed 42
```

### Train a model
```bash
python train.py --model gan
python train.py --model vae      --epochs 300 --beta 0.5
python train.py --model cgan     --lambda-fm 10
python train.py --model cyclegan --lambda-cycle 10
```

All runs are reproducible via `--seed` (default 42).  
Checkpoints are saved to `checkpoints/<model>_best.pt` whenever validation CSR improves.

### Evaluate & generate
```bash
python evaluate.py --model gan      --checkpoint checkpoints/gan_best.pt
python evaluate.py --model vae      --checkpoint checkpoints/vae_best.pt
python evaluate.py --model cgan     --checkpoint checkpoints/cgan_best.pt
python evaluate.py --model cyclegan --checkpoint checkpoints/cyclegan_best.pt
```

Predictions for `data/example_input.txt` are written to `output/<model>_predictions.txt`.
