# Dates Generator

Generative models that produce a valid calendar date satisfying four input conditions:
**day-of-week**, **month**, **leap-year flag**, and **decade**.

A pure algorithmic solver is also included as a 100%-accurate baseline.

## Project structure

```
dates-generator/
├── data/
│   ├── data.txt                   # full dataset
│   └── example_input.txt          # input conditions for generation
├── checkpoints/                   # saved model weights (created at training time)
├── output/                        # generated predictions (created at inference time)
├── report_figures/                # figures used in the PDF report
├── src/
│   ├── data/
│   │   ├── encoding.py            # one-hot encode/decode + constrained_decode
│   │   └── dataset.py             # PyTorch Dataset + DataLoader factory
│   ├── models/
│   │   ├── gan.py                 # Basic GAN
│   │   ├── vae.py                 # Variational Autoencoder
│   │   ├── cgan.py                # Conditional GAN (projection discriminator)
│   │   └── cyclegan.py            # Cycle GAN (conditions ↔ dates)
│   ├── training/
│   │   ├── train_gan.py
│   │   ├── train_vae.py
│   │   ├── train_cgan.py
│   │   └── train_cyclegan.py
│   └── utils.py                   # shared Condition Satisfaction Rate evaluator
├── solver.py                      # pure algorithmic solver (no ML)
├── train.py                       # unified training CLI
├── evaluate.py                    # evaluation + batch generation CLI
└── generate_report.py             # generates report.pdf with figures
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
| Day of week | one-hot MON...SUN     | 7   |
| Month       | one-hot JAN...DEC     | 12  |
| Leap year   | one-hot [False, True] | 2   |
| Decade      | one-hot 180...220     | 41  |

Total condition vector: **62-dim**.
Date (variable parts only): day-of-month (31) + year-in-decade (10) = **41-dim**.

## Models

| Model             | Key idea |
|-------------------|----------|
| **Basic GAN**     | MLP generator + discriminator, conditioned via concatenation |
| **VAE**           | Encoder maps to latent (mu, sigma^2); decoder samples from prior at inference |
| **Conditional GAN** | Dense condition embedding; projection discriminator + spectral norm + feature-matching loss |
| **Cycle GAN**     | Domains A=conditions, B=dates; cycle consistency enforces A->B->A and B->A->B |
| **Solver**        | Enumerate + filter valid dates deterministically; 100% CSR by construction |

## Evaluation metric

Since multiple valid dates exist per condition set, **exact-match accuracy is meaningless**.
The primary metric is the **Condition Satisfaction Rate (CSR)**:

| Metric    | Description |
|-----------|-------------|
| `day`     | Generated date falls on the required weekday |
| `month`   | Generated date is in the required month |
| `leap`    | Generated year has the required leap-year status |
| `decade`  | Generated year is in the required decade |
| **`all`** | **All four conditions satisfied simultaneously** |

## Results (test set, seed=42)

### Unconstrained decoding

| Model     | day    | month   | leap    | decade  | **all** |
|-----------|-------:|--------:|--------:|--------:|--------:|
| GAN       | 14.19% | 100.00% | 99.88%  | 100.00% | **14.18%** |
| VAE       | 13.89% | 100.00% | 100.00% | 100.00% | **13.89%** |
| cGAN      | 14.22% | 99.45%  | 77.48%  | 99.45%  | **10.82%** |
| CycleGAN  | 14.69% | 100.00% | 74.53%  | 100.00% | **10.85%** |

Without constrained decoding, all models hover near **~14% day accuracy (≈ 1/7)**,
confirming that none of them learned the weekday constraint from training alone.
Month and decade are near-perfect across the board. The cGAN and CycleGAN additionally
struggle with leap year (~75%) due to class imbalance in the training data.

### Constrained decoding (`--constrained`)

| Model     | day      | month    | leap    | decade   | **all**    |
|-----------|:--------:|:--------:|:-------:|:--------:|:----------:|
| GAN       | **100%** | **100%** | 92.52%  | **100%** | **92.52%** |
| VAE       | **100%** | **100%** | **100%**| **100%** | **100.00%**|
| cGAN      | **100%** | **100%** | 75.01%  | **100%** | **75.01%** |
| CycleGAN  | **100%** | **100%** | 68.78%  | **100%** | **68.78%** |

Constrained decoding enforces the weekday condition at inference time by enumerating
all 310 (day-of-month × year-in-decade) combinations, discarding those that produce the
wrong weekday, and selecting the highest-probability valid combination.
Day accuracy reaches **100% across all models** instantly.
The **VAE achieves a perfect 100% all-CSR** — the only remaining failures in the other
models are in the leap-year condition, which is a training issue (class imbalance), not
a decoding issue.

### Condition satisfaction by condition — summary

```
Metric      GAN     VAE     cGAN    CycleGAN    (constrained decoding)
--------  ------  ------  ------  ----------
day        100%    100%    100%      100%       <- fixed by constrained decoding
month      100%    100%    100%      100%       <- learned perfectly
leap        93%    100%     75%       69%       <- bottleneck for cGAN / CycleGAN
decade     100%    100%    100%      100%       <- learned perfectly
all         93%    100%     75%       69%       <- headline metric
```

### Key findings

- **Day of week was never learned during training** across any architecture. The weekday
  is a modular arithmetic function of the joint (day, month, year) triple. Since the
  output is factorised into independent softmax heads (day-of-month and year-in-decade),
  no gradient signal explicitly penalises weekday mismatch. All models plateau at ~1/7.
- **Constrained decoding fully solves the weekday problem** at zero training cost.
- **VAE + constrained decoding = 100% CSR** — the best overall result.
- **cGAN and CycleGAN struggle with leap year** due to the ~25% natural frequency of
  leap years in the data. Enabling the weighted sampler (`use_weighted_sampler=True`)
  during retraining is expected to resolve this.

## Setup

**1. Check your CUDA version**
```bash
nvidia-smi
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
```

**3. Install GPU-enabled PyTorch** (pick the line matching your CUDA version)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8
```

**4. Install remaining dependencies**
```bash
pip install -r requirements.txt
```

**5. Verify GPU**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Usage

### Algorithmic solver (no ML)
```bash
python solver.py                                  # uses data/example_input.txt
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

### Evaluate and generate dates
```bash
# Standard evaluation
python evaluate.py --model gan      --checkpoint checkpoints/gan_best.pt
python evaluate.py --model vae      --checkpoint checkpoints/vae_best.pt
python evaluate.py --model cgan     --checkpoint checkpoints/cgan_best.pt
python evaluate.py --model cyclegan --checkpoint checkpoints/cyclegan_best.pt

# With constrained decoding (enforces weekday at inference, recommended)
python evaluate.py --model vae --checkpoint checkpoints/vae_best.pt --constrained
```

Predictions are written to `output/<model>_predictions.txt`.

### Generate the PDF report
```bash
pip install reportlab matplotlib   # one-time install
python generate_report.py          # writes report.pdf
```
