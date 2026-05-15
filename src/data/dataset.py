"""PyTorch Dataset and DataLoader factory for the dates dataset."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .encoding import (
    encode_conditions,
    encode_date,
    parse_data_line,
    _LEAP_END,
    _MONTH_END,
    _DAY_END,
    LEAP_DIM,
)

Sample = Tuple[torch.Tensor, torch.Tensor]  # (cond 62-dim, date 41-dim)


class DateDataset(Dataset[Sample]):
    def __init__(self, samples: List[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def load_samples(path: str | Path) -> List[Sample]:
    """Parse data.txt and return a list of (cond_tensor, date_tensor) pairs."""
    samples: List[Sample] = []
    with open(path, "r") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            day, month, leap, decade, date_str = parse_data_line(raw)
            d_str, m_str, y_str = date_str.split("-")
            cond     = encode_conditions(day, month, leap, decade)
            date_vec = encode_date(int(d_str), int(y_str))
            samples.append((cond, date_vec))
    return samples


def _leap_weights(samples: List[Sample]) -> List[float]:
    """Assign higher weight to leap-year samples to counter class imbalance."""
    weights: List[float] = []
    for cond, _ in samples:
        leap_idx = int(cond[_MONTH_END:_LEAP_END].argmax().item())
        weights.append(3.0 if leap_idx == 1 else 1.0)
    return weights


def make_loaders(
    path: str | Path,
    batch_size: int = 128,
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader).

    The split is reproducible: data is shuffled with a fixed seed before
    partitioning, so train/val/test sets are always identical for the same seed.
    """
    generator = torch.Generator().manual_seed(seed)
    all_samples = load_samples(path)
    n = len(all_samples)

    perm    = torch.randperm(n, generator=generator).tolist()
    n_train = int(n * split[0])
    n_val   = int(n * split[1])

    train_s = [all_samples[i] for i in perm[:n_train]]
    val_s   = [all_samples[i] for i in perm[n_train : n_train + n_val]]
    test_s  = [all_samples[i] for i in perm[n_train + n_val :]]

    sampler: WeightedRandomSampler | None = None
    if use_weighted_sampler:
        w = _leap_weights(train_s)
        sampler = WeightedRandomSampler(
            weights=w,
            num_samples=len(w),
            replacement=True,
            generator=generator,
        )

    train_loader = DataLoader(
        DateDataset(train_s),
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        generator=generator if sampler is None else None,
    )
    val_loader = DataLoader(
        DateDataset(val_s),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        DateDataset(test_s),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
