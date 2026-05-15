"""Shared evaluation utilities used by all training scripts."""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader

from .data.encoding import (
    check_conditions,
    constrained_decode,
    decode_date,
    DAY_TOKENS,
    MONTH_TOKENS,
    DECADE_MIN,
    _DAY_END,
    _MONTH_END,
    _LEAP_END,
)

GeneratorFn = Callable[[torch.Tensor], torch.Tensor]


def evaluate_csr(
    generator_fn: GeneratorFn,
    loader: DataLoader,
    device: torch.device,
    constrained: bool = False,
) -> dict[str, float]:
    """Compute Condition Satisfaction Rates on a DataLoader.

    Args:
        generator_fn: callable that accepts a condition tensor (N, 62) already
                      on *device* and returns a date tensor (N, 41).
        loader:       DataLoader yielding (cond, date) batches.
        device:       target device for inference.
        constrained:  if True, use constrained_decode to enforce the weekday
                      condition at inference time.

    Returns:
        Dict with keys 'day', 'month', 'leap', 'decade', 'all', each in [0, 1].
    """
    decode_fn = constrained_decode if constrained else decode_date
    counts: dict[str, int] = {"day": 0, "month": 0, "leap": 0, "decade": 0, "all": 0}
    total = 0

    with torch.no_grad():
        for cond_batch, _ in loader:
            cond_batch = cond_batch.to(device)
            fake_batch = generator_fn(cond_batch)
            n = cond_batch.size(0)

            for i in range(n):
                ci = cond_batch[i].cpu()
                fi = fake_batch[i].cpu()

                day    = DAY_TOKENS[int(ci[:_DAY_END].argmax().item())]
                month  = MONTH_TOKENS[int(ci[_DAY_END:_MONTH_END].argmax().item())]
                leap_i = int(ci[_MONTH_END:_LEAP_END].argmax().item())
                leap   = "True" if leap_i == 1 else "False"
                decade = str(int(ci[_LEAP_END:].argmax().item()) + DECADE_MIN)

                try:
                    date_str = decode_fn(ci, fi)
                    result   = check_conditions(date_str, day, month, leap, decade)
                    for k in counts:
                        if result[k]:
                            counts[k] += 1
                except (ValueError, IndexError):
                    pass   # invalid calendar date counts as all-fail

                total += 1

    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}
