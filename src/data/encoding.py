"""One-hot encoding / decoding utilities for conditions and dates."""

import torch
from datetime import date
from typing import Tuple

# ── Vocabulary ────────────────────────────────────────────────────────────────
DAY_TOKENS   = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MONTH_TOKENS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
LEAP_TOKENS  = ["False", "True"]

DECADE_MIN = 180   # decade code for 1800–1809
DECADE_MAX = 220   # decade code for 2200 (only 2200 is valid)
NUM_DECADES = DECADE_MAX - DECADE_MIN + 1  # 41

# ── Dimension constants ───────────────────────────────────────────────────────
DAY_DIM          = 7
MONTH_DIM        = 12
LEAP_DIM         = 2
DECADE_DIM       = NUM_DECADES      # 41
COND_DIM         = DAY_DIM + MONTH_DIM + LEAP_DIM + DECADE_DIM  # 62

DAY_OF_MONTH_DIM = 31
YEAR_IN_DECADE_DIM = 10
DATE_DIM         = DAY_OF_MONTH_DIM + YEAR_IN_DECADE_DIM  # 41

# ── Slice helpers (avoids magic numbers throughout the codebase) ──────────────
_DAY_END    = DAY_DIM
_MONTH_END  = _DAY_END + MONTH_DIM
_LEAP_END   = _MONTH_END + LEAP_DIM
# _DECADE_END = COND_DIM


def _onehot(index: int, size: int) -> torch.Tensor:
    v = torch.zeros(size, dtype=torch.float32)
    v[index] = 1.0
    return v


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def encode_conditions(day: str, month: str, leap: str, decade: str) -> torch.Tensor:
    """Encode four condition tokens into a 62-dim one-hot float tensor."""
    day_idx    = DAY_TOKENS.index(day)
    month_idx  = MONTH_TOKENS.index(month)
    leap_idx   = LEAP_TOKENS.index(leap)
    decade_idx = int(decade) - DECADE_MIN

    return torch.cat([
        _onehot(day_idx,    DAY_DIM),
        _onehot(month_idx,  MONTH_DIM),
        _onehot(leap_idx,   LEAP_DIM),
        _onehot(decade_idx, DECADE_DIM),
    ])  # (62,)


def encode_date(day_of_month: int, year: int) -> torch.Tensor:
    """Encode the variable parts of a date into a 41-dim one-hot float tensor.

    Month is omitted because it duplicates the month condition.
    Decade is omitted because it duplicates the decade condition.
    """
    dom_idx        = day_of_month - 1        # 1–31 → 0–30
    year_in_decade = year % 10               # 0–9

    return torch.cat([
        _onehot(dom_idx,        DAY_OF_MONTH_DIM),
        _onehot(year_in_decade, YEAR_IN_DECADE_DIM),
    ])  # (41,)


def decode_date(cond: torch.Tensor, date_vec: torch.Tensor) -> str:
    """Reconstruct a 'dd-mm-yyyy' string from condition + date tensors."""
    month_idx  = int(cond[_DAY_END:_MONTH_END].argmax().item())
    month      = month_idx + 1

    decade_idx = int(cond[_LEAP_END:].argmax().item())
    decade     = decade_idx + DECADE_MIN

    day_of_month  = int(date_vec[:DAY_OF_MONTH_DIM].argmax().item()) + 1
    year_in_decade = int(date_vec[DAY_OF_MONTH_DIM:].argmax().item())
    year           = decade * 10 + year_in_decade

    return f"{day_of_month:02d}-{month:02d}-{year:04d}"


def parse_conditions(line: str) -> Tuple[str, str, str, str]:
    """Parse a conditions-only line: '[MON] [DEC] [False] [196]'."""
    tokens = line.strip().split()
    return (
        tokens[0].strip("[]"),
        tokens[1].strip("[]"),
        tokens[2].strip("[]"),
        tokens[3].strip("[]"),
    )


def parse_data_line(line: str) -> Tuple[str, str, str, str, str]:
    """Parse a full data line: '[MON] [DEC] [False] [196] 3-12-1962'."""
    tokens = line.strip().split()
    return (
        tokens[0].strip("[]"),
        tokens[1].strip("[]"),
        tokens[2].strip("[]"),
        tokens[3].strip("[]"),
        tokens[4],
    )


def check_conditions(
    date_str: str,
    day: str,
    month: str,
    leap: str,
    decade: str,
) -> dict[str, bool]:
    """Return per-condition satisfaction booleans for a generated date string.

    Keys: 'day', 'month', 'leap', 'decade', 'all'.
    Raises ValueError for calendar-invalid dates (e.g. Feb 30).
    """
    parts = date_str.split("-")
    d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
    dt = date(y, m, d)  # raises ValueError for invalid calendar dates

    results: dict[str, bool] = {
        "day":    DAY_TOKENS[dt.weekday()] == day,
        "month":  MONTH_TOKENS[m - 1]      == month,
        "leap":   str(is_leap_year(y))     == leap,
        "decade": str(y // 10)             == decade,
    }
    results["all"] = all(results.values())
    return results
