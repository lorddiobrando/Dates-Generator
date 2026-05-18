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


def constrained_decode(cond: torch.Tensor, date_vec: torch.Tensor) -> str:
    """Decode a date while enforcing the weekday condition.

    Enumerates all 310 (day_of_month x year_in_decade) combinations, discards
    any that produce the wrong weekday or are calendar-invalid, then returns the
    combination with the highest joint probability.  Falls back to unconstrained
    decode if no valid combination exists (should not happen in practice).
    """
    target_weekday = int(cond[:_DAY_END].argmax().item())       # 0=MON … 6=SUN
    month          = int(cond[_DAY_END:_MONTH_END].argmax().item()) + 1
    decade         = int(cond[_LEAP_END:].argmax().item()) + DECADE_MIN

    dom_probs = date_vec[:DAY_OF_MONTH_DIM]   # (31,) — probability per day-of-month
    yid_probs = date_vec[DAY_OF_MONTH_DIM:]   # (10,) — probability per year-in-decade

    best_score: float = -1.0
    best_date:  str | None = None

    for yid in range(YEAR_IN_DECADE_DIM):
        year = decade * 10 + yid
        for dom in range(DAY_OF_MONTH_DIM):
            day_of_month = dom + 1
            try:
                d = date(year, month, day_of_month)
            except ValueError:
                continue                          # calendar-invalid (e.g. Feb 30)

            if d.weekday() != target_weekday:
                continue                          # wrong weekday — skip

            score = dom_probs[dom].item() * yid_probs[yid].item()
            if score > best_score:
                best_score = score
                best_date  = f"{day_of_month:02d}-{month:02d}-{year:04d}"

    return best_date if best_date is not None else decode_date(cond, date_vec)


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


# ── Joint softmax constants (310-dim date space) ──────────────────────────────
JOINT_DIM = DAY_OF_MONTH_DIM * YEAR_IN_DECADE_DIM  # 31 * 10 = 310


def pair_to_index(dom_idx: int, yid: int) -> int:
    """Map (day-of-month index, year-in-decade) → flat index in [0, 310)."""
    return dom_idx * YEAR_IN_DECADE_DIM + yid


def index_to_pair(idx: int) -> Tuple[int, int]:
    """Map flat index back to (dom_idx, yid)."""
    return idx // YEAR_IN_DECADE_DIM, idx % YEAR_IN_DECADE_DIM


def encode_date_joint(day_of_month: int, year: int) -> torch.Tensor:
    """One-hot encode a specific date in 310-dim joint (dom × yid) space."""
    dom_idx = day_of_month - 1
    yid     = year % 10
    vec     = torch.zeros(JOINT_DIM, dtype=torch.float32)
    vec[pair_to_index(dom_idx, yid)] = 1.0
    return vec


def build_calendar_mask(cond: torch.Tensor) -> torch.Tensor:
    """Return a 310-dim boolean mask.

    True  → (dom, yid) pair is a calendar-valid date with the correct leap status.
    False → impossible date (e.g. Feb 30) or wrong leap year.

    The weekday condition is deliberately NOT masked here; the model must learn
    the weekday constraint from the soft-label targets.
    """
    month     = int(cond[_DAY_END:_MONTH_END].argmax().item()) + 1
    leap_flag = int(cond[_MONTH_END:_LEAP_END].argmax().item()) == 1
    decade    = int(cond[_LEAP_END:].argmax().item()) + DECADE_MIN

    mask = torch.zeros(JOINT_DIM, dtype=torch.bool)
    for yid in range(YEAR_IN_DECADE_DIM):
        year = decade * 10 + yid
        if is_leap_year(year) != leap_flag:
            continue
        for dom in range(DAY_OF_MONTH_DIM):
            try:
                date(year, month, dom + 1)
                mask[pair_to_index(dom, yid)] = True
            except ValueError:
                pass
    return mask


def build_soft_label(cond: torch.Tensor) -> torch.Tensor:
    """Return a 310-dim uniform soft-label over all valid (dom, yid) pairs.

    A pair is valid when it produces a real calendar date that satisfies ALL
    four conditions: correct month (from cond), correct leap status, correct
    decade, AND the correct day-of-week.  The result is normalised to sum=1.

    If no valid pair exists (edge case) the returned tensor is all-zeros.
    """
    target_weekday = int(cond[:_DAY_END].argmax().item())
    month          = int(cond[_DAY_END:_MONTH_END].argmax().item()) + 1
    leap_flag      = int(cond[_MONTH_END:_LEAP_END].argmax().item()) == 1
    decade         = int(cond[_LEAP_END:].argmax().item()) + DECADE_MIN

    soft = torch.zeros(JOINT_DIM, dtype=torch.float32)
    for yid in range(YEAR_IN_DECADE_DIM):
        year = decade * 10 + yid
        if is_leap_year(year) != leap_flag:
            continue
        for dom in range(DAY_OF_MONTH_DIM):
            day_of_month = dom + 1
            try:
                d = date(year, month, day_of_month)
            except ValueError:
                continue
            if d.weekday() == target_weekday:
                soft[pair_to_index(dom, yid)] = 1.0

    total = soft.sum()
    if total > 0:
        soft /= total
    return soft


def decode_joint(cond: torch.Tensor, joint_vec: torch.Tensor) -> str:
    """Reconstruct a 'dd-mm-yyyy' string from condition + 310-dim joint vector.

    Takes the argmax of joint_vec, maps it back to (dom_idx, yid), and
    combines with month and decade from the condition tensor.
    """
    month  = int(cond[_DAY_END:_MONTH_END].argmax().item()) + 1
    decade = int(cond[_LEAP_END:].argmax().item()) + DECADE_MIN

    best_idx     = int(joint_vec.argmax().item())
    dom_idx, yid = index_to_pair(best_idx)
    day_of_month = dom_idx + 1
    year         = decade * 10 + yid
    return f"{day_of_month:02d}-{month:02d}-{year:04d}"


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
