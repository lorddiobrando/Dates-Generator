#!/usr/bin/env python3
"""Pure algorithmic date solver — no ML required.

Usage:
    python solver.py                             # reads data/example_input.txt
    python solver.py data/example_input.txt
    python solver.py data/example_input.txt --seed 0
"""

from __future__ import annotations

import argparse
import random
from datetime import date
from typing import Optional

from src.data.encoding import (
    DAY_TOKENS,
    MONTH_TOKENS,
    check_conditions,
    is_leap_year,
    parse_conditions,
)


def _days_in_month(month: int, year: int) -> int:
    if month in (4, 6, 9, 11):
        return 30
    if month == 2:
        return 29 if is_leap_year(year) else 28
    return 31


def solve(
    day: str,
    month: str,
    leap: str,
    decade: str,
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """Return a random valid date (dd-mm-yyyy) satisfying all four conditions.

    Returns None if no valid date exists for the combination (e.g. a decade
    that has no leap years when leap=True is required).
    """
    if rng is None:
        rng = random.Random()

    target_weekday = DAY_TOKENS.index(day)
    target_month   = MONTH_TOKENS.index(month) + 1
    target_leap    = leap == "True"
    decade_start   = int(decade) * 10
    decade_end     = min(decade_start + 9, 2200)

    valid: list[str] = []
    for year in range(decade_start, decade_end + 1):
        if is_leap_year(year) != target_leap:
            continue
        for d in range(1, _days_in_month(target_month, year) + 1):
            if date(year, target_month, d).weekday() == target_weekday:
                valid.append(f"{d:02d}-{target_month:02d}-{year:04d}")

    return rng.choice(valid) if valid else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithmic date solver")
    parser.add_argument("input", nargs="?", default="data/example_input.txt")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.input) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    passed = 0
    for line in lines:
        day, month, leap, decade = parse_conditions(line)
        result = solve(day, month, leap, decade, rng=rng)
        if result is not None:
            checks = check_conditions(result, day, month, leap, decade)
            ok     = checks["all"]
        else:
            ok = False
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"{line}  →  {result or 'NO_SOLUTION':>14}  [{status}]")

    print(f"\n{passed}/{len(lines)} fully satisfied")


if __name__ == "__main__":
    main()
