"""Isotonic calibration by pool-adjacent-violators (PAV), gated by sample size.

Below ``min_points`` resolved forecasts the calibrator is the identity. Stacked on
top of the fitted extremizing alpha, a second calibrator overfits on small
histories (see the ablation in CLAUDE.md), so it only switches on with a lot of data.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MIN_POINTS = 1000
CLIP_LO, CLIP_HI = 0.01, 0.99


@dataclass
class Calibrator:
    """Piecewise-linear monotone map. With no knots it is the identity."""

    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    n: int = 0  # points it was fitted on (or offered, if gated off)

    @property
    def is_identity(self) -> bool:
        return not self.xs

    def apply(self, p: float) -> float:
        if self.is_identity:
            return p
        value = float(np.interp(p, self.xs, self.ys))
        return min(max(value, CLIP_LO), CLIP_HI)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"xs": self.xs, "ys": self.ys, "n": self.n}, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Calibrator:
        """A missing file means no calibrator has been fitted yet: the identity."""
        path = Path(path)
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())
        return cls(xs=list(d["xs"]), ys=list(d["ys"]), n=int(d["n"]))


def fit_calibrator(
    probs: Sequence[float], outcomes: Sequence[int], min_points: int = MIN_POINTS
) -> Calibrator:
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes must be the same length")
    n = len(probs)
    if n == 0 or n < min_points:
        return Calibrator(n=n)

    x = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)

    # Start with one block per distinct forecast value, in increasing order.
    # A block is [sum_y, count, min_x, max_x].
    blocks: list[list[float]] = []
    for value in np.unique(x):
        mask = x == value
        blocks.append([float(y[mask].sum()), float(mask.sum()), float(value), float(value)])

    # Pass 1: classic PAV on the block means. Ties are pooled too, so a run of
    # identical outcomes becomes one block rather than many single points.
    stack = _pool(blocks, lambda lo, hi: _mean(lo) >= _mean(hi))
    # Pass 2: each block's value is Laplace-smoothed, (sum_y + 1) / (count + 2),
    # so it is never exactly 0 or 1. Smoothing pulls small blocks toward 0.5 more
    # than large ones, which can reorder neighbours; pool again until monotone.
    stack = _pool(stack, lambda lo, hi: _value(lo) > _value(hi))

    xs = [(b[2] + b[3]) / 2 for b in stack]
    ys = [_value(b) for b in stack]
    return Calibrator(xs=xs, ys=ys, n=n)


def _pool(
    blocks: list[list[float]], violates: Callable[[list[float], list[float]], bool]
) -> list[list[float]]:
    stack: list[list[float]] = []
    for block in blocks:
        stack.append(block)
        while len(stack) > 1 and violates(stack[-2], stack[-1]):
            hi = stack.pop()
            lo = stack.pop()
            stack.append([lo[0] + hi[0], lo[1] + hi[1], lo[2], hi[3]])
    return stack


def _mean(block: list[float]) -> float:
    return block[0] / block[1]


def _value(block: list[float]) -> float:
    return (block[0] + 1) / (block[1] + 2)
