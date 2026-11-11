"""Proper scoring rules, skill against the market, calibration and the
contamination check. Lower Brier and log loss are better."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np

EPS = 1e-4
MIN_N_FOR_BOOTSTRAP = 10
CONTAMINATION_RATIO = 0.75


def brier(p: Sequence[float], y: Sequence[int]) -> float:
    p_, y_ = _arrays(p, y)
    return float(np.mean((p_ - y_) ** 2))


def log_loss(p: Sequence[float], y: Sequence[int]) -> float:
    p_, y_ = _arrays(p, y)
    p_ = np.clip(p_, EPS, 1 - EPS)
    return float(-np.mean(y_ * np.log(p_) + (1 - y_) * np.log(1 - p_)))


def brier_skill_vs(p: Sequence[float], market: Sequence[float], y: Sequence[int]) -> float:
    """1 - Brier(p) / Brier(market). The headline number: > 0 means we beat the market."""
    reference = brier(market, y)
    if reference == 0:
        return 0.0 if brier(p, y) == 0 else float("-inf")
    return 1 - brier(p, y) / reference


@dataclass(frozen=True)
class BootstrapBSS:
    bss: float
    ci_low: float  # 2.5th percentile
    ci_high: float  # 97.5th percentile
    p_beat_market: float  # share of resamples with BSS > 0
    n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_bss(
    p: Sequence[float],
    market: Sequence[float],
    y: Sequence[int],
    n_resamples: int = 5000,
    seed: int = 0,
) -> BootstrapBSS | None:
    """Bootstrap the Brier skill score over questions. None when n < 10."""
    p_, y_ = _arrays(p, y)
    m_, _ = _arrays(market, y)
    n = len(y_)
    if n < MIN_N_FOR_BOOTSTRAP:
        return None
    err_p = (p_ - y_) ** 2
    err_m = (m_ - y_) ** 2
    rng = np.random.default_rng(seed)
    scores = np.empty(n_resamples)
    chunk = 250  # keeps the index matrix small for large n
    for start in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - start)
        idx = rng.integers(0, n, size=(size, n))
        num = err_p[idx].mean(axis=1)
        den = err_m[idx].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores[start : start + size] = np.where(den > 0, 1 - num / den, np.nan)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return None
    return BootstrapBSS(
        bss=brier_skill_vs(p_, m_, y_),
        ci_low=float(np.percentile(scores, 2.5)),
        ci_high=float(np.percentile(scores, 97.5)),
        p_beat_market=float(np.mean(scores > 0)),
        n=n,
    )


def calibration_table(
    p: Sequence[float], y: Sequence[int], bins: int = 10
) -> list[dict[str, Any]]:
    """Equal-width bins with count, mean forecast and observed rate (None if empty)."""
    p_, y_ = _arrays(p, y) if len(p) else (np.array([]), np.array([]))
    index = np.minimum((p_ * bins).astype(int), bins - 1)  # p = 1.0 goes in the top bin
    table: list[dict[str, Any]] = []
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        table.append(
            {
                "lo": b / bins,
                "hi": (b + 1) / bins,
                "n": count,
                "mean_forecast": float(p_[mask].mean()) if count else None,
                "observed": float(y_[mask].mean()) if count else None,
            }
        )
    return table


def contamination_check(
    p: Sequence[float],
    y: Sequence[int],
    resolved_at: Sequence[datetime],
    cutoff: datetime,
) -> dict[str, Any]:
    """Compare Brier on questions resolved before vs after the model's cutoff.

    If the model is much better on pre-cutoff questions (Brier below 0.75x the
    post-cutoff Brier), it may be remembering outcomes rather than forecasting.
    """
    pre = [i for i, t in enumerate(resolved_at) if t < cutoff]
    post = [i for i, t in enumerate(resolved_at) if t >= cutoff]
    brier_pre = brier([p[i] for i in pre], [y[i] for i in pre]) if pre else None
    brier_post = brier([p[i] for i in post], [y[i] for i in post]) if post else None
    suspicious = (
        brier_pre is not None
        and brier_post is not None
        and brier_pre < CONTAMINATION_RATIO * brier_post
    )
    return {
        "n_pre": len(pre),
        "n_post": len(post),
        "brier_pre": brier_pre,
        "brier_post": brier_post,
        "suspicious": bool(suspicious),
    }


def _arrays(p: Sequence[float], y: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    p_ = np.asarray(p, dtype=float)
    y_ = np.asarray(y, dtype=float)
    if p_.shape != y_.shape or p_.size == 0:
        raise ValueError("forecasts and outcomes must be non-empty and the same length")
    return p_, y_
