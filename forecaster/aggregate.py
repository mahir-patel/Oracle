"""Combining probabilities. Everything happens in log-odds, where independent
evidence adds and where "more extreme" is a simple multiplication."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

EPS = 1e-4  # probabilities are clipped to [EPS, 1 - EPS] before taking logits


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def trimmed_mean_logodds(probs: Sequence[float], trim: float = 0.2) -> float:
    """Mean of the logits after dropping ``int(n * trim)`` from each end.

    Nothing is dropped if that would leave no samples.
    """
    if not probs:
        raise ValueError("need at least one probability")
    logits = sorted(logit(p) for p in probs)
    k = int(len(logits) * trim)
    if len(logits) - 2 * k >= 1:
        logits = logits[k : len(logits) - k]
    return sigmoid(sum(logits) / len(logits))


def extremize(p: float, alpha: float) -> float:
    """Push p away from 50% (alpha > 1) or toward it (alpha < 1)."""
    return sigmoid(alpha * logit(p))


def blend_with_market(p: float, market: float | None, weight: float) -> float:
    """Weighted mean of model and market in log-odds; ``weight`` is the market's share."""
    if market is None:
        return p
    return sigmoid((1 - weight) * logit(p) + weight * logit(market))


def disagreement(probs: Sequence[float]) -> float:
    """Standard deviation of the sample logits (0 for a single sample)."""
    if len(probs) < 2:
        return 0.0
    return float(np.std([logit(p) for p in probs]))


@dataclass(frozen=True)
class Fit:
    alpha: float
    market_weight: float
    log_loss: float


ALPHA_GRID = np.round(np.arange(0.8, 2.5 + 1e-9, 0.05), 2)
WEIGHT_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)


def fit_alpha_and_weight(
    raw: Sequence[float], market: Sequence[float], outcomes: Sequence[int]
) -> Fit:
    """Grid search for the (alpha, market weight) pair that minimises log loss of
    ``sigmoid((1 - w) * alpha * logit(raw) + w * logit(market))``."""
    if not (len(raw) == len(market) == len(outcomes)) or len(raw) == 0:
        raise ValueError("raw, market and outcomes must be non-empty and the same length")
    lr = _logits(raw)
    lm = _logits(market)
    y = np.asarray(outcomes, dtype=float)

    best = Fit(alpha=1.0, market_weight=0.0, log_loss=math.inf)
    for alpha in ALPHA_GRID:
        for w in WEIGHT_GRID:
            z = (1 - w) * alpha * lr + w * lm
            p = np.clip(1 / (1 + np.exp(-z)), EPS, 1 - EPS)
            loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
            if loss < best.log_loss - 1e-12:
                best = Fit(float(alpha), float(w), loss)
    return best


def _logits(probs: Sequence[float]) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))
