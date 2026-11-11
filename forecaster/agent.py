"""Forecasting agents. The real LLM agent arrives in S4; ``FakeAgent`` is the
offline stand-in used by tests and the simulation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Protocol

import numpy as np

from forecaster.aggregate import logit, sigmoid
from forecaster.models import Question, Sample

PROB_LO, PROB_HI = 0.01, 0.99


class ForecastAgent(Protocol):
    """What the pipeline needs from an agent. The counters are running totals."""

    model: str
    cost_usd: float
    n_searches: int
    parse_failures: int

    def forecast(self, q: Question, n: int, now: datetime) -> list[Sample]: ...


class FakeAgent:
    """A noisy, hedging forecaster with access to the true probability.

    Each sample is ``sigmoid(hedge * logit(truth) + N(0, noise))`` clipped to
    [0.01, 0.99]: ``hedge`` < 1 pulls it toward 50% the way LLMs do, and
    ``noise`` is independent per sample. It costs nothing and never fails.
    """

    model = "fake"

    def __init__(
        self,
        truth_fn: Callable[[Question], float],
        noise: float = 0.5,
        hedge: float = 0.7,
        seed: int | Sequence[int] = 0,
    ) -> None:
        self.truth_fn = truth_fn
        self.noise = noise
        self.hedge = hedge
        self.rng = np.random.default_rng(seed)
        self.cost_usd = 0.0
        self.n_searches = 0
        self.parse_failures = 0

    def forecast(self, q: Question, n: int, now: datetime) -> list[Sample]:
        centre = self.hedge * logit(self.truth_fn(q))
        samples = []
        for i in range(n):
            p = sigmoid(centre + float(self.rng.normal(0, self.noise)))
            p = min(max(p, PROB_LO), PROB_HI)
            samples.append(Sample(persona=f"fake_{i}", probability=p, rationale="simulated"))
        return samples
