"""Glue: one question in, one Forecast out; plus market selection.

The agent never sees the market price. The blend with the market happens here,
afterwards, with a weight fitted on resolved data.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from forecaster.agent import ForecastAgent
from forecaster.aggregate import blend_with_market, disagreement, extremize, trimmed_mean_logodds
from forecaster.calibrate import Calibrator
from forecaster.config import Config
from forecaster.models import Forecast, Question, ensure_utc
from forecaster.sizing import decide


@dataclass(frozen=True)
class Params:
    """The fitted numbers. Until a refit writes params.json, the config priors apply."""

    alpha: float
    market_weight: float
    n_fit: int = 0  # resolved forecasts they were fitted on; 0 = priors

    @classmethod
    def from_config(cls, config: Config) -> Params:
        return cls(alpha=config.extremize_alpha, market_weight=config.market_weight)

    @classmethod
    def load(cls, path: str | Path, config: Config) -> Params:
        path = Path(path)
        if not path.exists():
            return cls.from_config(config)
        d = json.loads(path.read_text())
        return cls(float(d["alpha"]), float(d["market_weight"]), int(d.get("n_fit", 0)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def forecast_question(
    q: Question,
    agent: ForecastAgent,
    calibrator: Calibrator,
    params: Params,
    now: datetime,
    config: Config,
) -> Forecast | None:
    """samples -> trimmed mean -> extremize -> calibrate -> blend with market -> bet.

    Returns None when the agent produced no valid sample.
    """
    cost_before = agent.cost_usd
    searches_before = agent.n_searches
    failures_before = agent.parse_failures
    started = time.perf_counter()
    samples = agent.forecast(q, config.n_samples, now)
    latency = time.perf_counter() - started
    if not samples:
        return None

    probs = [s.probability for s in samples]
    raw = trimmed_mean_logodds(probs, config.trim_fraction)
    calibrated = calibrator.apply(extremize(raw, params.alpha))
    final = blend_with_market(calibrated, q.mid, params.market_weight)
    bet = decide(final, q.yes_bid, q.yes_ask, q.venue, config, q.fee_rate)

    return Forecast(
        question=q,
        samples=samples,
        raw_prob=raw,
        calibrated_prob=calibrated,
        final_prob=final,
        market_prob=q.mid,
        created_at=now,
        model=agent.model,
        cost_usd=agent.cost_usd - cost_before,
        bet=bet,
        disagreement=disagreement(probs),
        latency_s=latency,
        n_searches=agent.n_searches - searches_before,
        parse_failures=agent.parse_failures - failures_before,
    )


def select_questions(
    questions: Iterable[Question], now: datetime, config: Config
) -> list[Question]:
    """Keep markets worth forecasting.

    - 1 to 60 days to close: under a day leaves no time to be right, over 60
      makes feedback too slow.
    - enough liquidity for the quote to mean something.
    - a two-sided quote with a tight spread: wide spreads eat the edge.
    - mid away from 0 and 1: near-certain markets have no room for edge.
    """
    now = ensure_utc(now)
    kept = []
    for q in questions:
        days = (q.close_time - now).total_seconds() / 86400
        if not config.min_days_to_close <= days <= config.max_days_to_close:
            continue
        if q.liquidity < config.min_liquidity:
            continue
        mid, spread = q.mid, q.spread
        if mid is None or spread is None or spread > config.max_spread + 1e-9:
            continue
        if not config.min_mid <= mid <= config.max_mid:
            continue
        kept.append(q)
    return kept
