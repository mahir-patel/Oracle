"""Glue: one question in, one Forecast out; plus market selection.

The agent never sees the market price. The blend with the market happens here,
afterwards, with a weight fitted on resolved data.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from forecaster.agent import ForecastAgent
from forecaster.aggregate import blend_with_market, disagreement, extremize, trimmed_mean_logodds
from forecaster.calibrate import Calibrator
from forecaster.config import Config
from forecaster.ledger import Ledger
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


# ---------- the daily pick ----------


@dataclass(frozen=True)
class BandCount:
    """What happened in one close-time band of a daily pick."""

    lo_days: float
    hi_days: float
    eligible: int  # events open to us in this band
    quota: int
    picked: int  # after redistribution


def bands(config: Config) -> list[tuple[float, float, float]]:
    """(lo_days, hi_days, share) per band, shortest first."""
    edges = [float(x) for x in config.band_days.split(",")]
    shares = [float(x) for x in config.band_shares.split(",")]
    if len(edges) != len(shares) + 1 or edges != sorted(edges) or abs(sum(shares) - 1) > 1e-6:
        raise ValueError("band_days needs one more ascending edge than band_shares, summing to 1")
    return [(edges[i], edges[i + 1], shares[i]) for i in range(len(shares))]


def quotas(max_n: int, shares: Sequence[float]) -> list[int]:
    """Split ``max_n`` by the shares; leftover slots go to the largest
    remainders, shortest band first on a tie (10 -> 5/3/2, 5 -> 3/1/1)."""
    exact = [max_n * s for s in shares]
    out = [int(x + 1e-9) for x in exact]
    order = sorted(range(len(shares)), key=lambda i: (-(exact[i] - out[i]), i))
    for i in order[: max_n - sum(out)]:
        out[i] += 1
    return out


def pick_daily(
    selected: Iterable[Question],
    ledger: Ledger,
    max_n: int,
    today: datetime,
    config: Config,
) -> tuple[list[Question], list[BandCount]]:
    """Choose which selected markets are forecast today (SPEC 6.9).

    - quotas by time to close, so most questions resolve soon;
    - one market per event, and none from an event already in the ledger,
      because markets of one event resolve together and are not independent;
    - a shuffle seeded by the UTC date, a rule fixed in advance so nobody can
      cherry-pick easy questions;
    - slots a band cannot fill go to the other bands, shortest first.
    """
    today = ensure_utc(today)
    seen_markets: set[tuple[str, str]] = set()
    seen_events: set[tuple[str, str]] = set()
    for r in ledger.records():
        if r["type"] == "forecast":
            q = r["question"]
            seen_markets.add((q["venue"], q["market_id"]))
            if q.get("event_id"):
                seen_events.add((q["venue"], q["event_id"]))

    def event_key(q: Question) -> tuple[str, str]:
        # A market with no known event counts as an event of its own.
        return (q.venue, q.event_id or f"market:{q.market_id}")

    plan = bands(config)
    pools: list[list[Question]] = [[] for _ in plan]
    for q in sorted(selected, key=lambda q: (q.venue, q.market_id)):
        if (q.venue, q.market_id) in seen_markets or event_key(q) in seen_events:
            continue
        seen_markets.add((q.venue, q.market_id))  # the same market fetched twice
        days = (q.close_time - today).total_seconds() / 86400
        for i, (lo, hi, _) in enumerate(plan):
            if lo <= days < hi or (i == len(plan) - 1 and days == hi):
                pools[i].append(q)
                break

    day = today.strftime("%Y-%m-%d")
    for i, pool in enumerate(pools):
        random.Random(f"{day}:{i}").shuffle(pool)

    quota = quotas(max_n, [share for _, _, share in plan])
    taken: set[tuple[str, str]] = set()
    picked: list[list[Question]] = [[] for _ in plan]

    def take(i: int, count: int) -> int:
        got = 0
        while got < count and pools[i]:
            q = pools[i].pop(0)
            if event_key(q) not in taken:
                taken.add(event_key(q))
                picked[i].append(q)
                got += 1
        return got

    eligible = [len({event_key(q) for q in pool}) for pool in pools]
    left = sum(n - take(i, n) for i, n in enumerate(quota))
    for i in range(len(plan)):  # redistribution, shortest band first
        left -= take(i, left)

    counts = [
        BandCount(lo, hi, eligible[i], quota[i], len(picked[i]))
        for i, (lo, hi, _) in enumerate(plan)
    ]
    return [q for band in picked for q in band], counts
