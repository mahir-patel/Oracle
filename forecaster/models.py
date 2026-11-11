"""Plain data types shared by every module: Question, Sample, Forecast."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def ensure_utc(dt: datetime) -> datetime:
    """Datetimes are always UTC-aware; a naive value is taken to be UTC."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


@dataclass
class Question:
    venue: str  # "kalshi" | "polymarket" | "sim"
    market_id: str
    title: str
    rules: str  # full resolution text
    close_time: datetime  # UTC
    yes_bid: float | None = None  # probability units, 0-1
    yes_ask: float | None = None
    category: str = "other"
    liquidity: float = 0.0  # USD
    url: str = ""
    # Taker fee coefficient for this market (fee = rate * contracts * P * (1 - P)).
    # None means "use the venue's default"; see sizing.py.
    fee_rate: float | None = None

    def __post_init__(self) -> None:
        self.close_time = ensure_utc(self.close_time)

    @property
    def mid(self) -> float | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return (self.yes_bid + self.yes_ask) / 2

    @property
    def spread(self) -> float | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return self.yes_ask - self.yes_bid

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["close_time"] = self.close_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Question:
        return cls(**{**d, "close_time": datetime.fromisoformat(d["close_time"])})


@dataclass
class Sample:
    persona: str
    probability: float
    rationale: str
    base_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Forecast:
    question: Question
    samples: list[Sample]
    raw_prob: float  # aggregate of the samples
    calibrated_prob: float  # after extremizing and the (gated) calibrator
    final_prob: float  # after the market blend
    market_prob: float | None  # the mid at forecast time
    created_at: datetime
    model: str
    cost_usd: float = 0.0
    bet: dict[str, Any] = field(default_factory=dict)
    # Measurements logged with every forecast (rule 5).
    disagreement: float = 0.0  # std of the sample logits
    latency_s: float = 0.0
    n_searches: int = 0
    parse_failures: int = 0

    def __post_init__(self) -> None:
        self.created_at = ensure_utc(self.created_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question.to_dict(),
            "samples": [s.to_dict() for s in self.samples],
            "raw_prob": self.raw_prob,
            "calibrated_prob": self.calibrated_prob,
            "final_prob": self.final_prob,
            "market_prob": self.market_prob,
            "created_at": self.created_at.isoformat(),
            "model": self.model,
            "cost_usd": self.cost_usd,
            "bet": self.bet,
            "disagreement": self.disagreement,
            "latency_s": self.latency_s,
            "n_searches": self.n_searches,
            "parse_failures": self.parse_failures,
        }
