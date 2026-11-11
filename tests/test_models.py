from datetime import UTC, datetime, timedelta, timezone

import pytest

from forecaster.models import Forecast, Question, Sample


def make_question(**overrides: object) -> Question:
    base: dict[str, object] = dict(
        venue="sim",
        market_id="m1",
        title="Will it rain?",
        rules="Resolves YES if it rains.",
        close_time=datetime(2026, 9, 1, tzinfo=UTC),
        yes_bid=0.40,
        yes_ask=0.44,
    )
    base.update(overrides)
    return Question(**base)  # type: ignore[arg-type]


def test_mid_and_spread() -> None:
    q = make_question()
    assert q.mid == pytest.approx(0.42)
    assert q.spread == pytest.approx(0.04)


@pytest.mark.parametrize("missing", ["yes_bid", "yes_ask"])
def test_mid_and_spread_are_none_without_both_quotes(missing: str) -> None:
    q = make_question(**{missing: None})
    assert q.mid is None
    assert q.spread is None


def test_defaults() -> None:
    q = make_question()
    assert q.category == "other"
    assert q.liquidity == 0.0


def test_close_time_is_normalised_to_utc() -> None:
    naive = make_question(close_time=datetime(2026, 9, 1, 12, 0))
    assert naive.close_time == datetime(2026, 9, 1, 12, 0, tzinfo=UTC)

    eastern = timezone(timedelta(hours=-4))
    q = make_question(close_time=datetime(2026, 9, 1, 8, 0, tzinfo=eastern))
    assert q.close_time.utcoffset() == timedelta(0)
    assert q.close_time == datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def test_question_dict_round_trip() -> None:
    q = make_question(category="weather", liquidity=1234.5, url="https://example.test/m1")
    d = q.to_dict()
    assert d["close_time"] == "2026-09-01T00:00:00+00:00"
    assert Question.from_dict(d) == q


def test_forecast_to_dict_is_json_ready() -> None:
    f = Forecast(
        question=make_question(),
        samples=[Sample("outside_view", 0.3, "base rate", base_rate=0.25)],
        raw_prob=0.3,
        calibrated_prob=0.28,
        final_prob=0.35,
        market_prob=0.42,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        model="fake",
    )
    d = f.to_dict()
    assert d["created_at"] == "2026-08-01T00:00:00+00:00"
    assert d["question"]["market_id"] == "m1"
    assert d["samples"] == [
        {"persona": "outside_view", "probability": 0.3, "rationale": "base rate", "base_rate": 0.25}
    ]
    assert d["bet"] == {} and d["cost_usd"] == 0.0
