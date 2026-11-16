from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from forecaster.agent import FakeAgent
from forecaster.aggregate import blend_with_market, extremize, trimmed_mean_logodds
from forecaster.calibrate import Calibrator
from forecaster.config import Config
from forecaster.ledger import Ledger
from forecaster.models import Question, Sample
from forecaster.pipeline import (
    Params,
    forecast_question,
    pick_daily,
    quotas,
    select_questions,
)

CFG = Config()
NOW = datetime(2026, 8, 1, tzinfo=UTC)


def make_question(market_id: str = "m1", **overrides: object) -> Question:
    base: dict[str, object] = dict(
        venue="sim",
        market_id=market_id,
        title=f"Question {market_id}?",
        rules="Resolves YES if it happens.",
        close_time=NOW + timedelta(days=10),
        yes_bid=0.39,
        yes_ask=0.41,
        liquidity=5000.0,
    )
    base.update(overrides)
    return Question(**base)  # type: ignore[arg-type]


# ---------- FakeAgent ----------


def test_fake_agent_is_seeded_and_free() -> None:
    q = make_question()
    a = FakeAgent(lambda _: 0.7, seed=1).forecast(q, 5, NOW)
    b = FakeAgent(lambda _: 0.7, seed=1).forecast(q, 5, NOW)
    c = FakeAgent(lambda _: 0.7, seed=2).forecast(q, 5, NOW)
    assert a == b and a != c
    assert len(a) == 5
    assert all(0.01 <= s.probability <= 0.99 for s in a)
    agent = FakeAgent(lambda _: 0.7)
    assert agent.cost_usd == 0 and agent.model == "fake"


def test_fake_agent_hedges_toward_half() -> None:
    q = make_question()
    exact = FakeAgent(lambda _: 0.9, noise=0.0, hedge=1.0).forecast(q, 1, NOW)[0]
    hedged = FakeAgent(lambda _: 0.9, noise=0.0, hedge=0.7).forecast(q, 1, NOW)[0]
    assert exact.probability == pytest.approx(0.9)
    assert 0.5 < hedged.probability < 0.9
    extreme = FakeAgent(lambda _: 0.9999, noise=0.0, hedge=1.0).forecast(q, 1, NOW)[0]
    assert extreme.probability == 0.99


# ---------- forecast_question ----------


class StubAgent:
    """Returns fixed probabilities and charges a fixed cost per call."""

    model = "stub"

    def __init__(self, probs: list[float]) -> None:
        self.probs = probs
        self.cost_usd = 1.0  # spend from earlier questions must not be re-counted
        self.n_searches = 3
        self.parse_failures = 0

    def forecast(self, q: Question, n: int, now: datetime) -> list[Sample]:
        self.cost_usd += 0.25
        self.n_searches += 7
        self.parse_failures += 1
        return [Sample(f"p{i}", p, "r") for i, p in enumerate(self.probs)]


def test_pipeline_applies_each_step_in_order() -> None:
    probs = [0.55, 0.60, 0.62, 0.65, 0.95]
    params = Params(alpha=1.5, market_weight=0.25)
    q = make_question()
    f = forecast_question(q, StubAgent(probs), Calibrator(), params, NOW, CFG)
    assert f is not None
    raw = trimmed_mean_logodds(probs, 0.2)
    assert f.raw_prob == pytest.approx(raw)
    assert f.calibrated_prob == pytest.approx(extremize(raw, 1.5))
    assert f.final_prob == pytest.approx(blend_with_market(extremize(raw, 1.5), 0.40, 0.25))
    assert f.market_prob == pytest.approx(0.40)
    assert f.created_at == NOW and f.model == "stub"
    assert f.disagreement > 0
    assert f.bet["side"] == "yes"  # model well above a 40% market


def test_pipeline_records_only_this_questions_cost_and_counts() -> None:
    f = forecast_question(
        make_question(), StubAgent([0.5]), Calibrator(), Params(1.0, 0.0), NOW, CFG
    )
    assert f is not None
    assert f.cost_usd == pytest.approx(0.25)
    assert f.n_searches == 7
    assert f.parse_failures == 1
    assert f.latency_s >= 0


def test_pipeline_uses_the_calibrator() -> None:
    cal = Calibrator(xs=[0.0, 1.0], ys=[0.2, 0.2], n=2000)
    f = forecast_question(make_question(), StubAgent([0.9]), cal, Params(1.0, 0.0), NOW, CFG)
    assert f is not None
    assert f.raw_prob == pytest.approx(0.9)
    assert f.calibrated_prob == pytest.approx(0.2)


def test_pipeline_without_a_quote_keeps_the_model_view() -> None:
    q = make_question(yes_bid=None, yes_ask=None)
    f = forecast_question(q, StubAgent([0.7]), Calibrator(), Params(1.0, 0.5), NOW, CFG)
    assert f is not None
    assert f.market_prob is None
    assert f.final_prob == pytest.approx(0.7)
    assert f.bet["side"] is None


def test_pipeline_returns_none_without_samples() -> None:
    assert (
        forecast_question(make_question(), StubAgent([]), Calibrator(), Params(1, 0), NOW, CFG)
        is None
    )


def test_end_to_end_with_fake_agent_writes_a_valid_ledger(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    agent = FakeAgent(lambda q: 0.2 if q.market_id.endswith("0") else 0.8, seed=3)
    params = Params.from_config(CFG)
    for i in range(20):
        f = forecast_question(make_question(f"m{i}"), agent, Calibrator(), params, NOW, CFG)
        assert f is not None
        ledger.append_forecast(f)
    for i in range(0, 20, 2):
        ledger.append_resolution("sim", f"m{i}", i % 3 == 0, NOW + timedelta(days=11))

    assert ledger.verify()
    rows = ledger.joined()
    assert len(rows) == 20
    assert sum(r["outcome"] is not None for r in rows) == 10
    row = rows[0]
    assert len(row["samples"]) == CFG.n_samples
    assert 0 < row["final_prob"] < 1
    assert row["model"] == "fake" and row["cost_usd"] == 0
    assert "side" in row["bet"]
    assert row["question"]["fee_rate"] is None


# ---------- Params ----------


def test_params_default_to_config_priors_and_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "data" / "params.json"
    assert Params.load(path, CFG) == Params(alpha=1.3, market_weight=0.5, n_fit=0)
    Params(alpha=1.45, market_weight=0.3, n_fit=120).save(path)
    assert Params.load(path, CFG) == Params(alpha=1.45, market_weight=0.3, n_fit=120)


# ---------- select_questions ----------


def test_select_keeps_a_good_market() -> None:
    assert select_questions([make_question()], NOW, CFG) == [make_question()]


@pytest.mark.parametrize(
    "overrides",
    [
        {"close_time": NOW + timedelta(hours=23)},  # closes too soon
        {"close_time": NOW + timedelta(days=61)},  # too far out
        {"close_time": NOW - timedelta(days=1)},  # already closed
        {"liquidity": 499.0},
        {"yes_bid": None},
        {"yes_ask": None},
        {"yes_bid": 0.30, "yes_ask": 0.41},  # spread 0.11
        {"yes_bid": 0.01, "yes_ask": 0.03},  # mid 0.02
        {"yes_bid": 0.97, "yes_ask": 0.99},  # mid 0.98
    ],
)
def test_select_rejects(overrides: dict[str, object]) -> None:
    assert select_questions([make_question(**overrides)], NOW, CFG) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"close_time": NOW + timedelta(days=1)},
        {"close_time": NOW + timedelta(days=60)},
        {"liquidity": 500.0},
        {"yes_bid": 0.30, "yes_ask": 0.40},  # spread exactly 0.10
        {"yes_bid": 0.02, "yes_ask": 0.04},  # mid exactly 0.03
        {"yes_bid": 0.96, "yes_ask": 0.98},  # mid exactly 0.97
    ],
)
def test_select_boundaries_are_inclusive(overrides: dict[str, object]) -> None:
    assert len(select_questions([make_question(**overrides)], NOW, CFG)) == 1


# ---------- pick_daily ----------


def band_questions(short: int, mid: int, long: int, events: bool = True) -> list[Question]:
    out = []
    for name, count, days in (("s", short, 5), ("m", mid, 20), ("l", long, 45)):
        for i in range(count):
            out.append(
                make_question(
                    f"{name}{i}",
                    close_time=NOW + timedelta(days=days),
                    event_id=f"ev-{name}{i}" if events else "",
                )
            )
    return out


def test_quotas_split_by_share() -> None:
    shares = [0.5, 0.3, 0.2]
    assert quotas(10, shares) == [5, 3, 2]
    assert quotas(5, shares) == [3, 1, 1]
    assert quotas(1, shares) == [1, 0, 0]
    assert quotas(0, shares) == [0, 0, 0]
    assert all(sum(quotas(n, shares)) == n for n in range(40))


def test_pick_daily_fills_the_quotas(tmp_path: Path) -> None:
    picked, counts = pick_daily(band_questions(20, 20, 20), Ledger(tmp_path / "l"), 10, NOW, CFG)
    assert [(c.eligible, c.quota, c.picked) for c in counts] == [(20, 5, 5), (20, 3, 3), (20, 2, 2)]
    assert [q.market_id[0] for q in picked] == list("sssssmmmll")


def test_pick_daily_redistributes_unfilled_slots_shortest_first(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l")
    # The long band is empty: its 2 slots go to the short band.
    _, counts = pick_daily(band_questions(20, 20, 0), ledger, 10, NOW, CFG)
    assert [c.picked for c in counts] == [7, 3, 0]
    # The short band has 1: its 4 spare slots go to the medium band first.
    _, counts = pick_daily(band_questions(1, 20, 20), ledger, 10, NOW, CFG)
    assert [c.picked for c in counts] == [1, 7, 2]
    # Not enough anywhere: take everything there is.
    picked, counts = pick_daily(band_questions(1, 2, 3), ledger, 10, NOW, CFG)
    assert [c.picked for c in counts] == [1, 2, 3] and len(picked) == 6


def test_pick_daily_takes_one_market_per_event(tmp_path: Path) -> None:
    questions = [
        make_question(f"m{i}", close_time=NOW + timedelta(days=5), event_id="fed-december")
        for i in range(5)
    ]
    questions.append(make_question("other", close_time=NOW + timedelta(days=5), event_id="x"))
    # A different venue with the same event id is a different event.
    questions.append(
        make_question("k", venue="kalshi", close_time=NOW + timedelta(days=5), event_id="x")
    )
    picked, counts = pick_daily(questions, Ledger(tmp_path / "l"), 10, NOW, CFG)
    assert len(picked) == 3 and counts[0].eligible == 3
    assert len({(q.venue, q.event_id) for q in picked}) == 3
    # Markets with no known event are each their own event.
    picked, _ = pick_daily(band_questions(4, 0, 0, events=False), Ledger(tmp_path / "l"), 10, NOW, CFG)
    assert len(picked) == 4


def test_pick_daily_skips_markets_and_events_already_in_the_ledger(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    done = make_question("done", close_time=NOW + timedelta(days=5), event_id="ev-done")
    forecast = forecast_question(done, FakeAgent(lambda _: 0.5), Calibrator(), Params(1, 0), NOW, CFG)
    assert forecast is not None
    ledger.append_forecast(forecast)
    sibling = make_question("sibling", close_time=NOW + timedelta(days=5), event_id="ev-done")
    fresh = make_question("fresh", close_time=NOW + timedelta(days=5), event_id="ev-new")
    picked, _ = pick_daily([done, sibling, fresh], ledger, 10, NOW, CFG)
    assert [q.market_id for q in picked] == ["fresh"]


def test_pick_daily_is_fixed_by_the_date_not_the_input_order(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l")
    questions = band_questions(30, 30, 30)
    a, _ = pick_daily(questions, ledger, 10, NOW, CFG)
    b, _ = pick_daily(list(reversed(questions)), ledger, 10, NOW + timedelta(hours=5), CFG)
    c, _ = pick_daily(questions, ledger, 10, NOW + timedelta(days=1), CFG)
    assert [q.market_id for q in a] == [q.market_id for q in b]
    assert [q.market_id for q in a] != [q.market_id for q in c]
