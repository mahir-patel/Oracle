import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import anthropic
import httpx
import pytest

from forecaster import cli
from forecaster.agent import FakeAgent
from forecaster.config import Config
from forecaster.ledger import Ledger
from forecaster.models import Question, Sample

NOW = datetime(2026, 11, 15, 6, 30, tzinfo=UTC)


class PaidAgent:
    """Answers 0.30 on every sample and charges a fixed price per question."""

    model = "test-model"

    def __init__(self, config: Config, price: float = 0.10, fail_on: int | None = None) -> None:
        self.config = config
        self.price = price
        self.fail_on = fail_on
        self.calls = 0
        self.cost_usd = 0.0
        self.n_searches = 0
        self.parse_failures = 0

    def forecast(self, q: Question, n: int, now: datetime) -> list[Sample]:
        self.calls += 1
        if self.calls == self.fail_on:
            raise anthropic.APIError("credit balance is too low", request=None, body=None)  # type: ignore[arg-type]
        self.cost_usd += self.price
        if q.market_id.startswith("silent"):
            return []
        return [Sample("outside_view", 0.30, "because") for _ in range(n)]


def question(market_id: str, days: float = 5, venue: str = "polymarket", **kw: Any) -> Question:
    return Question(
        venue=venue,
        market_id=market_id,
        title=f"Will {market_id} happen?",
        rules="Resolves YES if it happens.",
        close_time=NOW + timedelta(days=days),
        yes_bid=0.39,
        yes_ask=0.41,
        liquidity=5000.0,
        url=f"https://example.com/{market_id}",
        event_id=f"ev-{market_id}",
        **kw,
    )


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A CLI wired to temp files, a fixed clock, fake markets and a fake agent."""
    for name in ("ledger", "params", "calibrator", "report"):
        suffix = "jsonl" if name == "ledger" else "json"
        monkeypatch.setenv(f"ORACLE_{name.upper()}_PATH", str(tmp_path / f"{name}.{suffix}"))
    monkeypatch.setenv("ORACLE_SITE_PATH", str(tmp_path / "site" / "index.html"))
    monkeypatch.setenv("ORACLE_SPENT_BEFORE_LEDGER", "0")
    state: dict[str, Any] = {
        "tmp": tmp_path,
        "ledger": Ledger(tmp_path / "ledger.jsonl"),
        "now": NOW,
        "markets": {"kalshi": [], "polymarket": [question(f"p{i}") for i in range(8)]},
        "outcomes": {},
        "fetches": [],
        "agents": [],
        "agent_kwargs": {},
    }

    def fake_fetch(venue: str, scan: int, client: Any, lo: datetime, hi: datetime) -> list[Question]:
        state["fetches"].append((venue, round((lo - state["now"]).days), round((hi - state["now"]).days)))
        found = state["markets"][venue]
        if isinstance(found, Exception):
            raise found
        return [q for q in found if lo <= q.close_time <= hi]

    def fake_resolve(venue: str, market_id: str, client: Any = None) -> int | None:
        outcome = state["outcomes"].get(market_id)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def fake_agent(config: Config) -> PaidAgent:
        agent = PaidAgent(config, **state["agent_kwargs"])
        state["agents"].append(agent)
        return agent

    monkeypatch.setattr(cli, "utc_now", lambda: state["now"])
    monkeypatch.setattr(cli, "fetch_open", fake_fetch)
    monkeypatch.setattr(cli, "resolve", fake_resolve)
    monkeypatch.setattr(cli, "make_agent", fake_agent)
    return state


def forecasts(world: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in world["ledger"].records() if r["type"] == "forecast"]


# ---------- forecast ----------


def test_forecast_logs_the_picked_questions(world: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["forecast", "--max", "3"]) == 0
    logged = forecasts(world)
    assert len(logged) == 3 and world["ledger"].verify()
    assert all(r["model"] == "test-model" and r["cost_usd"] == pytest.approx(0.10) for r in logged)
    # One fetch per venue per close-time band.
    assert sorted(world["fetches"]) == sorted(
        (v, lo, hi) for v in ("kalshi", "polymarket") for lo, hi in ((1, 14), (14, 30), (30, 60))
    )
    out = capsys.readouterr().out
    assert "band 1-14 days: 8 eligible, quota 1, picked 3" in out  # other bands are empty
    assert "run: 3 logged of 3 attempted (3 picked); spent $0.300 of $1.00 cap" in out


def test_forecast_never_repeats_a_market(world: dict[str, Any]) -> None:
    cli.main(["forecast", "--max", "5"])
    cli.main(["forecast", "--max", "5"])
    ids = [r["question"]["market_id"] for r in forecasts(world)]
    assert len(ids) == 8 == len(set(ids))  # only 8 exist


def test_forecast_one_venue_and_a_failed_fetch(world: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    world["markets"]["kalshi"] = httpx.ConnectError("down")
    assert cli.main(["forecast", "--venue", "both", "--max", "2"]) == 0
    assert len(forecasts(world)) == 2
    assert "kalshi 1-14 days: fetch failed, skipped" in capsys.readouterr().out
    world["fetches"].clear()
    cli.main(["forecast", "--venue", "polymarket", "--max", "1"])
    assert {v for v, _, _ in world["fetches"]} == {"polymarket"}


def test_budget_guard_stops_before_a_question_it_cannot_afford(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ORACLE_MAX_SPEND_PER_RUN", "0.35")
    monkeypatch.setenv("ORACLE_FIRST_QUESTION_COST_ESTIMATE", "0.20")
    cli.main(["forecast", "--max", "8"])
    # 0.10 each: after 3 questions 0.05 is left, less than the 0.10 average.
    assert len(forecasts(world)) == 3
    out = capsys.readouterr().out
    assert "budget guard: $0.05 left, a question costs about $0.10; stopping" in out
    assert "spent $0.300 of $0.35 cap" in out


def test_first_question_is_skipped_if_the_estimate_does_not_fit(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORACLE_MAX_SPEND_PER_RUN", "0.15")
    cli.main(["forecast", "--max", "3"])
    assert forecasts(world) == [] and world["agents"][0].calls == 0


def test_total_budget_caps_the_run_and_then_stops_everything(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ORACLE_MAX_SPEND_TOTAL", "10")
    monkeypatch.setenv("ORACLE_SPENT_BEFORE_LEDGER", "9.55")
    monkeypatch.setenv("ORACLE_FIRST_QUESTION_COST_ESTIMATE", "0.10")
    cli.main(["forecast", "--max", "8"])
    # 0.45 of the total is left, so the run's cap is 0.45, not 1.00.
    assert world["agents"][0].config.max_spend_per_run == pytest.approx(0.45)
    assert len(forecasts(world)) == 4
    assert "this run may spend $0.45" in capsys.readouterr().out

    monkeypatch.setenv("ORACLE_SPENT_BEFORE_LEDGER", "9.60")  # 9.60 + 0.40 logged = 10
    world["fetches"].clear()
    assert cli.main(["forecast", "--max", "8"]) == 0
    assert len(forecasts(world)) == 4 and world["fetches"] == []
    assert "total budget reached" in capsys.readouterr().out


def test_question_with_no_valid_sample_is_not_logged(world: dict[str, Any]) -> None:
    world["markets"]["polymarket"] = [question("silent1"), question("p1")]
    cli.main(["forecast", "--max", "2"])
    assert [r["question"]["market_id"] for r in forecasts(world)] == ["p1"]


def test_api_error_keeps_what_was_logged_and_fails_the_run(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    world["agent_kwargs"] = {"fail_on": 3}
    assert cli.main(["forecast", "--max", "5"]) == 1
    assert len(forecasts(world)) == 2 and world["ledger"].verify()
    out = capsys.readouterr().out
    assert "API error, run stopped" in out and "run: 2 logged of 3 attempted" in out


def test_missing_api_key_stops_before_anything_is_fetched(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()  # drop the fake agent, so the real factory runs
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ORACLE_LEDGER_PATH", str(world["tmp"] / "ledger.jsonl"))
    monkeypatch.setattr(cli, "fetch_open", lambda *a, **k: pytest.fail("must not fetch"))
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY is not set"):
        cli.main(["forecast"])


# ---------- resolve ----------


def test_resolve_logs_results_of_closed_markets_only(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    world["markets"]["polymarket"] = [
        question("yes", days=2),
        question("no", days=3),
        question("pending", days=4),
        question("broken", days=5),
        question("open", days=40),
    ]
    cli.main(["forecast", "--max", "5"])
    assert len(forecasts(world)) == 5
    world["now"] = NOW + timedelta(days=10)
    world["outcomes"] = {"yes": 1, "no": 0, "pending": None, "broken": httpx.ReadTimeout("slow"), "open": 1}
    assert cli.main(["resolve"]) == 0
    rows = {r["question"]["market_id"]: r["outcome"] for r in world["ledger"].joined()}
    # "open" has not closed, so it is not even looked up.
    assert rows == {"yes": 1, "no": 0, "pending": None, "broken": None, "open": None}
    out = capsys.readouterr().out
    assert "broken: lookup failed, will retry" in out
    assert "resolve: 2 resolved, 2 closed and still waiting" in out

    world["outcomes"]["broken"] = 0
    cli.main(["resolve"])  # the next run picks it up, and logs nothing twice
    assert sum(r["type"] == "resolution" for r in world["ledger"].records()) == 3
    assert world["ledger"].verify()


# ---------- refit ----------


def seed_resolved(world: dict[str, Any], n: int) -> None:
    world["markets"]["polymarket"] = [question(f"q{i}", days=2) for i in range(n)]
    for _ in range(n // 8 + 1):
        cli.main(["forecast", "--max", "8"])
    world["now"] = NOW + timedelta(days=5)
    world["outcomes"] = {f"q{i}": int(i % 3 == 0) for i in range(n)}
    cli.main(["resolve"])


def test_refit_needs_30_clean_rows(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORACLE_MAX_SPEND_PER_RUN", "100")
    monkeypatch.setenv("ORACLE_MODEL", "test-model")
    seed_resolved(world, 29)
    assert cli.main(["refit"]) == 1
    assert not (world["tmp"] / "params.json").exists()


def test_refit_writes_params_and_a_gated_calibrator(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORACLE_MAX_SPEND_PER_RUN", "100")
    monkeypatch.setenv("ORACLE_MODEL", "test-model")
    seed_resolved(world, 32)
    assert cli.main(["refit"]) == 0
    params = json.loads((world["tmp"] / "params.json").read_text())
    assert params["n_fit"] == 32 and 0.8 <= params["alpha"] <= 2.5
    assert json.loads((world["tmp"] / "calibrator.json").read_text())["xs"] == []  # below the gate

    # Forecasts by another model are never pooled into this model's fit.
    monkeypatch.setenv("ORACLE_MODEL", "some-other-model")
    assert cli.main(["refit"]) == 1


# ---------- report, verify, simulate ----------


def test_report_writes_the_json_and_the_page(world: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["forecast", "--max", "2"])
    assert cli.main(["report"]) == 0
    saved = json.loads((world["tmp"] / "report.json").read_text())
    assert (saved["n_total"], saved["n_open"]) == (2, 2)
    page = (world["tmp"] / "site" / "index.html").read_text()
    assert page.count("https://example.com/p") == 2
    assert "# Oracle report" in capsys.readouterr().out


def test_verify_exit_code(world: dict[str, Any]) -> None:
    assert cli.main(["verify"]) == 0  # an empty ledger is valid
    cli.main(["forecast", "--max", "2"])
    assert cli.main(["verify"]) == 0
    path = world["tmp"] / "ledger.jsonl"
    path.write_text(path.read_text().replace('"final_prob":0.', '"final_prob":0.9', 1))
    assert cli.main(["verify"]) == 1


def test_simulate_prints_both_phases(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["simulate", "--n", "60", "--seed", "7"]) == 0
    out = capsys.readouterr().out
    assert "Phase 2 (fitted)" in out and "hash chain OK" in out


def test_fake_agent_runs_through_the_cli(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "make_agent", lambda config: FakeAgent(lambda q: 0.6, seed=1))
    assert cli.main(["forecast", "--max", "2"]) == 0
    assert len(forecasts(world)) == 2
