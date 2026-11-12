import threading
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest

from forecaster.agent import Agent, parse_forecast
from forecaster.config import Config
from forecaster.models import Question
from forecaster.prompts import PERSONAS, SYSTEM_PROMPT

NOW = datetime(2026, 11, 11, 9, 30, tzinfo=UTC)
GOOD = 'Reasoning...\n<forecast>{"probability": 0.37, "base_rate": 0.25, "rationale": "Because."}</forecast>'


def make_question() -> Question:
    return Question(
        venue="polymarket",
        market_id="m1",
        title="Will it happen?",
        rules="Resolves YES if it happens.",
        close_time=datetime(2026, 12, 1, tzinfo=UTC),
        yes_bid=0.613,
        yes_ask=0.647,
    )


def reply(
    text: str | list[str] = GOOD,
    tokens_in: int = 1000,
    tokens_out: int = 500,
    searches: int = 2,
    stop_reason: str = "end_turn",
) -> SimpleNamespace:
    parts = [text] if isinstance(text, str) else text
    content: list[Any] = [SimpleNamespace(type="thinking", thinking="")]
    for part in parts:
        content.append(SimpleNamespace(type="server_tool_use", name="web_search"))
        content.append(SimpleNamespace(type="text", text=part))
    usage = SimpleNamespace(
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=None,
        server_tool_use=SimpleNamespace(web_search_requests=searches) if searches else None,
    )
    return SimpleNamespace(content=content, usage=usage, stop_reason=stop_reason)


def api_error(status: int) -> anthropic.APIStatusError:
    response: Any = SimpleNamespace(status_code=status, headers={}, request=None)
    cls = {429: anthropic.RateLimitError, 400: anthropic.BadRequestError}.get(
        status, anthropic.InternalServerError
    )
    return cls("boom", response=response, body=None)


class FakeClient:
    """Stands in for ``anthropic.Anthropic``. ``script`` is a reply, an exception,
    or a function of the request; a list is consumed one item per call."""

    def __init__(self, script: Any = None) -> None:
        self.script = script if script is not None else reply()
        self.calls: list[dict[str, Any]] = []
        self.messages = self
        self._lock = threading.Lock()

    def create(self, **kwargs: Any) -> Any:
        with self._lock:
            self.calls.append(kwargs)
            item = self.script.pop(0) if isinstance(self.script, list) else self.script
        if callable(item):
            item = item(kwargs)
        if isinstance(item, Exception):
            raise item
        return item


def make_agent(script: Any = None, **overrides: Any) -> tuple[Agent, FakeClient]:
    client = FakeClient(script)
    slept: list[float] = []
    agent = Agent(Config(**overrides), client=client, sleep=slept.append, keep_transcripts=True)
    agent.slept = slept  # type: ignore[attr-defined]
    return agent, client


# ---------- parser ----------


def test_parse_valid_block() -> None:
    s = parse_forecast(GOOD, "outside_view")
    assert s is not None
    assert (s.persona, s.probability, s.base_rate, s.rationale) == (
        "outside_view", 0.37, 0.25, "Because.",
    )  # fmt: skip


def test_parse_clamps_the_probability() -> None:
    high = parse_forecast('<forecast>{"probability": 1.0, "rationale": "sure"}</forecast>', "p")
    low = parse_forecast('<forecast>{"probability": 0, "rationale": "no"}</forecast>', "p")
    assert high is not None and high.probability == 0.99
    assert low is not None and low.probability == 0.01
    assert high.base_rate is None


def test_parse_tolerates_whitespace_newlines_and_braces_in_the_rationale() -> None:
    text = '<forecast>\n {"probability": 0.6,\n "base_rate": null, "rationale": "a {b} c"}\n</forecast>'
    s = parse_forecast(text, "p")
    assert s is not None and s.probability == 0.6 and s.rationale == "a {b} c"


def test_parse_uses_the_last_block() -> None:
    text = 'Format is <forecast>{"probability": 0.XX}</forecast>. ' + GOOD
    s = parse_forecast(text, "p")
    assert s is not None and s.probability == 0.37


def test_parse_drops_an_out_of_range_base_rate_but_keeps_the_sample() -> None:
    s = parse_forecast('<forecast>{"probability": 0.4, "base_rate": 40}</forecast>', "p")
    assert s is not None and s.base_rate is None and s.rationale == ""


@pytest.mark.parametrize(
    "text",
    [
        "",
        "I think about 40%.",
        '<forecast>{"probability": 0.4, "rationale": "cut off',
        "<forecast>{probability: 0.4}</forecast>",
        '<forecast>{"probability": "high"}</forecast>',
        '<forecast>{"probability": true}</forecast>',
        '<forecast>{"probability": NaN}</forecast>',
        '<forecast>{"rationale": "no number"}</forecast>',
        "<forecast>[0.4]</forecast>",
    ],
)
def test_parse_missing_or_garbled_block_is_none(text: str) -> None:
    assert parse_forecast(text, "p") is None


# ---------- request ----------


def test_request_follows_the_spec() -> None:
    agent, client = make_agent()
    agent.forecast(make_question(), 1, NOW)
    (call,) = client.calls
    assert call["model"] == "claude-fable-5-2" == agent.model
    assert call["max_tokens"] == 4000
    assert call["system"] == SYSTEM_PROMPT
    assert call["tools"] == [
        {
            "type": "web_search_20260318",
            "name": "web_search",
            "max_uses": 4,
            "blocked_domains": ["polymarket.com", "kalshi.com"],
        }
    ]
    # Fable 5.2 rejects sampling parameters and explicit thinking budgets.
    assert set(call) == {"model", "max_tokens", "system", "tools", "messages"}
    (message,) = call["messages"]
    assert message["role"] == "user"
    assert "QUESTION: Will it happen?" in message["content"]


def test_the_agent_never_sees_the_market_price() -> None:
    agent, client = make_agent()
    agent.forecast(make_question(), 5, NOW)
    sent = repr(client.calls)
    for hidden in ("0.613", "0.647", "0.63", "yes_bid", "yes_ask"):
        assert hidden not in sent


def test_no_blocked_domains_key_when_the_setting_is_empty() -> None:
    agent, client = make_agent(search_blocked_domains="")
    agent.forecast(make_question(), 1, NOW)
    assert "blocked_domains" not in client.calls[0]["tools"][0]


def test_one_pass_per_persona_in_order() -> None:
    def by_persona(request: dict[str, Any]) -> Any:
        angle = request["messages"][0]["content"].split("ANGLE FOR THIS PASS: ")[1]
        index = list(PERSONAS.values()).index(angle)
        return reply(f'<forecast>{{"probability": 0.{index + 1}}}</forecast>')

    agent, client = make_agent(by_persona)
    samples = agent.forecast(make_question(), 5, NOW)
    assert [s.persona for s in samples] == list(PERSONAS)
    assert [s.probability for s in samples] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert len(client.calls) == 5


def test_text_blocks_are_joined_without_separators() -> None:
    agent, _ = make_agent(reply(['<forecast>{"probability": 0.', '42, "rationale": "x"}</fore', "cast>"]))
    (sample,) = agent.forecast(make_question(), 1, NOW)
    assert sample.probability == 0.42


# ---------- cost and counters ----------


def test_cost_counts_tokens_and_searches() -> None:
    agent, _ = make_agent(reply(tokens_in=100_000, tokens_out=10_000, searches=3))
    agent.forecast(make_question(), 2, NOW)
    per_pass = 100_000 * 8 / 1e6 + 10_000 * 40 / 1e6 + 3 * 0.01
    assert per_pass == pytest.approx(1.23)
    assert agent.cost_usd == pytest.approx(2 * per_pass)
    assert agent.n_searches == 6
    assert (agent.input_tokens, agent.output_tokens) == (200_000, 20_000)
    assert agent.parse_failures == 0


def test_cost_without_searches() -> None:
    agent, _ = make_agent(reply(tokens_in=1000, tokens_out=1000, searches=0))
    agent.forecast(make_question(), 1, NOW)
    assert agent.cost_usd == pytest.approx(0.048)
    assert agent.n_searches == 0


def test_parse_failure_drops_the_sample_but_still_costs() -> None:
    agent, _ = make_agent([reply(), reply("No block here.", stop_reason="max_tokens"), reply()])
    samples = agent.forecast(make_question(), 3, NOW)
    assert len(samples) == 2
    assert agent.parse_failures == 1
    assert agent.cost_usd == pytest.approx(3 * (0.008 + 0.02 + 0.02))
    assert agent.transcripts is not None
    failed = [t for t in agent.transcripts if not t["parsed"]]
    assert [(t["stop_reason"], t["text"]) for t in failed] == [("max_tokens", "No block here.")]


def test_refusal_counts_as_a_parse_failure() -> None:
    agent, _ = make_agent(reply([], stop_reason="refusal"))
    assert agent.forecast(make_question(), 1, NOW) == []
    assert agent.parse_failures == 1


# ---------- paused turns ----------


def test_paused_turn_is_resumed_and_both_requests_are_billed() -> None:
    first = reply("Searching. ", searches=4, stop_reason="pause_turn")
    agent, client = make_agent([first, reply(searches=1)])
    (sample,) = agent.forecast(make_question(), 1, NOW)
    assert sample.probability == 0.37
    assert len(client.calls) == 2
    resumed = client.calls[1]["messages"]
    assert [m["role"] for m in resumed] == ["user", "assistant"]
    assert resumed[1]["content"] is first.content  # sent back unchanged
    assert agent.n_searches == 5
    assert agent.cost_usd == pytest.approx(2 * (0.008 + 0.02) + 0.05)


def test_a_turn_that_keeps_pausing_is_given_up_on() -> None:
    agent, client = make_agent(reply("still going", stop_reason="pause_turn"))
    assert agent.forecast(make_question(), 1, NOW) == []
    assert len(client.calls) == 4
    assert agent.parse_failures == 1


# ---------- retries ----------


def test_retryable_errors_back_off_then_succeed() -> None:
    timeout = anthropic.APITimeoutError(request=None)  # type: ignore[arg-type]
    agent, client = make_agent([api_error(429), timeout, reply()])
    assert len(agent.forecast(make_question(), 1, NOW)) == 1
    assert len(client.calls) == 3
    assert agent.slept == [2.0, 4.0]  # type: ignore[attr-defined]
    assert agent.api_failures == 0


def test_three_failed_attempts_drop_the_sample() -> None:
    agent, client = make_agent(api_error(529))
    assert agent.forecast(make_question(), 2, NOW) == []
    assert len(client.calls) == 6
    assert agent.api_failures == 2
    assert agent.parse_failures == 0
    assert agent.cost_usd == 0.0


def test_errors_a_retry_cannot_fix_are_raised() -> None:
    agent, client = make_agent(api_error(400))
    with pytest.raises(anthropic.BadRequestError):
        agent.forecast(make_question(), 1, NOW)
    assert len(client.calls) == 1
    assert agent.slept == []  # type: ignore[attr-defined]


# ---------- spending cap ----------


def test_no_request_starts_once_the_cap_is_reached() -> None:
    agent, client = make_agent(reply(tokens_in=100_000, tokens_out=10_000), max_spend_per_run=1.0)
    assert not agent.budget_exhausted
    assert len(agent.forecast(make_question(), 1, NOW)) == 1  # $1.22 spent
    assert agent.budget_exhausted
    assert agent.forecast(make_question(), 5, NOW) == []
    assert len(client.calls) == 1
    assert agent.parse_failures == 0


def test_a_paused_turn_is_not_resumed_over_the_cap() -> None:
    paused = reply("...", tokens_in=200_000, stop_reason="pause_turn")
    agent, client = make_agent(paused, max_spend_per_run=1.0)
    assert agent.forecast(make_question(), 1, NOW) == []
    assert len(client.calls) == 1


def test_zero_samples_requested() -> None:
    agent, client = make_agent()
    assert agent.forecast(make_question(), 0, NOW) == []
    assert client.calls == []


# ---------- pipeline ----------


def test_real_agent_through_the_pipeline() -> None:
    from forecaster.calibrate import Calibrator, fit_calibrator
    from forecaster.pipeline import Params, forecast_question

    config = Config()
    agent, _ = make_agent([reply(), reply("garbled"), reply(), reply(), reply()])
    identity: Calibrator = fit_calibrator([], [], config.calibrator_min_points)
    f = forecast_question(make_question(), agent, identity, Params.from_config(config), NOW, config)
    assert f is not None
    assert len(f.samples) == 4 and f.parse_failures == 1
    assert f.raw_prob == pytest.approx(0.37)
    assert f.n_searches == 10
    assert f.cost_usd == pytest.approx(agent.cost_usd) and f.cost_usd > 0
    assert f.model == "claude-fable-5-2"

