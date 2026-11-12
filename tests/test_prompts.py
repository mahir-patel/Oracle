import re
from datetime import UTC, datetime
from pathlib import Path

from forecaster import prompts
from forecaster.models import Question

NOW = datetime(2026, 11, 11, 9, 30, tzinfo=UTC)


def make_question(rules: str = "Resolves YES if the rate is cut.") -> Question:
    return Question(
        venue="kalshi",
        market_id="FED-26DEC",
        title="Will the Fed cut rates in December?",
        rules=rules,
        close_time=datetime(2026, 12, 16, 19, 0, tzinfo=UTC),
        yes_bid=0.613,
        yes_ask=0.647,
        liquidity=12345.0,
        url="https://kalshi.com/markets/FED",
    )


def test_system_prompt_and_personas_match_the_spec_verbatim() -> None:
    spec = (Path(__file__).parent.parent / "SPEC.md").read_text()
    section = spec.split("### 6.2")[1].split("### 6.3")[0]
    block = re.search(r"System prompt:\n```\n(.*?)\n```", section, re.DOTALL)
    assert block and prompts.SYSTEM_PROMPT == block.group(1)

    listed = re.findall(r"^- `(\w+)`: (.+)$", section, re.MULTILINE)
    assert listed == list(prompts.PERSONAS.items())
    assert len(listed) == 5


def test_personas_cycle_in_order() -> None:
    names = [prompts.persona_for(i)[0] for i in range(7)]
    assert names == [
        "outside_view", "inside_view", "devils_advocate", "resolution_lawyer", "status_quo",
        "outside_view", "inside_view",
    ]  # fmt: skip
    assert prompts.persona_for(2)[1] == prompts.PERSONAS["devils_advocate"]


def test_user_message_layout() -> None:
    text = prompts.user_message(make_question(), prompts.PERSONAS["status_quo"], NOW)
    assert text.splitlines() == [
        "Today is 2026-11-11 09:30 UTC.",
        "QUESTION: Will the Fed cut rates in December?",
        "RESOLUTION RULES: Resolves YES if the rate is cut.",
        "CLOSES: 2026-12-16 19:00 UTC",
        f"ANGLE FOR THIS PASS: {prompts.PERSONAS['status_quo']}",
    ]


def test_user_message_flags_missing_rules() -> None:
    text = prompts.user_message(make_question(rules="  "), "angle", NOW)
    assert "RESOLUTION RULES: (none provided — infer from the question" in text


def test_user_message_never_contains_the_market_price() -> None:
    """Rule 2: the agent forms its view without seeing the price."""
    q = make_question()
    text = prompts.user_message(q, "angle", NOW) + prompts.SYSTEM_PROMPT
    for hidden in ("0.613", "0.647", "0.63", "61", "64", "12345", "kalshi.com"):
        assert hidden not in text
