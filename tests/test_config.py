from datetime import UTC, datetime

import pytest

from forecaster.config import Config, load_config


def test_defaults_match_spec() -> None:
    c = load_config({})
    assert c == Config()
    assert c.model == "claude-fable-5-2"
    assert c.model_cutoff == datetime(2026, 7, 1, tzinfo=UTC)
    assert c.n_samples == 5
    assert c.max_web_searches == 4
    assert (c.price_in, c.price_out, c.price_per_search) == (8.0, 40.0, 0.01)
    assert c.max_tokens == 4000
    assert (c.cache_write_multiplier, c.cache_read_multiplier) == (1.25, 0.025)
    assert c.web_search_tool == "web_search_20260318"
    assert c.search_blocked_domains == "polymarket.com,kalshi.com"
    assert c.trim_fraction == 0.2
    assert c.extremize_alpha == 1.3
    assert c.market_weight == 0.5
    assert c.kelly_fraction == 0.25
    assert c.min_edge == 0.04
    assert c.max_stake_fraction == 0.05
    assert c.bankroll == 1000.0
    assert c.max_spend_per_run == 5.0
    assert c.ledger_path == "data/ledger.jsonl"


def test_env_overrides_are_typed() -> None:
    c = load_config(
        {
            "ORACLE_MAX_SPEND_PER_RUN": "2.5",
            "ORACLE_N_SAMPLES": "3",
            "ORACLE_MODEL": "other-model",
            "ORACLE_MODEL_CUTOFF": "2026-03-01",
            "UNRELATED": "ignored",
        }
    )
    assert c.max_spend_per_run == 2.5
    assert c.n_samples == 3 and isinstance(c.n_samples, int)
    assert c.model == "other-model"
    assert c.model_cutoff == datetime(2026, 3, 1, tzinfo=UTC)


def test_cutoff_is_always_utc_aware() -> None:
    c = load_config({"ORACLE_MODEL_CUTOFF": "2026-03-01T12:00:00+00:00"})
    assert c.model_cutoff.tzinfo is not None


def test_bad_value_names_the_variable() -> None:
    with pytest.raises(ValueError, match="ORACLE_N_SAMPLES"):
        load_config({"ORACLE_N_SAMPLES": "five"})
