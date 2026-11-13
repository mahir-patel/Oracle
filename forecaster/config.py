"""All settings live here. Every field can be overridden by an environment
variable named ``ORACLE_<FIELD_NAME_UPPERCASE>``, e.g. ``ORACLE_MAX_SPEND_PER_RUN=2``.

The Anthropic API key is deliberately not a setting: the SDK reads
``ANTHROPIC_API_KEY`` from the environment itself, so it never passes through here.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from typing import Any

ENV_PREFIX = "ORACLE_"


@dataclass(frozen=True)
class Config:
    # --- model ---
    # Haiku 5.5 since 2026-11-13: the whole project has a 50 USD budget, and
    # Fable 5.2 cost about 5.50 USD per question in the live check.
    model: str = "claude-haiku-5-5"
    # Questions resolved before this instant may be remembered by the model, so
    # they are excluded from scoring, fitting and the public record (rule 1).
    model_cutoff: datetime = datetime(2026, 7, 1, tzinfo=UTC)
    n_samples: int = 5
    max_web_searches: int = 2  # search fees are most of the cost on Haiku
    max_tokens: int = 4000  # per sample; covers thinking, search code and the reply
    # Server-side web search tool version (checked against the docs 2026-11-11).
    web_search_tool: str = "web_search_20260318"
    # Comma-separated domains the search may not return. The venues' own pages
    # show the live price, which the agent must not see (rule 2).
    search_blocked_domains: str = "polymarket.com,kalshi.com"
    # Comma-separated `allowed_callers` for the search tool; empty = API default
    # (searches run from filtering code). Set to "direct" if the API asks for it.
    search_allowed_callers: str = ""
    # Thinking mode sent with each request; empty sends none. Haiku 5.5 does not
    # think unless asked, and a forecast needs the reasoning.
    thinking: str = "adaptive"

    # --- cost tracking (USD) ---
    # Haiku 5.5 list prices per million tokens (docs 2026-11-13). Thinking is
    # billed as output.
    price_in: float = 0.10
    price_out: float = 0.50
    # Haiku 5.5 charges a request whose prompt is over 100,000 tokens (cached
    # tokens included) at higher prices. 0 switches the second tier off.
    long_prompt_tokens: int = 100_000
    price_in_long: float = 0.50
    price_out_long: float = 2.50
    # Prompt caching: a write costs 1.25x the input price, a read 0.1x.
    cache_write_multiplier: float = 1.25
    cache_read_multiplier: float = 0.1
    price_per_search: float = 0.01  # $10 per 1,000 searches; failed searches are free
    max_spend_per_run: float = 1.0
    first_question_cost_estimate: float = 0.20

    # --- aggregation (priors, replaced by fitted values in params.json) ---
    trim_fraction: float = 0.2
    extremize_alpha: float = 1.3
    market_weight: float = 0.5

    # --- calibration ---
    calibrator_min_points: int = 1000

    # --- paper sizing ---
    kelly_fraction: float = 0.25
    min_edge: float = 0.04
    max_stake_fraction: float = 0.05
    bankroll: float = 1000.0

    # --- market selection ---
    min_days_to_close: float = 1.0
    max_days_to_close: float = 60.0
    min_liquidity: float = 500.0
    max_spread: float = 0.10
    min_mid: float = 0.03
    max_mid: float = 0.97

    # --- refit ---
    min_refit_rows: int = 30

    # --- paths ---
    ledger_path: str = "data/ledger.jsonl"
    params_path: str = "data/params.json"
    calibrator_path: str = "data/calibrator.json"
    report_path: str = "data/report.json"
    site_path: str = "docs/index.html"


def _parse(raw: str, default: Any) -> Any:
    """Convert an environment string to the type of the field's default."""
    if isinstance(default, datetime):
        parsed = datetime.fromisoformat(raw)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Defaults overridden by any ``ORACLE_*`` variables present in ``env``."""
    env = os.environ if env is None else env
    base = Config()
    overrides: dict[str, Any] = {}
    for f in fields(Config):
        key = ENV_PREFIX + f.name.upper()
        if key in env:
            try:
                overrides[f.name] = _parse(env[key], getattr(base, f.name))
            except ValueError as exc:
                raise ValueError(f"{key}={env[key]!r} is not a valid value: {exc}") from exc
    return replace(base, **overrides)
