"""Read-only clients for Kalshi and Polymarket: list open markets as Questions and
look up how a market resolved. Public endpoints, no API keys, no order placement.

Field mappings were checked against live responses on 2026-11-10; see CLAUDE.md
for the differences from the original spec.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import httpx

from forecaster.models import Question, ensure_utc

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
POLYMARKET_BASE = "https://gamma-api.polymarket.com"
KALSHI_PAGE = 200
POLYMARKET_PAGE = 100  # the Gamma API returns at most 100 markets per request
YES_NO = ["Yes", "No"]
GET_ATTEMPTS = 3
GET_BACKOFF_SECONDS = 1.0
_sleep = time.sleep  # replaced in tests

Raw = dict[str, Any]


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(
        timeout=20.0,
        transport=transport,
        headers={"User-Agent": "oracle-forecaster", "Accept": "application/json"},
    )


def _get(client: httpx.Client, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
    """GET with retries. Both venues return an occasional 500 on a request that
    works a second later; one of those must not end a daily run.

    429, 5xx and network errors are tried ``GET_ATTEMPTS`` times (1 s, then 2 s
    apart). Anything else, and the last failure, raises ``httpx.HTTPError``.
    """
    for attempt in range(GET_ATTEMPTS):
        last = attempt == GET_ATTEMPTS - 1
        try:
            response = client.get(url, params=params)
        except httpx.TransportError:
            if last:
                raise
        else:
            if last or not (response.status_code == 429 or response.status_code >= 500):
                response.raise_for_status()
                return response
        _sleep(GET_BACKOFF_SECONDS * 2**attempt)
    raise AssertionError("unreachable")


# ---------- shared parsing ----------


def _parse_time(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quotes(bid: float | None, ask: float | None) -> tuple[float | None, float | None]:
    """A bid of 0 or an ask of 1 is how both venues show an empty side of the book."""
    if bid is not None and not 0 < bid < 1:
        bid = None
    if ask is not None and not 0 < ask < 1:
        ask = None
    return bid, ask


# ---------- Kalshi ----------


def kalshi_to_question(m: Raw) -> Question | None:
    """Map one Kalshi market. Returns None for anything that is not a usable binary market."""
    close_time = _parse_time(m.get("close_time"))
    if not m.get("ticker") or close_time is None or m.get("market_type", "binary") != "binary":
        return None

    # Prices are dollar strings ("0.1700"). Older responses used integer cents.
    bid = _to_float(m.get("yes_bid_dollars"))
    ask = _to_float(m.get("yes_ask_dollars"))
    if bid is None and m.get("yes_bid") is not None:
        bid = float(m["yes_bid"]) / 100
    if ask is None and m.get("yes_ask") is not None:
        ask = float(m["yes_ask"]) / 100
    bid, ask = _quotes(bid, ask)

    # The API no longer returns a liquidity figure. Open interest (contracts
    # outstanding, each worth up to $1) is the stand-in for "is this market real".
    liquidity = _to_float(m.get("liquidity_dollars"))
    if liquidity is None and m.get("liquidity") is not None:
        liquidity = float(m["liquidity"]) / 100
    if liquidity is None:
        notional = _to_float(m.get("notional_value_dollars")) or 1.0
        liquidity = (_to_float(m.get("open_interest_fp")) or 0.0) * notional

    # In a multi-outcome event every market can share one title ("Who will the
    # next Pope be?"); the sub-title says which outcome this market is about.
    title = str(m.get("title", "")).strip()
    sub = str(m.get("yes_sub_title") or "").strip()
    if sub and sub.lower() not in title.lower():
        title = f"{title} [{sub}]"

    rules = "\n".join(part for part in (m.get("rules_primary"), m.get("rules_secondary")) if part)
    return Question(
        venue="kalshi",
        market_id=m["ticker"],
        title=title,
        rules=rules,
        close_time=close_time,
        yes_bid=bid,
        yes_ask=ask,
        category=str(m.get("category") or "other").lower(),
        liquidity=liquidity,
        url=f"https://kalshi.com/markets/{m.get('event_ticker', m['ticker'])}",
        fee_rate=None,  # standard taker rate; see sizing.DEFAULT_FEE_RATE
    )


def fetch_kalshi(
    client: httpx.Client,
    scan: int,
    min_close: datetime | None = None,
    max_close: datetime | None = None,
) -> list[Question]:
    """Read up to ``scan`` open markets, following the cursor.

    ``mve_filter=exclude`` drops multivariate "combo" markets, which otherwise
    fill the first pages and have no rules text.
    """
    questions: list[Question] = []
    seen = 0
    cursor = ""
    while seen < scan:
        params: dict[str, Any] = {
            "status": "open",
            "limit": min(KALSHI_PAGE, scan - seen),
            "mve_filter": "exclude",
        }
        if min_close is not None:
            params["min_close_ts"] = int(ensure_utc(min_close).timestamp())
        if max_close is not None:
            params["max_close_ts"] = int(ensure_utc(max_close).timestamp())
        if cursor:
            params["cursor"] = cursor
        response = _get(client, f"{KALSHI_BASE}/markets", params=params)
        body = response.json()
        markets = body.get("markets") or []
        seen += len(markets)
        for m in markets:
            q = kalshi_to_question(m)
            if q is not None:
                questions.append(q)
        cursor = body.get("cursor") or ""
        if not cursor or not markets:
            break
    return questions


def resolve_kalshi(client: httpx.Client, ticker: str) -> int | None:
    response = _get(client, f"{KALSHI_BASE}/markets/{ticker}")
    result = (response.json().get("market") or {}).get("result")
    if result == "yes":
        return 1
    if result == "no":
        return 0
    return None  # not settled yet, or settled to something other than yes/no


# ---------- Polymarket ----------


def _json_list(value: Any) -> list[Any] | None:
    """``outcomes`` and ``outcomePrices`` arrive as JSON encoded inside a string."""
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def polymarket_to_question(m: Raw) -> Question | None:
    """Map one Polymarket market. Returns None unless it is a plain Yes/No market."""
    close_time = _parse_time(m.get("endDate"))
    if m.get("id") is None or close_time is None or _json_list(m.get("outcomes")) != YES_NO:
        return None

    bid, ask = _quotes(_to_float(m.get("bestBid")), _to_float(m.get("bestAsk")))

    # Fees are per market. No fee flag at all means we do not know: use the default.
    fee_rate: float | None
    if m.get("feesEnabled") is False:
        fee_rate = 0.0
    elif m.get("feesEnabled") is True:
        fee_rate = _to_float((m.get("feeSchedule") or {}).get("rate"))
    else:
        fee_rate = None

    # There is no category field any more; the fee type ("politics_fees") carries it.
    category = m.get("category")
    if not category and isinstance(m.get("feeType"), str):
        category = m["feeType"].removesuffix("_fees")

    liquidity = _to_float(m.get("liquidityNum"))
    if liquidity is None:
        liquidity = _to_float(m.get("liquidity")) or 0.0
    return Question(
        venue="polymarket",
        market_id=str(m["id"]),
        title=str(m.get("question", "")).strip(),
        rules=str(m.get("description") or ""),
        close_time=close_time,
        yes_bid=bid,
        yes_ask=ask,
        category=str(category or "other").lower(),
        liquidity=liquidity,
        url=f"https://polymarket.com/market/{m.get('slug', '')}",
        fee_rate=fee_rate,
    )


def fetch_polymarket(
    client: httpx.Client,
    scan: int,
    min_close: datetime | None = None,
    max_close: datetime | None = None,
) -> list[Question]:
    """Read up to ``scan`` open markets, most traded in the last 24 hours first."""
    questions: list[Question] = []
    offset = 0
    while offset < scan:
        limit = min(POLYMARKET_PAGE, scan - offset)
        params: dict[str, Any] = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
            "order": "volume24hr",
            "ascending": "false",
        }
        if min_close is not None:
            params["end_date_min"] = ensure_utc(min_close).strftime("%Y-%m-%dT%H:%M:%SZ")
        if max_close is not None:
            params["end_date_max"] = ensure_utc(max_close).strftime("%Y-%m-%dT%H:%M:%SZ")
        response = _get(client, f"{POLYMARKET_BASE}/markets", params=params)
        markets = response.json()
        for m in markets:
            q = polymarket_to_question(m)
            if q is not None:
                questions.append(q)
        offset += len(markets)
        if len(markets) < limit:
            break
    return questions


def resolve_polymarket(client: httpx.Client, market_id: str) -> int | None:
    response = _get(client, f"{POLYMARKET_BASE}/markets/{market_id}")
    m = response.json()
    if not m.get("closed"):
        return None
    raw_prices = _json_list(m.get("outcomePrices")) or []
    prices = [p for p in (_to_float(x) for x in raw_prices) if p is not None]
    if len(prices) != len(raw_prices) or not prices:
        return None
    if prices[0] > 0.99:
        return 1
    if any(p > 0.99 for p in prices):
        return 0
    return None  # closed but split or disputed: not a clean yes/no


# ---------- venue dispatch ----------


def fetch_open(
    venue: str,
    scan: int,
    client: httpx.Client,
    min_close: datetime | None = None,
    max_close: datetime | None = None,
) -> list[Question]:
    if venue == "kalshi":
        return fetch_kalshi(client, scan, min_close, max_close)
    if venue == "polymarket":
        return fetch_polymarket(client, scan, min_close, max_close)
    raise ValueError(f"unknown venue {venue!r}")


def resolve(venue: str, market_id: str, client: httpx.Client | None = None) -> int | None:
    """1 for YES, 0 for NO, None while unresolved. Network errors raise httpx.HTTPError."""
    owned = client is None
    client = client or make_client()
    try:
        if venue == "kalshi":
            return resolve_kalshi(client, market_id)
        if venue == "polymarket":
            return resolve_polymarket(client, market_id)
        raise ValueError(f"unknown venue {venue!r}")
    finally:
        if owned:
            client.close()
