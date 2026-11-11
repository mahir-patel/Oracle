import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from forecaster import markets
from forecaster.config import Config
from forecaster.pipeline import select_questions

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def client_for(handler: Any) -> httpx.Client:
    return markets.make_client(httpx.MockTransport(handler))


def json_response(payload: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


# ---------- Kalshi mapping ----------


def kalshi_page1() -> list[dict[str, Any]]:
    return load("kalshi_markets_page1.json")["markets"]


def test_kalshi_mapping() -> None:
    q = markets.kalshi_to_question(kalshi_page1()[0])
    assert q is not None
    assert q.venue == "kalshi"
    assert q.market_id == "KXVENUEPERFORMANCEYE-27JAN01-MSG"
    assert q.title.startswith("Will Kanye West / Ye perform at Madison Square Garden (MSG)")
    assert q.yes_bid == pytest.approx(0.17) and q.yes_ask == pytest.approx(0.21)
    assert q.mid == pytest.approx(0.19) and q.spread == pytest.approx(0.04)
    assert q.close_time == datetime(2027, 1, 8, 15, 0, tzinfo=UTC)
    assert q.close_time.tzinfo is not None
    assert q.liquidity == pytest.approx(2211.90)  # open interest x $1 notional
    assert q.url.startswith("https://kalshi.com/markets/KXVENUEPERFORMANCEYE-")
    assert q.category == "other"  # the markets endpoint carries no category
    assert q.fee_rate is None


def test_kalshi_rules_join_primary_and_secondary() -> None:
    raw = kalshi_page1()[0]
    q = markets.kalshi_to_question(raw)
    assert q is not None
    assert q.rules.startswith(raw["rules_primary"])
    if raw["rules_secondary"]:
        assert q.rules == raw["rules_primary"] + "\n" + raw["rules_secondary"]
    only_primary = markets.kalshi_to_question({**raw, "rules_secondary": ""})
    assert only_primary is not None and only_primary.rules == raw["rules_primary"]


def test_kalshi_empty_book_sides_become_none() -> None:
    no_bid = markets.kalshi_to_question(kalshi_page1()[1])  # yes_bid_dollars "0.0000"
    assert no_bid is not None
    assert no_bid.yes_bid is None and no_bid.yes_ask == pytest.approx(0.08)
    assert no_bid.mid is None and no_bid.spread is None

    no_ask = markets.kalshi_to_question(load("kalshi_markets_page2.json")["markets"][0])
    assert no_ask is not None  # yes_ask_dollars "1.0000"
    assert no_ask.yes_bid == pytest.approx(0.06) and no_ask.yes_ask is None


def test_kalshi_subtitle_is_added_when_the_title_lacks_it() -> None:
    raw = {**kalshi_page1()[0], "title": "Who will the next Pope be?", "yes_sub_title": "A. Name"}
    q = markets.kalshi_to_question(raw)
    assert q is not None and q.title == "Who will the next Pope be? [A. Name]"
    same = markets.kalshi_to_question(kalshi_page1()[0])
    assert same is not None and "[" not in same.title  # sub-title already in the title


def test_kalshi_legacy_cent_fields_still_work() -> None:
    raw = {
        k: v
        for k, v in kalshi_page1()[0].items()
        if k not in ("yes_bid_dollars", "yes_ask_dollars", "open_interest_fp")
    }
    q = markets.kalshi_to_question({**raw, "yes_bid": 17, "yes_ask": 21, "liquidity": 123456})
    assert q is not None
    assert (q.yes_bid, q.yes_ask) == (pytest.approx(0.17), pytest.approx(0.21))
    assert q.liquidity == pytest.approx(1234.56)


def test_kalshi_skips_unusable_markets() -> None:
    raw = kalshi_page1()[0]
    assert markets.kalshi_to_question({**raw, "market_type": "scalar"}) is None
    assert markets.kalshi_to_question({**raw, "close_time": ""}) is None
    assert markets.kalshi_to_question({**raw, "close_time": "not a date"}) is None
    assert markets.kalshi_to_question({k: v for k, v in raw.items() if k != "ticker"}) is None


# ---------- Kalshi fetch ----------


def test_kalshi_fetch_follows_the_cursor() -> None:
    requests: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url)
        assert request.url.path == "/trade-api/v2/markets"
        if request.url.params.get("cursor") == "PAGE2CURSOR":
            return json_response(load("kalshi_markets_page2.json"))
        assert "cursor" not in request.url.params
        return json_response(load("kalshi_markets_page1.json"))

    with client_for(handler) as client:
        questions = markets.fetch_kalshi(
            client,
            scan=400,
            min_close=datetime(2026, 11, 12, tzinfo=UTC),
            max_close=datetime(2027, 1, 9, tzinfo=UTC),
        )
    assert len(requests) == 2
    assert [q.market_id for q in questions] == [
        "KXVENUEPERFORMANCEYE-27JAN01-MSG",
        "KXUSDMXNMAX-01JAN27-T24",
        "KXHOLIDAYRETAIL-27JAN08-T400000",
        "KXSIERRAOPEN-27-03JAN27",
    ]
    params = requests[0].params
    assert params["status"] == "open" and params["mve_filter"] == "exclude"
    assert params["limit"] == "200"
    assert params["min_close_ts"] == str(int(datetime(2026, 11, 12, tzinfo=UTC).timestamp()))
    assert params["max_close_ts"] == str(int(datetime(2027, 1, 9, tzinfo=UTC).timestamp()))
    assert requests[1].params["limit"] == "200"


def test_kalshi_fetch_stops_at_scan() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.params["limit"] == "3"
        return json_response(load("kalshi_markets_page1.json"))

    with client_for(handler) as client:
        questions = markets.fetch_kalshi(client, scan=3)
    assert calls == 1 and len(questions) == 3


def test_fetched_kalshi_questions_pass_through_selection() -> None:
    with client_for(lambda r: json_response(load("kalshi_markets_page2.json") | {"markets": kalshi_page1()})) as c:
        questions = markets.fetch_kalshi(c, scan=10)
    now = datetime(2026, 11, 20, tzinfo=UTC)
    kept = select_questions(questions, now, Config())
    # the no-bid market has no mid; the other two are liquid with tight spreads
    assert [q.market_id for q in kept] == [
        "KXVENUEPERFORMANCEYE-27JAN01-MSG",
        "KXHOLIDAYRETAIL-27JAN08-T400000",
    ]


# ---------- Kalshi resolution ----------


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [("kalshi_market_yes.json", 1), ("kalshi_market_no.json", 0), ("kalshi_market_open.json", None)],
)
def test_kalshi_resolution(fixture: str, expected: int | None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/trade-api/v2/markets/SOME-TICKER"
        return json_response(load(fixture))

    with client_for(handler) as client:
        assert markets.resolve("kalshi", "SOME-TICKER", client) == expected


@pytest.mark.parametrize("result", ["", "void", "scalar", None])
def test_kalshi_unusual_results_are_unresolved(result: str | None) -> None:
    payload = copy.deepcopy(load("kalshi_market_yes.json"))
    payload["market"]["result"] = result
    with client_for(lambda r: json_response(payload)) as client:
        assert markets.resolve("kalshi", "T", client) is None


# ---------- Polymarket mapping ----------


def poly_list() -> list[dict[str, Any]]:
    return load("polymarket_markets.json")


def test_polymarket_mapping() -> None:
    q = markets.polymarket_to_question(poly_list()[0])
    assert q is not None
    assert q.venue == "polymarket"
    assert q.market_id == "3215008" and isinstance(q.market_id, str)
    assert q.title.startswith("Will the Fed increase interest rates by 25 bps")
    assert q.rules == poly_list()[0]["description"] and len(q.rules) > 50
    assert q.close_time == datetime(2026, 12, 9, 23, 59, tzinfo=UTC)
    assert (q.yes_bid, q.yes_ask) == (0.85, 0.86)
    assert q.liquidity == pytest.approx(poly_list()[0]["liquidityNum"])
    assert q.url == "https://polymarket.com/market/" + poly_list()[0]["slug"]
    assert q.category == "economics"  # from feeType "economics_fees"
    assert q.fee_rate == 0.05


def test_polymarket_fee_rate_per_market() -> None:
    tech = markets.polymarket_to_question(poly_list()[1])
    free = markets.polymarket_to_question(poly_list()[2])  # feesEnabled false
    assert tech is not None and tech.fee_rate == 0.04 and tech.category == "tech"
    assert free is not None and free.fee_rate == 0.0 and free.category == "other"
    unknown = markets.polymarket_to_question(
        {k: v for k, v in poly_list()[0].items() if k not in ("feesEnabled", "feeSchedule")}
    )
    assert unknown is not None and unknown.fee_rate is None  # falls back to the venue default


def test_polymarket_keeps_only_yes_no_markets() -> None:
    assert poly_list()[3]["outcomes"] == '["Heat", "Jazz"]'
    assert markets.polymarket_to_question(poly_list()[3]) is None
    raw = poly_list()[0]
    assert markets.polymarket_to_question({**raw, "outcomes": '["No", "Yes"]'}) is None
    assert markets.polymarket_to_question({**raw, "outcomes": "not json"}) is None
    assert markets.polymarket_to_question({**raw, "endDate": None}) is None


def test_polymarket_missing_quotes() -> None:
    raw = {k: v for k, v in poly_list()[0].items() if k not in ("bestBid", "bestAsk")}
    q = markets.polymarket_to_question(raw)
    assert q is not None and q.yes_bid is None and q.yes_ask is None
    empty = markets.polymarket_to_question({**poly_list()[0], "bestBid": 0, "bestAsk": 1})
    assert empty is not None and empty.mid is None


# ---------- Polymarket fetch ----------


def test_polymarket_fetch_pages_by_offset_and_filters() -> None:
    seen: list[httpx.URL] = []
    full_page = [poly_list()[0]] * 100

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        assert request.url.path == "/markets"
        return json_response(full_page if request.url.params["offset"] == "0" else poly_list())

    with client_for(handler) as client:
        questions = markets.fetch_polymarket(
            client,
            scan=400,
            min_close=datetime(2026, 11, 12, tzinfo=UTC),
            max_close=datetime(2027, 1, 9, tzinfo=UTC),
        )
    assert [u.params["offset"] for u in seen] == ["0", "100"]  # short second page ends it
    assert len(questions) == 100 + 3  # the Heat/Jazz market is dropped
    p = seen[0].params
    assert (p["active"], p["closed"], p["limit"]) == ("true", "false", "100")
    assert (p["order"], p["ascending"]) == ("volume24hr", "false")
    assert p["end_date_min"] == "2026-11-12T00:00:00Z"
    assert p["end_date_max"] == "2027-01-09T00:00:00Z"


def test_polymarket_fetch_stops_at_scan() -> None:
    limits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        limits.append(request.url.params["limit"])
        return json_response([poly_list()[0]] * int(request.url.params["limit"]))

    with client_for(handler) as client:
        questions = markets.fetch_polymarket(client, scan=150)
    assert limits == ["100", "50"] and len(questions) == 150


# ---------- Polymarket resolution ----------


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("polymarket_market_yes.json", 1),
        ("polymarket_market_no.json", 0),
        ("polymarket_market_open.json", None),
    ],
)
def test_polymarket_resolution(fixture: str, expected: int | None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/markets/12345"
        return json_response(load(fixture))

    with client_for(handler) as client:
        assert markets.resolve("polymarket", "12345", client) == expected


@pytest.mark.parametrize(
    "prices",
    [
        '["0.5", "0.5"]',  # resolved 50/50
        '["0.62", "0.38"]',  # closed while disputed, not yet final
        '["0.99", "0.01"]',  # exactly 0.99 is not above the threshold
        "[]",
        "not json",
        '["abc", "1"]',
        None,
    ],
)
def test_polymarket_disputed_or_unclear_is_unresolved(prices: str | None) -> None:
    payload = {**load("polymarket_market_yes.json"), "outcomePrices": prices}
    assert payload["closed"] is True
    with client_for(lambda r: json_response(payload)) as client:
        assert markets.resolve("polymarket", "1", client) is None


def test_polymarket_open_market_is_unresolved_even_at_extreme_prices() -> None:
    payload = {**load("polymarket_market_open.json"), "outcomePrices": '["0.999", "0.001"]'}
    assert payload["closed"] is False
    with client_for(lambda r: json_response(payload)) as client:
        assert markets.resolve("polymarket", "1", client) is None


# ---------- dispatch and errors ----------


def test_fetch_open_dispatches_by_venue() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "gamma-api.polymarket.com":
            return json_response(poly_list())
        assert request.url.host == "api.elections.kalshi.com"
        return json_response(load("kalshi_markets_page2.json"))

    with client_for(handler) as client:
        assert {q.venue for q in markets.fetch_open("kalshi", 10, client)} == {"kalshi"}
        assert {q.venue for q in markets.fetch_open("polymarket", 10, client)} == {"polymarket"}
        with pytest.raises(ValueError):
            markets.fetch_open("nope", 10, client)
        with pytest.raises(ValueError):
            markets.resolve("nope", "x", client)


def test_http_errors_raise_so_the_caller_can_skip_and_retry() -> None:
    kalshi_404 = {"error": {"code": "not_found", "message": "not found"}}
    with client_for(lambda r: json_response(kalshi_404, status=404)) as client:
        with pytest.raises(httpx.HTTPError):
            markets.resolve("kalshi", "MISSING", client)
        with pytest.raises(httpx.HTTPError):
            markets.fetch_kalshi(client, 10)

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down", request=request)

    with client_for(boom) as client:
        with pytest.raises(httpx.HTTPError):
            markets.resolve("polymarket", "1", client)
        with pytest.raises(httpx.HTTPError):
            markets.fetch_polymarket(client, 10)


def test_all_datetimes_are_utc_aware() -> None:
    raws = [markets.kalshi_to_question(m) for m in kalshi_page1()]
    raws += [markets.polymarket_to_question(m) for m in poly_list()]
    for q in raws:
        if q is not None:
            assert q.close_time.utcoffset() is not None
            assert q.close_time.utcoffset().total_seconds() == 0  # type: ignore[union-attr]
