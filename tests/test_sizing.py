import pytest

from forecaster.config import Config
from forecaster.sizing import decide, fee, pnl

CFG = Config()


# ---------- fees ----------


def test_kalshi_fee_one_contract() -> None:
    assert fee("kalshi", 1, 0.5) == 0.02  # ceil(1.75 cents)
    assert fee("kalshi", 1, 0.1) == 0.01  # ceil(0.63 cents)
    assert fee("kalshi", 1, 0.9) == 0.01  # symmetric in P


def test_kalshi_fee_is_rounded_per_order_not_per_contract() -> None:
    assert fee("kalshi", 100, 0.5) == 1.75  # not 100 * 0.02 = 2.00
    assert fee("kalshi", 100, 0.1) == 0.63  # not 100 * 0.01 = 1.00
    assert fee("kalshi", 3, 0.5) == 0.06  # ceil(5.25 cents)
    assert fee("kalshi", 7, 0.1) == 0.05  # ceil(4.41 cents)


def test_kalshi_fee_matches_published_range_for_100_contracts() -> None:
    # kalshi.com/fee-schedule lists $0.07 - $1.75 per 100 contracts
    assert fee("kalshi", 100, 0.01) == 0.07
    assert fee("kalshi", 100, 0.99) == 0.07
    assert fee("kalshi", 100, 0.50) == 1.75


def test_polymarket_fee_uses_the_market_rate() -> None:
    # figures from docs.polymarket.com/trading/fees, 100 shares at $0.50
    assert fee("polymarket", 100, 0.5, fee_rate=0.07) == pytest.approx(1.75)
    assert fee("polymarket", 100, 0.5, fee_rate=0.05) == pytest.approx(1.25)
    assert fee("polymarket", 100, 0.5, fee_rate=0.04) == pytest.approx(1.00)
    assert fee("polymarket", 100, 0.5, fee_rate=0.0) == 0.0  # geopolitics
    assert fee("polymarket", 100, 0.5) == pytest.approx(1.25)  # default when unknown
    assert fee("polymarket", 1, 0.37, fee_rate=0.04) == 0.00932  # 5 decimal places


def test_fee_edge_cases() -> None:
    assert fee("sim", 100, 0.5) == 0.0
    assert fee("kalshi", 0, 0.5) == 0.0
    assert fee("kalshi", 10, 0.5, fee_rate=0.035) == 0.09  # ceil(8.75 cents)


# ---------- decide ----------


def test_no_bet_without_a_quote() -> None:
    for bid, ask in ((None, 0.5), (0.5, None), (None, None)):
        bet = decide(0.9, bid, ask, "kalshi", CFG)
        assert bet["side"] is None and bet["contracts"] == 0
        assert bet["reason"] == "no quote"


def test_no_bet_below_the_minimum_edge() -> None:
    # YES costs 0.51 + 0.02 fee = 0.53; p = 0.56 gives edge 0.03 < 0.04
    bet = decide(0.56, 0.49, 0.51, "kalshi", CFG)
    assert bet["side"] is None
    assert bet["edge"] == pytest.approx(0.03)
    assert "below minimum" in bet["reason"]
    assert pnl(bet, 1) == 0.0 and pnl(bet, 0) == 0.0


def test_fee_can_remove_the_edge() -> None:
    assert decide(0.555, 0.49, 0.51, "sim", CFG)["side"] == "yes"  # 0.045 without fees
    assert decide(0.555, 0.49, 0.51, "kalshi", CFG)["side"] is None  # 0.025 with them


def test_picks_yes_when_forecast_is_above_the_market() -> None:
    bet = decide(0.70, 0.49, 0.51, "sim", CFG)
    assert bet["side"] == "yes"
    assert bet["quote"] == 0.51
    assert bet["win_prob"] == pytest.approx(0.70)
    assert bet["edge"] == pytest.approx(0.19)


def test_picks_no_when_forecast_is_below_the_market() -> None:
    bet = decide(0.30, 0.49, 0.51, "sim", CFG)
    assert bet["side"] == "no"
    assert bet["quote"] == pytest.approx(0.51)  # 1 - bid
    assert bet["win_prob"] == pytest.approx(0.70)
    assert bet["edge"] == pytest.approx(0.19)


def test_kelly_math_below_the_cap() -> None:
    # sim venue (no fee): cost 0.50, p 0.56 -> edge 0.06, f* = 0.06 / 0.50 = 0.12
    # stake = 0.25 * 0.12 * 1000 = 30 -> 60 contracts
    bet = decide(0.56, 0.48, 0.50, "sim", CFG)
    assert bet["kelly"] == pytest.approx(0.12)
    assert bet["stake_fraction"] == pytest.approx(0.03)
    assert bet["contracts"] == 60
    assert bet["fee"] == 0.0 and bet["price"] == pytest.approx(0.50)


def test_stake_is_capped() -> None:
    # f* = 0.4 / 0.5 = 0.8 -> quarter Kelly 0.2, capped at 0.05 -> $50 -> 100 contracts
    bet = decide(0.90, 0.48, 0.50, "sim", CFG)
    assert bet["stake_fraction"] == pytest.approx(0.05)
    assert bet["contracts"] == 100


def test_kalshi_bet_records_the_order_fee_in_the_price() -> None:
    # per-contract cost for sizing: 0.50 + 0.02 = 0.52; f* = 0.38/0.48, capped -> $50
    bet = decide(0.90, 0.48, 0.50, "kalshi", CFG)
    assert bet["contracts"] == int(50 / 0.52) == 96
    assert bet["fee"] == 1.68  # 0.07 * 96 * 0.25, already whole cents
    assert bet["price"] == pytest.approx(0.50 + 1.68 / 96)
    assert bet["edge"] == pytest.approx(0.90 - 0.52)


def test_market_fee_rate_is_used() -> None:
    free = decide(0.90, 0.48, 0.50, "polymarket", CFG, fee_rate=0.0)
    charged = decide(0.90, 0.48, 0.50, "polymarket", CFG, fee_rate=0.07)
    assert free["fee"] == 0.0 and free["price"] == pytest.approx(0.50)
    assert charged["fee"] > 0 and charged["edge"] < free["edge"]


def test_no_bet_when_stake_buys_less_than_one_contract() -> None:
    tiny = Config(bankroll=5.0)
    bet = decide(0.70, 0.49, 0.51, "sim", tiny)
    assert bet["side"] is None and "one contract" in bet["reason"]


# ---------- P&L ----------


def test_pnl_win_and_loss() -> None:
    yes = {"side": "yes", "contracts": 10, "price": 0.40}
    assert pnl(yes, 1) == pytest.approx(6.0)
    assert pnl(yes, 0) == pytest.approx(-4.0)
    no = {"side": "no", "contracts": 10, "price": 0.40}
    assert pnl(no, 0) == pytest.approx(6.0)
    assert pnl(no, 1) == pytest.approx(-4.0)


def test_pnl_is_net_of_fees() -> None:
    bet = decide(0.90, 0.48, 0.50, "kalshi", CFG)
    # 96 contracts bought at 0.50 plus a 1.68 fee
    assert pnl(bet, 1) == pytest.approx(96 * 0.50 - 1.68)
    assert pnl(bet, 0) == pytest.approx(-(96 * 0.50) - 1.68)
