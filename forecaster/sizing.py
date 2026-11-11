"""Paper-trading decisions: fees, edge, fractional Kelly and P&L.

No order is ever placed. A "bet" is a dict recorded in the ledger so that paper
P&L can be computed once the market resolves.

Fees (verified 2026-11-10; sources in CLAUDE.md). Both venues charge takers
``rate * contracts * P * (1 - P)``, where P is the price paid per contract:
  - Kalshi: rate 0.07 on most markets, rounded UP to the next cent per order.
  - Polymarket: rate set per market (0 to 0.07), rounded to 5 decimal places.
We always cross the spread, so we are always the taker.
"""

from __future__ import annotations

import math
from typing import Any

from forecaster.config import Config

DEFAULT_FEE_RATE: dict[str, float] = {
    "kalshi": 0.07,
    "polymarket": 0.05,  # the "Other / General" rate, used when a market's own rate is unknown
    "sim": 0.0,
}

Bet = dict[str, Any]


def fee(venue: str, contracts: int, price: float, fee_rate: float | None = None) -> float:
    """Taker fee in USD for one order of ``contracts`` at ``price`` per contract."""
    if contracts <= 0:
        return 0.0
    rate = DEFAULT_FEE_RATE.get(venue, 0.0) if fee_rate is None else fee_rate
    raw = rate * contracts * price * (1 - price)
    if venue == "kalshi":
        # Round to 1e-9 first so float noise (175.00000000000003) cannot add a cent.
        return math.ceil(round(raw * 100, 9)) / 100
    return round(raw, 5)


def decide(
    p: float,
    bid: float | None,
    ask: float | None,
    venue: str,
    config: Config,
    fee_rate: float | None = None,
) -> Bet:
    """Choose YES, NO or no bet for a forecast ``p`` against a quoted market.

    The edge test and Kelly use the per-contract cost (quote + fee for one
    contract). The recorded bet uses the real fee for the whole order.
    """
    if bid is None or ask is None:
        return _no_bet("no quote", edge=None)

    sides = []
    for side, quote, win_p in (("yes", ask, p), ("no", 1 - bid, 1 - p)):
        cost = quote + fee(venue, 1, quote, fee_rate)
        sides.append((win_p - cost, side, quote, win_p, cost))
    edge, side, quote, win_p, cost = max(sides, key=lambda s: s[0])

    if edge < config.min_edge:
        return _no_bet(f"edge {edge:.3f} below minimum {config.min_edge:.3f}", edge=edge)
    if cost >= 1:
        return _no_bet("cost per contract is not below the payout", edge=edge)

    kelly = (win_p - cost) / (1 - cost)
    stake_fraction = min(config.kelly_fraction * kelly, config.max_stake_fraction)
    stake = stake_fraction * config.bankroll
    contracts = int(stake / cost)
    if contracts < 1:
        return _no_bet("stake is smaller than one contract", edge=edge)

    order_fee = fee(venue, contracts, quote, fee_rate)
    return {
        "side": side,
        "contracts": contracts,
        "quote": quote,
        "fee": order_fee,
        "price": quote + order_fee / contracts,  # all-in cost per contract
        "win_prob": win_p,
        "edge": edge,
        "kelly": kelly,
        "stake_fraction": stake_fraction,
        "reason": "bet",
    }


def pnl(bet: Bet, outcome: int) -> float:
    """Paper profit in USD, net of fees (``price`` already includes them)."""
    contracts = bet.get("contracts", 0)
    if not bet.get("side") or contracts <= 0:
        return 0.0
    won = (bet["side"] == "yes") == (outcome == 1)
    return contracts * (1 - bet["price"]) if won else -contracts * bet["price"]


def _no_bet(reason: str, edge: float | None) -> Bet:
    return {"side": None, "contracts": 0, "edge": edge, "reason": reason}
