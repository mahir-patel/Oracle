# Fixtures

Real responses from the venues' public read-only APIs, recorded 2026-11-10/11 (UTC).

- `kalshi_markets_page*.json`: `GET /trade-api/v2/markets?status=open&mve_filter=exclude&...`,
  cut down to a few markets and split into two pages. The `cursor` values are made up.
- `kalshi_market_{yes,no,open}.json`: `GET /trade-api/v2/markets/{ticker}`.
- `polymarket_markets.json`: `GET /markets?active=true&closed=false&...` on the Gamma API, a few
  markets. The last entry (outcomes Heat/Jazz) comes from a `closed=true` listing and is there
  to test that non-Yes/No markets are skipped.
- `polymarket_market_{yes,no,open}.json`: `GET /markets/{id}`.

Trimmed, values untouched: bulky fields the client never reads were removed (Kalshi
`custom_strike`, `price_ranges`, `mve_selected_legs`; Polymarket images, `events`, reward and
token-id fields). Disputed / malformed cases are built inside the tests by editing a copy of a
real fixture, so every file here is genuine data.
