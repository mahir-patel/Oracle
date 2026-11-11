# Oracle — living memory

A fresh session must be able to continue from this file plus `SPEC.md` alone.
Updated in the last commit of every stage.

## Current stage

**S2 complete** (tagged `s2`). Next: **S3** — `markets` (Kalshi + Polymarket clients,
resolution lookup) + fixture tests, then ask the user to run one live fetch.
Never start a stage until the user says "next".

## How to work here

- Environment: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"`
  (the machine's default `python3` is 3.13/3.14; 3.12 comes from uv).
- Tests: `.venv/bin/pytest -q`. No network in tests.
- Commits: Conventional Commits, one per logical unit, only when tests pass. Tag each stage
  (`s0`…`s6`), then `git push && git push --tags` (remote `origin` exists).
- Allowed dependencies: `anthropic`, `httpx`, `numpy`, `pytest`. Ask before adding anything.

## Decisions made (and why)

- **`pytest` is a `dev` extra, not a runtime dependency.** The daily workflow only needs the
  runtime packages; CI test jobs install `.[dev]`.
- **`[tool.setuptools] packages = ["forecaster"]` is explicit.** With `data/`, `docs/` and
  `tests/` at the top level, setuptools' automatic discovery refuses to build.
- **A one-line smoke test exists from S0.** `pytest` exits with code 5 on zero tests, which
  would read as a failure in CI.
- **The `oracle` console script is added in S5**, together with `cli.py`, so the entry point
  never points at a module that does not exist.
- **`spec.md` was renamed to `SPEC.md`.** It was committed in lowercase, which only matched the
  spec's name on a case-insensitive filesystem.

- **Config is a frozen dataclass built by `load_config(env)`** (S1). Functions receive a
  `Config` rather than reading globals, so tests pass their own without touching the
  environment. Env names are `ORACLE_<FIELD>` in upper case. Beyond the spec's table it also
  holds the selection thresholds (6.9), the calibrator gate (6.5), the refit minimum (6.12)
  and the report/site paths, because the spec wants every setting in one place.
- **`Forecast` has four fields beyond the spec's list** (S1): `disagreement`, `latency_s`,
  `n_searches`, `parse_failures`. Rule 5 requires them in the ledger and the `Forecast` is
  what gets logged.
- **`Ledger.verify()` is stricter than the spec** (S1): besides the hash chain it requires
  every line to be byte-for-byte canonical JSON, and rejects blank lines and a missing final
  newline. Without that, an edit that parses to the same values would pass.
- **`Ledger.append()` rejects NaN/Infinity** (S1), which are not valid JSON.
- **`joined()` uses the latest resolution record** for a market if there is more than one
  (S1), so a venue correcting a result can be appended without editing history.
- **The chain cannot detect truncation of the newest records** (S1). Git history of
  `data/ledger.jsonl` covers that; say so in the README's verification section (S6).

- **Polymarket is not fee-free; `Question` has a nullable `fee_rate`** (S2, user chose this
  over a flat rate). See "Fee schedules" below. S3 must fill it from the API.
- **Isotonic fit pools twice** (S2). Pass 1 is classic PAV on block means with ties pooled.
  Pass 2 pools again on the Laplace-smoothed values, because smoothing pulls a small block
  toward 0.5 more than a large one and can break monotonicity (1 point at y=0 smooths to
  0.333, above a 100-point block at 0.30).
- **`fit_alpha_and_weight` fits `sigmoid((1-w)*alpha*logit(raw) + w*logit(market))`** (S2),
  i.e. exactly what the pipeline computes when the calibrator is the identity.
- **`Params` (alpha, market_weight, n_fit) lives in `pipeline.py`** (S2) and falls back to the
  config priors when `data/params.json` does not exist.
- **Simulations write their ledger to a temp directory** unless given a path (S2), so they can
  never touch `data/ledger.jsonl`.
- **Simulated chronology is strict** (S2): the second half of the markets starts 3 days later,
  so every phase-1 market has resolved before the first phase-2 forecast is made.
- **`Ledger.last_hash()` reads the file tail** (S2); appends were quadratic before.
- **Not built yet, on purpose:** the pre-question budget guard (amendment 5) belongs to the
  `forecast` command and is due in S5 with `cli.py`; search-fee cost tracking is due in S4.

## Spec amendments agreed after S0 (SPEC.md is updated to match)

1. **Search fees count as cost.** `PRICE_PER_SEARCH` setting; `cost_usd` and the spending cap
   include searches. The 0.01 default is a placeholder: **confirm the price from the Anthropic
   docs in S4.**
2. **A bet's `price` = quote + fee**, so paper P&L is net of fees.
3. **Kalshi fee is rounded up per order**: ceil(0.07 x contracts x P x (1-P) x 100) / 100.
   The edge check uses the per-contract cost (fee for 1 contract). **Verify against Kalshi's
   current fee schedule in S2 and cite the source here.**
4. **Simulation acceptance is over 5 fixed seeds (7, 8, 9, 10, 11), n = 1200 each (600/600).**
   Pass if final < model-only < market Brier on average, and the pooled phase-2 BSS CI lower
   bound > 0. Never hunt for a passing seed. **Report per-seed results here in S2, including
   any seed that fails on its own.**
5. **Budget guard.** Before each question, estimate its cost as the run's average so far
   ($0.50 for the first, `FIRST_QUESTION_COST_ESTIMATE`) and skip it if the remaining budget
   is smaller. Overshoot of up to one question is accepted. Log actual spend vs cap at the end.

## Fee schedules (verified 2026-11-10)

Both venues charge takers `rate x contracts x P x (1-P)` per order.

- **Kalshi.** Source: https://kalshi.com/fee-schedule (read 2026-11-10). "Most markets": fee
  multiplier 1, taker fee range $0.07-$1.75 per 100 contracts, no maker fee. That matches
  0.07 x 100 x P x (1-P) rounded up to the cent ($0.0693 -> $0.07 at 1c; $1.75 at 50c). The
  page lists "No upcoming fee changes". About 160 series are non-standard: most only add
  maker fees (irrelevant to us as takers), a few have no fees at all (we overstate those,
  which is the conservative direction).
  **Not confirmed from a primary source:** the exact rounding wording. The fee-schedule PDF
  (https://kalshi.com/docs/kalshi-fee-schedule.pdf) returned HTTP 429. One third-party guide
  claims rounding to $0.0001 rather than to the cent; if so we overstate by under 1c per order.
- **Polymarket.** Source: https://docs.polymarket.com/trading/fees and
  https://docs.polymarket.com/market-data/market-details (read 2026-11-10). Taker fee =
  C x feeRate x p x (1-p), rounded to 5 decimals; makers never pay. Rates: crypto 0.07;
  sports, economics, culture, weather, other 0.05; finance, politics, mentions, tech 0.04;
  geopolitics / world events 0. The Gamma market object exposes `feesEnabled` and
  `feeSchedule.rate` (the docs' examples disagree on whether these sit at the top level or
  under `trading`: **check the raw JSON in S3**). `feeSchedule.exponent` exists (example
  value 1); the docs do not say how it enters the formula, we assume 1.
  The spec's "Polymarket fee = 0" was out of date.

## Findings (numbers)

### S2 simulation acceptance (n = 1200 per seed, 600/600; phase-2 numbers, out of sample)

| seed | alpha | w | final | model-only | market | BSS | 95% CI |
|---|---|---|---|---|---|---|---|
| 7 | 1.60 | 0.30 | 0.1827 | 0.1828 | 0.1930 | +0.053 | +0.021..+0.087 |
| 8 | 1.40 | 0.35 | 0.1776 | 0.1792 | 0.1855 | +0.042 | +0.007..+0.077 |
| 9 | 1.05 | 0.60 | 0.1993 | 0.1957 | 0.2077 | +0.040 | +0.025..+0.055 |
| 10 | 1.45 | 0.00 | 0.1899 | 0.1899 | 0.1947 | +0.025 | -0.027..+0.075 |
| 11 | 1.35 | 0.25 | 0.1748 | 0.1779 | 0.1782 | +0.019 | -0.023..+0.059 |
| **mean** | | | **0.1849** | **0.1851** | **0.1918** | | |

Pooled phase 2 (n = 3000): **BSS +0.036, 95% CI +0.020..+0.052**, P(beat market) 1.000.
**Acceptance: PASS** (mean final < mean model-only < mean market; pooled CI lower bound > 0).

Seeds that fail on their own, reported as agreed:
- **Seed 9:** final (0.1993) is *worse* than model-only (0.1957). The fit chose w = 0.60 and
  alpha = 1.05 on phase 1, where the market happened to beat the model, and that did not
  carry over.
- **Seed 10:** the fit chose w = 0, so final equals model-only (no strict improvement), and
  its own BSS CI includes zero (-0.027).
- **Seed 11:** its own BSS CI includes zero (-0.023).

What this says:
- **Beating the market is the robust result**: all 5 seeds have BSS > 0 and the pooled CI is
  clear of zero. It comes almost entirely from undoing the hedge (alpha > 1).
- **The market blend adds very little here** (mean 0.1849 vs 0.1851) and the fitted weight is
  unstable at n = 600 (0.00 to 0.60 across seeds). Expect the same in production: do not
  read much into the fitted `market_weight` until n is in the thousands.
- Single-seed CIs at n = 600 are too wide to detect a 2-5% skill edge two times in five. That
  is the same power problem the live record will have; hence the n >= 200 credibility rule,
  which is if anything optimistic.
- Paper P&L in the simulation is large and positive (about +$2,500 to +$6,300 per 600
  questions on a $1,000 flat bankroll) but means little: the synthetic market is far noisier
  than a real one and there is no slippage.

### S2 isotonic-calibrator ablation (SPEC 6.5)

Setup: fit alpha and w on the first n_train questions; fit an ungated isotonic calibrator on
the extremized training forecasts; score the next 5,000 questions with and without it. Seeds
7-11. Delta = Brier with calibrator minus Brier without (positive = calibrator is WORSE).

| n_train | mean delta, final | mean delta, model-only | seeds where final got worse |
|---|---|---|---|
| 300 | +0.00118 | +0.00165 | 4 / 5 |
| 600 | +0.00034 | +0.00069 | 4 / 5 |
| 1000 | +0.00044 | +0.00082 | 5 / 5 |
| 3000 | +0.00021 | +0.00041 | 4 / 5 |

- **The prototype's finding reproduces:** stacked on the fitted alpha, isotonic calibration
  makes out-of-sample Brier worse at n = 300 and 600 (4 of 5 seeds each). The harm shrinks
  with n but the calibrator never helps, even at 3,000.
- **Caveat that matters:** in this world the agent's miscalibration is *exactly* a
  logit-linear hedge, which alpha fixes completely, so isotonic has nothing left to fix and
  can only add variance. The simulation therefore cannot tell us where the gate should be;
  1000 is a judgement call. Real LLM miscalibration may not be logit-linear. Keep the rule:
  enable only with a large history and keep it only if the live report shows it helps.
- Best single run was a -0.00012 improvement (seed 8, n = 3000): noise.

## Known issues

- Kalshi's exact fee rounding wording is unconfirmed from the PDF (see "Fee schedules").
- Kalshi series with non-standard or zero fees are charged the standard 0.07 rate.

## Operating rhythm (after the build; from SPEC section 10)

- **Daily (automatic):** resolve → forecast → report → verify → commit.
- **Weekly (user + Claude):** read the report. Look at the worst-scoring resolved forecasts
  and read their rationales: rules misread, bad base rate, stale news, or bad luck? Write the
  findings in this file.
- **Refit:** only at 30, 100, 300 and 1,000 clean resolved forecasts. Never tune on fewer.
  Every prompt or parameter change is its own commit, judged by later out-of-sample BSS.
- **Budget:** about 10 questions a day at roughly $0.10–0.50 each. Watch `total_cost_usd`.
- **Credibility threshold:** no public claim of beating the market until n ≥ 200 and the BSS
  CI lower bound is > 0.

## Next steps

1. S3: `forecaster/markets.py` with recorded fixtures and `httpx.MockTransport` tests. Check
   both venues' current API docs first and report field differences. Fill `Question.fee_rate`
   for Polymarket from `feesEnabled` / `feeSchedule.rate`. Then ask the user to run one live
   fetch. Retry the Kalshi fee-schedule PDF to confirm the rounding wording.
