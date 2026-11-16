# Oracle — living memory

A fresh session must be able to continue from this file plus `SPEC.md` alone.
Updated in the last commit of every stage.

## Current stage

**S5 built, not tagged (2026-11-15).** `cli.py`, `report.py`, `pick_daily`, the page and
`.github/workflows/daily.yml` are in; 230 tests pass; `oracle simulate` and `oracle report`
work locally. The S5 gate still needs three things only the user can do:
1. add the repository secret `ANTHROPIC_API_KEY` (GitHub: Settings > Secrets and variables >
   Actions). **The user pasted a key into the chat on 2026-11-15; it was never written to any
   file here, and they were told to revoke it and make a new one.**
2. enable GitHub Pages (Settings > Pages > Deploy from a branch > `main`, folder `/docs`);
3. run the `daily` workflow once by hand (Actions > daily > Run workflow).
Then: check the run and the page (https://mahir-patel.github.io/Oracle/), tag `s5`, push tags.

Model is Haiku 5.5 (user decision: portfolio piece, **total budget 50 USD**, about 6.10
spent on live checks before the ledger). The S4 adjustments (3 searches, 8,000 tokens, the
search-budget line) get their first live run in that workflow run.

S3 live-fetch gate: the user ran the smoke fetch on 2026-11-11. Kalshi 400 fetched / 72
selected; Polymarket 355 Yes/No of 400 scanned / 199 selected. Mids, liquidity, close dates,
fee rates and categories all looked right; no field differences needed fixing.

**Agreed for S5 (user said yes to all three, 2026-11-11; SPEC 6.9 `pick_daily`):** (1) quotas
by time to close, 50/30/20% of each day's picks for 1-14 / 14-30 / 30-60 days; (2) one market
per event, via a new `Question.event_id` (Kalshi `event_ticker`, Polymarket `events[0].id`),
also skipping events already in the ledger; (3) seeded random pick within each band, seed from
the UTC date; (4) one fetch per venue per band. Unfilled slots from ANY band are redistributed
to the other bands, shortest first, so the daily total is met whenever enough eligible markets
exist. Every run logs per-band eligible / quota / picked. No category excluded up front. Build notes: the Polymarket fixtures were trimmed
of `events`, so re-record or add that field; `forecast` must fetch once per band because
Kalshi lists latest-closing markets first.

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

- **`markets.py` field mappings follow live responses, not the spec's notes** (S3). Differences,
  all recorded in SPEC 6.1:
  - Kalshi prices are dollar strings (`yes_bid_dollars`); legacy cent fields still parse.
  - Kalshi has no `liquidity` field. **Open interest in USD stands in for it** (the 500
    threshold then means "at least 500 contracts outstanding"). This is a different quantity
    from Polymarket's `liquidityNum` (resting order-book depth), so the two venues' liquidity
    numbers are not comparable.
  - Kalshi has no market-level `category` (it is on the event). All Kalshi questions are
    "other" for now. Roadmap item 5 (category analysis) will need one `GET /events/{ticker}`
    per forecast question; not built.
  - Kalshi needs `mve_filter=exclude`, or combo markets fill the scan.
  - Kalshi `yes_sub_title` is appended to the title in brackets when the title does not
    contain it. Multi-outcome events can give every market the same title ("Who will the
    next Pope be?"); the price of safety is some redundant suffixes like "[Above 24]".
  - Polymarket pages cap at 100 (offset pagination); ordered by 24h volume, most traded first.
  - Polymarket has no `category`; it is derived from `feeType`. Fee-free markets
    (geopolitics) have no `feeType`, so they show as "other".
  - Polymarket `fee_rate`: `feeSchedule.rate` when `feesEnabled` is true, 0.0 when false,
    None (venue default 0.05) when the flag is absent. `feeSchedule.exponent` is ignored
    (always 1 in everything seen so far).
  - A bid of 0 or an ask of 1 means an empty side of the book and maps to None on both venues.
- **`fetch_open` takes an optional close-time window, applied server-side** (S3).
- **Fixtures are real recorded responses, trimmed** (S3); see `tests/fixtures/README.md`.
  Disputed/odd cases are made inside the tests by editing a copy of a real fixture.
- **HTTP and network errors raise `httpx.HTTPError`** from `fetch_*` and `resolve` (S3). The
  S5 `resolve` command must catch it and skip the market until the next run.
- **Market GETs retry 429, 5xx and network errors** (added 2026-11-12, `markets._get`): 3
  attempts, 1 s then 2 s apart, then `httpx.HTTPError` as before. Polymarket returned a 500
  on page 2 of a scan during the S4 live check; the identical request worked a minute later.

- **Web search tool is `web_search_20260318`** (S4, docs read 2026-11-11:
  https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool). It runs
  searches inside server-side code that filters results before the model reads them; no
  charge beyond tokens. Searches are counted from `usage.server_tool_use.web_search_requests`.
  $10 per 1,000 searches; a failed search is not billed.
- **Prices are 8 / 40 USD per M tokens** (S4), not the spec's 10 / 50: Fable 5.2 is cheaper
  than 5.1. Thinking tokens are billed as output. Training data cutoff "Jun 2026", so
  `MODEL_CUTOFF` 2026-07-01 stands.
- **The search blocks `polymarket.com` and `kalshi.com`** (S4, my addition, not in the
  original spec). Their pages show the live price, and a search for the question title would
  find them, which breaks rule 2. Other prediction sites are not blocked.
- **No model fallback on refusals** (S4). The API offers one; it would log another model's
  forecast under this model's name. A refusal is a parse failure.
- **The `Agent` sends no `effort` setting** (S4), so the API default (`high`) applies. This is
  the first lever to try if the live cost per question is too high.
- **`max_tokens` 4000 may be tight** (S4): it has to cover thinking, the search-filtering
  code and the reply. Truncation shows up as `stop_reason: max_tokens` and a parse failure;
  check the stop reasons in the live run.
- **API errors** (S4): 429, 5xx and network errors are retried (3 attempts, 2 s then 4 s,
  SDK retries off), then the sample is dropped and counted in `agent.api_failures`. Other
  4xx errors are raised so a bad key stops the run. The S5 `forecast` command must let that
  propagate after writing what it has.
- **The parser takes the last `<forecast>` block** (S4), since the reasoning may quote the
  format before the real answer.
- **`scripts/live_check.py` is temporary** (S4): delete it in S5 once `oracle forecast` exists.
- **The installed `anthropic` SDK (2.8) uses `httpx2`, not `httpx`.** `markets.py` uses
  `httpx` directly, which is on the allowed list; tests build SDK errors without importing
  `httpx2`.

- **Model is `claude-haiku-5-5`, 5 samples, 2 searches each** (2026-11-13, user decision on a
  50 USD total budget; replaces the S4 Fable prices above). Docs read 2026-11-13: training
  cutoff "Jun 2026", same as Fable, so `MODEL_CUTOFF` stands. Prices 0.10 / 0.50 USD per M;
  **a request whose prompt exceeds 100,000 tokens pays 0.50 / 2.50**, and the Fable run used
  104,450 input tokens in one request, so the cost tracker applies the higher tier per
  response (`LONG_PROMPT_TOKENS`). Cache read is 0.1x. Expected about $0.15 per question,
  most of it search fees. Never pool Fable and Haiku forecasts in one skill number.
- **`thinking={"type": "adaptive"}` is now sent** (`THINKING` setting): Haiku 5.5's thinking
  is not always on. Untested live; a 400 on the next run means set `ORACLE_THINKING=`.
- **`SEARCH_ALLOWED_CALLERS` setting, empty by default.** The docs do not say whether Haiku
  5.5 can run searches from filtering code. If the API returns a 400 asking for it, set
  `ORACLE_SEARCH_ALLOWED_CALLERS=direct`.
- **Per-run cap is now $1; S5 must add `MAX_SPEND_TOTAL` = 50**, checked against the sum of
  `cost_usd` in the ledger before each run, and pick about 5 questions a day.
- **Searches failing "on a tool limit"** was `max_uses` being exceeded by parallel searches
  from code (see the Haiku live check). The user message now states the budget.

- **`pick_daily` takes `config` and returns `(picks, per-band counts)`** (S5). Quotas use
  whole slots first, then largest remainders (10 -> 5/3/2, 5 -> 3/1/1). A market with no
  `event_id` counts as its own event. The shuffle seed is `"<UTC date>:<band index>"` after
  sorting by (venue, market_id), so the pick does not depend on fetch order.
- **Polymarket `events` comes only from the list endpoint** (S5); `GET /markets/{id}` omits
  it. Two list fixtures carry the real field; the single-market fixtures do not need it.
- **Total budget** (S5): the run's cap = min(`MAX_SPEND_PER_RUN`, `MAX_SPEND_TOTAL` -
  `SPENT_BEFORE_LEDGER` - ledger cost). Samples of a question that produced no forecast are
  not in the ledger, so the total can understate by that question's cost (rare, cents).
- **`forecast` appends each forecast as it is made** (S5) and exits 1 on a non-retried API
  error; the workflow still reports, verifies and commits (`if: always()`), but never commits
  a ledger that fails `verify`.
- **`refit` and the daily default** (S5): `refit` uses only rows of the current `MODEL`;
  `--max` and the workflow default are 5 a day (budget), overridable with the repository
  variable `ORACLE_DAILY_MAX`. `refit` is manual, never in the workflow.
- **The page's headline makes no claim below n = 200** (S5, `report.CREDIBLE_N`), and the
  page has no scripts or external assets; venue text is escaped and only http(s) links are
  made clickable.
- **`report.json` carries `generated_at`**, so the daily run always has something to commit.
- **Not built, could be asked for:** a test workflow on push (the spec lists only
  `daily.yml`); a filter for markets that are already decided (see the Haiku live check).

## Spec amendments agreed after S0 (SPEC.md is updated to match)

1. **Search fees count as cost.** `PRICE_PER_SEARCH` setting; `cost_usd` and the spending cap
   include searches. Confirmed in S4: 0.01 USD per search.
2. **A bet's `price` = quote + fee**, so paper P&L is net of fees.
3. **Kalshi fee is rounded up per order**: ceil(0.07 x contracts x P x (1-P) x 100) / 100.
   The edge check uses the per-contract cost (fee for 1 contract). **Verify against Kalshi's
   current fee schedule in S2 and cite the source here.**
4. **Simulation acceptance is over 5 fixed seeds (7, 8, 9, 10, 11), n = 1200 each (600/600).**
   Pass if final < model-only < market Brier on average, and the pooled phase-2 BSS CI lower
   bound > 0. Never hunt for a passing seed. **Report per-seed results here in S2, including
   any seed that fails on its own.**
5. **Budget guard.** Before each question, estimate its cost as the run's average so far
   ($0.20 for the first, `FIRST_QUESTION_COST_ESTIMATE`) and skip it if the remaining budget
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

### S4 live check, first attempt (2026-11-12): far over the cost target

One Polymarket question ("Will Bitcoin reach $76,000 November 9-15?"), 5 samples, then the
account ran out of credit on question 2 (HTTP 400, raised as designed).

- **Cost $4.645 for one question** against a target of $0.10-0.50 and a $5 run cap.
- Latency 88 s; 20 searches (every sample used all 4); parse failures 0 of 5.
- Samples 0.09 / 0.13 / 0.07 / 0.15 / 0.17, raw 0.121, final 0.062, market 0.055, no bet.
- Token counts were lost (the script saved only at the end; fixed). Bound: searches are
  $0.20, output is at most 5 x 4,000 tokens = $0.80 unless turns were paused, so **input is
  at least about $3.6, i.e. 90,000+ input tokens per sample**. Search results are the cost.
- Not yet known: tokens per pass, cached share, stop reasons. The next run prints them.
- **Applied 2026-11-12: automatic prompt caching** (`cache_control={"type": "ephemeral"}` on
  every request). A Fable 5.2 cache read costs 0.025x input ($0.20 vs $8 per M), a write
  1.25x. It changes the bill only, not what the model sees. Certain to help on resumed
  (`pause_turn`) turns; the docs do NOT say whether it applies between the search iterations
  inside one request, so the saving is unmeasured until the next live run (`cached_tokens`).
  Cost tracking now prices fresh / written / read input separately.
- **Proposed, not built: Batch API for the daily run** (50% off everything, web search
  supported, most batches done in under 1 hour, 24 h limit, no effect on output). Needs the
  user's yes: `forecast` becomes submit-then-collect, and the market price must be read when
  the result is logged.
- The user's rule (2026-11-12): cheapest way that does not compromise quality. So Fable 5.2,
  5 samples and 4 searches stay unless a measurement shows a cut costs nothing.
- Docs facts (2026-11-12): the search tool has no parameter limiting result size; effort is
  `output_config={"effort": ...}`, default `high`; Haiku 5.5 is $0.10 / $0.50 per M.
- Other levers, none applied (each needs the user's yes): `N_SAMPLES` 5 -> 3,
  `MAX_WEB_SEARCHES` 4 -> 2, an `effort` setting, `max_content_tokens`-style limits if the
  tool has one (check the docs).

### S4 live check, one-sample diagnostic (2026-11-13, caching on)

Polymarket 814613 (Taiwan local elections, DPP most seats), persona `outside_view`, 1 sample.

- **Cost $1.099 for ONE sample**: input 104,450 tokens = $0.836 (76%), output 5,582 tokens =
  $0.223 (20%), 4 searches = $0.04. So 5 samples are about $5.50 per question, in line with
  the first run ($4.65).
- **Caching saved nothing: 0 cached tokens, 0 cache writes.** It does not apply between the
  search steps inside one request. It stays on (free, helps resumed turns) but is not a fix.
- Latency 163 s for the one sample. 1 request, stop reason `end_turn`, parsed 1 of 1.
- Output 5,582 > `max_tokens` 4,000: the limit is evidently not on the whole server-side
  turn. No truncation seen in 6 samples so far.
- **Quality problem:** the reply says only its first two searches returned results and
  "every later one failed on a tool limit", while 4 searches were billed. Cause unknown; the
  transcript now records every reply block (tool calls, inputs, error codes) to find it.
  Suspects: `max_uses` 4 being hit by searches issued from the filtering code, or a
  code-execution limit.
- Forecast 0.08 vs market 0.357 (final 0.132, paper bet NO, 75 contracts). One sample, no
  conclusion to draw.
- **Where this leaves cost:** about $5.50 a question as specified; $2.75 with the Batch API.
  The $0.10-0.50 target is not reachable with 5 searching Fable samples.

### S4 live check on Haiku 5.5 (2026-11-15): the gate run

3 questions x 5 samples, 2 searches per sample, `max_tokens` 4000, adaptive thinking.

| question | samples | raw / final / market | cost | latency | searches |
|---|---|---|---|---|---|
| Kalshi: Dems win >= 46 Trump-carried House seats | 0.95 0.95 0.04 0.97 (1 lost) | 0.824 / 0.940 / 0.970 | $0.127 | 42 s | 10 |
| Polymarket: Bitcoin reaches $70,000 in November | 0.28 0.30 0.36 0.35 0.27 | 0.309 / 0.301 / 0.345 | $0.123 | 20 s | 10 |
| Polymarket: Bournemouth win on 2026-11-21 | 0.40 0.43 0.42 0.47 0.43 | 0.427 / 0.450 / 0.495 | $0.084 | 13 s | 7 |

- **Total $0.335, about $0.11 per question** (567,082 input tokens of which roughly a third
  cached, 36,264 output, 27 searches = $0.27 of the cost). No request came near the
  100,000-token price tier (largest 67,164). No paper bet: all edges under 0.04.
- **Parse rate 14 / 15.** The one failure stopped at `max_tokens` while still thinking after
  its searches (4,703 output tokens). Fixed: `MAX_TOKENS` 8000.
- **Adaptive thinking and code-driven search both work on Haiku 5.5** (35 thinking blocks, 23
  code-execution calls, no error blocks, no API failures).
- **One badly wrong sample:** `resolution_lawyer` on the House question said 0.04 against
  0.95-0.97 from the others. Its reply: "My search tool hit its limit, so I could only
  retrieve one page". The model fires several searches at once from code and those over
  `max_uses` fail. Same symptom as the Fable run. Fixed two ways: `MAX_WEB_SEARCHES` 3, and
  the user message now states the budget. Not yet re-run live.
- **The trimmed mean did not remove that outlier:** with 4 samples, 20% trimming drops
  nothing, so raw was 0.824. Working as specified; worth a look at the first weekly review.
- One sample (`devils_advocate`, Bournemouth) was billed 0 searches although its code called
  `web_search`: either those searches failed or code-driven searches are not always counted.
  Watch `n_searches` against the reply blocks.
- The House question was already decided (election on Nov 3, market at 0.97). Such
  near-settled markets teach little; `MAX_MID` 0.97 lets them through at the edge.
- Expected cost with 3 searches: about $0.16-0.18 per question, so the remaining budget
  covers roughly 240-270 forecasts.

## Known issues

- Kalshi returns the latest-closing markets first; `forecast` fetches once per close-time
  band to get round it (S5).
- Kalshi's exact fee rounding wording is unconfirmed from the PDF (see "Fee schedules").
- Kalshi series with non-standard or zero fees are charged the standard 0.07 rate.

## Operating rhythm (after the build; from SPEC section 10)

- **Daily (automatic):** resolve → forecast → report → verify → commit.
- **Weekly (user + Claude):** read the report. Look at the worst-scoring resolved forecasts
  and read their rationales: rules misread, bad base rate, stale news, or bad luck? Write the
  findings in this file.
- **Refit:** only at 30, 100, 300 and 1,000 clean resolved forecasts. Never tune on fewer.
  Every prompt or parameter change is its own commit, judged by later out-of-sample BSS.
- **Budget:** 50 USD in total; about 5 questions a day at roughly $0.15 each. Watch `total_cost_usd`.
- **Credibility threshold:** no public claim of beating the market until n ≥ 200 and the BSS
  CI lower bound is > 0.

## Next steps

1. S5 gate: secret, Pages, one manual workflow run (see Current stage); then tag `s5`.
2. S6: README (the interview document) and the final pass over this file.
