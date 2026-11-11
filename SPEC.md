# Oracle — Full Build Prompt

Paste everything below the line into Fable as your first message. Save a copy into the repo as `SPEC.md` once Stage 0 creates it.

---

You are a senior software engineer, ML engineer, data scientist and quant building a production system with me, end to end. This spec has every design decision already made, along with the reason for each. Follow it exactly. If you think a decision is wrong, say so and explain why BEFORE writing code for it. Don't silently deviate.

## 0. How we work (read first)

**Staged build.** We build in 7 stages (S0–S6, section 9). After each stage:
1. All tests pass (`pytest -q`).
2. Everything is committed (git rules below).
3. You post a stage summary: what was built, test results, any deviation from this spec, open questions.
4. **STOP and wait for me** to say "next". Never start the next stage on your own.

**Git rules: commit iteratively, never in one big dump.**
- S0 runs `git init`, sets `main` as the default branch, adds `.gitignore` and makes the first commit.
- Commit after **every logical unit** (one module plus its tests), not once per stage. A stage should usually have 3–8 commits.
- Only commit when tests pass. Never commit broken code to `main`.
- Use Conventional Commits: `feat(ledger): hash-chained append-only log`, `test(scoring): bootstrap CI`, `fix(...)`, `docs(...)`, `chore(...)`, `ci(...)`. The subject line is imperative and under 72 characters. The body explains *why* when it isn't obvious.
- At the end of each stage, tag it: `git tag s0`, `s1`, … `s6`.
- If a remote exists (`git remote -v`), push after each tag: `git push && git push --tags`. If none exists, remind me to create a GitHub repo and give me the exact commands.
- Never commit secrets, `.env`, API keys, `__pycache__` or virtualenvs. Data under `data/` IS committed, because the ledger is the public record. The only exception is `data/cache/`.
- Show me `git log --oneline` in every stage summary.

**Living memory.** Keep a `CLAUDE.md` at the repo root and update it in the last commit of every stage. It holds: the current stage; the decisions made and why; findings (numbers); known issues; next steps. A fresh session must be able to continue from `CLAUDE.md` + `SPEC.md` alone.

**Standards.**
- Python 3.12; type hints everywhere; dataclasses for data types.
- Dependencies: `anthropic`, `httpx`, `numpy`, `pytest`. Nothing else without asking me (no sklearn, no pandas, no LangChain).
- Readable over clever. Each module has one job.
- No network calls in tests. Use recorded fixtures and mocks.
- Read the API key from the `ANTHROPIC_API_KEY` environment variable. Put all settings in `forecaster/config.py`, overridable by environment variables prefixed `ORACLE_`.

## 1. Project

**Oracle** forecasts binary questions on Kalshi and Polymarket using an LLM agent with web search. It logs every forecast **before** the question resolves, in a tamper-evident ledger, and measures whether it beats market prices. The deliverable is a public, verifiable track record: a GitHub repo plus a GitHub Pages site.

**Paper trading only.** Write no order-placement code. Real-money access to these venues is restricted in some countries and provinces; that's my decision to make later, not part of this build.

## 2. Non-negotiable rules (the reason each exists matters)

1. **No leakage.** A forecast may only use information available at forecast time.
   - Live forecasting with web search is leak-free by construction (the web only holds the past).
   - Backtests are NOT: the model may remember outcomes from training. Any question that **resolved before `MODEL_CUTOFF`** is contaminated. Exclude it from all scoring, fitting and the public record.
   - Build a contamination check: if Brier on pre-cutoff questions is < 0.75 × Brier on post-cutoff questions, flag it as suspicious.
2. **The agent never sees the market price.** It forms an independent view. Blending with the market happens afterwards, in code, with a weight fitted on data. *Why:* showing the price anchors the model, which destroys the independent signal that makes blending worthwhile.
3. **Append-only, hash-chained ledger.** No forecast is ever edited or deleted. *Why:* the track record is only credible if it's provably unaltered.
4. **Every tunable number is fitted only on resolved, post-cutoff data, in chronological order.** The scores in reports always use the probability that was logged at forecast time, so they're out of sample by construction.
5. **Measure everything:** dollar cost per forecast, parse failures, latency, and the number of web searches.
6. **Hard spending cap per run** (`ORACLE_MAX_SPEND_PER_RUN`, default $5). Abort the run cleanly once it's reached. Before each question, estimate its cost as the run's average cost per question so far ($0.50 for the first) and skip it if the remaining budget is below that estimate. Samples run in parallel, so overshooting by up to one question's cost is accepted. Log actual spend vs the cap at the end of the run.

## 3. Repo layout

```
oracle/
  forecaster/
    __init__.py
    config.py      # all settings
    models.py      # Question, Sample, Forecast
    markets.py     # Kalshi + Polymarket clients, resolution lookup
    prompts.py     # system prompt + personas (tunable params)
    agent.py       # Agent (real) + FakeAgent (simulation/tests)
    aggregate.py   # logit math, trimmed mean, extremize, blend, fit
    calibrate.py   # isotonic (PAV), gated
    sizing.py      # fees, edge, fractional Kelly, P&L
    ledger.py      # hash-chained JSONL
    scoring.py     # Brier, log loss, BSS, bootstrap, calibration, contamination
    pipeline.py    # question -> forecast glue + market selection
    simulate.py    # offline synthetic world
    report.py      # metrics + docs/index.html
    cli.py         # entry point
  tests/
    fixtures/      # recorded API responses (JSON)
  data/            # ledger.jsonl, params.json, calibrator.json, report.json (committed)
  docs/            # index.html (GitHub Pages)
  .github/workflows/daily.yml
  SPEC.md  CLAUDE.md  README.md  pyproject.toml  .gitignore
```

## 4. Data types (`models.py`)

- **Question:** `venue` ("kalshi" | "polymarket" | "sim"), `market_id`, `title`, `rules` (full resolution text), `close_time` (UTC datetime), `yes_bid`, `yes_ask` (probability units 0–1, nullable), `category` (default "other"), `liquidity` (float, USD), `url`, `fee_rate` (nullable taker-fee coefficient for this market; None = venue default). Properties: `mid` = (bid+ask)/2; `spread` = ask−bid (None if either quote is missing). `to_dict()` writes `close_time` as ISO format.
- **Sample:** `persona`, `probability`, `rationale`, `base_rate` (nullable).
- **Forecast:** `question`, `samples`, `raw_prob` (aggregate of the samples), `calibrated_prob`, `final_prob` (after market blend), `market_prob` (the mid), `created_at`, `model`, `cost_usd`, `bet` (dict).

## 5. Config defaults (`config.py`) — with reasons

| Setting | Default | Why |
|---|---|---|
| `MODEL` | `claude-fable-5-2` | The model I'm using |
| `MODEL_CUTOFF` | `2026-07-01` (env-overridable) | Set to the model's published training cutoff; see rule 1 |
| `N_SAMPLES` | 5 | One per persona; balances diversity against cost |
| `MAX_WEB_SEARCHES` | 4 per sample | Caps cost; more searches give diminishing returns |
| `PRICE_IN` / `PRICE_OUT` | 10 / 50 USD per M tokens | For cost tracking; update if pricing changes |
| `PRICE_PER_SEARCH` | 0.01 USD (placeholder; confirm from the Anthropic docs in S4) | Web searches are billed per search on top of tokens; leaving them out understates cost |
| `FIRST_QUESTION_COST_ESTIMATE` | 0.50 USD | Pre-question budget guard before the run has an average (rule 6) |
| `TRIM_FRACTION` | 0.2 | Robust to one sample going off the rails |
| `EXTREMIZE_ALPHA` | 1.3 (before fitting) | LLMs hedge toward 50%; replaced by the fitted value |
| `MARKET_WEIGHT` | 0.5 (before fitting) | A neutral prior; replaced by the fitted value |
| `KELLY_FRACTION` | 0.25 | Full Kelly is far too volatile when your probabilities are uncertain |
| `MIN_EDGE` | 0.04 | Edge after fees and spread; smaller edges are noise |
| `MAX_STAKE_FRACTION` | 0.05 | Per-bet cap as a share of bankroll |
| `BANKROLL` | 1000 | Paper bankroll; no compounding in v1 |
| `MAX_SPEND_PER_RUN` | 5.0 | Hard dollar cap |
| `LEDGER_PATH` etc. | `data/ledger.jsonl`, `data/params.json`, `data/calibrator.json` | |

## 6. Module specs

### 6.1 `markets.py`
Use public read-only endpoints, no keys. **Before coding, check the venues' current API docs. The field names below are my best knowledge and may have changed.** Tell me about any differences.

- **Kalshi** `GET https://api.elections.kalshi.com/trade-api/v2/markets?status=open&limit=200&cursor=…`
  - Paginate with `cursor`.
  - Map: `ticker` → market_id; `title`; rules = `rules_primary` + "\n" + `rules_secondary`; `close_time`; `yes_bid`/`yes_ask` are in cents, so divide by 100 (if the API now returns dollar-string fields, use those); `liquidity` is in cents, so divide by 100; `category`; url = `https://kalshi.com/markets/{event_ticker}`.
  - Resolution: `GET /markets/{ticker}`; `result` "yes" → 1, "no" → 0, anything else → unresolved.
- **Polymarket** `GET https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=…`
  - Keep only markets where `outcomes` (a JSON string) == `["Yes","No"]`.
  - Map: `id`; `question` → title; `description` → rules; `endDate`; `bestBid`/`bestAsk`; `liquidityNum`; `category`; url = `https://polymarket.com/market/{slug}`. `fee_rate` = `feeSchedule.rate` if `feesEnabled`, else 0 (confirm the field placement against a live response in S3).
  - Resolution: `GET /markets/{id}`. If `closed`, parse `outcomePrices` (a JSON string): if the first price > 0.99 → 1; if any price > 0.99 → 0; otherwise unresolved (disputed or unclear).
- Expose `resolve(venue, market_id) -> int | None`.
- Datetimes are always UTC-aware.
- Tests: use recorded JSON fixtures and `httpx.MockTransport`. I'll run one live smoke test manually.

### 6.2 `prompts.py` — use these verbatim (they're tunable parameters)

System prompt:
```
You are a calibrated superforecaster. Your only goal is an accurate probability for a binary question, scored by Brier score and log loss. Overconfidence and hedging toward 50% are both penalised.

Method:
1. RESOLUTION: Restate exactly what makes this resolve YES, citing the rules. Note any technicality (source, timezone, threshold, deadline) that could flip the outcome.
2. BASE RATE: Find a reference class and estimate how often events like this happen. Give the number.
3. EVIDENCE: Search for the latest relevant information. Weigh each item by reliability and how much it should move you from the base rate.
4. TIME: Account for how much time remains before the deadline; status-quo outcomes are more likely the less time is left.
5. ANSWER: Commit to a probability between 0.01 and 0.99.

You are NOT shown the market price, on purpose. Form an independent view.

End your reply with exactly this block and nothing after it:
<forecast>{"probability": 0.XX, "base_rate": 0.XX, "rationale": "<= 60 words"}</forecast>
```

Personas (one per sample, cycled):
- `outside_view`: Lean primarily on base rates and reference classes; be sceptical of narratives.
- `inside_view`: Lean primarily on the specific current evidence and causal mechanisms.
- `devils_advocate`: First build the strongest case for the outcome you initially think is LESS likely, then decide.
- `resolution_lawyer`: Focus hardest on the exact resolution wording and edge cases in how it will be judged.
- `status_quo`: Ask what happens if nothing changes, then estimate the chance of a change before the deadline.

User message template: `Today is {now UTC}.` / `QUESTION: {title}` / `RESOLUTION RULES: {rules, or "(none provided — infer from the question, and flag the ambiguity)"}` / `CLOSES: {close_time}` / `ANGLE FOR THIS PASS: {persona text}`.

*Why personas instead of temperature:* they give real diversity of reasoning, which is what makes ensembling help. Newer models may also restrict sampling parameters.

### 6.3 `agent.py`
- `Agent.forecast(q, n, now) -> list[Sample]` runs n passes in parallel (thread pool, at most 5 workers).
- Each pass calls `messages.create` with `max_tokens=4000`, the system prompt, and the server-side web search tool: `{"type": "web_search_20250305", "name": "web_search", "max_uses": MAX_WEB_SEARCHES}`. Check the current tool version string in the Anthropic docs.
- Join the text blocks, then parse with the regex `<forecast>\s*(\{.*?\})\s*</forecast>` (DOTALL). Clamp the probability to [0.01, 0.99]. Drop a sample (return None) on a parse failure and count the failure.
- Track cost from `usage`: input and output tokens times the prices, plus the number of web searches times `PRICE_PER_SEARCH`. Add it to `agent.cost_usd`; this total is what the spending cap is checked against. Also log the number of searches.
- Retry API errors with exponential backoff (3 attempts). Respect the spending cap.
- **FakeAgent(truth_fn, noise=0.5, hedge=0.7, seed)**: each sample = sigmoid(hedge·logit(truth) + N(0, noise)), clipped to [0.01, 0.99]. It has `cost_usd = 0` and `model = "fake"`. It's used by tests and the simulation.

### 6.4 `aggregate.py` (all in log-odds; `EPS = 1e-4` clip)
- `trimmed_mean_logodds(probs, trim=0.2)`: sort the logits, drop `int(n·trim)` from each end (only if at least one remains), take the mean, apply sigmoid.
- `extremize(p, alpha)` = sigmoid(alpha·logit(p)).
- `blend_with_market(p, m, w)` = sigmoid((1−w)·logit(p) + w·logit(m)); return p if m is None.
- `disagreement(probs)` = the standard deviation of the logits. Log it with every forecast; it's the main feature for the future error predictor.
- `fit_alpha_and_weight(raw, market, outcomes)`: grid search with alpha ∈ [0.8, 2.5] step 0.05 and w ∈ [0, 1] step 0.05, minimising log loss. Return alpha, market_weight and log_loss.

### 6.5 `calibrate.py`
- Isotonic regression by pool-adjacent-violators, in numpy. The value for each block = (sum_y + 1)/(count + 2) (Laplace smoothing, so it never outputs exactly 0 or 1). Interpolate between block midpoints and clip to [0.01, 0.99]. Include save/load to JSON.
- **Gated: it's the identity function below `MIN_POINTS = 1000`.** *Finding from my prototype (reproduce it in the S2 simulation):* stacked on top of the fitted extremizing alpha, isotonic calibration made out-of-sample Brier WORSE at n = 300 and 600, across 4 seeds. Two calibrators in a row overfit. Only enable it with a large history, and keep it only if the report shows it helps.

### 6.6 `sizing.py`
- Both venues charge takers `rate · contracts · P · (1−P)` per order, where P is the price paid per contract (verified 2026-11-10; sources in CLAUDE.md). We always cross the spread, so we are always the taker.
  - **Kalshi:** rate 0.07 on most markets, rounded **up to the next cent per order**: ceil(0.07 · contracts · P · (1−P) · 100)/100.
  - **Polymarket:** no longer fee-free. The rate is set per market (0 for geopolitics, 0.04–0.07 elsewhere) and comes from `Question.fee_rate`; the fee is rounded to 5 decimal places. If a market's rate is unknown, use 0.05 (the "Other / General" rate).
  - `sim` venue: no fee.
- The edge check uses the per-contract cost: quote + `fee(1, P)`. The recorded bet uses the real order fee.
- `decide(p, bid, ask, venue)`:
  - YES costs ask + fee, with win probability p.
  - NO costs (1 − bid) + fee, with win probability 1 − p.
  - edge = win_p − cost. Pick the side with the higher edge.
  - If edge < MIN_EDGE, there's no bet; record the edge and the reason.
  - Kelly f* = (win_p − cost)/(1 − cost); stake = min(KELLY_FRACTION · f*, MAX_STAKE_FRACTION) · bankroll; contracts = int(stake / cost).
  - The bet records `quote`, `fee` (the per-order fee for `contracts`) and `price` = quote + fee/contracts, the all-in cost per contract.
  - If there's no quote, there's no bet.
- `pnl(bet, outcome)`: if it won, contracts · (1 − price); if it lost, −contracts · price. Because `price` includes the fee, paper P&L is net of fees.

### 6.7 `ledger.py`
- Canonical JSON: `sort_keys=True, separators=(",", ":"), default=str`. The genesis hash is 64 zeros.
- Each appended record gets `prev_hash`, then `hash` = sha256(canonical JSON of the record including `prev_hash`, excluding `hash`).
- Record types:
  - `forecast`: ts, question dict, samples, raw/calibrated/final/market probs, disagreement, model, cost_usd, bet, latency_s.
  - `resolution`: venue, market_id, outcome, ts.
- `verify()` recomputes the whole chain. `joined()` returns the latest forecast per (venue, market_id) joined to its resolution (outcome and resolved_at, or None).
- v1 decision: **one forecast per market** (skip markets already in the ledger). Re-forecasting as news arrives is on the roadmap.

### 6.8 `scoring.py`
- `brier`, `log_loss` (with EPS clip), and `brier_skill_vs(p, market, y)` = 1 − Brier(p)/Brier(market). **This is the headline number: > 0 means we beat the market.**
- `bootstrap_bss`: 5,000 resamples, seed 0. Return bss, the 95% CI (2.5/97.5 percentiles), P(beat market) = the share of resamples > 0, and n. Only report it when n ≥ 10.
- `calibration_table`: 10 equal-width bins with n, mean forecast and observed rate.
- `contamination_check`: Brier before vs after the cutoff, with the "suspicious" flag from rule 1.

### 6.9 `pipeline.py`
- `forecast_question(q, agent, calibrator, params, now)`: samples → trimmed mean (raw) → extremize(alpha) → calibrator → blend with q.mid (market_weight) → `decide` → Forecast. Return None if there are zero valid samples.
- `select_questions`: keep markets where:
  - time to close is between 1 and 60 days. Under 1 day leaves no time to be right; over 60 makes feedback too slow.
  - liquidity ≥ 500.
  - mid and spread both exist, and spread ≤ 0.10. Wide spreads eat the edge.
  - 0.03 ≤ mid ≤ 0.97. Near-certain markets have no room for edge.

### 6.10 `simulate.py` — the offline proof
Synthetic world, seeded:
- true p = clip(sigmoid(N(0, 1.5)), 0.02, 0.98)
- market mid = clip(sigmoid(logit(p) + N(0, 0.6)), 0.03, 0.97), with bid/ask = mid ∓ 0.01
- outcome ~ Bernoulli(p)
- FakeAgent with noise 0.5 and hedge 0.7
- close times are spread over the days after 2026-08-01, so everything is post-cutoff

Procedure:
1. Phase 1 (first half of n = 1200): params alpha = 1, w = 0, no calibrator. Log forecasts and resolutions.
2. Fit alpha and w on phase 1.
3. Phase 2 (second half): forecast with the fitted params.
4. Print a report for each phase plus the ledger verification.

**Acceptance (robust to the seed; never hunt for a passing seed):** run n = 1200 (600/600 split) for each of the 5 fixed seeds 7, 8, 9, 10, 11. Pass if, averaged across the 5 seeds, phase-2 final Brier < phase-2 model-only Brier < market Brier, and the BSS CI lower bound on the pooled phase-2 rows (all 5 seeds, 3,000 rows) is > 0. Report the per-seed results in CLAUDE.md, including any seed that fails on its own. My prototype (n = 600, one seed) got phase-2 Brier of 0.189 for the final forecast, 0.191 for the model alone and 0.203 for the market, with BSS +0.068 (CI +0.006..+0.125); that lower bound is too close to zero to rest a single-seed test on. Also run the calibrator ablation from 6.5 and record the result in CLAUDE.md.

### 6.11 `report.py`
- `build(rows)`: counts (total/resolved/open), total cost. Then **post-cutoff resolved rows only**: Brier for final/model-only/market, log loss, bootstrap BSS, calibration table, paper P&L and number of bets. Add the contamination check over all resolved rows.
- `to_markdown(report)`: a short console summary with a ⚠️ line if contamination is suspicious.
- `write_site(report, rows)` → `docs/index.html`. One self-contained, mobile-friendly page with no external assets. It includes:
  - the headline BSS with CI and n (or "not enough resolved questions yet");
  - resolved and open counts, paper P&L and total API cost;
  - a calibration plot as inline SVG (dot size by n, plus a diagonal reference line);
  - a table of open forecasts (title linked to the market, Oracle %, market %, close date, first 10 characters of the ledger hash);
  - a "how to verify" note pointing to `data/ledger.jsonl` and `oracle verify`;
  - a "not financial advice" footer.

### 6.12 `cli.py` (`python -m forecaster.cli …`, plus an `oracle` console script)
- `forecast --venue {kalshi,polymarket,both} --scan 400 --max 10`: fetch, select, skip markets already in the ledger, forecast, log. Print one line per question plus the run's spend.
- `resolve`: for unresolved ledger markets past their close time, look up the result and log it. A network error skips the market until the next run.
- `refit`: needs at least 30 clean (post-cutoff) resolved rows. It fits alpha and w, writes `data/params.json`, and fits the gated calibrator.
- `report`: writes `data/report.json` and `docs/index.html` and prints the markdown summary.
- `verify`: exits 1 if the chain is broken.
- `simulate --n 1200 --seed 7` (one seed; the 5-seed acceptance run lives in the test suite)

### 6.13 `.github/workflows/daily.yml`
- Runs on a daily cron at an off-peak UTC time (avoid :00), plus `workflow_dispatch`.
- `permissions: contents: write`; Python 3.12; install the package.
- Steps: `resolve` → `forecast --max ${{ vars.ORACLE_DAILY_MAX || 10 }}` → `report` → `verify`.
- Then commit `data/` and `docs/` if they changed, with the message `chore(data): daily run YYYY-MM-DD`, and push.
- `ANTHROPIC_API_KEY` comes from repo secrets. Add a concurrency group so runs never overlap.
- *Why:* git commit timestamps from GitHub plus the hash chain give third-party proof of when each forecast was made.

## 7. Testing requirements
- `ledger`: append, verify, **tamper detection** (edit one byte, then `verify()` is False), and `joined()` correctness.
- `aggregate`: logit/sigmoid round trip, trimmed mean ignores an outlier, extremize alpha = 1 is the identity, blend w = 0/1 edge cases, and the fit recovers a known alpha on synthetic data.
- `calibrate`: monotone output, identity below the gate, save/load round trip.
- `sizing`: no bet below the edge, correct side chosen, Kelly math, the cap, P&L for win/loss net of fees, Kalshi per-order fee values at P = 0.5 and 0.1 for 1 and many contracts.
- `scoring`: known Brier/log-loss values, BSS sign, bootstrap CI contains the point estimate, contamination flag.
- `markets`: fixtures → Question mapping and resolution parsing for both venues, including a disputed or unresolved case.
- `agent`: a mocked client returns canned text, and the parser handles valid, clamped and missing/garbled `<forecast>` blocks.
- `pipeline`: an end-to-end run with FakeAgent writes a valid ledger.
- `simulate`: the acceptance criteria in 6.10 hold (a test over the 5 fixed seeds).

## 8. README (written in S6; it's an interview document)
Sections:
1. What it is — one paragraph.
2. Live results — link to the Pages site.
3. How it works — a pipeline diagram in Mermaid.
4. **Design decisions & findings** — leakage and the cutoff; blind-to-market plus the fitted blend; personas vs temperature; log-odds aggregation; the extremizing result; the isotonic overfitting finding with numbers; the hash-chained ledger plus git timestamps.
5. How to verify the track record.
6. Run it yourself.
7. Costs.
8. Limitations — slow feedback, small n, paper P&L ignores slippage and market impact.
9. Roadmap.

## 9. Stages and acceptance criteria

| Stage | Build | Done when |
|---|---|---|
| **S0** | `git init`, pyproject, `.gitignore`, layout, SPEC.md, CLAUDE.md skeleton, empty README | First commit; `pytest` runs (0 tests is OK); tag `s0` |
| **S1** | `models`, `config`, `ledger` + tests | Tamper test passes; tag `s1` |
| **S2** | `aggregate`, `calibrate`, `scoring`, `sizing`, FakeAgent, `pipeline`, `simulate` + tests | 6.10 acceptance criteria met; calibrator ablation recorded in CLAUDE.md; tag `s2` |
| **S3** | `markets` + fixture tests | Both venues map and resolve correctly from fixtures. **Then ask me to run one live fetch** and fix any field differences; tag `s3` |
| **S4** | `prompts`, `agent` + mocked tests | Mocked tests pass. **Then ask me to run ONE live forecast on 3 questions** and show me the cost, latency, parse rate and the full rationale of one sample. Adjust and tag `s4` |
| **S5** | `cli`, `report`, Pages site, GitHub Actions | `simulate` and `report` work locally; workflow passes on `workflow_dispatch`; Pages site loads; tag `s5` |
| **S6** | README, final CLAUDE.md | README complete, including the findings section; tag `s6`, push |

## 10. After the build: operating rhythm (record this in CLAUDE.md)
- **Daily (automatic):** resolve → forecast → report → verify → commit.
- **Weekly (me + you):** read the report. Look at the worst-scoring resolved forecasts and read their rationales: was the error from misreading the rules, a bad base rate, stale news or bad luck? Write the findings in CLAUDE.md.
- **Refit:** only at 30, 100, 300 and 1,000 clean resolved forecasts. Never tune on fewer. Every change to a prompt or parameter is its own commit and is judged by later out-of-sample BSS, never by how good the reasoning sounds.
- **Budget:** about 10 questions a day at roughly $0.10–0.50 each. Watch `total_cost_usd`.
- **Credibility threshold:** don't publicly claim to beat the market until n ≥ 200 and the BSS CI lower bound is > 0.

## 11. Roadmap (do NOT build until I ask; listed so the architecture leaves room)
1. **Error predictor:** a small logistic model on ledger features (disagreement, category, days to close, liquidity, spread, |model − market|) that predicts when the LLM is wrong. Use it to skip bets or shrink stakes. Validate it chronologically.
2. **Re-forecasting:** update forecasts as news arrives, with all versions kept in the ledger and scoring by time-weighted Brier.
3. **News-shock latency:** detect breaking news, re-forecast within seconds, and measure how much of the edge comes from speed vs judgment.
4. **Correlated exposure:** group related markets (for example, the same election) and cap exposure per group.
5. **Category analysis:** BSS per category, then stop forecasting categories with no edge.

Begin with **S0**. Confirm you've read the whole spec, list any decisions you disagree with, then build S0 and stop.
