# Oracle — living memory

A fresh session must be able to continue from this file plus `SPEC.md` alone.
Updated in the last commit of every stage.

## Current stage

**S0 complete** (tagged `s0`). Next: **S1** — `models`, `config`, `ledger` + tests.
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

## Open questions raised with the user (S0)

Awaiting answers; none blocks S1.

1. Web-search requests are billed per search on top of tokens. Cost tracking in 6.3 counts
   tokens only, so it would understate cost (rule 5). Proposed: add a `PRICE_PER_SEARCH`
   setting and include it. Verify the price in S4.
2. `pnl()` in 6.6 uses "price" without saying whether it includes the fee. Proposed: the bet
   records `price` = quote + fee (the all-in cost), so P&L is net of fees.
3. Kalshi's fee is rounded up per order, not per contract. The per-contract rounding in 6.6
   overstates fees on multi-contract bets (conservative). Proposed: keep it for the edge
   test, verify the schedule in S2.
4. The 6.10 acceptance needs the phase-2 BSS CI lower bound > 0; the prototype got +0.006,
   which is marginal at n = 300. If seed 7 fails under this implementation's random stream,
   report it rather than hunt for a seed.
5. The spending cap can only be checked between questions, so a run can overshoot by up to
   one question's cost (5 parallel samples).

## Findings (numbers)

None yet. S2 will record the simulation results and the isotonic-calibrator ablation here.

## Known issues

None.

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

1. S1: `forecaster/models.py`, `forecaster/config.py`, `forecaster/ledger.py` with tests
   (append, verify, tamper detection, `joined()`).
