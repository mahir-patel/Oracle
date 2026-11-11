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
