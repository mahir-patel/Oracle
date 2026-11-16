"""Command line entry point: ``oracle <command>`` or ``python -m forecaster.cli``.

    oracle resolve     look up results of forecast markets that have closed
    oracle forecast    pick today's questions, forecast them, log them
    oracle report      write data/report.json and docs/index.html
    oracle verify      recompute the ledger's hash chain (exit 1 if broken)
    oracle refit       fit alpha, market weight and the gated calibrator
    oracle simulate    run the offline synthetic world

Paper trading only: nothing here places an order.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import anthropic
import httpx

from forecaster import report as report_mod
from forecaster import simulate
from forecaster.agent import Agent, ForecastAgent
from forecaster.aggregate import extremize, fit_alpha_and_weight
from forecaster.calibrate import Calibrator, fit_calibrator
from forecaster.config import Config, load_config
from forecaster.ledger import Ledger
from forecaster.markets import fetch_open, make_client, resolve
from forecaster.models import Question
from forecaster.pipeline import Params, bands, forecast_question, pick_daily, select_questions

VENUES = ("kalshi", "polymarket")


def utc_now() -> datetime:
    return datetime.now(UTC)


def make_agent(config: Config) -> ForecastAgent:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set; nothing was spent")
    return Agent(config)


# ---------- forecast ----------


def spent_so_far(ledger: Ledger, config: Config) -> float:
    """Everything the project has spent: the ledger's forecasts plus the live
    checks made before the ledger existed. Samples of a question that produced
    no forecast are not in the ledger, so this can understate slightly."""
    logged = sum(r.get("cost_usd", 0.0) for r in ledger.records() if r["type"] == "forecast")
    return config.spent_before_ledger + logged


def fetch_candidates(venues: Sequence[str], scan: int, now: datetime, config: Config) -> list[Question]:
    """One fetch per venue per close-time band. Kalshi lists the latest-closing
    markets first, so a single fetch over 1-60 days would only see the far end."""
    selected: list[Question] = []
    with make_client() as client:
        for lo, hi, _ in bands(config):
            for venue in venues:
                try:
                    fetched = fetch_open(
                        venue, scan, client, now + timedelta(days=lo), now + timedelta(days=hi)
                    )
                except httpx.HTTPError as exc:
                    print(f"  {venue} {lo:g}-{hi:g} days: fetch failed, skipped ({exc})")
                    continue
                kept = select_questions(fetched, now, config)
                print(f"  {venue} {lo:g}-{hi:g} days: {len(fetched)} fetched, {len(kept)} selected")
                selected += kept
    return selected


def cmd_forecast(args: argparse.Namespace, config: Config) -> int:
    now = utc_now()
    ledger = Ledger(config.ledger_path)

    spent = spent_so_far(ledger, config)
    budget = min(config.max_spend_per_run, config.max_spend_total - spent)
    print(
        f"budget: ${spent:.2f} of ${config.max_spend_total:.2f} spent in total; "
        f"this run may spend ${max(budget, 0):.2f}"
    )
    if budget <= 0:
        print("total budget reached: no forecasts")
        return 0

    # The agent stops asking once the run's budget is used up.
    agent = make_agent(replace(config, max_spend_per_run=budget))
    venues = VENUES if args.venue == "both" else (args.venue,)
    selected = fetch_candidates(venues, args.scan, now, config)
    picked, counts = pick_daily(selected, ledger, args.max, now, config)
    for c in counts:
        print(
            f"  band {c.lo_days:g}-{c.hi_days:g} days: "
            f"{c.eligible} eligible, quota {c.quota}, picked {c.picked}"
        )

    params = Params.load(config.params_path, config)
    calibrator = Calibrator.load(config.calibrator_path)
    attempted = logged = 0
    status = 0
    try:
        for q in picked:
            # Budget guard: skip a question the remaining budget cannot cover.
            estimate = (
                agent.cost_usd / attempted if attempted else config.first_question_cost_estimate
            )
            if budget - agent.cost_usd < estimate:
                print(
                    f"budget guard: ${budget - agent.cost_usd:.2f} left, "
                    f"a question costs about ${estimate:.2f}; stopping"
                )
                break
            attempted += 1
            forecast = forecast_question(q, agent, calibrator, params, utc_now(), config)
            if forecast is None:
                print(f"- [{q.venue}] {q.title[:70]}: no valid sample, not logged")
                continue
            record = ledger.append_forecast(forecast)
            logged += 1
            bet = forecast.bet.get("side") or "no bet"
            print(
                f"- [{q.venue}] {q.title[:70]}: oracle {forecast.final_prob:.2f}, "
                f"market {forecast.market_prob:.2f}, {bet}, ${forecast.cost_usd:.3f}, "
                f"{forecast.latency_s:.0f}s, {record['hash'][:10]}"
            )
    except anthropic.APIError as exc:
        # A bad key or an empty account: everything forecast so far is already
        # in the ledger, so stop here and fail the run visibly.
        print(f"API error, run stopped: {exc}")
        status = 1

    print(
        f"run: {logged} logged of {attempted} attempted ({len(picked)} picked); "
        f"spent ${agent.cost_usd:.3f} of ${budget:.2f} cap"
    )
    return status


# ---------- resolve ----------


def cmd_resolve(args: argparse.Namespace, config: Config) -> int:
    now = utc_now()
    ledger = Ledger(config.ledger_path)
    due = [
        r
        for r in ledger.joined()
        if r["outcome"] is None and datetime.fromisoformat(r["question"]["close_time"]) <= now
    ]
    resolved = 0
    with make_client() as client:
        for r in due:
            q = r["question"]
            try:
                outcome = resolve(q["venue"], q["market_id"], client)
            except httpx.HTTPError as exc:
                print(f"- [{q['venue']}] {q['market_id']}: lookup failed, will retry ({exc})")
                continue
            if outcome is None:
                continue  # closed but not settled yet
            ledger.append_resolution(q["venue"], q["market_id"], outcome, now)
            resolved += 1
            print(f"- [{q['venue']}] {q['title'][:70]}: {'YES' if outcome else 'NO'}")
    print(f"resolve: {resolved} resolved, {len(due) - resolved} closed and still waiting")
    return 0


# ---------- refit ----------


def cmd_refit(args: argparse.Namespace, config: Config) -> int:
    cutoff = config.model_cutoff.isoformat()
    # Only this model's resolved, post-cutoff forecasts (rules 1 and 4).
    clean = [
        r
        for r in Ledger(config.ledger_path).joined()
        if r["outcome"] is not None and r["resolved_at"] >= cutoff and r["model"] == config.model
    ]
    if len(clean) < config.min_refit_rows:
        print(f"refit: {len(clean)} clean resolved forecasts, need {config.min_refit_rows}")
        return 1
    raw = [r["raw_prob"] for r in clean]
    outcomes = [r["outcome"] for r in clean]
    fit = fit_alpha_and_weight(raw, [r["market_prob"] for r in clean], outcomes)
    Params(fit.alpha, fit.market_weight, n_fit=len(clean)).save(config.params_path)
    calibrator = fit_calibrator(
        [extremize(p, fit.alpha) for p in raw], outcomes, config.calibrator_min_points
    )
    calibrator.save(config.calibrator_path)
    state = "identity (below the gate)" if calibrator.is_identity else "fitted"
    print(
        f"refit on {len(clean)} forecasts: alpha {fit.alpha:.2f}, "
        f"market weight {fit.market_weight:.2f}, calibrator {state}"
    )
    return 0


# ---------- report, verify, simulate ----------


def cmd_report(args: argparse.Namespace, config: Config) -> int:
    rows = Ledger(config.ledger_path).joined()
    rep = report_mod.build(rows, config, utc_now())
    report_mod.write_report(rep, config.report_path)
    report_mod.write_site(rep, rows, config.site_path)
    print(report_mod.to_markdown(rep))
    return 0


def cmd_verify(args: argparse.Namespace, config: Config) -> int:
    ledger = Ledger(config.ledger_path)
    if not ledger.verify():
        print(f"ledger BROKEN: {config.ledger_path} fails the hash chain check")
        return 1
    print(f"ledger OK: {len(ledger.records())} records, hash chain intact")
    return 0


def cmd_simulate(args: argparse.Namespace, config: Config) -> int:
    print(simulate.format_result(simulate.run(n=args.n, seed=args.seed, config=config)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oracle", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("forecast", help="pick today's questions, forecast and log them")
    p.add_argument("--venue", choices=[*VENUES, "both"], default="both")
    p.add_argument("--scan", type=int, default=400, help="markets read per venue per band")
    p.add_argument("--max", type=int, default=5, help="questions to forecast")
    p.set_defaults(run=cmd_forecast)

    for name, fn, text in (
        ("resolve", cmd_resolve, "log results of forecast markets that have closed"),
        ("refit", cmd_refit, "fit alpha, market weight and the gated calibrator"),
        ("report", cmd_report, "write the report and the public page"),
        ("verify", cmd_verify, "check the ledger's hash chain"),
    ):
        sub.add_parser(name, help=text).set_defaults(run=fn)

    p = sub.add_parser("simulate", help="run the offline synthetic world")
    p.add_argument("--n", type=int, default=simulate.ACCEPTANCE_N)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(run=cmd_simulate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args, load_config())


if __name__ == "__main__":
    sys.exit(main())
