"""Offline synthetic world: proof that the pipeline, ledger, fitting and scoring
work end to end before any money is spent on API calls.

World (seeded):
  true p   = clip(sigmoid(N(0, 1.5)), 0.02, 0.98)
  market   = clip(sigmoid(logit(p) + N(0, 0.6)), 0.03, 0.97), bid/ask = mid -/+ 0.01
  outcome  ~ Bernoulli(p)
  agent    = FakeAgent(noise 0.5, hedge 0.7): informative but hedged toward 50%

Procedure: phase 1 forecasts with neutral params (alpha 1, no market weight);
alpha and the market weight are fitted on phase 1; phase 2 forecasts with the
fitted params. Every phase-2 score is out of sample.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from forecaster.agent import FakeAgent
from forecaster.aggregate import Fit, extremize, fit_alpha_and_weight, trimmed_mean_logodds
from forecaster.calibrate import Calibrator, fit_calibrator
from forecaster.config import Config
from forecaster.ledger import Ledger, Record
from forecaster.models import Question
from forecaster.pipeline import Params, forecast_question
from forecaster.scoring import BootstrapBSS, bootstrap_bss, brier, log_loss
from forecaster.sizing import pnl

START = datetime(2026, 8, 1, tzinfo=UTC)  # after MODEL_CUTOFF, so nothing is contaminated
ACCEPTANCE_SEEDS = (7, 8, 9, 10, 11)
ACCEPTANCE_N = 1200


@dataclass
class World:
    questions: list[Question]
    truth: dict[str, float]
    outcomes: dict[str, int]


def make_world(n: int, seed: int) -> World:
    rng = np.random.default_rng([seed, 0])
    true_logit = rng.normal(0, 1.5, n)
    p = np.clip(_sigmoid(true_logit), 0.02, 0.98)
    mid = np.clip(_sigmoid(np.log(p / (1 - p)) + rng.normal(0, 0.6, n)), 0.03, 0.97)
    y = (rng.random(n) < p).astype(int)

    questions, truth, outcomes = [], {}, {}
    for i in range(n):
        market_id = f"sim-{seed}-{i:05d}"
        # One market closes every 2 hours. The second half starts 3 days later so
        # that every phase-1 market has resolved before any phase-2 forecast is made.
        close = START + timedelta(days=1, hours=2 * i) + timedelta(days=3 * (i >= n // 2))
        questions.append(
            Question(
                venue="sim",
                market_id=market_id,
                title=f"Synthetic question {i}",
                rules="Resolves YES with the hidden true probability.",
                close_time=close,
                yes_bid=float(mid[i]) - 0.01,
                yes_ask=float(mid[i]) + 0.01,
                category="sim",
                liquidity=10_000.0,
            )
        )
        truth[market_id] = float(p[i])
        outcomes[market_id] = int(y[i])
    return World(questions, truth, outcomes)


def phase_report(rows: Sequence[Record]) -> dict[str, Any]:
    """Scores for resolved rows, always using the probabilities logged at forecast time."""
    y = [r["outcome"] for r in rows]
    final = [r["final_prob"] for r in rows]
    model = [r["calibrated_prob"] for r in rows]
    market = [r["market_prob"] for r in rows]
    bets = [r for r in rows if r["bet"].get("side")]
    return {
        "n": len(rows),
        "brier_final": brier(final, y),
        "brier_model": brier(model, y),
        "brier_market": brier(market, y),
        "log_loss_final": log_loss(final, y),
        "log_loss_market": log_loss(market, y),
        "bss": bootstrap_bss(final, market, y),
        "n_bets": len(bets),
        "pnl": sum(pnl(r["bet"], r["outcome"]) for r in bets),
    }


@dataclass
class SimResult:
    seed: int
    n: int
    fit: Fit
    phase1: dict[str, Any]
    phase2: dict[str, Any]
    phase2_rows: list[Record]
    ledger_ok: bool
    n_records: int


def run(
    n: int = ACCEPTANCE_N,
    seed: int = 7,
    ledger_path: str | Path | None = None,
    config: Config | None = None,
) -> SimResult:
    """Run both phases through the real pipeline and ledger.

    The ledger goes to a temporary directory unless a path is given, so a
    simulation can never touch the real ``data/ledger.jsonl``.
    """
    config = config or Config()
    with tempfile.TemporaryDirectory() as tmp:
        ledger = Ledger(ledger_path if ledger_path is not None else Path(tmp) / "ledger.jsonl")
        world = make_world(n, seed)
        agent = FakeAgent(lambda q: world.truth[q.market_id], noise=0.5, hedge=0.7, seed=[seed, 1])
        half = n // 2
        phase1_ids = {q.market_id for q in world.questions[:half]}

        def play(questions: Sequence[Question], params: Params) -> None:
            for q in questions:
                now = q.close_time - timedelta(days=1)
                forecast = forecast_question(q, agent, Calibrator(), params, now, config)
                assert forecast is not None  # FakeAgent always returns samples
                ledger.append_forecast(forecast)
            for q in questions:
                ledger.append_resolution(
                    q.venue, q.market_id, world.outcomes[q.market_id], q.close_time
                )

        # Phase 1: neutral parameters, nothing fitted.
        play(world.questions[:half], Params(alpha=1.0, market_weight=0.0))

        # Fit on resolved, post-cutoff phase-1 rows only (rules 1 and 4).
        cutoff = config.model_cutoff.isoformat()
        clean = [
            r for r in ledger.joined() if r["outcome"] is not None and r["resolved_at"] >= cutoff
        ]
        fit = fit_alpha_and_weight(
            [r["raw_prob"] for r in clean],
            [r["market_prob"] for r in clean],
            [r["outcome"] for r in clean],
        )

        # Phase 2: the fitted parameters, scored out of sample.
        play(world.questions[half:], Params(fit.alpha, fit.market_weight, n_fit=len(clean)))

        rows = ledger.joined()
        phase1_rows = [r for r in rows if r["question"]["market_id"] in phase1_ids]
        phase2_rows = [r for r in rows if r["question"]["market_id"] not in phase1_ids]
        return SimResult(
            seed=seed,
            n=n,
            fit=fit,
            phase1=phase_report(phase1_rows),
            phase2=phase_report(phase2_rows),
            phase2_rows=phase2_rows,
            ledger_ok=ledger.verify(),
            n_records=len(ledger.records()),
        )


# ---------- acceptance over several seeds ----------


@dataclass
class Acceptance:
    results: list[SimResult]
    mean_brier_final: float
    mean_brier_model: float
    mean_brier_market: float
    pooled_bss: BootstrapBSS
    passed: bool


def acceptance(seeds: Sequence[int] = ACCEPTANCE_SEEDS, n: int = ACCEPTANCE_N) -> Acceptance:
    """SPEC 6.10: on average across the seeds, final < model-only < market Brier in
    phase 2, and the BSS CI lower bound on the pooled phase-2 rows is above zero."""
    results = [run(n=n, seed=s) for s in seeds]
    mean_final = float(np.mean([r.phase2["brier_final"] for r in results]))
    mean_model = float(np.mean([r.phase2["brier_model"] for r in results]))
    mean_market = float(np.mean([r.phase2["brier_market"] for r in results]))
    pooled = [row for r in results for row in r.phase2_rows]
    pooled_bss = bootstrap_bss(
        [r["final_prob"] for r in pooled],
        [r["market_prob"] for r in pooled],
        [r["outcome"] for r in pooled],
    )
    assert pooled_bss is not None
    passed = (
        mean_final < mean_model < mean_market
        and pooled_bss.ci_low > 0
        and all(r.ledger_ok for r in results)
    )
    return Acceptance(results, mean_final, mean_model, mean_market, pooled_bss, passed)


# ---------- calibrator ablation (SPEC 6.5) ----------


def _raw_forecasts(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw (trimmed-mean) forecasts, market mids and outcomes for a fresh world."""
    world = make_world(n, seed)
    agent = FakeAgent(lambda q: world.truth[q.market_id], noise=0.5, hedge=0.7, seed=[seed, 1])
    raw = [
        trimmed_mean_logodds([s.probability for s in agent.forecast(q, 5, START)], 0.2)
        for q in world.questions
    ]
    mid = [q.mid for q in world.questions]
    y = [world.outcomes[q.market_id] for q in world.questions]
    return np.array(raw), np.array(mid, dtype=float), np.array(y)


def calibrator_ablation(
    train_sizes: Sequence[int] = (300, 600, 1000, 3000),
    seeds: Sequence[int] = ACCEPTANCE_SEEDS,
    n_test: int = 5000,
) -> list[dict[str, Any]]:
    """Does isotonic calibration, stacked on the fitted alpha, help out of sample?

    For each training size and seed: fit alpha and the market weight on the first
    ``n_train`` questions, fit an (ungated) isotonic calibrator on the extremized
    training forecasts, then score the next ``n_test`` questions with and without it.
    A positive delta means the calibrator made Brier WORSE.
    """
    out = []
    for n_train in train_sizes:
        for seed in seeds:
            raw, mid, y = _raw_forecasts(n_train + n_test, seed)
            tr, te = slice(0, n_train), slice(n_train, None)
            fit = fit_alpha_and_weight(raw[tr], mid[tr], y[tr])
            ext_train = [extremize(float(p), fit.alpha) for p in raw[tr]]
            cal = fit_calibrator(ext_train, y[tr], min_points=0)

            ext = [extremize(float(p), fit.alpha) for p in raw[te]]
            with_cal = [cal.apply(p) for p in ext]
            w = fit.market_weight
            final_plain = _blend(np.array(ext), mid[te], w)
            final_cal = _blend(np.array(with_cal), mid[te], w)
            out.append(
                {
                    "n_train": n_train,
                    "seed": seed,
                    "alpha": fit.alpha,
                    "market_weight": w,
                    "model_plain": brier(ext, y[te]),
                    "model_calibrated": brier(with_cal, y[te]),
                    "final_plain": brier(final_plain, y[te]),
                    "final_calibrated": brier(final_cal, y[te]),
                }
            )
    for row in out:
        row["model_delta"] = row["model_calibrated"] - row["model_plain"]
        row["final_delta"] = row["final_calibrated"] - row["final_plain"]
    return out


# ---------- formatting ----------


def format_result(result: SimResult) -> str:
    lines = [
        f"Simulation: n={result.n}, seed={result.seed}",
        f"Fitted on phase 1: alpha={result.fit.alpha:.2f}, "
        f"market_weight={result.fit.market_weight:.2f}",
    ]
    for name, rep in (("Phase 1 (neutral params)", result.phase1), ("Phase 2 (fitted)", result.phase2)):
        lines.append(f"\n{name}: n={rep['n']}")
        lines.append(
            f"  Brier  final {rep['brier_final']:.4f} | model-only {rep['brier_model']:.4f}"
            f" | market {rep['brier_market']:.4f}"
        )
        lines.append(
            f"  Log loss  final {rep['log_loss_final']:.4f} | market {rep['log_loss_market']:.4f}"
        )
        bss = rep["bss"]
        if bss is not None:
            lines.append(
                f"  BSS vs market {bss.bss:+.3f} (95% CI {bss.ci_low:+.3f}..{bss.ci_high:+.3f}),"
                f" P(beat market) {bss.p_beat_market:.2f}"
            )
        lines.append(f"  Paper P&L ${rep['pnl']:+.2f} over {rep['n_bets']} bets")
    status = "OK" if result.ledger_ok else "BROKEN"
    lines.append(f"\nLedger: {result.n_records} records, hash chain {status}")
    return "\n".join(lines)


def format_acceptance(acc: Acceptance) -> str:
    lines = ["seed  alpha  w     final   model   market  BSS     CI low  CI high"]
    for r in acc.results:
        p2, bss = r.phase2, r.phase2["bss"]
        lines.append(
            f"{r.seed:<5} {r.fit.alpha:<6.2f} {r.fit.market_weight:<5.2f} "
            f"{p2['brier_final']:.4f}  {p2['brier_model']:.4f}  {p2['brier_market']:.4f}  "
            f"{bss.bss:+.3f}  {bss.ci_low:+.3f}  {bss.ci_high:+.3f}"
        )
    b = acc.pooled_bss
    lines.append(
        f"mean               {acc.mean_brier_final:.4f}  {acc.mean_brier_model:.4f}  "
        f"{acc.mean_brier_market:.4f}"
    )
    lines.append(
        f"pooled (n={b.n})    BSS {b.bss:+.3f}  CI {b.ci_low:+.3f}..{b.ci_high:+.3f}  "
        f"P(beat market) {b.p_beat_market:.3f}"
    )
    lines.append("ACCEPTANCE: " + ("PASS" if acc.passed else "FAIL"))
    return "\n".join(lines)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _blend(p: np.ndarray, market: np.ndarray, w: float) -> np.ndarray:
    eps = 1e-4
    p = np.clip(p, eps, 1 - eps)
    m = np.clip(market, eps, 1 - eps)
    return _sigmoid((1 - w) * np.log(p / (1 - p)) + w * np.log(m / (1 - m)))
