"""S4 gate: ONE live forecast on 3 questions. Spends real money (API key needed).

    ANTHROPIC_API_KEY=... .venv/bin/python scripts/live_check.py

Nothing is written to the ledger. The full output, including every sample's
reply text, is saved to data/cache/live_check.json (not committed).
The `forecast` command replaces this script in S5.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from forecaster.agent import Agent
from forecaster.calibrate import Calibrator
from forecaster.config import load_config
from forecaster.markets import fetch_open, make_client
from forecaster.models import Question
from forecaster.pipeline import Params, forecast_question, select_questions

PICKS = {"polymarket": 2, "kalshi": 1}
OUT = Path("data/cache/live_check.json")


def pick_questions(now: datetime) -> list[Question]:
    config = load_config()
    rng = random.Random(now.strftime("%Y-%m-%d"))
    picked: list[Question] = []
    with make_client() as client:
        for venue, count in PICKS.items():
            fetched = fetch_open(
                venue, 200, client, now + timedelta(days=1), now + timedelta(days=30)
            )
            selected = select_questions(fetched, now, config)
            print(f"{venue}: {len(fetched)} fetched, {len(selected)} selected")
            picked += rng.sample(selected, min(count, len(selected)))
    return picked


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set in this shell. Nothing was spent.")
        return 2
    config = load_config()
    now = datetime.now(UTC)
    questions = pick_questions(now)
    if not questions:
        print("no eligible markets")
        return 1

    agent = Agent(config, keep_transcripts=True)
    params = Params.load(config.params_path, config)
    calibrator = Calibrator.load(config.calibrator_path)
    rows = []
    for q in questions:
        if agent.budget_exhausted:
            print("spending cap reached, stopping")
            break
        print(f"\n[{q.venue}] {q.title}\n  closes {q.close_time:%Y-%m-%d}  {q.url}")
        f = forecast_question(q, agent, calibrator, params, datetime.now(UTC), config)
        if f is None:
            print("  no valid sample")
            continue
        probs = "  ".join(f"{s.persona}={s.probability:.2f}" for s in f.samples)
        print(f"  samples: {probs}")
        print(
            f"  raw {f.raw_prob:.3f}  final {f.final_prob:.3f}  market {f.market_prob:.3f}"
            f"  | cost ${f.cost_usd:.3f}  latency {f.latency_s:.0f}s"
            f"  searches {f.n_searches}  parse failures {f.parse_failures}"
        )
        print(f"  paper bet: {f.bet.get('side') or 'none'} ({f.bet.get('reason', '')})")
        rows.append(f.to_dict())

    transcripts = agent.transcripts or []
    passes = len(transcripts) + agent.api_failures
    parsed = sum(t["parsed"] for t in transcripts)
    print("\n=== totals ===")
    print(f"cost            ${agent.cost_usd:.3f} of ${config.max_spend_per_run:.2f} cap")
    print(f"tokens          {agent.input_tokens:,} in / {agent.output_tokens:,} out")
    print(f"searches        {agent.n_searches}")
    print(f"parse rate      {parsed}/{passes}")
    print(f"api failures    {agent.api_failures}")
    print(f"stop reasons    {dict(Counter(t['stop_reason'] for t in transcripts))}")

    sample = next((t for t in transcripts if t["parsed"]), transcripts[0] if transcripts else None)
    if sample:
        print(f"\n=== full reply of one sample ({sample['market_id']}, {sample['persona']}) ===")
        print(sample["text"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "ran_at": now.isoformat(),
                "model": agent.model,
                "cost_usd": agent.cost_usd,
                "input_tokens": agent.input_tokens,
                "output_tokens": agent.output_tokens,
                "n_searches": agent.n_searches,
                "api_failures": agent.api_failures,
                "forecasts": rows,
                "transcripts": transcripts,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nsaved to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
