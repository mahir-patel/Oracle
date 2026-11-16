import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from forecaster import report
from forecaster.config import Config
from forecaster.scoring import brier

CFG = Config()
NOW = datetime(2026, 11, 15, 6, 30, tzinfo=UTC)


def row(
    i: int,
    final: float = 0.7,
    market: float = 0.6,
    outcome: int | None = 1,
    resolved_at: datetime | None = datetime(2026, 9, 1, tzinfo=UTC),
    bet: dict[str, Any] | None = None,
    title: str | None = None,
    url: str = "https://example.com/m",
) -> dict[str, Any]:
    return {
        "type": "forecast",
        "ts": "2026-08-20T00:00:00+00:00",
        "question": {
            "venue": "sim",
            "market_id": f"m{i}",
            "title": title or f"Question {i}?",
            "close_time": (NOW + timedelta(days=i + 1)).isoformat(),
            "url": url,
        },
        "samples": [{"persona": "outside_view", "probability": final}] * 5,
        "raw_prob": final,
        "calibrated_prob": final + 0.05,
        "final_prob": final,
        "market_prob": market,
        "model": "claude-haiku-5-5",
        "cost_usd": 0.15,
        "latency_s": 20.0,
        "n_searches": 12,
        "parse_failures": 1 if i == 0 else 0,
        "bet": bet or {"side": None, "contracts": 0},
        "hash": f"{i:02d}abcdef0123456789",
        "outcome": outcome,
        "resolved_at": resolved_at.isoformat() if outcome is not None and resolved_at else None,
    }


def test_empty_ledger_builds_and_renders() -> None:
    r = report.build([], CFG, NOW)
    assert (r["n_total"], r["n_resolved"], r["n_open"], r["scores"]) == (0, 0, 0, None)
    assert "Not enough resolved questions yet" in report.to_markdown(r)
    page = report.render_site(r, [])
    assert "Not enough resolved questions yet" in page and "No open forecasts." in page


def test_counts_cost_and_scores_use_the_logged_probabilities() -> None:
    yes_bet = {"side": "yes", "contracts": 10, "price": 0.62}
    rows = [row(i, outcome=i % 2, bet=yes_bet if i < 4 else None) for i in range(12)]
    rows += [row(100 + i, outcome=None) for i in range(3)]
    r = report.build(rows, CFG, NOW)
    assert (r["n_total"], r["n_resolved"], r["n_open"], r["n_scored"]) == (15, 12, 3, 12)
    assert r["total_cost_usd"] == pytest.approx(15 * 0.15)
    assert r["parse_failures"] == 1 and r["samples"] == 75
    y = [i % 2 for i in range(12)]
    s = r["scores"]
    assert s["brier_final"] == pytest.approx(brier([0.7] * 12, y))
    assert s["brier_model"] == pytest.approx(brier([0.75] * 12, y))
    assert s["brier_market"] == pytest.approx(brier([0.6] * 12, y))
    assert s["bss"]["n"] == 12 and s["bss"]["ci_low"] <= s["bss"]["bss"] <= s["bss"]["ci_high"]
    # Bets on m0..m3: outcomes 0,1,0,1 -> two losses of 6.20, two wins of 3.80.
    assert r["n_bets"] == 4 and r["pnl_usd"] == pytest.approx(2 * 3.8 - 2 * 6.2)
    assert sum(b["n"] for b in r["calibration"]) == 12


def test_questions_resolved_before_the_cutoff_are_not_scored() -> None:
    before = datetime(2026, 6, 1, tzinfo=UTC)
    # Pre-cutoff: perfect forecasts (as if remembered). Post-cutoff: poor ones.
    rows = [row(i, final=0.99, outcome=1, resolved_at=before) for i in range(10)]
    rows += [row(50 + i, final=0.6, outcome=i % 2) for i in range(10)]
    r = report.build(rows, CFG, NOW)
    assert (r["n_resolved"], r["n_scored"]) == (20, 10)
    assert r["scores"]["brier_final"] == pytest.approx(brier([0.6] * 10, [i % 2 for i in range(10)]))
    assert r["contamination"]["suspicious"] is True
    assert (r["contamination"]["n_pre"], r["contamination"]["n_post"]) == (10, 10)
    assert "⚠️" in report.to_markdown(r) and "⚠️" in report.render_site(r, rows)


def test_headline_makes_no_claim_below_200_questions() -> None:
    good = [row(i, final=0.9 if i % 2 else 0.1, market=0.5, outcome=i % 2) for i in range(60)]
    text = report.headline(report.build(good, CFG, NOW))
    assert "n = 60" in text and "Too few to conclude" in text
    many = [row(i, final=0.9 if i % 2 else 0.1, market=0.5, outcome=i % 2) for i in range(220)]
    assert "Beating the market" in report.headline(report.build(many, CFG, NOW))
    worse = [row(i, final=0.5, market=0.9 if i % 2 else 0.1, outcome=i % 2) for i in range(220)]
    assert "Not shown to beat the market" in report.headline(report.build(worse, CFG, NOW))


def test_site_lists_open_forecasts_and_is_self_contained() -> None:
    rows = [row(1, outcome=None, final=0.31, market=0.44), row(2, outcome=1)]
    rows += [row(10 + i, outcome=i % 2) for i in range(10)]
    page = report.render_site(report.build(rows, CFG, NOW), rows)
    assert '<a href="https://example.com/m" rel="noopener">Question 1?</a>' in page
    assert "31%" in page and "44%" in page and "<code>01abcdef01</code>" in page
    assert "Question 2?" not in page  # resolved, so not in the open table
    assert "<svg" in page and "<circle" in page
    assert "oracle verify" in page and "data/ledger.jsonl" in page
    assert "Not financial advice" in page
    assert 'name="viewport"' in page
    # No scripts and nothing loaded from elsewhere.
    assert "<script" not in page and "<link" not in page and "<img" not in page
    assert 'src="' not in page and "@import" not in page


def test_site_escapes_text_from_the_venues() -> None:
    evil = row(1, outcome=None, title='<script>alert("x")</script>', url="javascript:alert(1)")
    page = report.render_site(report.build([evil], CFG, NOW), [evil])
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "javascript:" not in page


def test_write_report_and_site(tmp_path: Path) -> None:
    rows = [row(i, outcome=i % 2) for i in range(12)]
    r = report.build(rows, CFG, NOW)
    report.write_report(r, tmp_path / "data" / "report.json")
    report.write_site(r, rows, tmp_path / "docs" / "index.html")
    saved = json.loads((tmp_path / "data" / "report.json").read_text())
    assert saved["n_total"] == 12 and saved["generated_at"] == NOW.isoformat()
    assert (tmp_path / "docs" / "index.html").read_text().startswith("<!doctype html>")
