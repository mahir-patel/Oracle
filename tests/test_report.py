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


def sample_rows() -> list[dict[str, Any]]:
    rows = [row(1, outcome=None, final=0.31, market=0.44), row(2, outcome=1)]
    rows += [row(10 + i, outcome=i % 2) for i in range(10)]
    for r in rows:
        r["prev_hash"] = "ff" * 32
        r["question"]["venue"] = "polymarket"
        r["samples"] = [
            {"persona": "outside_view", "probability": 0.2, "rationale": "Base rate is low."},
            {"persona": "status_quo", "probability": 0.4, "rationale": "Nothing has changed."},
        ]
    return rows


def test_site_has_four_sections_behind_tabs() -> None:
    rows = sample_rows()
    page = report.render_site(report.build(rows, CFG, NOW), rows)
    for key in ("overview", "how", "forecasts", "verify"):
        assert f'data-tab="{key}"' in page and f'data-panel="{key}"' in page
    assert 'name="viewport"' in page
    assert "oracle verify" in page and "data/ledger.jsonl" in page
    assert "Not financial advice" in page
    assert "<svg" in page and "<circle" in page  # the calibration plot


def test_site_is_one_file_with_no_external_requests() -> None:
    rows = sample_rows()
    page = report.render_site(report.build(rows, CFG, NOW), rows)
    assert page.count("<script>") == 1  # one inline script
    assert "<script src" not in page and 'src="' not in page
    assert "<link" not in page and "<img" not in page and "<iframe" not in page
    assert "@import" not in page and "url(" not in page
    assert "fetch(" not in page and "XMLHttpRequest" not in page
    # The design rules ban long dashes anywhere on the page.
    assert "\u2014" not in page and "\u2013" not in page


def test_every_forecast_is_listed_with_its_numbers_and_reasons() -> None:
    rows = sample_rows()
    page = report.render_site(report.build(rows, CFG, NOW), rows)
    assert page.count('<details class="fc"') == 12
    assert page.count('data-state="open"') == 1 and page.count('data-state="resolved"') == 11
    assert 'data-default="open"' in page
    one = page[page.index("Question 1?</span>") :].split("</details>")[0]
    assert "<b>31%</b> <span>vs 44%</span>" in one and "Closes" in one
    assert "Base rate is low." in one and "Outside view" in one and "Status quo" in one
    assert "<code>01abcdef01</code>" in one
    assert '<a href="https://example.com/m" rel="noopener">View on Polymarket</a>' in one
    assert "Resolved YES" in page and "Resolved NO" in page


def test_pipeline_walks_a_real_forecast_through_five_steps() -> None:
    rows = sample_rows()
    page = report.render_site(report.build(rows, CFG, NOW), rows)
    assert page.count('class="stage-btn"') == 5 and page.count('class="stage"') == 5
    order = [page.index(f"<b>{s}</b><span>") for s in ("Pick", "Research", "Combine", "Record", "Score")]
    assert order == sorted(order)
    # The open forecast furthest from the market is the one shown: Question 1.
    hero = page[page.index('class="feature"') :].split("</div></div>")[0]
    assert "Largest open disagreement" in hero and "Question 1?" in hero
    assert "12 questions were drawn on 20 Aug 2026" in page
    assert "The 2 estimates Oracle made" in page
    assert "<code>ffffffffff</code>" in page and "<code>01abcdef01</code>" in page
    assert "It is scored" in page  # still open, so no error bars yet

    resolved = [r for r in rows if r["outcome"] is not None]
    page = report.render_site(report.build(resolved, CFG, NOW), resolved)
    assert "Largest disagreement" in page and 'class="errs"' in page
    assert "The answer was" in page


def test_site_escapes_text_from_the_venues_and_the_model() -> None:
    evil = row(1, outcome=None, title='<script>alert("x")</script>', url="javascript:alert(1)")
    evil["samples"] = [
        {"persona": "<b>x</b>", "probability": 0.5, "rationale": "<img src=x onerror=alert(1)>"}
    ]
    page = report.render_site(report.build([evil], CFG, NOW), [evil])
    assert page.count("<script>") == 1 and "&lt;script&gt;" in page
    assert "<img" not in page and "&lt;img" in page and "<b>x</b>" not in page
    assert "javascript:" not in page


def test_write_report_and_site(tmp_path: Path) -> None:
    rows = [row(i, outcome=i % 2) for i in range(12)]
    r = report.build(rows, CFG, NOW)
    report.write_report(r, tmp_path / "data" / "report.json")
    report.write_site(r, rows, tmp_path / "docs" / "index.html")
    saved = json.loads((tmp_path / "data" / "report.json").read_text())
    assert saved["n_total"] == 12 and saved["generated_at"] == NOW.isoformat()
    assert (tmp_path / "docs" / "index.html").read_text().startswith("<!doctype html>")
