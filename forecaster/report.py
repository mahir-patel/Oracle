"""Metrics over the ledger, a console summary and the public page.

Scores always use the probabilities that were logged at forecast time, and only
questions resolved after the model's training cutoff are scored (rules 1, 4).
"""

from __future__ import annotations

import html
import json
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from forecaster.config import Config
from forecaster.models import ensure_utc
from forecaster.scoring import (
    bootstrap_bss,
    brier,
    calibration_table,
    contamination_check,
    log_loss,
)
from forecaster.sizing import pnl

Row = dict[str, Any]
Report = dict[str, Any]

CREDIBLE_N = 200  # no claim of beating the market below this many resolved questions


def _when(value: str) -> datetime:
    return ensure_utc(datetime.fromisoformat(value))


def build(rows: Sequence[Row], config: Config, now: datetime) -> Report:
    """``rows`` is ``Ledger.joined()``: one forecast per market with its outcome."""
    resolved = [r for r in rows if r["outcome"] is not None]
    clean = [r for r in resolved if _when(r["resolved_at"]) >= config.model_cutoff]
    n = len(rows)

    report: Report = {
        "generated_at": ensure_utc(now).isoformat(),
        "model_cutoff": config.model_cutoff.isoformat(),
        "models": dict(Counter(r["model"] for r in rows)),
        "n_total": n,
        "n_resolved": len(resolved),
        "n_open": n - len(resolved),
        "n_scored": len(clean),
        "total_cost_usd": sum(r.get("cost_usd", 0.0) for r in rows),
        # Rule 5: what the forecasts cost to make.
        "mean_cost_usd": _mean([r.get("cost_usd", 0.0) for r in rows]),
        "mean_latency_s": _mean([r.get("latency_s", 0.0) for r in rows]),
        "mean_searches": _mean([r.get("n_searches", 0) for r in rows]),
        "parse_failures": sum(r.get("parse_failures", 0) for r in rows),
        "samples": sum(len(r.get("samples", [])) for r in rows),
        "scores": None,
        "calibration": [],
        "n_bets": 0,
        "pnl_usd": 0.0,
    }

    if clean:
        y = [r["outcome"] for r in clean]
        final = [r["final_prob"] for r in clean]
        model = [r["calibrated_prob"] for r in clean]
        market = [r["market_prob"] for r in clean]
        bss = bootstrap_bss(final, market, y)
        report["scores"] = {
            "brier_final": brier(final, y),
            "brier_model": brier(model, y),
            "brier_market": brier(market, y),
            "log_loss_final": log_loss(final, y),
            "log_loss_market": log_loss(market, y),
            "bss": bss.to_dict() if bss else None,
        }
        report["calibration"] = calibration_table(final, y)
        bets = [r for r in clean if r["bet"].get("side")]
        report["n_bets"] = len(bets)
        report["pnl_usd"] = sum(pnl(r["bet"], r["outcome"]) for r in bets)

    report["contamination"] = contamination_check(
        [r["final_prob"] for r in resolved],
        [r["outcome"] for r in resolved],
        [_when(r["resolved_at"]) for r in resolved],
        config.model_cutoff,
    )
    return report


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def headline(report: Report) -> str:
    """One honest sentence about skill against the market."""
    scores = report["scores"]
    bss = scores["bss"] if scores else None
    if not bss:
        return f"Not enough resolved questions yet ({report['n_scored']} scored, 10 needed)."
    text = (
        f"Brier skill vs market {bss['bss']:+.3f} "
        f"(95% CI {bss['ci_low']:+.3f} to {bss['ci_high']:+.3f}, n = {bss['n']})."
    )
    if bss["n"] < CREDIBLE_N:
        return text + f" Too few to conclude anything: the bar is n >= {CREDIBLE_N}."
    if bss["ci_low"] > 0:
        return text + " Beating the market, with the whole interval above zero."
    return text + " Not shown to beat the market: the interval includes zero."


def to_markdown(report: Report) -> str:
    lines = [
        "# Oracle report",
        "",
        headline(report),
        "",
        f"- forecasts: {report['n_total']} "
        f"({report['n_resolved']} resolved, {report['n_open']} open, {report['n_scored']} scored)",
        f"- API cost: ${report['total_cost_usd']:.2f}",
    ]
    scores = report["scores"]
    if scores:
        lines += [
            f"- Brier: final {scores['brier_final']:.4f}, model only {scores['brier_model']:.4f}, "
            f"market {scores['brier_market']:.4f}",
            f"- log loss: final {scores['log_loss_final']:.4f}, "
            f"market {scores['log_loss_market']:.4f}",
            f"- paper P&L: ${report['pnl_usd']:+.2f} over {report['n_bets']} bets",
        ]
    if report["samples"]:
        lines.append(
            f"- parse failures: {report['parse_failures']} "
            f"(with {report['samples']} valid samples); "
            f"mean latency {report['mean_latency_s']:.0f} s, "
            f"mean searches {report['mean_searches']:.1f}"
        )
    c = report["contamination"]
    if c["suspicious"]:
        lines.append(
            f"- ⚠️ possible leakage: Brier {c['brier_pre']:.4f} on {c['n_pre']} questions resolved "
            f"before the model cutoff vs {c['brier_post']:.4f} on {c['n_post']} after it"
        )
    return "\n".join(lines) + "\n"


def write_report(report: Report, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


# ---------- the public page ----------

CSS = """
:root{color-scheme:light dark;--bg:#f5f5f7;--fg:#1d1d1f;--muted:#6e6e73;--line:rgba(0,0,0,.1);
--card:#fff;--accent:#06c;--press:rgba(0,0,0,.04)}
@media (prefers-color-scheme:dark){:root{--bg:#000;--fg:#f5f5f7;--muted:#98989d;
--line:rgba(255,255,255,.14);--card:#1c1c1e;--accent:#2997ff;--press:rgba(255,255,255,.06)}}
@media (prefers-contrast:more){:root{--muted:var(--fg);--line:var(--fg)}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);-webkit-font-smoothing:antialiased;
font:100%/1.5 -apple-system,BlinkMacSystemFont,"SF Pro Text",system-ui,"Segoe UI",sans-serif}
main{max-width:54rem;margin:0 auto;padding:3.5rem 1rem 4rem;animation:arrive .5s cubic-bezier(.2,.8,.2,1) both}
@keyframes arrive{from{opacity:0;transform:translateY(.5rem)}}
h1{font-size:clamp(2.25rem,6vw,3.5rem);line-height:1.05;letter-spacing:-.025em;font-weight:700;
margin:0 0 .75rem}
h2{font-size:1.5rem;line-height:1.15;letter-spacing:-.015em;font-weight:600;margin:3.5rem 0 .5rem}
p{margin:.5rem 0}
.muted{color:var(--muted)}
h1+.muted{font-size:1.1875rem;line-height:1.4;max-width:38rem}
.headline,.tile,.scroll,.plot{background:var(--card);border-radius:1.125rem}
.headline{font-size:1.3125rem;line-height:1.35;letter-spacing:-.01em;font-weight:600;
padding:1.5rem;margin:2rem 0 .75rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(9.5rem,1fr));gap:.75rem}
.tile{padding:1.125rem 1.25rem}
.tile b{display:block;font-size:1.75rem;line-height:1.15;letter-spacing:-.02em;font-weight:600;
font-variant-numeric:tabular-nums}
.tile span{color:var(--muted);font-size:.8125rem}
.plot{padding:1rem;margin-top:1rem}
.scroll{overflow-x:auto;margin-top:1rem;padding:.25rem 0}
table{border-collapse:collapse;width:100%;font-size:.9375rem;line-height:1.4}
th,td{text-align:left;padding:.75rem 1.25rem;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:last-child td{border-bottom:0}
tbody tr{transition:background-color .2s ease}
tbody tr:hover{background:var(--press)}
th{color:var(--muted);font-size:.75rem;font-weight:600;white-space:nowrap}
td:first-child{min-width:15rem}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
a{color:var(--accent);text-decoration:none;transition:opacity .1s ease-out}
a:hover{text-decoration:underline}
a:active{opacity:.6}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:.25rem}
code{font:.8125rem ui-monospace,"SF Mono",Menlo,monospace;color:var(--muted)}
p code{color:inherit}
svg{max-width:26rem;width:100%;height:auto;display:block;margin:0 auto}
svg text{fill:var(--muted);font:11px -apple-system,system-ui,sans-serif}
footer{margin-top:4rem;color:var(--muted);font-size:.8125rem;border-top:1px solid var(--line);
padding-top:1rem}
.flow{list-style:none;margin:1rem 0 0;padding:0;display:grid;gap:1.75rem;counter-reset:step}
.flow li{position:relative;background:var(--card);border-radius:1.125rem;padding:1.125rem 1.25rem;
counter-increment:step}
.flow li::before{content:counter(step);display:grid;place-items:center;width:1.75rem;height:1.75rem;
border-radius:50%;background:var(--accent);color:var(--card);font-size:.875rem;font-weight:600;
margin-bottom:.625rem}
.flow b{display:block;font-size:1.0625rem;letter-spacing:-.01em}
.flow span{display:block;color:var(--muted);font-size:.875rem;line-height:1.4;margin-top:.25rem}
.flow li:not(:last-child)::after{content:"\\2193";position:absolute;left:50%;bottom:-1.5rem;
transform:translateX(-50%);color:var(--muted);line-height:1}
@media (min-width:56rem){.flow{grid-template-columns:repeat(5,1fr);gap:1.5rem}
.flow li{padding:1rem}
.flow li:not(:last-child)::after{content:"\\2192";left:auto;right:-1.25rem;bottom:auto;top:50%;
transform:translateY(-50%)}}
@media (prefers-reduced-motion:reduce){main{animation:none}tbody tr,a{transition:none}}
"""


def calibration_svg(table: Sequence[dict[str, Any]]) -> str:
    """Observed rate against mean forecast, one dot per bin sized by its count."""
    size, pad = 340, 46
    span = size - 2 * pad

    def x(p: float) -> float:
        return pad + p * span

    def y(p: float) -> float:
        return size - pad - p * span

    parts = [
        f'<svg viewBox="0 0 {size} {size}" role="img" '
        'aria-label="Calibration: observed rate against forecast">',
        f'<rect x="{pad}" y="{pad}" width="{span}" height="{span}" fill="none" '
        'stroke="var(--line)"/>',
        f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="var(--muted)" '
        'stroke-dasharray="4 4"/>',
    ]
    for tick in (0, 0.5, 1):
        parts.append(
            f'<text x="{x(tick)}" y="{size - pad + 16}" text-anchor="middle">{tick:.0%}</text>'
        )
        parts.append(f'<text x="{pad - 6}" y="{y(tick) + 4}" text-anchor="end">{tick:.0%}</text>')
    parts.append(f'<text x="{size / 2}" y="{size - 4}" text-anchor="middle">forecast</text>')
    parts.append(
        f'<text x="12" y="{size / 2}" text-anchor="middle" '
        f'transform="rotate(-90 12 {size / 2})">observed</text>'
    )
    filled = [b for b in table if b["n"]]
    biggest = max((b["n"] for b in filled), default=1)
    for b in filled:
        radius = 3 + 9 * (b["n"] / biggest) ** 0.5
        parts.append(
            f'<circle cx="{x(b["mean_forecast"]):.1f}" cy="{y(b["observed"]):.1f}" '
            f'r="{radius:.1f}" fill="var(--accent)" fill-opacity="0.75">'
            f'<title>forecast {b["mean_forecast"]:.0%}, observed {b["observed"]:.0%}, '
            f'n = {b["n"]}</title></circle>'
        )
    parts.append("</svg>")
    return "".join(parts)


# The pipeline in plain words, in the order it runs.
STEPS = [
    ("Pick", "Questions are drawn at random from Kalshi and Polymarket, fixed by the date."),
    (
        "Research",
        "An AI searches the web and estimates the chance several times, "
        "without seeing the market price.",
    ),
    ("Combine", "The estimates are averaged, then blended with the market price."),
    ("Record", "The forecast is logged publicly before the outcome is known."),
    ("Score", "Once settled, Oracle and the market are scored against what happened."),
]
FLOW = (
    '<ol class="flow">'
    + "".join(f"<li><b>{title}</b><span>{text}</span></li>" for title, text in STEPS)
    + "</ol>"
)


def _pct(p: float | None) -> str:
    return "–" if p is None else f"{p:.0%}"


def _link(title: str, url: str) -> str:
    text = html.escape(title)
    if url.startswith(("https://", "http://")):
        return f'<a href="{html.escape(url, quote=True)}" rel="noopener">{text}</a>'
    return text


def render_site(report: Report, rows: Sequence[Row]) -> str:
    """One self-contained page: no scripts, no external assets."""
    tiles = [
        (str(report["n_resolved"]), "resolved"),
        (str(report["n_open"]), "open"),
        (f"${report['pnl_usd']:+,.2f}", f"paper P&L, {report['n_bets']} bets"),
        (f"${report['total_cost_usd']:,.2f}", "API cost"),
    ]
    tile_html = "".join(f'<div class="tile"><b>{b}</b><span>{s}</span></div>' for b, s in tiles)

    open_rows = sorted(
        (r for r in rows if r["outcome"] is None), key=lambda r: r["question"]["close_time"]
    )
    body = "".join(
        "<tr>"
        f"<td>{_link(r['question']['title'], r['question'].get('url', ''))}</td>"
        f'<td class="n">{_pct(r["final_prob"])}</td>'
        f'<td class="n">{_pct(r["market_prob"])}</td>'
        f'<td class="n">{r["question"]["close_time"][:10]}</td>'
        f"<td><code>{html.escape(r['hash'][:10])}</code></td>"
        "</tr>"
        for r in open_rows
    )
    table = (
        '<div class="scroll"><table><thead><tr><th>Question</th><th>Oracle</th><th>Market</th>'
        f"<th>Closes</th><th>Ledger hash</th></tr></thead><tbody>{body}</tbody></table></div>"
        if open_rows
        else '<p class="muted">No open forecasts.</p>'
    )
    calibration = (
        calibration_svg(report["calibration"])
        if any(b["n"] for b in report["calibration"])
        else '<p class="muted">Appears once questions have resolved.</p>'
    )
    models = ", ".join(html.escape(m) for m in sorted(report["models"])) or "none yet"
    warning = ""
    if report["contamination"]["suspicious"]:
        warning = (
            '<p class="headline">⚠️ Scores on questions resolved before the model\'s training '
            "cutoff are much better than after it. That pattern suggests the model remembers "
            "outcomes; those questions are excluded from the scores above.</p>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Oracle: AI forecasts vs prediction markets</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Oracle</h1>
<p class="muted">An AI forecaster with web search, scored against Kalshi and Polymarket prices.
Every forecast is logged before the outcome is known. Paper trading only.</p>
<p class="headline">{html.escape(headline(report))}</p>
{warning}
<div class="tiles">{tile_html}</div>

<h2>How it works</h2>
<p class="muted">The same five steps run automatically every day.</p>
{FLOW}

<h2>Calibration</h2>
<p class="muted">Of the questions Oracle gave about X%, how many happened? Dots on the dashed
line are perfectly calibrated; bigger dots hold more questions.</p>
<div class="plot">{calibration}</div>

<h2>Open forecasts</h2>
{table}

<h2>How to verify</h2>
<p>Each forecast is one line of <a href="https://github.com/mahir-patel/Oracle/blob/main/data/ledger.jsonl"><code>data/ledger.jsonl</code></a>.
Every line carries the hash of the line before it, so editing or deleting an old forecast
breaks the chain. Clone the repository and run <code>oracle verify</code> to recompute it.
The git history of that file shows when each forecast was committed.</p>

<footer>
<p>Model: {models}. Questions resolved before {report["model_cutoff"][:10]} are never scored.
Updated {report["generated_at"][:16].replace("T", " ")} UTC.</p>
<p>Not financial advice. No real money is traded; paper results ignore slippage and
market impact.</p>
</footer>
</main>
</body>
</html>
"""


def write_site(report: Report, rows: Sequence[Row], path: str | Path = "docs/index.html") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_site(report, rows))
