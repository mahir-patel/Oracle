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
#
# One self-contained file: inline CSS and one inline script, no external
# requests. With scripts off, every section is shown stacked. Text that comes
# from the venues or the model is always escaped.

REPO_URL = "https://github.com/mahir-patel/Oracle"

CSS = """
:root{color-scheme:light dark;--bg:#f6f7f8;--surface:#fdfdfd;--fg:#15171c;--muted:#5b616e;
--line:rgba(21,23,28,.11);--accent:#0a7a58;--on-accent:#fdfdfd;--soft:rgba(10,122,88,.12);
--nav:rgba(246,247,248,.78);--r:14px;--ease:cubic-bezier(.16,1,.3,1)}
@media (prefers-color-scheme:dark){:root{--bg:#0c0d10;--surface:#15171b;--fg:#eceef2;
--muted:#939aa8;--line:rgba(236,238,242,.13);--accent:#3fd19c;--on-accent:#0c0d10;
--soft:rgba(63,209,156,.14);--nav:rgba(12,13,16,.72)}}
@media (prefers-contrast:more){:root{--muted:var(--fg);--line:var(--fg)}}
*{box-sizing:border-box}
[hidden]{display:none!important}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);-webkit-font-smoothing:antialiased;
font:100%/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:72rem;margin:0 auto;padding:0 1.25rem}
a{color:var(--accent)}
code,.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
h1,h2,h3,p{margin:0}
h1{font-size:clamp(2.25rem,5.2vw,4rem);line-height:1.02;letter-spacing:-.035em;font-weight:680}
h2{font-size:clamp(1.6rem,3vw,2.25rem);line-height:1.1;letter-spacing:-.025em;font-weight:650}
.lede{font-size:1.125rem;line-height:1.5;color:var(--muted);max-width:34rem}
.muted{color:var(--muted)}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px;border-radius:6px}

/* navigation */
.nav{position:sticky;top:0;z-index:10;background:var(--nav);border-bottom:1px solid var(--line);
backdrop-filter:blur(18px) saturate(160%);-webkit-backdrop-filter:blur(18px) saturate(160%)}
@media (prefers-reduced-transparency:reduce){.nav{background:var(--bg);backdrop-filter:none;
-webkit-backdrop-filter:none}}
.nav .wrap{display:flex;align-items:center;gap:1.5rem;height:3.75rem}
.brand{font-weight:700;letter-spacing:-.02em;font-size:1.0625rem}
.tabs{display:none;gap:.25rem;overflow-x:auto;scrollbar-width:none;margin-left:auto}
.js .tabs{display:flex}
.tabs::-webkit-scrollbar{display:none}
.tab,.chip,.btn{font:inherit;cursor:pointer;border-radius:999px;border:1px solid transparent;
white-space:nowrap;transition:background-color .2s var(--ease),color .2s var(--ease),
transform .12s var(--ease),border-color .2s var(--ease)}
.tab{flex:none;background:none;color:var(--muted);padding:.4rem .85rem;font-size:.9375rem;font-weight:550}
.tab:hover{color:var(--fg)}
.tab[aria-selected=true]{background:var(--fg);color:var(--bg)}
.tab:active,.chip:active,.btn:active{transform:scale(.97)}

/* panels */
.panel{padding:3.5rem 0 4.5rem}
.panel+.panel{border-top:1px solid var(--line)}
.js .panel+.panel{border-top:0}
.js .panel.enter{animation:rise .5s var(--ease) both}
@keyframes rise{from{opacity:0;transform:translateY(.75rem)}}
.sub{margin-top:.75rem}

/* overview */
.hero{display:grid;gap:2.5rem;align-items:center}
@media (min-width:60rem){.hero{grid-template-columns:1.15fr .85fr;gap:4rem}}
.hero .lede{margin-top:1.25rem}
.cta{display:none;gap:.75rem;margin-top:2rem;flex-wrap:wrap}
.js .cta{display:flex}
.btn{padding:.7rem 1.25rem;font-weight:600;font-size:1rem;background:var(--accent);
color:var(--on-accent);text-decoration:none}
.btn.ghost{background:none;color:var(--fg);border-color:var(--line)}
.btn.ghost:hover{border-color:var(--fg)}
.feature{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);
padding:1.5rem;box-shadow:0 18px 50px -28px rgba(10,60,45,.35)}
.eyebrow{font-size:.75rem;font-weight:650;letter-spacing:.09em;text-transform:uppercase;
color:var(--muted)}
.feature h3{font-size:1.1875rem;line-height:1.3;letter-spacing:-.012em;margin:.5rem 0 1.5rem;
font-weight:620}
.pair{display:flex;justify-content:space-between;margin-top:1rem;font-size:.875rem;
color:var(--muted)}
.pair b{display:block;font-size:2.25rem;line-height:1;letter-spacing:-.03em;color:var(--fg);
font-variant-numeric:tabular-nums;font-weight:650}
.pair .o b{color:var(--accent)}
.pair .m{text-align:right}
.verdict{margin-top:4.5rem;font-size:clamp(1.25rem,2.4vw,1.75rem);line-height:1.3;
letter-spacing:-.02em;font-weight:600;max-width:52rem}
.stats{display:grid;grid-template-columns:repeat(2,1fr);margin-top:2rem;border-top:1px solid var(--line)}
@media (min-width:44rem){.stats{grid-template-columns:repeat(4,1fr)}}
.stat{padding:1.25rem 1rem 0 0}
.stat b{display:block;font-size:2rem;line-height:1.1;letter-spacing:-.03em;font-weight:650;
font-variant-numeric:tabular-nums}
.stat span{color:var(--muted);font-size:.875rem}
.calib{display:grid;gap:2rem;margin-top:4.5rem;align-items:center}
@media (min-width:52rem){.calib{grid-template-columns:22rem 1fr;gap:3.5rem}}
.calib p{margin-top:.75rem;max-width:32rem}
.plot svg{width:100%;height:auto;display:block;max-width:24rem}
.plot svg text{fill:var(--muted);font:11px system-ui,sans-serif}
.blank{border:1px dashed var(--line);border-radius:var(--r);padding:1.5rem;color:var(--muted)}

/* the two-marker scale: filled = Oracle, ring = market */
.duo{position:relative;display:block;height:1.25rem}
.duo::before{content:"";position:absolute;left:0;right:0;top:50%;height:1px;background:var(--line)}
.duo .gap{position:absolute;left:var(--lo);width:var(--w);top:calc(50% - 2px);height:4px;
border-radius:2px;background:var(--soft);transform-origin:left}
.duo .mk{position:absolute;top:50%;width:.8rem;height:.8rem;margin:-.4rem 0 0 -.4rem;
border-radius:50%}
.duo .o{left:var(--o);background:var(--accent)}
.duo .m{left:var(--m);background:var(--surface);border:2px solid var(--muted)}
.duo.big{height:2rem}
.duo.big .mk{width:1.1rem;height:1.1rem;margin:-.55rem 0 0 -.55rem}
.enter .feature .mk,.is-active .duo .mk{animation:drop .6s var(--ease) both .15s}
.enter .feature .gap,.is-active .duo .gap{animation:grow .7s var(--ease) both .45s}
@keyframes drop{from{opacity:0;transform:translateY(-.9rem)}}
@keyframes grow{from{transform:scaleX(0)}}
@keyframes pop{from{opacity:0;transform:scale(.4)}}
.key{display:flex;gap:1.25rem;font-size:.8125rem;color:var(--muted);align-items:center}
.key i{display:inline-block;width:.7rem;height:.7rem;border-radius:50%;margin-right:.4rem;
vertical-align:-1px}
.key .o{background:var(--accent)}
.key .m{border:2px solid var(--muted)}

/* how it works */
.pipe{display:grid;gap:1.5rem;margin-top:2.5rem}
@media (min-width:56rem){.pipe{grid-template-columns:17rem 1fr;gap:2.5rem;align-items:start}}
.pipe-nav{display:none;gap:.5rem;overflow-x:auto;scrollbar-width:none}
.js .pipe-nav{display:flex}
@media (min-width:56rem){.js .pipe-nav{flex-direction:column;gap:0;overflow:visible}}
.stage-btn{flex:none;position:relative;font:inherit;text-align:left;cursor:pointer;background:none;border:0;
color:var(--muted);padding:.6rem .9rem;border-radius:10px;white-space:nowrap;overflow:hidden;
transition:color .2s var(--ease),background-color .2s var(--ease)}
.stage-btn b{display:block;font-size:1.0625rem;letter-spacing:-.01em;font-weight:620}
.stage-btn span{display:none;font-size:.875rem;line-height:1.4;margin-top:.15rem;white-space:normal}
@media (min-width:56rem){.stage-btn{padding:1rem 1.1rem}.stage-btn span{display:block}}
.stage-btn:hover{color:var(--fg)}
.stage-btn[aria-selected=true]{color:var(--fg);background:var(--surface)}
.stage-btn::after{content:"";position:absolute;left:0;bottom:0;height:2px;width:100%;
background:var(--accent);transform:scaleX(0);transform-origin:left}
.stage-btn[aria-selected=true]::after{transform:scaleX(1)}
.stage-btn.auto::after{animation:fill 7s linear both}
.pipe:hover .stage-btn.auto::after,.pipe:focus-within .stage-btn.auto::after{animation-play-state:paused}
@keyframes fill{from{transform:scaleX(0)}to{transform:scaleX(1)}}
.stage{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);
padding:1.5rem;min-height:23rem}
.stage+.stage{margin-top:1rem}
.js .stage+.stage{margin-top:0}
@media (min-width:44rem){.stage{padding:2rem}}
.stage h3{font-size:1.375rem;letter-spacing:-.02em;font-weight:640}
.stage>p{color:var(--muted);margin-top:.5rem;max-width:36rem}
.viz{margin-top:1.75rem}
.cap{font-size:.875rem;color:var(--muted);margin-top:1rem}
.cap strong{color:var(--fg);font-weight:600}
.grid{display:grid;grid-template-columns:repeat(15,1fr);gap:.4rem;max-width:36rem}
.grid i{aspect-ratio:1;border-radius:5px;background:var(--line)}
.grid i.on{background:var(--accent)}
.grid i.me{outline:2px solid var(--accent);outline-offset:2px}
.is-active .grid i.on{animation:pop .5s var(--ease) both;animation-delay:calc(var(--i)*90ms + .2s)}
.srow{display:grid;grid-template-columns:8.5rem 1fr 3rem;gap:.75rem;align-items:center;
padding:.55rem 0;font-size:1rem}
.srow .ax{position:relative;height:1rem}
.srow .ax::before{content:"";position:absolute;left:0;right:0;top:50%;height:1px;background:var(--line)}
.srow .ax i{position:absolute;top:50%;left:var(--p);width:.7rem;height:.7rem;
margin:-.35rem 0 0 -.35rem;border-radius:50%;background:var(--accent)}
.srow b{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.is-active .srow .ax i{animation:drop .55s var(--ease) both;animation-delay:calc(var(--i)*140ms + .2s)}
.is-active .srow b{animation:rise .5s var(--ease) both;animation-delay:calc(var(--i)*140ms + .35s)}
.ends{display:flex;justify-content:space-between;font-size:.75rem;color:var(--muted);margin-top:.25rem}
.blend{position:relative;height:7.5rem;margin:0 .6rem}
.blend::before{content:"";position:absolute;left:0;right:0;top:3.75rem;height:1px;background:var(--line)}
.blend .pt{position:absolute;top:3.75rem;left:var(--p);width:.5rem;height:.5rem;
margin:-.25rem 0 0 -.25rem;border-radius:50%;background:var(--muted);opacity:.45}
.blend .tag{position:absolute;left:var(--p);transform:translateX(-50%);font-size:.8125rem;
white-space:nowrap;text-align:center;color:var(--muted)}
.blend .tag b{display:block;font-size:1.0625rem;color:var(--fg);font-variant-numeric:tabular-nums}
.blend .tag::after{content:"";position:absolute;left:50%;width:1px;background:var(--line)}
.blend .t1{top:0}.blend .t1::after{top:100%;height:1.1rem}
.blend .t2{top:4.6rem}.blend .t2::after{bottom:100%;height:.5rem}
.blend .dot{position:absolute;top:3.75rem;left:var(--p);width:1rem;height:1rem;
margin:-.5rem 0 0 -.5rem;border-radius:50%}
.blend .model{background:var(--surface);border:2px solid var(--accent)}
.blend .market{background:var(--surface);border:2px solid var(--muted)}
.blend .final{background:var(--accent);width:1.3rem;height:1.3rem;margin:-.65rem 0 0 -.65rem}
.is-active .blend .pt{animation:pop .4s var(--ease) both;animation-delay:calc(var(--i)*70ms + .1s)}
.is-active .blend .model,.is-active .blend .t1{animation:drop .5s var(--ease) both .6s}
.is-active .blend .market{animation:drop .5s var(--ease) both .9s}
.is-active .blend .final{animation:pop .6s var(--ease) both 1.3s}
.finalnote{font-size:1.0625rem;margin-top:.5rem}
.finalnote b{color:var(--accent);font-variant-numeric:tabular-nums}
.chain{display:flex;align-items:center;flex-wrap:wrap;gap:.5rem 0}
.block{border:1px solid var(--line);border-radius:10px;padding:1rem 1.25rem;font-size:.875rem;
color:var(--muted);background:var(--bg)}
.block code{display:block;font-size:1.125rem;color:var(--fg);margin-top:.25rem}
.block.now{border-color:var(--accent);background:var(--soft)}
.block.next{border-style:dashed;background:none}
.join{width:3rem;height:2px;background:var(--muted);transform-origin:left;opacity:.6}
.is-active .block{animation:rise .5s var(--ease) both}
.is-active .block.now{animation-delay:.5s}.is-active .block.next{animation-delay:1s}
.is-active .join{animation:grow .4s var(--ease) both .35s}
.is-active .join+.block+.join{animation-delay:.85s}
.errs{display:grid;gap:.6rem;max-width:28rem}
.err{display:grid;grid-template-columns:5rem 1fr 3.5rem;gap:.75rem;align-items:center;font-size:.9375rem}
.err i{height:.6rem;border-radius:3px;background:var(--muted);width:var(--w);min-width:2px;
transform-origin:left}
.err.o i{background:var(--accent)}
.err b{text-align:right;font-variant-numeric:tabular-nums}
.is-active .err i{animation:grow .7s var(--ease) both .3s}

/* forecasts */
.bar{display:none;flex-wrap:wrap;gap:.5rem 1.5rem;align-items:center;margin-top:2rem}
.js .bar{display:flex}
.group{display:flex;gap:.35rem;align-items:center}
.group>span{font-size:.8125rem;color:var(--muted);margin-right:.25rem}
.chip{background:none;border-color:var(--line);color:var(--fg);padding:.35rem .8rem;font-size:.875rem}
.chip:hover{border-color:var(--fg)}
.chip[aria-pressed=true]{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.bar .key{margin-left:auto}
.list{margin-top:1.25rem;border-top:1px solid var(--line)}
.fc{border-bottom:1px solid var(--line)}
.fc summary{display:grid;grid-template-columns:1fr auto;gap:.5rem 1.25rem;align-items:center;
padding:1rem .25rem;cursor:pointer;list-style:none;transition:background-color .2s var(--ease)}
.fc summary::-webkit-details-marker{display:none}
.fc summary:hover{background:var(--soft)}
.fc-title{font-weight:560;line-height:1.35}
.fc-viz{grid-column:1/-1}
.fc-num{font-variant-numeric:tabular-nums;white-space:nowrap;text-align:right}
.fc-num b{color:var(--accent);font-size:1.0625rem}
.fc-num span,.fc-when{color:var(--muted);font-size:.875rem}
.fc-when{grid-column:1/-1;white-space:nowrap}
@media (min-width:56rem){.fc summary{grid-template-columns:minmax(0,1fr) 15rem 7.5rem 8.5rem;
padding:1.1rem .5rem}.fc-viz,.fc-when{grid-column:auto}.fc-when{text-align:right}}
.fc-body{padding:.25rem .25rem 1.5rem;max-width:52rem}
@media (min-width:56rem){.fc-body{padding:.25rem .5rem 1.75rem}}
.fc[open] .fc-body{animation:rise .4s var(--ease) both}
.fc-body>p{color:var(--muted);font-size:.9375rem}
.samples{display:grid;gap:1rem;margin:1.25rem 0}
@media (min-width:44rem){.samples{grid-template-columns:1fr 1fr;gap:1.25rem 2rem}}
.sample h4{margin:0;font-size:.875rem;font-weight:620;display:flex;justify-content:space-between}
.sample h4 b{color:var(--accent);font-variant-numeric:tabular-nums}
.sample p{font-size:.9375rem;color:var(--muted);margin-top:.25rem}
.meta{display:flex;flex-wrap:wrap;gap:.35rem 1.5rem;font-size:.875rem;color:var(--muted)}
.meta code{color:var(--fg)}

/* verify */
.verify{display:grid;gap:2.5rem;margin-top:2rem}
@media (min-width:56rem){.verify{grid-template-columns:1fr 1fr;gap:4rem}}
.verify p{margin-top:.75rem;color:var(--muted);max-width:34rem}
pre{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:var(--r);
padding:1.25rem;overflow-x:auto;font-size:.875rem;line-height:1.7}
footer{border-top:1px solid var(--line);padding:1.5rem 0 3rem;color:var(--muted);font-size:.8125rem}
footer p+p{margin-top:.35rem}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;
transition:none!important}}
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


JS = r"""
(function(){
var d=document,h=d.documentElement;h.classList.add('js');
var still=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function all(s,c){return [].slice.call((c||d).querySelectorAll(s))}
var tabs=all('[role=tab][data-tab]'),panels=all('[data-panel]'),counted=false;

function countUp(){
  if(still)return;
  all('[data-count]').forEach(function(el){
    var to=parseFloat(el.dataset.count),dec=+(el.dataset.dec||0),pre=el.dataset.pre||'',t0=null;
    function step(t){
      if(!t0)t0=t;
      var p=Math.min((t-t0)/900,1),e=1-Math.pow(1-p,3);
      el.textContent=pre+(to*e).toFixed(dec).replace(/\B(?=(\d{3})+(?!\d))/g,',');
      if(p<1)requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  });
}

var btns=all('.stage-btn'),stages=all('.stage'),auto=!still;
function stage(i,run){
  btns.forEach(function(b,j){
    var on=j===i;
    b.setAttribute('aria-selected',on?'true':'false');b.tabIndex=on?0:-1;
    b.classList.remove('auto');
    if(on&&auto&&run){void b.offsetWidth;b.classList.add('auto');}
  });
  stages.forEach(function(s,j){
    var on=j===i;s.hidden=!on;s.classList.remove('is-active');
    if(on){void s.offsetWidth;s.classList.add('is-active');}
  });
}
btns.forEach(function(b,i){
  b.addEventListener('click',function(){auto=false;stage(i,false);});
  b.addEventListener('animationend',function(e){
    if(e.animationName==='fill'&&auto)stage((i+1)%btns.length,true);
  });
  b.addEventListener('keydown',function(e){
    var k=(e.key==='ArrowDown'||e.key==='ArrowRight')?1:(e.key==='ArrowUp'||e.key==='ArrowLeft')?-1:0;
    if(!k)return;
    e.preventDefault();auto=false;
    var n=(i+k+btns.length)%btns.length;stage(n,false);btns[n].focus();
  });
});

function show(id,user){
  if(!panels.some(function(p){return p.dataset.panel===id;}))id='overview';
  tabs.forEach(function(t){
    var on=t.dataset.tab===id;
    t.setAttribute('aria-selected',on?'true':'false');t.tabIndex=on?0:-1;
  });
  panels.forEach(function(p){
    var on=p.dataset.panel===id;p.hidden=!on;p.classList.remove('enter');
    if(on){void p.offsetWidth;p.classList.add('enter');}
  });
  if(user)history.replaceState(null,'','#'+id);
  window.scrollTo(0,0);
  if(id==='overview'&&!counted){counted=true;countUp();}
  if(btns.length){
    if(id==='how')stage(0,true);
    else btns.forEach(function(b){b.classList.remove('auto');});
  }
}
tabs.forEach(function(t,i){
  t.addEventListener('click',function(){show(t.dataset.tab,true);});
  t.addEventListener('keydown',function(e){
    var k=e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0;
    if(!k)return;
    var n=tabs[(i+k+tabs.length)%tabs.length];n.focus();show(n.dataset.tab,true);
  });
});
all('[data-goto]').forEach(function(a){
  a.addEventListener('click',function(e){e.preventDefault();show(a.dataset.goto,true);});
});
window.addEventListener('hashchange',function(){show(location.hash.slice(1),false);});

var list=d.querySelector('[data-list]');
if(list){
  var rows=all('.fc',list),empty=d.querySelector('[data-empty]'),state=list.dataset.default,sort='close';
  function apply(){
    var n=0;
    rows.slice().sort(function(a,b){
      if(sort==='gap')return b.dataset.gap-a.dataset.gap;
      return a.dataset.close<b.dataset.close?-1:a.dataset.close>b.dataset.close?1:0;
    }).forEach(function(r){list.appendChild(r);});
    rows.forEach(function(r){var on=state==='all'||r.dataset.state===state;r.hidden=!on;if(on)n++;});
    if(empty)empty.hidden=n>0;
    all('[data-filter]').forEach(function(b){b.setAttribute('aria-pressed',b.dataset.filter===state?'true':'false');});
    all('[data-sort]').forEach(function(b){b.setAttribute('aria-pressed',b.dataset.sort===sort?'true':'false');});
  }
  all('[data-filter]').forEach(function(b){b.addEventListener('click',function(){state=b.dataset.filter;apply();});});
  all('[data-sort]').forEach(function(b){b.addEventListener('click',function(){sort=b.dataset.sort;apply();});});
  apply();
}
show(location.hash.slice(1)||'overview',false);
})();
"""

PERSONA_LABELS = {
    "outside_view": "Outside view",
    "inside_view": "Inside view",
    "devils_advocate": "Devil's advocate",
    "resolution_lawyer": "Rules lawyer",
    "status_quo": "Status quo",
}
VENUE_LABELS = {"kalshi": "Kalshi", "polymarket": "Polymarket"}

# The pipeline in plain words, in the order it runs: (name, one line, fuller text).
STAGES = [
    (
        "Pick",
        "Questions are drawn at random.",
        "Each day a few open questions are drawn at random from Kalshi and Polymarket. "
        "The draw is fixed by the date, so nobody can choose easy ones.",
    ),
    (
        "Research",
        "An AI estimates the chance, several times.",
        "An AI model searches the web and estimates the chance of YES, each time from a "
        "different angle. It is never shown the market price.",
    ),
    (
        "Combine",
        "The estimates become one number.",
        "The estimates are averaged into one probability, which is then blended with the "
        "market price.",
    ),
    (
        "Record",
        "It is logged before the outcome.",
        "The forecast goes into a public log before the outcome is known. Each entry carries "
        "the fingerprint of the one before it, so an old entry cannot be changed unnoticed.",
    ),
    (
        "Score",
        "Oracle and the market are compared.",
        "When the question is settled, Oracle and the market are both scored against what "
        "happened. The smaller error wins.",
    ),
]


def _pct(p: float | None) -> str:
    return "n/a" if p is None else f"{p:.0%}"


def _esc(text: Any) -> str:
    return html.escape(str(text))


def _day(iso: str) -> str:
    """'2026-11-17T...' -> '17 Nov 2026'."""
    return _when(iso).strftime("%d %b %Y").lstrip("0")


def _market_link(q: dict[str, Any]) -> str:
    url = str(q.get("url") or "")
    if not url.startswith(("https://", "http://")):
        return ""
    venue = VENUE_LABELS.get(q.get("venue", ""), "the market")
    return f'<a href="{html.escape(url, quote=True)}" rel="noopener">View on {venue}</a>'


def _duo(oracle: float, market: float | None, big: bool = False) -> str:
    """A 0-100% line with Oracle (filled) and the market (ring) marked on it."""
    o = oracle * 100
    m = o if market is None else market * 100
    label = f"Oracle {o:.0f} percent, market {m:.0f} percent"
    return (
        f'<span class="duo{" big" if big else ""}" role="img" aria-label="{label}" '
        f'style="--o:{o:.1f}%;--m:{m:.1f}%;--lo:{min(o, m):.1f}%;--w:{abs(o - m):.1f}%">'
        '<i class="gap"></i><i class="mk m"></i><i class="mk o"></i></span>'
    )


def _featured(rows: Sequence[Row]) -> Row | None:
    """The open forecast that disagrees most with the market (else any forecast)."""
    priced = [r for r in rows if r.get("market_prob") is not None]
    pool = [r for r in priced if r["outcome"] is None] or priced
    return max(pool, key=lambda r: abs(r["final_prob"] - r["market_prob"]), default=None)


def _feature_card(row: Row | None) -> str:
    if row is None:
        return '<div class="blank">The first forecasts appear after the next daily run.</div>'
    label = "Largest open disagreement" if row["outcome"] is None else "Largest disagreement"
    return (
        '<div class="feature">'
        f'<span class="eyebrow">{label}</span>'
        f"<h3>{_esc(row['question']['title'])}</h3>"
        f"{_duo(row['final_prob'], row['market_prob'], big=True)}"
        '<div class="pair">'
        f'<span class="o"><b>{_pct(row["final_prob"])}</b>Oracle</span>'
        f'<span class="m"><b>{_pct(row["market_prob"])}</b>Market</span>'
        "</div></div>"
    )


def _stage_visual(index: int, row: Row, rows: Sequence[Row]) -> str:
    """The picture for one pipeline stage, drawn from a real ledger entry."""
    q = row["question"]
    samples = row.get("samples", [])
    if index == 0:
        same_day = sum(1 for r in rows if r["ts"][:10] == row["ts"][:10])
        picked = {(i * 13 + 7) % 60: i for i in range(min(same_day, 60))}
        cells = "".join(
            f'<i class="on{" me" if picked[c] == 0 else ""}" style="--i:{picked[c]}"></i>'
            if c in picked
            else "<i></i>"
            for c in range(60)
        )
        noun = "question was" if same_day == 1 else "questions were"
        return (
            f'<div class="grid" aria-hidden="true">{cells}</div>'
            f'<p class="cap">{same_day} {noun} drawn on {_day(row["ts"])}. '
            f"The outlined one: <strong>{_esc(q['title'])}</strong></p>"
        )
    if index == 1:
        lines = "".join(
            f'<div class="srow" style="--i:{i}">'
            f"<span>{_esc(PERSONA_LABELS.get(s['persona'], s['persona']))}</span>"
            f'<span class="ax"><i style="--p:{s["probability"] * 100:.1f}%"></i></span>'
            f"<b>{_pct(s['probability'])}</b></div>"
            for i, s in enumerate(samples)
        )
        return (
            f"{lines}"
            '<div class="srow"><span></span><span class="ends"><span>0%</span><span>100%</span>'
            "</span><b></b></div>"
            f'<p class="cap">The {len(samples)} estimates Oracle made for this question.</p>'
        )
    if index == 2:
        model, market, final = row["calibrated_prob"], row.get("market_prob"), row["final_prob"]
        points = "".join(
            f'<i class="pt" style="--p:{s["probability"] * 100:.1f}%;--i:{i}"></i>'
            for i, s in enumerate(samples)
        )
        market_bits = (
            f'<i class="dot market" style="--p:{market * 100:.1f}%"></i>'
            f'<span class="tag t2" style="--p:{market * 100:.1f}%"><b>{_pct(market)}</b>Market</span>'
            if market is not None
            else ""
        )
        return (
            f'<div class="blend">{points}'
            f'<span class="tag t1" style="--p:{model * 100:.1f}%"><b>{_pct(model)}</b>Oracle alone</span>'
            f'<i class="dot model" style="--p:{model * 100:.1f}%"></i>'
            f"{market_bits}"
            f'<i class="dot final" style="--p:{final * 100:.1f}%"></i></div>'
            f'<p class="finalnote">Final forecast: <b>{_pct(final)}</b></p>'
            '<p class="cap">Small grey dots are the single estimates. The solid dot is the '
            "blend of Oracle and the market.</p>"
        )
    if index == 3:
        prev = str(row.get("prev_hash") or "")
        first = not prev.strip("0")
        before = (
            '<div class="block">Start of the record<code>first entry</code></div>'
            if first
            else f'<div class="block">Entry before<code>{_esc(prev[:10])}</code></div>'
        )
        return (
            f'<div class="chain">{before}<i class="join"></i>'
            f'<div class="block now">This forecast<code>{_esc(row["hash"][:10])}</code></div>'
            '<i class="join"></i><div class="block next">Next entry<code>builds on this</code></div>'
            "</div>"
            f'<p class="cap">Logged on {_day(row["ts"])}. The codes are the first 10 characters '
            "of each entry's fingerprint in the ledger.</p>"
        )
    outcome = row["outcome"]
    if outcome is None or row.get("market_prob") is None:
        return (
            f"{_duo(row['final_prob'], row.get('market_prob'), big=True)}"
            '<div class="ends"><span>0% (NO)</span><span>100% (YES)</span></div>'
            f'<p class="cap">This question closes on {_day(q["close_time"])}. It is scored '
            "once the result is known: whoever sits closer to the outcome wins.</p>"
        )
    err_o = (row["final_prob"] - outcome) ** 2
    err_m = (row["market_prob"] - outcome) ** 2
    scale = max(err_o, err_m, 1e-9)
    return (
        '<div class="errs">'
        f'<div class="err o"><span>Oracle</span><i style="--w:{err_o / scale:.1%}"></i>'
        f"<b>{err_o:.3f}</b></div>"
        f'<div class="err"><span>Market</span><i style="--w:{err_m / scale:.1%}"></i>'
        f"<b>{err_m:.3f}</b></div></div>"
        f'<p class="cap">The answer was <strong>{"YES" if outcome else "NO"}</strong>. '
        "Bars show each side's squared error; shorter is better.</p>"
    )


def _pipeline(row: Row | None, rows: Sequence[Row]) -> str:
    buttons = "".join(
        f'<button class="stage-btn" role="tab" type="button" aria-selected="false">'
        f"<b>{name}</b><span>{line}</span></button>"
        for name, line, _ in STAGES
    )
    panels = "".join(
        f'<div class="stage" role="tabpanel"><h3>{name}</h3><p>{text}</p>'
        + (f'<div class="viz">{_stage_visual(i, row, rows)}</div>' if row else "")
        + "</div>"
        for i, (name, _, text) in enumerate(STAGES)
    )
    return (
        f'<div class="pipe"><div class="pipe-nav" role="tablist" aria-label="Steps">{buttons}</div>'
        f"<div>{panels}</div></div>"
    )


def _forecast_row(r: Row) -> str:
    q = r["question"]
    outcome = r["outcome"]
    market = r.get("market_prob")
    gap = abs(r["final_prob"] - market) if market is not None else 0.0
    when = (
        f"Closes {_day(q['close_time'])}"
        if outcome is None
        else f"Resolved {'YES' if outcome else 'NO'}"
    )
    samples = "".join(
        '<div class="sample">'
        f"<h4>{_esc(PERSONA_LABELS.get(s['persona'], s['persona']))}"
        f"<b>{_pct(s['probability'])}</b></h4>"
        f"<p>{_esc(s.get('rationale', ''))}</p></div>"
        for s in r.get("samples", [])
    )
    bet = r.get("bet", {})
    bet_text = (
        f"Paper bet: {str(bet['side']).upper()}, {bet.get('contracts', 0)} contracts"
        if bet.get("side")
        else "Paper bet: none"
    )
    meta = [bet_text, f"Ledger entry <code>{_esc(r['hash'][:10])}</code>", _market_link(q)]
    return (
        f'<details class="fc" data-state="{"open" if outcome is None else "resolved"}" '
        f'data-close="{_esc(q["close_time"])}" data-gap="{gap:.4f}">'
        "<summary>"
        f'<span class="fc-title">{_esc(q["title"])}</span>'
        f'<span class="fc-viz">{_duo(r["final_prob"], market)}</span>'
        f'<span class="fc-num"><b>{_pct(r["final_prob"])}</b> <span>vs {_pct(market)}</span></span>'
        f'<span class="fc-when">{when}</span>'
        "</summary>"
        '<div class="fc-body">'
        f"<p>Oracle alone said {_pct(r['calibrated_prob'])}. Blended with the market price of "
        f"{_pct(market)}, the logged forecast is {_pct(r['final_prob'])}.</p>"
        f'<div class="samples">{samples}</div>'
        f'<div class="meta">{"".join(f"<span>{m}</span>" for m in meta if m)}</div>'
        "</div></details>"
    )


def render_site(report: Report, rows: Sequence[Row]) -> str:
    featured = _featured(rows)
    n_open = report["n_open"]

    stats = [
        (f'<b data-count="{report["n_resolved"]}">{report["n_resolved"]:,}</b>', "resolved"),
        (f'<b data-count="{n_open}">{n_open:,}</b>', "open"),
        (f"<b>${report['pnl_usd']:+,.2f}</b>", f"paper P&amp;L, {report['n_bets']} bets"),
        (
            f'<b data-count="{report["total_cost_usd"]:.2f}" data-dec="2" data-pre="$">'
            f"${report['total_cost_usd']:,.2f}</b>",
            "API cost",
        ),
    ]
    stats_html = "".join(f'<div class="stat">{b}<span>{s}</span></div>' for b, s in stats)

    plot = (
        f'<div class="plot">{calibration_svg(report["calibration"])}</div>'
        if any(b["n"] for b in report["calibration"])
        else '<div class="blank">Appears once questions have resolved.</div>'
    )
    warning = ""
    if report["contamination"]["suspicious"]:
        warning = (
            '<p class="verdict">⚠️ Scores on questions resolved before the model\'s training '
            "cutoff are much better than after it. That pattern suggests the model remembers "
            "outcomes; those questions are excluded from the scores.</p>"
        )

    ordered = sorted(rows, key=lambda r: r["question"]["close_time"])
    forecasts = "".join(_forecast_row(r) for r in ordered)
    default = "open" if n_open else "all"
    if rows:
        listing = (
            '<div class="bar">'
            '<div class="group"><span>Show</span>'
            '<button class="chip" type="button" data-filter="open">Open</button>'
            '<button class="chip" type="button" data-filter="resolved">Resolved</button>'
            '<button class="chip" type="button" data-filter="all">All</button></div>'
            '<div class="group"><span>Sort</span>'
            '<button class="chip" type="button" data-sort="close">Closing soonest</button>'
            '<button class="chip" type="button" data-sort="gap">Biggest gap</button></div>'
            '<div class="key"><span><i class="o"></i>Oracle</span>'
            '<span><i class="m"></i>Market</span></div></div>'
            f'<div class="list" data-list data-default="{default}">{forecasts}</div>'
            '<p class="blank" data-empty hidden>Nothing here yet.</p>'
        )
    else:
        listing = '<p class="blank" style="margin-top:2rem">No open forecasts.</p>'

    how_sub = (
        "Follow one real forecast from the record through every step."
        if featured
        else "The same five steps run automatically every day."
    )
    models = ", ".join(_esc(m) for m in sorted(report["models"])) or "none yet"
    tabs = "".join(
        f'<button class="tab" role="tab" type="button" data-tab="{key}" '
        f'aria-selected="false">{label}</button>'
        for key, label in (
            ("overview", "Overview"),
            ("how", "How it works"),
            ("forecasts", "Forecasts"),
            ("verify", "Verify"),
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Oracle: AI forecasts vs prediction markets</title>
<meta name="description" content="An AI forecaster scored against Kalshi and Polymarket prices, with every forecast logged before the outcome is known.">
<style>{CSS}</style>
</head>
<body>
<header class="nav"><div class="wrap">
<span class="brand">Oracle</span>
<nav class="tabs" role="tablist" aria-label="Sections">{tabs}</nav>
</div></header>
<main class="wrap">

<section class="panel" data-panel="overview">
<div class="hero">
<div>
<h1>Can an AI beat the prediction markets?</h1>
<p class="lede">Oracle forecasts real questions daily, logs each one before the outcome is known,
and scores itself against market prices. Paper trading only.</p>
<div class="cta">
<a class="btn" href="#forecasts" data-goto="forecasts">See the forecasts</a>
<a class="btn ghost" href="#how" data-goto="how">How it works</a>
</div>
</div>
{_feature_card(featured)}
</div>
<p class="verdict">{_esc(headline(report))}</p>
{warning}
<div class="stats">{stats_html}</div>
<div class="calib">
{plot}
<div>
<h2>Is it honest about its odds?</h2>
<p class="muted">Of the questions Oracle gave about X%, how many happened? Dots on the dashed
line are perfectly calibrated; bigger dots hold more questions.</p>
</div>
</div>
</section>

<section class="panel" data-panel="how">
<h2>One forecast, start to finish</h2>
<p class="lede sub">{how_sub}</p>
{_pipeline(featured, rows)}
</section>

<section class="panel" data-panel="forecasts">
<h2>Every forecast on record</h2>
<p class="lede sub">Select a question to read what the AI found and why it landed where it did.</p>
{listing}
</section>

<section class="panel" data-panel="verify">
<h2>Check the record yourself</h2>
<div class="verify">
<div>
<p>Each forecast is one line of
<a href="{REPO_URL}/blob/main/data/ledger.jsonl"><code>data/ledger.jsonl</code></a>.
Every line carries the fingerprint of the line before it, so editing or deleting an old forecast
breaks the chain.</p>
<p>The <a href="{REPO_URL}/commits/main/data/ledger.jsonl">git history of that file</a> shows
when each forecast was committed, by GitHub's clock and not Oracle's.</p>
</div>
<div>
<pre><code>git clone {REPO_URL}.git
cd Oracle
pip install .
oracle verify</code></pre>
<p>The last command recomputes the whole chain and reports whether it is intact.</p>
</div>
</div>
</section>

</main>
<footer><div class="wrap">
<p>Model: {models}. Questions resolved before {report["model_cutoff"][:10]} are never scored.
Updated {report["generated_at"][:16].replace("T", " ")} UTC.</p>
<p>Not financial advice. No real money is traded; paper results ignore slippage and
market impact.</p>
</div></footer>
<script>{JS}</script>
</body>
</html>
"""


def write_site(report: Report, rows: Sequence[Row], path: str | Path = "docs/index.html") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_site(report, rows))
