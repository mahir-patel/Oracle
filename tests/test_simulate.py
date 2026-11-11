from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from forecaster import simulate
from forecaster.config import Config
from forecaster.ledger import Ledger


def test_world_matches_the_spec() -> None:
    world = simulate.make_world(2000, seed=3)
    truth = np.array(list(world.truth.values()))
    mids = np.array([q.mid for q in world.questions])
    assert truth.min() >= 0.02 and truth.max() <= 0.98
    assert mids.min() >= 0.03 - 1e-9 and mids.max() <= 0.97 + 1e-9
    assert all(q.spread == pytest.approx(0.02) for q in world.questions)
    assert set(world.outcomes.values()) == {0, 1}
    # outcomes follow the true probabilities
    assert np.mean(list(world.outcomes.values())) == pytest.approx(truth.mean(), abs=0.03)
    # the market is informative but noisy
    assert 0.7 < np.corrcoef(truth, mids)[0, 1] < 0.99


def test_world_is_seeded_and_post_cutoff() -> None:
    a, b = simulate.make_world(50, seed=1), simulate.make_world(50, seed=1)
    assert a.questions == b.questions and a.outcomes == b.outcomes
    assert simulate.make_world(50, seed=2).truth != a.truth
    cutoff = Config().model_cutoff
    assert all(q.close_time > datetime(2026, 8, 1, tzinfo=UTC) > cutoff for q in a.questions)


def test_phase_two_forecasts_come_after_every_phase_one_resolution() -> None:
    """Chronology: the fit may only use outcomes known before phase 2 starts."""
    result = simulate.run(n=100, seed=1)
    world = simulate.make_world(100, seed=1)
    last_phase1_close = max(q.close_time for q in world.questions[:50])
    first_phase2_forecast = min(datetime.fromisoformat(r["ts"]) for r in result.phase2_rows)
    assert first_phase2_forecast > last_phase1_close


def test_run_writes_a_valid_ledger_at_the_given_path(tmp_path: Path) -> None:
    path = tmp_path / "sim.jsonl"
    result = simulate.run(n=100, seed=1, ledger_path=path)
    ledger = Ledger(path)
    assert result.ledger_ok and ledger.verify()
    assert result.n_records == len(ledger.records()) == 200  # forecast + resolution each
    assert result.phase1["n"] == result.phase2["n"] == 50
    # phase 1 uses neutral params: no extremizing, no market blend
    phase1 = ledger.joined()[:50]
    for r in phase1:
        assert r["calibrated_prob"] == pytest.approx(r["raw_prob"])
        assert r["final_prob"] == pytest.approx(r["raw_prob"])
    text = simulate.format_result(result)
    assert "Phase 2 (fitted)" in text and "hash chain OK" in text


def test_run_is_deterministic() -> None:
    a, b = simulate.run(n=200, seed=5), simulate.run(n=200, seed=5)
    assert a.fit == b.fit
    assert a.phase2["brier_final"] == b.phase2["brier_final"]
    assert simulate.run(n=200, seed=6).phase2["brier_final"] != a.phase2["brier_final"]


def test_default_run_does_not_touch_the_real_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    simulate.run(n=40, seed=1)
    assert not (tmp_path / "data").exists()


def test_acceptance_over_five_fixed_seeds() -> None:
    """SPEC 6.10. The seeds are fixed in the spec; never change them to make this pass."""
    assert simulate.ACCEPTANCE_SEEDS == (7, 8, 9, 10, 11)
    assert simulate.ACCEPTANCE_N == 1200
    acc = simulate.acceptance()
    assert len(acc.results) == 5
    assert all(r.phase2["n"] == 600 and r.ledger_ok for r in acc.results)
    assert acc.mean_brier_final < acc.mean_brier_model < acc.mean_brier_market
    assert acc.pooled_bss.n == 3000
    assert acc.pooled_bss.ci_low > 0
    assert acc.passed
    # the fit undoes the agent's hedging (hedge 0.7 -> alpha above 1) in every seed
    assert all(r.fit.alpha > 1 for r in acc.results)
    assert "ACCEPTANCE: PASS" in simulate.format_acceptance(acc)


def test_calibrator_ablation_does_not_help_on_small_histories() -> None:
    """SPEC 6.5: stacked on the fitted alpha, isotonic calibration makes
    out-of-sample Brier worse on average at n = 300 and 600."""
    rows = simulate.calibrator_ablation(train_sizes=(300, 600))
    assert len(rows) == 10
    for n_train in (300, 600):
        subset = [r for r in rows if r["n_train"] == n_train]
        assert np.mean([r["final_delta"] for r in subset]) > 0
        assert np.mean([r["model_delta"] for r in subset]) > 0
        assert sum(r["final_delta"] > 0 for r in subset) >= 4
