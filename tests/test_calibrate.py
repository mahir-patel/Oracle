from pathlib import Path

import numpy as np
import pytest

from forecaster.calibrate import MIN_POINTS, Calibrator, fit_calibrator


def _data(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts that are underconfident: truth is more extreme than p."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    logit = np.log(p / (1 - p))
    truth = 1 / (1 + np.exp(-2 * logit))
    y = (rng.random(n) < truth).astype(int)
    return p, y


def test_identity_below_the_gate() -> None:
    p, y = _data(MIN_POINTS - 1)
    cal = fit_calibrator(p, y)
    assert cal.is_identity
    assert cal.n == MIN_POINTS - 1
    for v in (0.0, 0.123, 0.5, 1.0):
        assert cal.apply(v) == v


def test_default_gate_is_1000() -> None:
    assert MIN_POINTS == 1000
    p, y = _data(1000)
    assert not fit_calibrator(p, y).is_identity


def test_empty_input_is_identity() -> None:
    assert fit_calibrator([], [], min_points=0).is_identity


def test_output_is_monotone_and_clipped() -> None:
    p, y = _data(2000)
    cal = fit_calibrator(p, y)
    grid = np.linspace(0, 1, 501)
    out = np.array([cal.apply(float(g)) for g in grid])
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0.01 and out.max() <= 0.99
    assert np.all(np.diff(cal.ys) >= 0)


def test_monotone_when_a_small_block_sits_below_a_large_one() -> None:
    """One lone 0 below a big block at rate 0.3: smoothing the lone point gives
    1/3 > 0.3, which must be pooled away rather than left as a dip."""
    p = [0.1] + [0.5] * 100
    y = [0] + [1] * 30 + [0] * 70
    cal = fit_calibrator(p, y, min_points=0)
    assert cal.apply(0.1) <= cal.apply(0.5)
    assert np.all(np.diff(cal.ys) >= 0)


def test_learns_the_direction_of_miscalibration() -> None:
    p, y = _data(5000)
    cal = fit_calibrator(p, y)
    assert cal.apply(0.8) > 0.85
    assert cal.apply(0.2) < 0.15
    assert cal.apply(0.5) == pytest.approx(0.5, abs=0.08)


def test_never_outputs_zero_or_one() -> None:
    cal = fit_calibrator([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], min_points=0)
    assert cal.apply(0.0) == pytest.approx(1 / 4)  # (0 + 1) / (2 + 2)
    assert cal.apply(1.0) == pytest.approx(3 / 4)


def test_repeated_forecast_values() -> None:
    cal = fit_calibrator([0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], min_points=0)
    assert cal.xs == [0.5]
    assert cal.apply(0.3) == pytest.approx(0.5)


def test_save_load_round_trip(tmp_path: Path) -> None:
    p, y = _data(1500)
    cal = fit_calibrator(p, y)
    path = tmp_path / "data" / "calibrator.json"
    cal.save(path)
    loaded = Calibrator.load(path)
    assert loaded == cal
    assert loaded.apply(0.37) == cal.apply(0.37)


def test_identity_round_trip_and_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "calibrator.json"
    assert Calibrator.load(path).is_identity
    Calibrator(n=12).save(path)
    loaded = Calibrator.load(path)
    assert loaded.is_identity and loaded.n == 12


def test_length_mismatch() -> None:
    with pytest.raises(ValueError):
        fit_calibrator([0.5], [1, 0])
