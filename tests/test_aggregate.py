import math

import numpy as np
import pytest

from forecaster.aggregate import (
    EPS,
    blend_with_market,
    disagreement,
    extremize,
    fit_alpha_and_weight,
    logit,
    sigmoid,
    trimmed_mean_logodds,
)


@pytest.mark.parametrize("p", [0.01, 0.2, 0.5, 0.77, 0.99])
def test_logit_sigmoid_round_trip(p: float) -> None:
    assert sigmoid(logit(p)) == pytest.approx(p)


def test_logit_clips_and_sigmoid_is_stable() -> None:
    assert logit(0.0) == pytest.approx(logit(EPS))
    assert logit(1.0) == pytest.approx(logit(1 - EPS))
    assert sigmoid(-1000) == 0.0
    assert sigmoid(1000) == 1.0
    assert sigmoid(0) == 0.5


def test_trimmed_mean_ignores_an_outlier() -> None:
    assert trimmed_mean_logodds([0.3, 0.3, 0.3, 0.3, 0.99]) == pytest.approx(0.3)
    assert trimmed_mean_logodds([0.01, 0.3, 0.3, 0.3, 0.3]) == pytest.approx(0.3)


def test_trimmed_mean_is_a_logodds_mean() -> None:
    assert trimmed_mean_logodds([0.2, 0.8], trim=0.0) == pytest.approx(0.5)
    expected = sigmoid((logit(0.1) + logit(0.5)) / 2)
    assert trimmed_mean_logodds([0.1, 0.5], trim=0.0) == pytest.approx(expected)


def test_trimmed_mean_small_n_keeps_samples() -> None:
    assert trimmed_mean_logodds([0.4]) == pytest.approx(0.4)
    # n=2 with a heavy trim would drop everything, so nothing is dropped
    assert trimmed_mean_logodds([0.2, 0.8], trim=0.5) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        trimmed_mean_logodds([])


def test_extremize() -> None:
    assert extremize(0.7, 1.0) == pytest.approx(0.7)
    assert extremize(0.7, 1.5) > 0.7
    assert extremize(0.3, 1.5) < 0.3
    assert extremize(0.5, 2.0) == pytest.approx(0.5)
    assert extremize(0.7, 0.5) < 0.7


def test_blend_edge_cases() -> None:
    assert blend_with_market(0.3, 0.8, 0.0) == pytest.approx(0.3)
    assert blend_with_market(0.3, 0.8, 1.0) == pytest.approx(0.8)
    assert blend_with_market(0.3, None, 0.5) == 0.3
    assert blend_with_market(0.2, 0.8, 0.5) == pytest.approx(0.5)
    assert 0.3 < blend_with_market(0.3, 0.8, 0.25) < 0.8


def test_disagreement() -> None:
    assert disagreement([0.4]) == 0.0
    assert disagreement([0.4, 0.4, 0.4]) == pytest.approx(0.0)
    expected = np.std([logit(0.2), logit(0.8)])
    assert disagreement([0.2, 0.8]) == pytest.approx(expected)


def _world(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(0, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-true_logit))).astype(int)
    return true_logit, y


def test_fit_recovers_known_alpha() -> None:
    """Forecasts hedged by a factor 0.6 need alpha = 1/0.6 = 1.67 to undo it.

    The market here is pure noise, so the fit should give it no weight.
    """
    true_logit, y = _world(20000, seed=1)
    raw = 1 / (1 + np.exp(-0.6 * true_logit))
    noise_market = 1 / (1 + np.exp(-np.random.default_rng(9).normal(0, 1.5, len(y))))
    fit = fit_alpha_and_weight(raw, noise_market, y)
    assert fit.market_weight == 0.0
    assert fit.alpha == pytest.approx(1 / 0.6, abs=0.1)
    assert math.isfinite(fit.log_loss)


def test_fit_with_flat_market_recovers_effective_alpha() -> None:
    """A market stuck at 50% has logit 0, so only alpha * (1 - w) is identified."""
    true_logit, y = _world(20000, seed=1)
    raw = 1 / (1 + np.exp(-0.6 * true_logit))
    fit = fit_alpha_and_weight(raw, np.full(len(y), 0.5), y)
    assert fit.alpha * (1 - fit.market_weight) == pytest.approx(1 / 0.6, abs=0.1)


def test_fit_puts_weight_on_the_better_source() -> None:
    true_logit, y = _world(5000, seed=2)
    rng = np.random.default_rng(3)
    truth = 1 / (1 + np.exp(-true_logit))
    noise = 1 / (1 + np.exp(-rng.normal(0, 1.5, len(y))))
    assert fit_alpha_and_weight(noise, truth, y).market_weight >= 0.9
    fit = fit_alpha_and_weight(truth, noise, y)
    assert fit.market_weight <= 0.1
    assert fit.alpha == pytest.approx(1.0, abs=0.1)


def test_fit_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        fit_alpha_and_weight([], [], [])
    with pytest.raises(ValueError):
        fit_alpha_and_weight([0.5], [0.5, 0.5], [1])
