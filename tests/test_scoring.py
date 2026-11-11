import math
from datetime import UTC, datetime

import numpy as np
import pytest

from forecaster.scoring import (
    bootstrap_bss,
    brier,
    brier_skill_vs,
    calibration_table,
    contamination_check,
    log_loss,
)


def test_brier_known_values() -> None:
    assert brier([1.0, 0.0], [1, 0]) == 0.0
    assert brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert brier([0.8, 0.3], [1, 0]) == pytest.approx((0.04 + 0.09) / 2)
    assert brier([0.0], [1]) == 1.0


def test_log_loss_known_values() -> None:
    assert log_loss([0.5, 0.5], [1, 0]) == pytest.approx(math.log(2))
    assert log_loss([0.8], [1]) == pytest.approx(-math.log(0.8))
    assert log_loss([0.8], [0]) == pytest.approx(-math.log(0.2))
    # clipped, so a confident miss is large but finite
    assert log_loss([0.0], [1]) == pytest.approx(-math.log(1e-4))


def test_empty_or_mismatched_input_raises() -> None:
    with pytest.raises(ValueError):
        brier([], [])
    with pytest.raises(ValueError):
        log_loss([0.5], [1, 0])


def test_bss_sign() -> None:
    y = [1, 0, 1, 0]
    good, bad = [0.9, 0.1, 0.8, 0.2], [0.6, 0.4, 0.5, 0.5]
    assert brier_skill_vs(good, bad, y) > 0
    assert brier_skill_vs(bad, good, y) < 0
    assert brier_skill_vs(good, good, y) == pytest.approx(0.0)
    assert brier_skill_vs([0.5, 0.5], [0.0, 1.0], [1, 0]) == pytest.approx(0.75)


def _sim(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < truth).astype(int)
    market = np.clip(truth + rng.normal(0, 0.2, n), 0.01, 0.99)
    return truth, market, y


def test_bootstrap_ci_contains_the_point_estimate() -> None:
    p, market, y = _sim(400, seed=1)
    result = bootstrap_bss(p, market, y)
    assert result is not None
    assert result.n == 400
    assert result.bss == pytest.approx(brier_skill_vs(p, market, y))
    assert result.ci_low < result.bss < result.ci_high
    assert result.bss > 0 and result.p_beat_market > 0.95


def test_bootstrap_is_deterministic_and_symmetric() -> None:
    p, market, y = _sim(200, seed=2)
    assert bootstrap_bss(p, market, y) == bootstrap_bss(p, market, y)
    worse = bootstrap_bss(market, p, y)
    assert worse is not None and worse.bss < 0 and worse.p_beat_market < 0.05


def test_bootstrap_needs_ten_rows() -> None:
    p, market, y = _sim(9, seed=3)
    assert bootstrap_bss(p, market, y) is None
    p, market, y = _sim(10, seed=3)
    assert bootstrap_bss(p, market, y) is not None


def test_calibration_table() -> None:
    table = calibration_table([0.05, 0.05, 0.55, 0.95, 1.0], [0, 1, 1, 1, 1])
    assert len(table) == 10
    assert [row["n"] for row in table] == [2, 0, 0, 0, 0, 1, 0, 0, 0, 2]
    assert table[0]["mean_forecast"] == pytest.approx(0.05)
    assert table[0]["observed"] == pytest.approx(0.5)
    assert table[1]["mean_forecast"] is None and table[1]["observed"] is None
    assert table[9]["mean_forecast"] == pytest.approx(0.975)
    assert (table[3]["lo"], table[3]["hi"]) == (0.3, 0.4)
    assert sum(row["n"] for row in calibration_table([], [])) == 0


CUTOFF = datetime(2026, 7, 1, tzinfo=UTC)
BEFORE = datetime(2026, 5, 1, tzinfo=UTC)
AFTER = datetime(2026, 9, 1, tzinfo=UTC)


def test_contamination_flag_when_pre_cutoff_is_too_good() -> None:
    # near-perfect before the cutoff, mediocre after
    p = [0.99, 0.01, 0.6, 0.4]
    y = [1, 0, 1, 0]
    result = contamination_check(p, y, [BEFORE, BEFORE, AFTER, AFTER], CUTOFF)
    assert result["n_pre"] == 2 and result["n_post"] == 2
    assert result["brier_pre"] == pytest.approx(0.0001)
    assert result["brier_post"] == pytest.approx(0.16)
    assert result["suspicious"] is True


def test_contamination_not_flagged_when_similar() -> None:
    p = [0.6, 0.4, 0.6, 0.4]
    y = [1, 0, 1, 0]
    result = contamination_check(p, y, [BEFORE, BEFORE, AFTER, AFTER], CUTOFF)
    assert result["suspicious"] is False


def test_contamination_threshold_is_three_quarters() -> None:
    # pre Brier 0.09, post Brier 0.16: ratio 0.5625 < 0.75 -> flagged
    flagged = contamination_check([0.7, 0.6], [1, 1], [BEFORE, AFTER], CUTOFF)
    assert flagged["suspicious"] is True
    # pre Brier 0.1444, post 0.16: ratio 0.9 -> fine
    fine = contamination_check([0.62, 0.6], [1, 1], [BEFORE, AFTER], CUTOFF)
    assert fine["suspicious"] is False


def test_contamination_needs_both_sides() -> None:
    result = contamination_check([0.9], [1], [AFTER], CUTOFF)
    assert result["brier_pre"] is None and result["suspicious"] is False
    # a question resolved exactly at the cutoff counts as post-cutoff
    assert contamination_check([0.9], [1], [CUTOFF], CUTOFF)["n_post"] == 1
