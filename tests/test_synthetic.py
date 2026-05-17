"""Synthetic field generator tests."""

from __future__ import annotations

import numpy as np

from pulse_ep_decay import generate_synthetic_field
from pulse_ep_decay.synthetic import DEFAULT_AMPLITUDE, DEFAULT_BASELINE


def test_unreachable_get_nan() -> None:
    d = np.array([0.0, 5.0, 10.0, np.inf, np.inf])
    scores = generate_synthetic_field(d, sigma_true_mm=8.0)
    assert np.isnan(scores[3])
    assert np.isnan(scores[4])
    assert np.isfinite(scores[:3]).all()


def test_max_at_origin() -> None:
    rng = np.random.default_rng(0)
    d = np.linspace(0.0, 60.0, 200)
    scores = generate_synthetic_field(d, sigma_true_mm=10.0, noise_level=0.0, rng=rng)
    assert np.argmax(scores) == 0


def test_missing_fraction_respected() -> None:
    rng = np.random.default_rng(0)
    d = np.linspace(0.0, 60.0, 1000)
    scores = generate_synthetic_field(
        d, sigma_true_mm=10.0, missing_fraction=0.10, rng=rng
    )
    missing = int(np.isnan(scores).sum())
    # 10 % of 1000 reachable vertices should be NaN — allow small jitter.
    assert 80 <= missing <= 120


def test_score_range_clipped() -> None:
    rng = np.random.default_rng(0)
    d = np.linspace(0.0, 60.0, 500)
    scores = generate_synthetic_field(
        d, sigma_true_mm=8.0, noise_level=0.3, rng=rng,
        amplitude=DEFAULT_AMPLITUDE, baseline=DEFAULT_BASELINE,
    )
    finite = scores[np.isfinite(scores)]
    assert finite.min() >= 50.0
    assert finite.max() <= 100.0
