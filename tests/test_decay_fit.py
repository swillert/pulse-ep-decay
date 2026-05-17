"""Gaussian-decay fit tests."""

from __future__ import annotations

import numpy as np
import pytest

from pulse_ep_decay import (
    DEFAULT_AMPLITUDE,
    DEFAULT_BASELINE,
    fit_gaussian_decay,
    gaussian_decay,
)


def test_perfect_gaussian_recovers_sigma() -> None:
    """A clean A·exp(-d²/(2σ²)) + B field fits back to its parameters."""
    d = np.linspace(0, 60, 600)
    sigma_true = 8.0
    A_true = 35.0
    B_true = 65.0
    scores = gaussian_decay(d, A_true, sigma_true, B_true)

    fit = fit_gaussian_decay(d, scores)
    assert fit is not None
    assert fit.sigma_mm == pytest.approx(sigma_true, rel=0.01)
    assert fit.amplitude == pytest.approx(A_true, rel=0.02)
    assert fit.baseline == pytest.approx(B_true, rel=0.02)
    assert fit.r_squared > 0.99


def test_noisy_gaussian_within_reasonable_bias() -> None:
    """Adding 5 % relative noise should still recover σ within ±15 %."""
    rng = np.random.default_rng(123)
    d = np.linspace(0, 60, 4_000)
    sigma_true = 10.0
    noise = rng.normal(0.0, 0.05 * DEFAULT_AMPLITUDE, size=d.shape)
    scores = gaussian_decay(d, DEFAULT_AMPLITUDE, sigma_true, DEFAULT_BASELINE) + noise

    fit = fit_gaussian_decay(d, scores)
    assert fit is not None
    assert fit.sigma_mm == pytest.approx(sigma_true, rel=0.15)
    assert fit.r_squared > 0.85


def test_insufficient_data_returns_none() -> None:
    fit = fit_gaussian_decay(np.array([1.0, 2.0]), np.array([10.0, 20.0]))
    assert fit is None


def test_nans_and_infs_are_filtered() -> None:
    d = np.linspace(0, 60, 600)
    scores = gaussian_decay(d, 35.0, 8.0, 65.0)
    # Inject some non-finite entries; the fit should still succeed.
    d[5] = np.inf
    scores[10] = np.nan
    fit = fit_gaussian_decay(d, scores)
    assert fit is not None
    assert fit.sigma_mm == pytest.approx(8.0, rel=0.02)


def test_predict_round_trip() -> None:
    d = np.linspace(0, 60, 600)
    scores = gaussian_decay(d, 35.0, 8.0, 65.0)
    fit = fit_gaussian_decay(d, scores)
    assert fit is not None
    np.testing.assert_allclose(
        fit.predict(np.array([0.0, 8.0, 16.0])),
        gaussian_decay(np.array([0.0, 8.0, 16.0]), fit.amplitude, fit.sigma_mm, fit.baseline),
    )
