"""End-to-end σ-recovery tests on a synthetic sphere.

These are the acceptance tests for the method paper: given a known σ₀,
the pipeline must recover σ̂ with small bias under noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from pulse_ep_decay import recover_sigma, run_sigma_recovery

pytest.importorskip("potpourri3d")


def test_recover_sigma_on_clean_field(synthetic_sphere) -> None:
    """Clean synthetic Gaussian on a sphere → σ̂ matches σ₀ within ~10 %."""
    vertices, triangles = synthetic_sphere

    # Place a Gaussian centred on vertex 0 with σ₀ = 6 mm.
    from pulse_ep_decay import generate_synthetic_field, heat_distances

    d = heat_distances(vertices, triangles, origin_idx=0)
    sigma_true = 6.0
    scores = generate_synthetic_field(d, sigma_true_mm=sigma_true, noise_level=0.0)

    fit = recover_sigma(vertices, triangles, scores, method="heat")
    assert fit is not None
    assert fit.sigma_mm == pytest.approx(sigma_true, rel=0.10)
    assert fit.r_squared > 0.95


def test_run_sigma_recovery_smoke(synthetic_sphere) -> None:
    """A 1-σ x 1-noise x 3-trial sweep returns the expected number of structured results."""
    vertices, triangles = synthetic_sphere
    results = run_sigma_recovery(
        vertices,
        triangles,
        sigma_true_mm=[8.0],
        noise_levels=[0.05],
        n_trials=3,
        method="heat",
        seed=7,
    )
    assert len(results) == 3
    for r in results:
        assert r.sigma_true_mm == 8.0
        assert r.noise_level == 0.05
        assert r.true_origin >= 0
        if r.fit is not None:
            assert r.sigma_hat_mm is not None
            assert r.bias_mm is not None
