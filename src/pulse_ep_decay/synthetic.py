"""Synthetic radial score-field generator for σ-recovery validation.

Given a vector of geodesic distances from a (synthetic) true origin and a
prescribed σ₀, generates a noisy Gaussian field with realistic
amplitude / baseline characteristics matching the pace-mapping
distribution observed clinically.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "generate_synthetic_field",
    "DEFAULT_AMPLITUDE",
    "DEFAULT_BASELINE",
]

# Defaults match the clinical pace-mapping score distribution.
DEFAULT_BASELINE: float = 65.0  # %
DEFAULT_AMPLITUDE: float = 35.0  # percentage points above baseline


def generate_synthetic_field(
    geodesic_distances: np.ndarray,
    sigma_true_mm: float,
    *,
    noise_level: float = 0.0,
    missing_fraction: float = 0.0,
    amplitude: float = DEFAULT_AMPLITUDE,
    baseline: float = DEFAULT_BASELINE,
    clip_range: tuple[float, float] = (50.0, 100.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Per-vertex synthetic score field ``A·exp(-d²/(2σ₀²)) + B + ε``.

    Vertices whose geodesic distance is non-finite (i.e. unreachable
    from the chosen origin) receive ``np.nan``. A configurable fraction
    of reachable vertices is additionally set to ``np.nan`` to simulate
    measurement gaps.

    Parameters
    ----------
    geodesic_distances : (N,) array
        Distance from the synthetic origin to each mesh vertex (mm).
        Non-finite entries are treated as unreachable.
    sigma_true_mm : float
        Width parameter of the synthetic field (the ground truth that
        the fit should recover).
    noise_level : float, default 0.0
        Relative noise standard deviation, expressed as a fraction of
        ``amplitude``. ``0.05`` ≙ 5 % of amplitude as Gaussian noise.
    missing_fraction : float, default 0.0
        Fraction of reachable vertices to set to ``np.nan`` after noise
        application (simulating dropouts).
    amplitude, baseline : float
        Defaults match the clinical pace-mapping score distribution.
    clip_range : (lo, hi)
        Realistic clipping window for pace-mapping scores in %.
    rng : np.random.Generator, optional
        For reproducibility. If ``None``, a new default RNG is created.

    Returns
    -------
    (N,) array
        Per-vertex scores, with ``np.nan`` for unreachable or missing.
    """
    if rng is None:
        rng = np.random.default_rng()

    d = np.asarray(geodesic_distances, dtype=np.float64)
    n = len(d)
    scores = np.full(n, np.nan)

    reachable = np.isfinite(d)
    d_r = d[reachable]

    s = amplitude * np.exp(-(d_r**2) / (2.0 * sigma_true_mm**2)) + baseline

    if noise_level > 0:
        noise_std = noise_level * amplitude
        s = s + rng.normal(0.0, noise_std, size=len(s))

    s = np.clip(s, clip_range[0], clip_range[1])
    scores[reachable] = s

    if missing_fraction > 0:
        reach_idx = np.where(reachable)[0]
        n_missing = int(missing_fraction * len(reach_idx))
        if n_missing > 0:
            miss_idx = rng.choice(reach_idx, size=n_missing, replace=False)
            scores[miss_idx] = np.nan

    return scores
