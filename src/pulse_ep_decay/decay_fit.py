"""Gaussian decay model and fitter for radial score profiles.

Model::

    f(d) = A * exp(-d^2 / (2 sigma^2)) + B

with ``d`` a geodesic distance from the score-maximum, ``A`` the
amplitude over baseline, ``B`` the baseline at large distance, and
``sigma`` (mm) the width parameter that serves as a physical-unit
spatial-resolution measure.

The fit uses distance-binned means rather than raw per-vertex scores,
which both stabilises the fit against per-vertex noise and matches the
pipeline used in the clinical analysis of the method paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "gaussian_decay",
    "DecayFitResult",
    "fit_gaussian_decay",
]


def gaussian_decay(d: np.ndarray, A: float, sigma: float, B: float) -> np.ndarray:
    """Gaussian-decay model ``A * exp(-d^2 / (2 sigma^2)) + B``."""
    return A * np.exp(-(d ** 2) / (2.0 * sigma ** 2)) + B


@dataclass(frozen=True)
class DecayFitResult:
    """Outcome of a Gaussian-decay fit on a radial profile.

    Attributes
    ----------
    sigma_mm : float
        Fitted width parameter (always non-negative).
    amplitude : float
        Fitted ``A`` (peak-over-baseline).
    baseline : float
        Fitted ``B`` (asymptotic score at large distance).
    r_squared : float
        Coefficient of determination of the fit on the binned profile.
    n_bins : int
        Number of distance bins that contributed to the fit.
    distance_bin_centres_mm : np.ndarray
        Bin-centre distances used for the fit (mm).
    binned_scores : np.ndarray
        Mean score per bin (same length as ``distance_bin_centres_mm``).
    """

    sigma_mm: float
    amplitude: float
    baseline: float
    r_squared: float
    n_bins: int
    distance_bin_centres_mm: np.ndarray
    binned_scores: np.ndarray

    def predict(self, distances: np.ndarray) -> np.ndarray:
        """Model prediction at arbitrary distances."""
        return gaussian_decay(
            np.asarray(distances), self.amplitude, self.sigma_mm, self.baseline
        )


def fit_gaussian_decay(
    distances: np.ndarray,
    scores: np.ndarray,
    *,
    bin_width_mm: float = 2.0,
    min_vertices_per_bin: int = 3,
    min_bins_for_fit: int = 4,
    sigma_bounds_mm: tuple[float, float] = (0.1, 200.0),
    max_evaluations: int = 5000,
) -> DecayFitResult | None:
    """Fit ``A·exp(-d²/(2σ²)) + B`` to a (distance, score) cloud.

    Bins the cloud into ``bin_width_mm`` rings of distance, averages
    scores per bin, and runs a non-linear least-squares fit on the
    binned profile. NaN and infinite values are filtered.

    Parameters
    ----------
    distances : (N,) array
        Geodesic distance of each vertex from the score-maximum (mm).
        ``inf`` and ``nan`` entries are skipped.
    scores : (N,) array
        Per-vertex scores (e.g. pace-mapping similarity, in %).
        ``nan`` entries are skipped.
    bin_width_mm : float, default 2.0
        Width of each distance bin.
    min_vertices_per_bin : int, default 3
        Bins with fewer contributing vertices are discarded.
    min_bins_for_fit : int, default 4
        Below this number of valid bins, the fit is not attempted and
        ``None`` is returned.
    sigma_bounds_mm : (lo, hi)
        Lower / upper bound for ``sigma`` in the optimiser.
    max_evaluations : int
        ``scipy.optimize.curve_fit`` ``maxfev``.

    Returns
    -------
    DecayFitResult or None
        ``None`` when the data is insufficient or the optimiser fails to
        converge.
    """
    d = np.asarray(distances, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)

    valid = np.isfinite(d) & ~np.isnan(s)
    d = d[valid]
    s = s[valid]
    if len(d) < 10:
        return None

    max_dist = float(d.max())
    bins = np.arange(0.0, max_dist + bin_width_mm, bin_width_mm)
    bin_idx = np.digitize(d, bins) - 1
    n_bins_total = len(bins) - 1

    bin_means = np.full(n_bins_total, np.nan)
    for k in range(n_bins_total):
        mask = bin_idx == k
        if np.sum(mask) >= min_vertices_per_bin:
            bin_means[k] = np.mean(s[mask])

    valid_bins = ~np.isnan(bin_means)
    if int(valid_bins.sum()) < min_bins_for_fit:
        return None

    centres = (np.arange(n_bins_total) + 0.5) * bin_width_mm
    d_fit = centres[valid_bins]
    s_fit = bin_means[valid_bins]

    A0 = float(s_fit.max() - s_fit.min())
    B0 = float(s_fit.min())
    try:
        popt, _ = curve_fit(
            gaussian_decay,
            d_fit,
            s_fit,
            p0=[A0, 10.0, B0],
            bounds=(
                [0.0, sigma_bounds_mm[0], -np.inf],
                [np.inf, sigma_bounds_mm[1], np.inf],
            ),
            maxfev=max_evaluations,
        )
    except (RuntimeError, ValueError):
        return None

    A_hat, sigma_hat, B_hat = popt
    residuals = s_fit - gaussian_decay(d_fit, *popt)
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((s_fit - np.mean(s_fit)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return DecayFitResult(
        sigma_mm=float(abs(sigma_hat)),
        amplitude=float(A_hat),
        baseline=float(B_hat),
        r_squared=float(r_squared),
        n_bins=int(valid_bins.sum()),
        distance_bin_centres_mm=d_fit,
        binned_scores=s_fit,
    )
