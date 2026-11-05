"""Radial decay profile: origin rule, 2-mm binning, Gaussian and exponential fits,
synthetic fields.

Steps 1, 3 and 4 of the method (Algorithm 1): :func:`find_origin` (maximum with
the plateau-centre tie rule), :func:`bin_profile` (2-mm bins, at least three
vertices), :func:`fit_gaussian` (three parameters, sigma in [0.1, 200] mm) and
:func:`fit_exponential` for the AIC comparison.  :func:`synth_field` builds
the synthetic score fields of the validation (Gaussian and the non-Gaussian
truths of Table S6).  Superseded variants live in :mod:`pulse_ep_decay.legacy`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Pipeline parameters (must match the manuscript)
# ---------------------------------------------------------------------------
BIN_WIDTH_MM = 2.0
MIN_VERTICES_PER_BIN = 3
SIGMA_INIT_MM = 10.0
SIGMA_BOUNDS_MM = (0.1, 200.0)
SYNTH_A = 35.0
SYNTH_B = 65.0
SYNTH_CLIP = (50.0, 100.0)
SYNTH_MASK_FRACTION = 0.05


# ---------------------------------------------------------------------------
# Synthetic fields
# ---------------------------------------------------------------------------
def gaussian_profile(d, sigma):
    return np.exp(-(d**2) / (2 * sigma**2))


def exponential_profile(d, fwhm):
    # exp(-d/lambda): half maximum at d = lambda*ln2  -> FWHM = 2*lambda*ln2
    lam = fwhm / (2 * np.log(2))
    return np.exp(-d / lam)


def lorentzian_profile(d, fwhm):
    gamma = fwhm / 2
    return 1.0 / (1.0 + (d / gamma) ** 2)


def raised_cosine_profile(d, fwhm):
    # 0.5*(1+cos(pi d / R)) for d<R, zero beyond; half max at d=R/2 -> FWHM=R
    R = fwhm
    p = 0.5 * (1 + np.cos(np.pi * np.clip(d, 0, R) / R))
    p[d >= R] = 0.0
    return p


PROFILES = {
    "gaussian": lambda d, fwhm: gaussian_profile(d, fwhm / 2.354820045),
    "exponential": exponential_profile,
    "lorentzian": lorentzian_profile,
    "raised_cosine": raised_cosine_profile,
}


def synth_field(
    distances: np.ndarray, profile: str, fwhm: float, noise: float, rng: np.random.Generator
) -> np.ndarray:
    """Synthetic score field, identical noise/clip/mask protocol as the
    manuscript (A=35, B=65, N(0,(eta*A)^2), clip [50,100], 5 % masked)."""
    finite = np.isfinite(distances)
    d = np.where(finite, distances, 0.0)
    s = SYNTH_A * PROFILES[profile](d, fwhm) + SYNTH_B
    if noise > 0:
        s = s + rng.normal(0, noise * SYNTH_A, size=s.shape)
    s = np.clip(s, *SYNTH_CLIP)
    s[~finite] = np.nan
    mask = rng.random(len(s)) < SYNTH_MASK_FRACTION
    s[mask] = np.nan
    return s


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def find_origin(scores: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> int:
    """Step 1: origin = vertex of maximum score.

    Tie rule (R1, Oct 2026): if several vertices share the maximum score
    (integer scores, or synthetic fields clipped at 100), the tied vertices
    are grouped into connected regions on the mesh edge graph; the largest
    region is taken (on equal size, the one containing the lowest vertex
    index) and the origin is its vertex nearest to the region's centroid
    ("plateau centre").  A unique maximum is returned unchanged.
    """
    s = np.asarray(scores, dtype=float)
    mx = np.nanmax(s)
    tied = np.flatnonzero(s == mx)
    if len(tied) == 1:
        return int(tied[0])
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    V = np.asarray(vertices, dtype=float)
    T = np.asarray(faces, dtype=int)
    e = np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    is_tied = np.zeros(len(s), bool)
    is_tied[tied] = True
    e = e[is_tied[e[:, 0]] & is_tied[e[:, 1]]]
    pos = -np.ones(len(s), int)
    pos[tied] = np.arange(len(tied))
    ee = pos[e]
    adj = coo_matrix((np.ones(len(ee)), (ee[:, 0], ee[:, 1])), shape=(len(tied), len(tied)))
    _, lab = connected_components(adj, directed=False)
    big = tied[lab == np.bincount(lab).argmax()]
    return int(big[np.argmin(np.linalg.norm(V[big] - V[big].mean(0), axis=1))])


def bin_profile(distances: np.ndarray, scores: np.ndarray):
    """Step 3: 2 mm binning over the full geodesic range, bins with < 3
    vertices excluded. Returns (bin_centres, mean_scores, counts)."""
    ok = np.isfinite(distances) & np.isfinite(scores)
    d, s = np.clip(distances[ok], 0, None), scores[ok]  # heat method may yield -1e-9 at source
    k = np.floor(d / BIN_WIDTH_MM).astype(int)
    nbins = k.max() + 1
    counts = np.bincount(k, minlength=nbins)
    sums = np.bincount(k, weights=s, minlength=nbins)
    valid = counts >= MIN_VERTICES_PER_BIN
    centres = (np.arange(nbins) + 0.5) * BIN_WIDTH_MM
    return centres[valid], sums[valid] / counts[valid], counts[valid]


def gauss_model(d, A, sigma, B):
    return A * np.exp(-(d**2) / (2 * sigma**2)) + B


def exp_model(d, A, lam, B):
    return A * np.exp(-d / lam) + B


def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _aic(y, yhat, k):
    n = len(y)
    rss = np.sum((y - yhat) ** 2)
    return n * np.log(rss / n) + 2 * k


def fit_gaussian(centres, means):
    """Step 4: identical start values and bounds as the manuscript."""
    A0 = means.max() - means.min()
    B0 = means.min()
    p0 = [A0, SIGMA_INIT_MM, B0]
    bounds = ([-np.inf, SIGMA_BOUNDS_MM[0], -np.inf], [np.inf, SIGMA_BOUNDS_MM[1], np.inf])
    try:
        popt, _ = curve_fit(gauss_model, centres, means, p0=p0, bounds=bounds, maxfev=20000)
        yhat = gauss_model(centres, *popt)
        return dict(
            A=popt[0],
            sigma=popt[1],
            B=popt[2],
            r2=_r2(means, yhat),
            aic=_aic(means, yhat, 3),
            converged=True,
        )
    except Exception:
        return dict(A=np.nan, sigma=np.nan, B=np.nan, r2=np.nan, aic=np.nan, converged=False)


def fit_exponential(centres, means):
    """Alternative model for AIC comparison (same protocol, lambda instead
    of sigma)."""
    A0 = means.max() - means.min()
    B0 = means.min()
    p0 = [A0, SIGMA_INIT_MM, B0]
    bounds = ([-np.inf, SIGMA_BOUNDS_MM[0], -np.inf], [np.inf, SIGMA_BOUNDS_MM[1], np.inf])
    try:
        popt, _ = curve_fit(exp_model, centres, means, p0=p0, bounds=bounds, maxfev=20000)
        yhat = exp_model(centres, *popt)
        return dict(
            A=popt[0],
            lam=popt[1],
            B=popt[2],
            r2=_r2(means, yhat),
            aic=_aic(means, yhat, 3),
            converged=True,
        )
    except Exception:
        return dict(A=np.nan, lam=np.nan, B=np.nan, r2=np.nan, aic=np.nan, converged=False)


GAUSS_FWHM_FACTOR = 2.354820045  # 2*sqrt(2 ln 2)

# kept importable from here for the study scripts (v43); see pulse_ep_decay.legacy
from pulse_ep_decay.legacy import find_origin_first, fit_gengauss, fwhm_from  # noqa: E402, F401
