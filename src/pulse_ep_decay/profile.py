"""Radial decay profile: binning, Gaussian and alternative fits, synthetic fields.

The pipeline of the method paper:
the four-step pipeline origin -> geodesic distances -> 2 mm binning ->
three-parameter Gaussian fit, the generalised-Gaussian and exponential fits
with AIC, the origin refinement and the synthetic profiles of the validation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from pulse_ep_decay.mesh import (
    nearest_vertex,  # noqa: F401  (re-exported for callers of the old module)
)

# ---------------------------------------------------------------------------
# Pipeline parameters (must match the manuscript / run_full_validation.py)
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


def find_origin_first(scores: np.ndarray) -> int:
    """Submitted / pre-R1 rule: first vertex (index order) at the maximum.
    Kept only for the tie sensitivity analysis."""
    return int(np.nanargmax(scores))


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


def run_pipeline(scores: np.ndarray, dist_solver, origin: int | None = None):
    """Full pipeline on a score field. If `origin` is None it is found by
    the maximum with the plateau-centre tie rule (as in the manuscript,
    using the solver's mesh); otherwise the given vertex is used."""
    if origin is None:
        origin = find_origin(scores, dist_solver.vertices, dist_solver.triangles)
    d = dist_solver(origin)
    centres, means, counts = bin_profile(d, scores)
    g = fit_gaussian(centres, means)
    g.update(origin=origin, n_bins=len(centres))
    return g, (centres, means, counts)


GAUSS_FWHM_FACTOR = 2.354820045  # 2*sqrt(2 ln 2)


# ---------------------------------------------------------------------------
# Robust origin refinement (TODO 10.1)
# ---------------------------------------------------------------------------
def refine_origin(scores, vertices, dist_solver, n_iter=3, max_shift_mm=15.0):
    """Iterative origin refinement. After an initial argmax fit, the origin is
    re-estimated as the score-weighted centroid of all vertices within ~1 sigma
    (score >= B + A*exp(-1/2)) AND within min(2 sigma, 15 mm) geodesic distance of
    the current origin (a global centroid is pulled away by secondary regions on
    real maps), snapped to the nearest vertex; distances are
    recomputed and the profile refitted, until the origin stops moving.
    Returns (fit_result, origin)."""
    o = find_origin(scores, dist_solver.vertices, dist_solver.triangles)
    res, _ = run_pipeline(scores, dist_solver, origin=o)
    for _ in range(n_iter):
        if not res["converged"]:
            break
        thr = res["B"] + res["A"] * np.exp(-0.5)
        d_o = dist_solver(o)
        local = np.isfinite(d_o) & (
            d_o <= min(2.0 * res["sigma"], max_shift_mm)
        )  # local neighbourhood only
        cand = np.where((np.nan_to_num(scores, nan=-np.inf) >= thr) & local)[0]
        if len(cand) < 5:
            break
        w = scores[cand] - res["B"]
        w = np.clip(w, 1e-6, None)
        c = (vertices[cand] * w[:, None]).sum(0) / w.sum()
        o_new = nearest_vertex(vertices, c)
        if o_new == o:
            break
        o = o_new
        res, _ = run_pipeline(scores, dist_solver, origin=o)
    res["origin"] = o
    return res, o


# ---------------------------------------------------------------------------
# Pipeline v3: generalised Gaussian decay model (TODO 14, option C)
#   S(d) = A * exp(-(d/alpha)^beta) + B ;  FWHM = 2 alpha (ln 2)^(1/beta)
#   beta = 2 -> Gaussian (alpha = sqrt(2) sigma), beta = 1 -> exponential
# ---------------------------------------------------------------------------
BETA_BOUNDS = (0.3, 4.0)
LN2 = np.log(2.0)


def gengauss_model(d, A, alpha, beta, B):
    return A * np.exp(-((d / alpha) ** beta)) + B


def fwhm_from(alpha, beta):
    return 2.0 * alpha * LN2 ** (1.0 / beta)


def sigma_equiv_from_fwhm(fwhm):
    return fwhm / GAUSS_FWHM_FACTOR


def fit_gengauss(centres, means, alpha_init=None):
    """Four-parameter generalised-Gaussian fit on the binned profile. Start
    values: A0, alpha0 from a Gaussian pre-fit (alpha = sqrt(2) sigma),
    beta0 = 2 (Gaussian), B0 = min. Returns dict with A, alpha, beta, B, fwhm,
    r2, aic, converged."""
    A0 = means.max() - means.min()
    B0 = means.min()
    if alpha_init is None:
        g = fit_gaussian(centres, means)
        alpha_init = (
            np.sqrt(2.0) * g["sigma"] if g["converged"] and np.isfinite(g["sigma"]) else 10.0
        )
    alpha_init = float(np.clip(alpha_init, 0.5, 250))
    best = None
    for beta0 in (2.0, 1.0):  # two starts: Gaussian and exponential
        try:
            popt, _ = curve_fit(
                gengauss_model,
                centres,
                means,
                p0=[A0, alpha_init, beta0, B0],
                bounds=(
                    [-np.inf, 0.1, BETA_BOUNDS[0], -np.inf],
                    [np.inf, 300.0, BETA_BOUNDS[1], np.inf],
                ),
                maxfev=20000,
            )
            yhat = gengauss_model(centres, *popt)
            rss = float(np.sum((means - yhat) ** 2))
            if best is None or rss < best[0]:
                best = (rss, popt, yhat)
        except Exception:
            continue
    if best is None:
        return dict(
            A=np.nan,
            alpha=np.nan,
            beta=np.nan,
            B=np.nan,
            fwhm=np.nan,
            r2=np.nan,
            aic=np.nan,
            converged=False,
        )
    _, p, yhat = best
    A, alpha, beta, B = p
    return dict(
        A=A,
        alpha=alpha,
        beta=beta,
        B=B,
        fwhm=fwhm_from(alpha, beta),
        r2=_r2(means, yhat),
        aic=_aic(means, yhat, 4),
        converged=True,
    )


def refine_origin_v3(scores, vertices, dist_solver, n_iter=3, max_shift_mm=15.0):
    """Origin refinement for the v3 pipeline: after each generalised-Gaussian
    fit, the origin is moved to the score-weighted centroid of the vertices
    above half maximum (score >= B + A/2) within min(FWHM, max_shift) geodesic
    distance of the current origin, snapped to the nearest vertex."""
    o = find_origin(scores, dist_solver.vertices, dist_solver.triangles)
    d = dist_solver(o)
    c, m, _ = bin_profile(d, scores)
    res = fit_gengauss(c, m)
    for _ in range(n_iter):
        if not res["converged"] or not np.isfinite(res["fwhm"]):
            break
        thr = res["B"] + res["A"] / 2.0
        local = np.isfinite(d) & (d <= min(res["fwhm"], max_shift_mm))
        cand = np.where((np.nan_to_num(scores, nan=-np.inf) >= thr) & local)[0]
        if len(cand) < 5:
            break
        w = np.clip(scores[cand] - res["B"], 1e-6, None)
        cen = (vertices[cand] * w[:, None]).sum(0) / w.sum()
        o_new = nearest_vertex(vertices, cen)
        if o_new == o:
            break
        o = o_new
        d = dist_solver(o)
        c, m, _ = bin_profile(d, scores)
        res = fit_gengauss(c, m)
    res["origin"] = o
    res["bins"] = (c, m)
    return res


def bootstrap_gengauss(centres, means, n_boot=500, seed=0):
    """Bin-resampling bootstrap; returns 95 % CI of FWHM and beta."""
    rng = np.random.default_rng(seed)
    n = len(centres)
    fw, be = [], []
    for _ in range(n_boot):
        idx = np.unique(rng.integers(0, n, n))
        if len(idx) < 5:
            continue
        g = fit_gengauss(centres[idx], means[idx])
        if g["converged"] and np.isfinite(g["fwhm"]):
            fw.append(g["fwhm"])
            be.append(g["beta"])
    fw, be = np.array(fw), np.array(be)
    if len(fw) < 20:
        return (np.nan, np.nan, np.nan, np.nan, len(fw))
    return (
        np.percentile(fw, 2.5),
        np.percentile(fw, 97.5),
        np.percentile(be, 2.5),
        np.percentile(be, 97.5),
        len(fw),
    )


def synth_gengauss(distances, fwhm, beta, noise, rng):
    """Synthetic field with generalised-Gaussian truth (A=35, B=65, clip, mask)."""
    alpha = fwhm / (2.0 * LN2 ** (1.0 / beta))
    finite = np.isfinite(distances)
    d = np.where(finite, distances, 0.0)
    s = SYNTH_A * np.exp(-((d / alpha) ** beta)) + SYNTH_B
    if noise > 0:
        s = s + rng.normal(0, noise * SYNTH_A, size=s.shape)
    s = np.clip(s, *SYNTH_CLIP)
    s[~finite] = np.nan
    s[rng.random(len(s)) < SYNTH_MASK_FRACTION] = np.nan
    return s
