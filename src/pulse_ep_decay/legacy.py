"""Superseded estimators kept for the sensitivity analyses of the supplement.

None of this is part of the method (Algorithm 1 in :mod:`pulse_ep_decay.analysis`).
Each function exists because one supplementary table re-runs an earlier
variant against the final pipeline:

- :func:`find_origin_first` -- the submitted (pre-R1) tie rule: first vertex at
  the maximum (Supplementary Table S19, scripts v51/v51b);
- :func:`smoothed_argmax` -- origin as the vertex with the highest 3-mm
  neighbourhood mean, the alternative origin rule compared in Table S5
  (script v47a);
- :func:`fit_gengauss` and helpers -- four-parameter generalised-Gaussian fit
  used to characterise the shape of the R^2-flagged clinical profiles
  (script v43).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

# imports of pulse_ep_decay.profile / .support happen inside the functions: both
# modules re-export names from here for the study scripts, so a top-level import
# would be circular.

__all__ = [
    "find_origin_first",
    "smoothed_argmax",
    "ORIGIN_SMOOTH_RADIUS_MM",
    "fit_gengauss",
    "gengauss_model",
    "fwhm_from",
    "sigma_equiv_from_fwhm",
]


def find_origin_first(scores: np.ndarray) -> int:
    """Submitted / pre-R1 rule: first vertex (index order) at the maximum.
    Kept only for the tie sensitivity analysis."""
    return int(np.nanargmax(scores))


ORIGIN_SMOOTH_RADIUS_MM = 3.0


def smoothed_argmax(vertices, scores, radius: float = ORIGIN_SMOOTH_RADIUS_MM, triangles=None) -> int:
    """Origin (Eq. 1): the vertex whose `radius`-neighbourhood has the highest
    mean score (NaN ignored).  Averaging before the argmax removes
    single-vertex noise; on the synthetic support validation it reduces the
    origin offset from ~sigma/2 to ~sigma/8 and the sigma bias for
    sigma <= 15 mm.

    With `triangles` the neighbourhood is geodesic — vertices within `radius`
    along the mesh edges (Dijkstra on the edge graph), so that a fold or the
    opposite wall cannot enter it; without, it is the Euclidean ball.  On the
    clinical meshes the two neighbourhoods coincide within the support
    (graph/Euclidean distance ratio median 1.00-1.02 at 3 mm, no vertex of a
    Euclidean ball unreachable along the surface).
    """
    finite = np.isfinite(scores)
    cand = np.where(finite)[0]
    sc = scores[cand]
    if triangles is not None:
        from scipy.sparse.csgraph import dijkstra

        # edge graph restricted to the candidate vertices (paths through vertices
        # without a score, i.e. outside the field, are not needed within 3 mm)
        sel = np.zeros(len(vertices), bool)
        sel[cand] = True
        T = np.asarray(triangles)
        T = T[sel[T].all(axis=1)]
        local = -np.ones(len(vertices), int)
        local[cand] = np.arange(len(cand))
        from pulse_ep_decay.support import edge_graph

        G = edge_graph(vertices[cand], local[T])
        means = np.empty(len(cand))
        chunk = 500  # dijkstra returns a dense (sources x candidates) block; keep it small
        for a in range(0, len(cand), chunk):
            D = dijkstra(G, indices=np.arange(a, min(a + chunk, len(cand))), limit=radius)
            means[a : a + chunk] = [sc[np.isfinite(row)].mean() for row in D]
    else:
        tree = cKDTree(vertices[cand])
        nb = tree.query_ball_point(vertices[cand], radius)
        means = np.array([sc[n].mean() for n in nb])
    return int(cand[int(np.argmax(means))])


# ---------------------------------------------------------------------------
# Generalised Gaussian  S(d) = A exp(-(d/alpha)^beta) + B ;  FWHM = 2 alpha (ln 2)^(1/beta)
#   beta = 2 -> Gaussian (alpha = sqrt(2) sigma), beta = 1 -> exponential
# ---------------------------------------------------------------------------
BETA_BOUNDS = (0.3, 4.0)
LN2 = np.log(2.0)


def gengauss_model(d, A, alpha, beta, B):
    return A * np.exp(-((d / alpha) ** beta)) + B


def fwhm_from(alpha, beta):
    return 2.0 * alpha * LN2 ** (1.0 / beta)


def sigma_equiv_from_fwhm(fwhm):
    from pulse_ep_decay.profile import GAUSS_FWHM_FACTOR

    return fwhm / GAUSS_FWHM_FACTOR


def fit_gengauss(centres, means, alpha_init=None):
    """Four-parameter generalised-Gaussian fit on the binned profile. Start
    values: A0, alpha0 from a Gaussian pre-fit (alpha = sqrt(2) sigma),
    beta0 = 2 (Gaussian), B0 = min. Returns dict with A, alpha, beta, B, fwhm,
    r2, aic, converged."""
    from pulse_ep_decay.profile import _aic, _r2, fit_gaussian

    A0 = means.max() - means.min()
    B0 = means.min()
    if alpha_init is None:
        g = fit_gaussian(centres, means)
        alpha_init = np.sqrt(2.0) * g["sigma"] if g["converged"] and np.isfinite(g["sigma"]) else 10.0
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
